import json
import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gliner import GLiNER

from kgg.config import KGGConfig
from kgg.models import Document
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator
from functools import lru_cache

@lru_cache(maxsize=1)
def get_gliner_model(labels: tuple[str, ...]):
    """Load GLiNER exactly once and return the HF model(+tokenizer)."""
    m = GLiNER.from_pretrained("urchade/gliner_large-v2.1").to('cuda')
    return m

def test_gliner_with_different_chunk_sizes(text, ner_labels, verbose=True):
    """
    Comprehensive evaluation of GLiNER with different chunking strategies.
    Tests a range of chunk sizes and overlap percentages to find optimal configuration.
    """
    if verbose:
        print(f"Testing GLiNER with {len(ner_labels)} labels and text length: {len(text)}")

    # Initialize base config
    config = KGGConfig()
    config.ner_labels = ner_labels
    config.ner_threshold = 0.5

    # Define model's maximum sequence length (from warning message)
    model_max_length = 384

    # Test with more granular chunk sizes up to the model's maximum length

    chunk_sizes = [96, 224, 384]


    overlap_percentages = [10, 30, 50]

    results = {}

    # Create document
    doc_id = "test_doc_001"
    document = Document(id=doc_id, text=text)

    # Define metrics for ranking configurations
    metrics = {
        "unique_entities": [],
        "entity_quality_score": [],
        "execution_time": [],
        "chunks": [],
        "chunk_size": [],
        "overlap": [],
        "config_name": []
    }

    # Suppress specific warnings during testing
    warnings.filterwarnings("ignore", message="Sentence of length .* has been truncated to .*")
    warnings.filterwarnings("ignore", message="Asking to truncate to max_length but no maximum length is provided.*")

    # First run a baseline with no chunking (if text is small enough)
    # This will serve as our "ground truth" for comparison
    baseline_entities = []
    baseline_run = False
    shared_model = get_gliner_model(tuple(config.ner_labels))


    # Only run baseline if the text isn't too large - max 10,000 chars to avoid memory issues
    if len(text) <= 10000:
        try:
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Running baseline (no chunking) for comparison")
                print(f"{'-' * 80}")

            generator = GLiNEREntitiesGenerator(config)
            generator.model = shared_model
            start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Try to process the whole document at once
            baseline_entities = generator.model.predict_entities(
                text[:model_max_length],  # Limited to model's max length
                config.ner_labels,
                threshold=config.ner_threshold,
                multi_label=True
            )

            baseline_time = time.time() - start_time
            baseline_run = True

            if verbose:
                print(f"Baseline run complete: found {len(baseline_entities)} entities in {baseline_time:.2f} seconds")
        except Exception as e:
            if verbose:
                print(f"Baseline run failed: {str(e)}")
            baseline_entities = []

    # Now run the chunking tests
    for chunk_size in chunk_sizes:
        for overlap_pct in overlap_percentages:
            # Calculate overlap size as percentage of chunk size
            overlap = int(chunk_size * overlap_pct / 100)

            if overlap >= chunk_size:
                continue  # Skip invalid configurations

            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Testing with chunk_size={chunk_size}, overlap={overlap} ({overlap_pct}% of chunk size)")
                print(f"{'-' * 80}")

            # Initialize generator with current settings
            generator = GLiNEREntitiesGenerator(config)
            generator.model = shared_model
            generator.chunk_size = chunk_size
            generator.overlap = overlap

            # Measure performance
            start_time = time.time()

            # Process document
            chunks = generator._split_into_chunks(document.text)
            if verbose:
                print(f"Document split into {len(chunks)} chunks")

            all_entities = []
            processed_chunks = 0
            skipped_chunks = 0

            # Process each chunk
            for i, (chunk_text, offset) in enumerate(chunks):
                if verbose and i % 10 == 0:  # Progress indicator
                    print(f"Processing chunk {i + 1}/{len(chunks)}")


                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    # Get entities for this chunk
                    chunk_entities = generator.model.predict_entities(
                        chunk_text,
                        config.ner_labels,
                        threshold=config.ner_threshold,
                        multi_label=True
                    )

                    # Adjust offsets
                    for entity in chunk_entities:
                        entity["start"] += offset
                        entity["end"] += offset
                        # Add source chunk info for analysis
                        entity["source_chunk"] = i

                    all_entities.extend(chunk_entities)
                    processed_chunks += 1

                except Exception as e:
                    if verbose:
                        print(f"Error processing chunk {i + 1}: {str(e)}")
                    skipped_chunks += 1
                    continue

            # Remove duplicates but keep track of how many duplicates were found
            pre_dedup_count = len(all_entities)
            unique_entities = generator._remove_duplicate_entities(all_entities)
            duplication_rate = (pre_dedup_count - len(unique_entities)) / pre_dedup_count if pre_dedup_count > 0 else 0

            # Calculate entity density (entities per token)
            tokens_processed = processed_chunks * chunk_size
            entity_density = len(unique_entities) / tokens_processed if tokens_processed > 0 else 0

            # Calculate time
            elapsed_time = time.time() - start_time

            # Calculate entity quality score - evaluate consistency between chunks
            # This score rewards configurations that find similar entities across chunk boundaries
            entity_quality_score = 0

            # Count entities by type and text to analyze consistency
            entity_counter = Counter()
            entity_positions = {}

            for entity in unique_entities:
                entity_key = f"{entity['label']}:{entity['text']}"
                entity_counter[entity_key] += 1

                # Track positions for entity distribution analysis
                if entity_key not in entity_positions:
                    entity_positions[entity_key] = []
                entity_positions[entity_key].append(entity['start'])

            # Reward consistently found entities (across multiple chunks)
            for entity_key, count in entity_counter.items():
                if count > 1:
                    entity_quality_score += np.log1p(count)  # logarithmic scaling

            # Analyze entity distribution - penalize if entities are only found in certain regions
            text_regions = 10  # Divide text into 10 regions
            region_size = len(text) / text_regions
            region_counts = [0] * text_regions

            for entity in unique_entities:
                region = min(int(entity['start'] / region_size), text_regions - 1)
                region_counts[region] += 1

            # Calculate distribution evenness (normalized entropy)
            total_regions_with_entities = sum(1 for count in region_counts if count > 0)
            distribution_score = total_regions_with_entities / text_regions

            # Combine metrics for final quality score
            entity_quality_score = entity_quality_score * (0.5 + 0.5 * distribution_score)

            # Store results
            key = f"chunk={chunk_size}, overlap={overlap}"
            results[key] = {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "overlap_pct": overlap_pct,
                "total_chunks": len(chunks),
                "processed_chunks": processed_chunks,
                "skipped_chunks": skipped_chunks,
                "total_entities_found": pre_dedup_count,
                "unique_entities": len(unique_entities),
                "duplication_rate": duplication_rate,
                "entity_density": entity_density,
                "entity_quality_score": entity_quality_score,
                "execution_time": elapsed_time,
                "entities_per_label": {},
                "baseline_comparison": None,
            }

            # If we have baseline results, compare against them
            if baseline_run:
                # Calculate overlap with baseline
                baseline_entity_keys = {f"{e['label']}:{e['text']}" for e in baseline_entities}
                test_entity_keys = {f"{e['label']}:{e['text']}" for e in unique_entities}

                # Find matches and unique entities
                matches = baseline_entity_keys.intersection(test_entity_keys)
                missed = baseline_entity_keys - test_entity_keys
                new_found = test_entity_keys - baseline_entity_keys

                # Calculate precision, recall, F1 against baseline
                precision = len(matches) / len(test_entity_keys) if test_entity_keys else 0
                recall = len(matches) / len(baseline_entity_keys) if baseline_entity_keys else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results[key]["baseline_comparison"] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "matches": len(matches),
                    "missed": len(missed),
                    "new_found": len(new_found)
                }

            # Store metrics for ranking
            metrics["unique_entities"].append(len(unique_entities))
            metrics["entity_quality_score"].append(entity_quality_score)
            metrics["execution_time"].append(elapsed_time)
            metrics["chunks"].append(processed_chunks)
            metrics["chunk_size"].append(chunk_size)
            metrics["overlap"].append(overlap)
            metrics["config_name"].append(key)

            # Count entities per label
            for entity in unique_entities:
                label = entity["label"]
                if label not in results[key]["entities_per_label"]:
                    results[key]["entities_per_label"][label] = 0
                results[key]["entities_per_label"][label] += 1

            # Print summary for this configuration
            if verbose:
                print(f"\nSummary for chunk_size={chunk_size}, overlap={overlap} ({overlap_pct}%):")
                print(f"  Total chunks: {len(chunks)}")
                print(f"  Processed chunks: {processed_chunks}")
                print(f"  Skipped chunks: {skipped_chunks}")
                print(f"  Total entities found: {pre_dedup_count}")
                print(f"  Unique entities after deduplication: {len(unique_entities)}")
                print(f"  Duplication rate: {duplication_rate:.2%}")
                print(f"  Entity density: {entity_density:.6f} entities per token")
                print(f"  Entity quality score: {entity_quality_score:.2f}")
                print(f"  Execution time: {elapsed_time:.2f} seconds")

                if baseline_run and results[key]["baseline_comparison"]:
                    bc = results[key]["baseline_comparison"]
                    print(f"  Baseline comparison:")
                    print(f"    Precision: {bc['precision']:.2f}")
                    print(f"    Recall: {bc['recall']:.2f}")
                    print(f"    F1: {bc['f1']:.2f}")
                    print(f"    Matches: {bc['matches']}")
                    print(f"    Missed: {bc['missed']}")
                    print(f"    New found: {bc['new_found']}")

                label_counts = results[key]["entities_per_label"]
                if label_counts:
                    top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("  Top 5 entity types found:")
                    for label, count in top_labels:
                        print(f"    {label}: {count}")

    # Create a dataframe for analysis
    df = pd.DataFrame({
        "Config": metrics["config_name"],
        "ChunkSize": metrics["chunk_size"],
        "Overlap": metrics["overlap"],
        "UniqueEntities": metrics["unique_entities"],
        "QualityScore": metrics["entity_quality_score"],
        "ExecutionTime": metrics["execution_time"],
        "Chunks": metrics["chunks"]
    })

    # Calculate performance metrics
    df["EntitiesPerSecond"] = df["UniqueEntities"] / df["ExecutionTime"]
    df["ProcessingEfficiency"] = df["UniqueEntities"] / df["Chunks"]

    # Create a comprehensive score
    # Normalize metrics for combined score
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val) if max_val > min_val else series

    df["NormEntities"] = normalize(df["UniqueEntities"])
    df["NormQuality"] = normalize(df["QualityScore"])
    df["NormSpeed"] = normalize(df["EntitiesPerSecond"])
    df["NormEfficiency"] = normalize(df["ProcessingEfficiency"])

    # Detect and flag potential anomalies
    mean_entities = df["UniqueEntities"].mean()
    std_entities = df["UniqueEntities"].std()
    df["IsAnomaly"] = (df["UniqueEntities"] > mean_entities + 2 * std_entities) | (
            df["UniqueEntities"] < mean_entities - 2 * std_entities)

    # Calculate combined score - weighted sum of normalized metrics
    # Penalize anomalies slightly in the scoring to prioritize consistent results
    df["Score"] = (
                          (df["NormEntities"] * 0.3) +
                          (df["NormQuality"] * 0.3) +
                          (df["NormSpeed"] * 0.2) +
                          (df["NormEfficiency"] * 0.2)
                  ) * (1.0 - 0.2 * df["IsAnomaly"])

    # Get top configurations
    top_configs = df.sort_values("Score", ascending=False).head(10)

    if verbose:
        print("\n\n===========================================================")
        print("TOP 10 CONFIGURATIONS BASED ON BALANCED PERFORMANCE METRICS")
        print("===========================================================")

        for i, (_, row) in enumerate(top_configs.iterrows(), 1):
            anomaly_flag = " ⚠️ ANOMALY" if row["IsAnomaly"] else ""
            print(f"\n#{i}: {row['Config']}{anomaly_flag}")
            print(f"  • Unique entities: {row['UniqueEntities']}")
            print(f"  • Entity quality score: {row['QualityScore']:.2f}")
            print(f"  • Processing time: {row['ExecutionTime']:.2f} seconds")
            print(f"  • Entities per second: {row['EntitiesPerSecond']:.2f}")
            print(f"  • Processing efficiency: {row['ProcessingEfficiency']:.2f} entities per chunk")
            print(f"  • Overall score: {row['Score']:.4f}")

    # Create visualization of results if there's enough data
        # Create visualization of results if there's enough data

            # Create visualization of results if there's enough data
        if len(df) > 1 and verbose:
            plt.style.use('seaborn-v0_8-whitegrid')  # Keep a clean style

            # --- Heatmap: Chunk Size vs Overlap, Color=Score (using Matplotlib only) ---
            try:
                # Pivot table for heatmap: Index=Overlap, Columns=ChunkSize, Values=Score
                pivot_table_score = df.pivot(index="Overlap", columns="ChunkSize", values="Score")
                # Pivot table for annotations (Unique Entities)
                pivot_table_entities = df.pivot(index="Overlap", columns="ChunkSize", values="UniqueEntities")

                # Get the data matrix and axis labels
                score_matrix = pivot_table_score.values
                entity_matrix = pivot_table_entities.values
                overlap_labels = pivot_table_score.index.tolist()
                chunk_size_labels = pivot_table_score.columns.tolist()

                fig, ax = plt.subplots(figsize=(12, 8))  # Adjust size for heatmap

                # Use imshow to display the score matrix
                im = ax.imshow(score_matrix, cmap="viridis", aspect='auto')  # 'auto' aspect ratio

                # Create colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Overall Performance Score", rotation=-90, va="bottom")

                # Set ticks and labels
                ax.set_xticks(np.arange(len(chunk_size_labels)))
                ax.set_yticks(np.arange(len(overlap_labels)))
                ax.set_xticklabels(chunk_size_labels)
                ax.set_yticklabels(overlap_labels)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                # Normalize score matrix for text color decision
                norm_score = (score_matrix - np.nanmin(score_matrix)) / (
                            np.nanmax(score_matrix) - np.nanmin(score_matrix))
                for i in range(len(overlap_labels)):
                    for j in range(len(chunk_size_labels)):
                        score_val = score_matrix[i, j]
                        entity_val = entity_matrix[i, j]
                        if not np.isnan(score_val):  # Check if value exists
                            # Choose text color based on background luminance
                            text_color = "white" if norm_score[i, j] < 0.5 else "black"
                            text = ax.text(j, i, f"S: {score_val:.2f}\nE: {int(entity_val)}",
                                           ha="center", va="center", color=text_color, fontsize=8)

                ax.set_title("GLiNER Performance Heatmap (Score by Chunk Size and Overlap)", fontsize=16, pad=20)
                ax.set_xlabel("Chunk Size (tokens)", fontsize=12, labelpad=10)
                ax.set_ylabel("Overlap Size (tokens)", fontsize=12, labelpad=10)

                fig.tight_layout()

                # Save the heatmap plot
                heatmap_filename = "gliner_heatmap_performance_mpl.png"
                plt.savefig(heatmap_filename, bbox_inches='tight')
                print(f"\nMatplotlib Heatmap plot saved as '{heatmap_filename}'")
                plt.close(fig)  # Close the figure

            except Exception as e:
                print(f"\nCould not generate Matplotlib heatmap: {e}")
                # If pivoting fails (e.g., duplicate index/column pairs), print df info
                if isinstance(e, ValueError):
                    print("DataFrame info for debugging heatmap pivot:")
                    print(df[['ChunkSize', 'Overlap', 'Score']].head())

            # --- Line Plot: Chunk Size vs Unique Entities (Grouped by Overlap) ---
            # Keep the existing line plot code here...
            # Ensure the 'OverlapPct' column exists in df from previous steps
            if 'OverlapPct' not in df.columns:
                print("Warning: 'OverlapPct' column missing from DataFrame for line plot legend.")
                # Attempt to calculate it if missing - requires overlap_pct in results dict
                # This is a fallback, ensure it's added correctly earlier
                try:
                    df['OverlapPct'] = df.apply(
                        lambda row: int(row['Overlap'] / row['ChunkSize'] * 100) if row['ChunkSize'] > 0 else 0,
                        axis=1)
                except KeyError:
                    print("Error: Cannot calculate 'OverlapPct' as 'Overlap' or 'ChunkSize' is missing.")
                    # Handle the absence of necessary columns gracefully, maybe skip the percentage in the label
                    df['OverlapPct'] = 0  # Assign a default or handle appropriately

            plt.figure(figsize=(14, 7))  # Adjusted size
            markers = ['o', 's', '^', 'D', 'v', '<', '>']
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(df["Overlap"].unique())))

            sorted_overlaps = sorted(df["Overlap"].unique())

            for i, overlap_val in enumerate(sorted_overlaps):
                subset = df[df["Overlap"] == overlap_val].sort_values("ChunkSize")
                if not subset.empty:
                    label_text = f"Overlap: {overlap_val}"
                    # Safely check if 'OverlapPct' exists and has valid data before using it
                    if 'OverlapPct' in subset.columns and not subset['OverlapPct'].isnull().all():
                        try:
                            overlap_pct_val = int(subset.iloc[0]['OverlapPct'])
                            label_text += f" ({overlap_pct_val}%)"
                        except (ValueError, TypeError):
                            pass  # Keep label without percentage if conversion fails

                    plt.plot(
                        subset["ChunkSize"],
                        subset["UniqueEntities"],
                        marker=markers[i % len(markers)],
                        linestyle='-',
                        linewidth=2.5,
                        markersize=8,
                        color=colors[i],
                        label=label_text
                    )

            plt.title("Impact of Chunk Size on Unique Entities Found (by Overlap)", fontsize=16, pad=15)
            plt.xlabel("Chunk Size (tokens)", fontsize=12, labelpad=10)
            plt.ylabel("Number of Unique Entities Found", fontsize=12, labelpad=10)
            plt.xticks(df["ChunkSize"].unique())
            plt.legend(title="Overlap Configuration", fontsize=10, title_fontsize=11)
            plt.grid(True, linestyle=':', alpha=0.6)

            line_filename = "gliner_line_entities_vs_chunksize.png"
            plt.savefig(line_filename, bbox_inches='tight')
            print(f"Line plot saved as '{line_filename}'")
            plt.close()  # Close the figure

            # (Optional) Display plots if not running in a headless environment
            # plt.show()

    # Save detailed results to JSON for later analysis
    with open("gliner_benchmark_results.json", "w") as f:
        # Convert results to serializable format
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                key: (value if not isinstance(value, np.float64) else float(value))
                for key, value in v.items() if key != "entities_per_label"
            }
            # Handle entities_per_label separately
            serializable_results[k]["entities_per_label"] = dict(v["entities_per_label"])

        json.dump(serializable_results, f, indent=2)

    # Return the best configuration for direct use
    if not top_configs.empty:
        best_config = top_configs.iloc[0]
        best_chunk_size = int(best_config["ChunkSize"])
        best_overlap = int(best_config["Overlap"])

        if verbose:
            print("\n=============================================================")
            print(f"RECOMMENDED CONFIGURATION: chunk_size={best_chunk_size}, overlap={best_overlap}")
            print("=============================================================")

        return results, top_configs, (best_chunk_size, best_overlap)

    return results, top_configs, (128, 32)  # Default if no configs worked


def generate_test_text(size_multiplier=1):
    """Generate a realistic test text with various entity types.
    Use size_multiplier to control the length of the text."""

    # Core paragraphs with diverse entity types
    paragraphs = [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. Based in Cupertino, California, "
        "the company designs and sells electronics such as the iPhone, iPad, and MacBook. In Q3 2023, they reported revenues "
        "of $81.8 billion. The company's stock (AAPL) trades on NASDAQ.",

        "Microsoft Corporation was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico in 1975. "
        "The company later moved its headquarters to Redmond, Washington. Their CEO, Satya Nadella, has led the company "
        "since February 2014. Microsoft's Azure cloud computing service competes with Amazon Web Services.",

        "Elon Musk is the CEO of multiple companies including Tesla Inc. and SpaceX. Tesla is headquartered in Austin, Texas "
        "while SpaceX operates from Hawthorne, California. SpaceX's Starship program aims to develop fully reusable launch vehicles "
        "for missions to Mars. Tesla produces electric vehicles like the Model S, Model 3, Model X, and Model Y.",

        "Google LLC is a subsidiary of Alphabet Inc. and was founded by Larry Page and Sergey Brin at Stanford University. "
        "The company's headquarters is in Mountain View, California. In October 2015, Sundar Pichai became the CEO of Google. "
        "Their products include the Chrome browser, Gmail, Google Maps, and the Android operating system.",

        "Amazon.com Inc. was founded by Jeff Bezos in Bellevue, Washington in 1994. The company started as an online bookstore "
        "but has since expanded to various other product categories and services including Amazon Web Services (AWS). "
        "Andy Jassy became CEO on July 5, 2021, after Bezos stepped down.",

        "Meta Platforms, formerly known as Facebook, was founded by Mark Zuckerberg while he was a student at Harvard University. "
        "The company owns several social media platforms including Instagram and WhatsApp. Meta's Reality Labs division focuses on "
        "virtual reality (VR) and augmented reality (AR) technologies, including the Oculus Quest headset.",

        "The COVID-19 pandemic, caused by SARS-CoV-2, began in December 2019 in Wuhan, China. By March 2020, the World Health Organization "
        "had declared it a global pandemic, affecting millions worldwide. Vaccines like Pfizer-BioNTech, Moderna, and Johnson & Johnson "
        "were developed in record time using mRNA technology.",

        "The International Space Station (ISS) orbits Earth at an altitude of approximately 408 kilometers. It has been continuously "
        "occupied since November 2000 and serves as a laboratory for scientific research. The ISS is a joint project between NASA, Roscosmos, "
        "JAXA, ESA, and CSA. In a typical day, astronauts aboard the station witness 16 sunrises and sunsets.",

        "Bitcoin was created in 2009 by an anonymous person or group using the pseudonym Satoshi Nakamoto. The cryptocurrency reached "
        "an all-time high value of over $64,000 in April 2021. Other popular cryptocurrencies include Ethereum, Cardano, and Solana. "
        "The total market capitalization of cryptocurrencies exceeded $2 trillion in 2021.",

        "The 2020 Summer Olympics, officially the Games of the XXXII Olympiad, were held in Tokyo, Japan from July 23 to August 8, 2021. "
        "They were postponed from 2020 due to the COVID-19 pandemic. The United States won the most medals (113), while China won the most "
        "gold medals (38). The 2024 Summer Olympics will be held in Paris, France.",

        "The European Union (EU) is a political and economic union of 27 member states. It was established by the Maastricht Treaty, "
        "which came into force on November 1, 1993. The EU has a combined population of about 447 million people and generates an "
        "estimated 18% of global GDP. Key institutions include the European Commission, European Parliament, and European Central Bank.",

        "Climate change has led to rising global temperatures, with 2016 and 2020 being the warmest years on record according to NASA. "
        "The Paris Agreement, signed in 2015, aims to limit global warming to well below 2°C compared to pre-industrial levels. "
        "Greenhouse gases like carbon dioxide, methane, and nitrous oxide contribute to the warming effect.",

        "Barack Obama served as the 44th President of the United States from 2009 to 2017. Born in Honolulu, Hawaii, he graduated from "
        "Columbia University and Harvard Law School. Before becoming president, he served as a U.S. Senator from Illinois. His vice president, "
        "Joe Biden, later became the 46th President of the United States.",

        "The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system. It consists of "
        "over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometers. It's home to more than 1,500 species of fish, "
        "400 types of coral, 4,000 types of mollusk, and 240 species of birds.",

        "Artificial Intelligence (AI) has advanced significantly with technologies like deep learning and natural language processing. "
        "Companies such as OpenAI, Google DeepMind, and Anthropic are developing large language models (LLMs) like GPT-4, Gemini, and Claude. "
        "These models can generate human-like text, translate languages, and even write computer code.",
    ]

    # Repeat paragraphs to make text longer
    long_text = ""
    repeats = max(1, size_multiplier)

    for _ in range(repeats):
        for p in paragraphs:
            long_text += p + " "

    return long_text


def optimize_gliner_settings(text=None, ner_labels=None, size_multiplier=3):
    """Run an optimization process to find the best GLiNER settings."""

    if text is None:
        text = generate_test_text(size_multiplier)

    if ner_labels is None:
        # Define a comprehensive set of NER labels
        ner_labels = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT",
            "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "FACILITY",
            "NORP", "GPE", "CARDINAL", "ORDINAL", "QUANTITY", "EMAIL", "URL",
            "PHONE_NUMBER", "SOFTWARE", "HARDWARE"
        ]

    print(f"Starting optimization with text of length {len(text)} and {len(ner_labels)} entity types")

    # Run tests
    results, top_configs, best_settings = test_gliner_with_different_chunk_sizes(text, ner_labels)

    # Return the recommended best settings
    return best_settings


if __name__ == "__main__":
    # Define a large set of NER labels
    ner_labels = [
        "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT",
        "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "FACILITY",
        "NORP", "GPE", "CARDINAL", "ORDINAL", "QUANTITY", "EMAIL", "URL",
        "PHONE_NUMBER", "SOFTWARE", "HARDWARE", "MEDICAL_CONDITION", "TREATMENT",
        "MEDICATION", "SYMPTOM", "JOB_TITLE", "SKILL", "EDUCATION", "DEGREE",
        "ACADEMIC_INSTITUTION", "COURSE", "CERTIFICATE", "INDUSTRY", "DEPARTMENT"
    ]

    # Generate test text - adjust multiplier for longer text
    # Higher multiplier = longer text = more thorough test, but slower
    test_text = generate_test_text(size_multiplier=5)
    print(f"Generated test text of {len(test_text)} characters")

    # Run tests
    results, top_configs, best_settings = test_gliner_with_different_chunk_sizes(test_text, ner_labels)

    print(f"\nBest settings found: chunk_size={best_settings[0]}, overlap={best_settings[1]}")
    print("Use these values in your production GLiNEREntitiesGenerator configuration")
