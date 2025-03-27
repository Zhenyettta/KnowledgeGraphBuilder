import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
from collections import Counter
from gliner import GLiNER
from kgg.config import KGGConfig
from kgg.models import Entity, Document
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator


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
    chunk_sizes = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]
    overlap_percentages = [10, 15, 20, 25, 30, 35, 40]

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

    # Only run baseline if the text isn't too large - max 10,000 chars to avoid memory issues
    if len(text) <= 10000:
        try:
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Running baseline (no chunking) for comparison")
                print(f"{'-' * 80}")

            generator = GLiNEREntitiesGenerator(config)
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
    if len(df) > 1 and verbose:
        plt.figure(figsize=(15, 10))

        # Create a scatter plot with different metrics
        scatter = plt.scatter(df["ChunkSize"], df["Overlap"],
                              s=df["UniqueEntities"] * 5 + 50,  # Scale bubbles for visibility
                              alpha=0.7,
                              c=df["Score"],  # Color by overall score
                              cmap="viridis")

        # Mark anomalies
        anomalies = df[df["IsAnomaly"]]
        if not anomalies.empty:
            plt.scatter(anomalies["ChunkSize"], anomalies["Overlap"],
                        s=anomalies["UniqueEntities"] * 5 + 50,
                        edgecolors='red',
                        facecolors='none',
                        linewidth=2,
                        alpha=0.7)

        # Highlight top configurations
        top_indices = top_configs.index[:5]  # Highlight top 5
        plt.scatter(df.loc[top_indices, "ChunkSize"],
                    df.loc[top_indices, "Overlap"],
                    s=200,
                    facecolors='none',
                    edgecolors='black',
                    linewidth=2)

        # Add labels for top 5
        for i, idx in enumerate(top_indices):
            plt.annotate(f"#{i + 1}",
                         (df.loc[idx, "ChunkSize"], df.loc[idx, "Overlap"]),
                         xytext=(5, 5),
                         textcoords="offset points",
                         fontweight="bold")

        plt.colorbar(scatter, label="Overall Score")
        plt.xlabel("Chunk Size")
        plt.ylabel("Overlap Size")
        plt.title("GLiNER Performance by Configuration\n(Bubble size = entities found, color = overall score)")
        plt.grid(True, alpha=0.3)

        # Add second plot showing entity counts vs chunk size
        plt.figure(figsize=(15, 6))

        for overlap_pct in sorted(df["Overlap"].unique()):
            subset = df[df["Overlap"] == overlap_pct]
            if not subset.empty:
                plt.plot(subset["ChunkSize"], subset["UniqueEntities"],
                         'o-', label=f"Overlap: {overlap_pct}")

        plt.xlabel("Chunk Size")
        plt.ylabel("Unique Entities Found")
        plt.title("Impact of Chunk Size on Entity Extraction")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save visualizations
        plt.savefig("gliner_performance_comparison.png")
        print("\nVisualizations saved as 'gliner_performance_comparison.png'")

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
