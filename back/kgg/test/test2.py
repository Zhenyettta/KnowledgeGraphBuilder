import random
import time
import uuid
import gc
import os
import psutil
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from kgg.models import Entity, Relation, Document, Node, Edge, KnowledgeGraph
from kgg.retriever import KnowledgeGraphRetriever

# Vocabulary for generating more varied text
TECH_VOCABULARY = {
    "FIELD": ["machine learning", "artificial intelligence", "data science", "computer vision",
              "natural language processing", "reinforcement learning", "deep learning",
              "bioinformatics", "robotics", "quantum computing", "cybersecurity",
              "distributed systems", "human-computer interaction", "computer graphics",
              "database systems", "knowledge representation", "information retrieval",
              "computational neuroscience", "computational linguistics", "operations research"],

    "TECHNIQUE": ["neural networks", "decision trees", "random forests", "support vector machines",
                  "gradient boosting", "k-means clustering", "principal component analysis",
                  "recurrent neural networks", "convolutional neural networks", "transformers",
                  "Bayesian networks", "genetic algorithms", "hidden Markov models",
                  "conditional random fields", "Q-learning", "A* search", "beam search",
                  "Monte Carlo methods", "long short-term memory", "attention mechanisms"],

    "LANGUAGE": ["Python", "R", "Java", "C++", "Julia", "JavaScript", "Go", "Scala",
                 "Swift", "Rust", "Kotlin", "MATLAB", "Ruby", "C#", "PHP", "TypeScript",
                 "Haskell", "Perl", "SQL", "Shell scripting"],

    "LIBRARY": ["TensorFlow", "PyTorch", "scikit-learn", "Keras", "Pandas", "NumPy",
                "SciPy", "Matplotlib", "Seaborn", "Plotly", "NLTK", "spaCy", "Hugging Face",
                "OpenCV", "Spark", "Hadoop", "Dask", "XGBoost", "LightGBM", "CatBoost"],

    "DATASET": ["ImageNet", "COCO", "MNIST", "CIFAR-10", "CIFAR-100", "SQuAD", "GLUE",
                "WikiText", "Penn Treebank", "Yelp Reviews", "Amazon Reviews", "MovieLens",
                "VQA", "LibriSpeech", "MS MARCO", "FashionMNIST", "UCI ML Repository",
                "IMDB Reviews", "Reuters Corpus", "WebText"],

    "PERSON": ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng", "Ian Goodfellow",
               "Fei-Fei Li", "Christopher Manning", "Judea Pearl", "Michael Jordan", "Daphne Koller",
               "Jeff Dean", "Demis Hassabis", "Andrej Karpathy", "Pieter Abbeel", "Chelsea Finn",
               "Kate Crawford", "Timnit Gebru", "Ilya Sutskever", "Kyunghyun Cho", "Alex Graves"],

    "ORGANIZATION": ["Google", "Meta", "Microsoft", "Apple", "Amazon", "DeepMind", "OpenAI",
                     "Stanford", "MIT", "Berkeley", "CMU", "Max Planck Institute", "Allen AI",
                     "Baidu", "Tencent", "IBM", "Intel", "NVIDIA", "Salesforce", "Samsung"]
}

VERBS = ["analyzes", "processes", "transforms", "generates", "enhances", "optimizes", "classifies",
         "predicts", "detects", "segments", "recognizes", "recommends", "models", "simulates",
         "evaluates", "measures", "extracts", "constructs", "interprets", "identifies"]

RELATION_TEMPLATES = [
    "{head} is used in {tail} for improved performance",
    "{head} serves as a foundation for advances in {tail}",
    "{head} significantly contributes to the field of {tail}",
    "{head} provides key insights for research in {tail}",
    "{head} enables new capabilities in {tail}",
    "{head} was developed by researchers working on {tail}",
    "{head} represents a breakthrough approach for {tail}",
    "{head} has transformed how we understand {tail}",
    "{head} offers substantial benefits when applied to {tail}",
    "{head} demonstrates impressive results in {tail} tasks",
    "{head} addresses key challenges in {tail}",
    "{head} provides a systematic framework for {tail}",
    "{head} introduces novel techniques for solving {tail} problems",
    "{head} establishes new performance benchmarks for {tail}",
    "{head} combines multiple methods to enhance {tail}",
    "{head} plays a central role in modern {tail} systems",
    "{head} leverages data-driven approaches to advance {tail}",
    "{head} implements efficient algorithms for {tail}",
    "{head} incorporates feedback mechanisms to improve {tail}",
    "{head} facilitates rapid development of {tail} applications"
]


def monitor_memory():
    """Monitor current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB


def generate_entity_text(label):
    """Generate entity text based on label."""
    if label in TECH_VOCABULARY:
        return random.choice(TECH_VOCABULARY[label])
    return f"Entity_{uuid.uuid4().hex[:8]}"


def generate_massive_kg(num_docs=1000, entities_per_doc=100, relations_ratio=0.1):
    """
    Generate a massive knowledge graph.
    relations_ratio: determines what fraction of possible entity pairs have relations
    """
    print(f"Generating massive knowledge graph with {num_docs} documents...")
    print(f"Estimated entities: {num_docs * entities_per_doc}")
    print(f"Estimated relations: ~{int(num_docs * entities_per_doc * entities_per_doc * relations_ratio / 2)}")

    # Entity labels with weighted probabilities
    entity_labels = list(TECH_VOCABULARY.keys())

    # Relation types
    relation_types = [
        "USES", "INFLUENCES", "DEPENDS_ON", "EVALUATES", "IMPLEMENTED_IN",
        "APPLIES_TO", "PART_OF", "VARIANT_OF", "PREDECESSOR_OF", "ALTERNATIVE_TO",
        "COLLABORATES_WITH", "TRAINED_ON", "ACHIEVES", "BENCHMARK_FOR", "COMPONENT_OF"
    ]

    # Initialize collections
    all_documents = []
    all_nodes = []
    all_edges = []
    entity_to_node = {}

    # Track progress
    total_entities = 0
    total_relations = 0

    # Generate documents with progress bar
    initial_memory = monitor_memory()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    for doc_idx in tqdm(range(num_docs), desc="Generating documents"):
        doc_id = f"doc_{doc_idx}"

        # Generate a document with paragraph structure
        paragraphs = []
        for _ in range(random.randint(3, 7)):
            paragraph_words = []
            for _ in range(random.randint(30, 100)):
                paragraph_words.append(random.choice(TECH_VOCABULARY[random.choice(entity_labels)]))
            paragraphs.append(" ".join(paragraph_words))

        doc_text = "\n\n".join(paragraphs)
        doc = Document(id=doc_id, text=doc_text)

        # Generate entities
        doc_entities = []
        entity_spans = []  # Track spans to avoid overlaps

        for ent_idx in range(entities_per_doc):
            # Try to find a non-overlapping span
            for attempt in range(10):  # Limit attempts to prevent infinite loops
                # Choose paragraph and position
                para_idx = random.randint(0, len(paragraphs) - 1)
                para_text = paragraphs[para_idx]
                if len(para_text) < 10:  # Skip very short paragraphs
                    continue

                # Calculate absolute span
                para_start = doc_text.find(para_text)
                start_idx = para_start + random.randint(0, max(0, len(para_text) - 20))
                end_idx = start_idx + random.randint(5, 15)

                if end_idx > len(doc_text):
                    continue

                # Check for overlap
                overlap = False
                for existing_start, existing_end in entity_spans:
                    if not (end_idx <= existing_start or start_idx >= existing_end):
                        overlap = True
                        break

                if not overlap:
                    entity_spans.append((start_idx, end_idx))
                    break

            if attempt == 9:  # If we couldn't find a non-overlapping span
                continue

            label = random.choice(entity_labels)
            entity_text = doc_text[start_idx:end_idx]

            entity = Entity(
                id=f"ent_{doc_idx}_{ent_idx}",
                document_id=doc_id,
                start_idx=start_idx,
                end_idx=end_idx,
                label=label,
                text=entity_text
            )
            doc_entities.append(entity)
            total_entities += 1

            # Create node for this entity
            node_id = f"node_{entity.id}"
            node = Node(id=node_id, entities=[entity])
            all_nodes.append(node)
            entity_to_node[entity.id] = node

        # Generate relations (only a small fraction of possible pairs to keep manageable)
        doc_relations = []
        max_relations = int(len(doc_entities) * (len(doc_entities) - 1) * relations_ratio / 2)
        entity_pairs_for_relations = set()

        for _ in range(min(max_relations, len(doc_entities) * 10)):
            # Try to find an unused pair
            for _ in range(5):  # Limit attempts
                head_idx = random.randint(0, len(doc_entities) - 1)
                tail_idx = random.randint(0, len(doc_entities) - 1)

                if head_idx != tail_idx and (head_idx, tail_idx) not in entity_pairs_for_relations:
                    entity_pairs_for_relations.add((head_idx, tail_idx))
                    break
            else:
                continue  # Couldn't find an unused pair

            head_entity = doc_entities[head_idx]
            tail_entity = doc_entities[tail_idx]
            relation_type = random.choice(relation_types)

            # Generate a more diverse description
            template = random.choice(RELATION_TEMPLATES)
            description = template.format(head=head_entity.text, tail=tail_entity.text)

            relation = Relation(
                id=f"rel_{doc_idx}_{len(doc_relations)}",
                document_id=doc_id,
                head=head_entity,
                tail=tail_entity,
                relation=relation_type,
                description=description
            )
            doc_relations.append(relation)
            total_relations += 1

            # Create edge for this relation
            edge = Edge(
                id=f"edge_{relation.id}",
                head=entity_to_node[head_entity.id],
                tail=entity_to_node[tail_entity.id],
                relation=relation
            )
            all_edges.append(edge)

        # Update document with entities and relations
        doc.entities = set(doc_entities)
        doc.relations = set(doc_relations)
        all_documents.append(doc)

        # Periodically report progress and force garbage collection
        if (doc_idx + 1) % 50 == 0:
            mem_usage = monitor_memory()
            print(
                f"Generated {doc_idx + 1}/{num_docs} docs, {total_entities} entities, {total_relations} relations, memory: {mem_usage:.2f} MB")
            gc.collect()

    # Final memory usage
    mem_usage = monitor_memory()
    print(f"Final memory usage: {mem_usage:.2f} MB")
    print(f"Generated {len(all_documents)} documents, {len(all_nodes)} nodes, and {len(all_edges)} edges")

    # Create and return the knowledge graph
    return KnowledgeGraph(
        documents=all_documents,
        nodes=all_nodes,
        edges=all_edges
    )


def test_indexing_performance(kg, batch_size=10000):
    """Test indexing performance."""
    print(f"\nTesting indexing performance with {len(kg.edges)} edges...")
    print(f"Using batch size of {batch_size}")

    # Measure indexing time
    gc.collect()  # Force garbage collection before timing

    # Create retriever with batch processing capability
    retriever = KnowledgeGraphRetriever()

    # Track memory before and during indexing
    mem_before = monitor_memory()
    print(f"Memory before indexing: {mem_before:.2f} MB")

    # Measure the time taken for indexing
    start_time = time.time()
    retriever.index(kg, batch_size)
    index_time = time.time() - start_time

    # Check memory after indexing
    mem_after = monitor_memory()
    print(f"Memory after indexing: {mem_after:.2f} MB")
    print(f"Memory increase: {mem_after - mem_before:.2f} MB")

    print(f"Indexing completed in {index_time:.2f} seconds")
    print(f"Performance: {len(kg.edges) / index_time:.2f} edges/second")
    print(f"Index size: {retriever.vector_store.index.ntotal}")

    # Run a few test queries to verify functionality
    test_queries = [
        "How are neural networks used in computer vision?",
        "What are the benefits of Python for machine learning?",
        "How do transformers improve natural language processing?"
    ]

    for query in test_queries:
        start_time = time.time()
        results = retriever.retrieve(query, k=3)
        query_time = time.time() - start_time
        print(f"\nQuery: '{query}' (took {query_time:.4f}s)")
        print(f"Top result: {results[0].page_content}")

    return retriever


def run_extreme_scale_test(batch_sizes=[500, 1000, 10000]):
    """Run the extreme-scale test with different batch sizes."""
    print("Starting extreme-scale test...")
    print(f"Initial memory usage: {monitor_memory():.2f} MB")

    try:
        # Generate a massive knowledge graph
        # You can adjust the parameters to control scale
        kg = generate_massive_kg(num_docs=500, entities_per_doc=200, relations_ratio=0.05)

        # Compare different batch sizes if specified
        if batch_sizes and len(batch_sizes) > 1:
            print("\nComparing different batch sizes for indexing performance:")
            results = {}

            # Only test with a subset of the graph to save time
            test_edges = kg.edges[:min(100000, len(kg.edges))]
            test_kg = KnowledgeGraph(
                documents=kg.documents,
                nodes=kg.nodes,
                edges=test_edges
            )

            for batch_size in batch_sizes:
                print(f"\nTesting batch size: {batch_size}")
                gc.collect()  # Force garbage collection

                retriever = KnowledgeGraphRetriever()
                start_time = time.time()
                retriever.index(test_kg, batch_size)
                index_time = time.time() - start_time

                results[batch_size] = {
                    'time': index_time,
                    'edges_per_second': len(test_kg.edges) / index_time,
                    'memory': monitor_memory()
                }

                print(
                    f"Batch size {batch_size}: {index_time:.2f}s, {results[batch_size]['edges_per_second']:.2f} edges/s")

                # Clean up
                del retriever
                gc.collect()

            # Report results
            print("\nBatch size comparison results:")
            for batch_size, metrics in sorted(results.items()):
                print(
                    f"Batch size {batch_size}: {metrics['time']:.2f}s, {metrics['edges_per_second']:.2f} edges/s, {metrics['memory']:.2f} MB")

            # Choose the best batch size for the full test
            best_batch_size = max(results.items(), key=lambda x: x[1]['edges_per_second'])[0]
            print(f"\nSelected best batch size: {best_batch_size} for full test")
        else:
            best_batch_size = batch_sizes[0] if batch_sizes else 10000

        # Test indexing and retrieve performance with the best/selected batch size
        retriever = test_indexing_performance(kg, batch_size=best_batch_size)

        # Run comprehensive query benchmark
        print("\nRunning comprehensive query benchmark...")
        query_times = []

        # Generate diverse queries
        query_templates = [
            "What is the relationship between {field1} and {field2}?",
            "How is {technique} used in {field}?",
            "Why is {language} popular for {field}?",
            "What are the best {technique} methods for {field}?",
            "How does {person} contribute to {field}?",
            "Compare {technique1} and {technique2} for {field}.",
            "What datasets are commonly used for {field}?",
            "How do {organization} researchers approach {field}?",
            "What are the limitations of {technique} in {field}?",
            "How has {field} evolved in recent years?"
        ]

        benchmark_queries = []
        for _ in range(100):  # 100 benchmark queries
            template = random.choice(query_templates)
            query = template.format(
                field=random.choice(TECH_VOCABULARY["FIELD"]),
                field1=random.choice(TECH_VOCABULARY["FIELD"]),
                field2=random.choice(TECH_VOCABULARY["FIELD"]),
                technique=random.choice(TECH_VOCABULARY["TECHNIQUE"]),
                technique1=random.choice(TECH_VOCABULARY["TECHNIQUE"]),
                technique2=random.choice(TECH_VOCABULARY["TECHNIQUE"]),
                language=random.choice(TECH_VOCABULARY["LANGUAGE"]),
                person=random.choice(TECH_VOCABULARY["PERSON"]),
                organization=random.choice(TECH_VOCABULARY["ORGANIZATION"])
            )
            benchmark_queries.append(query)

        # Run the benchmark
        for i, query in enumerate(benchmark_queries):
            if i % 10 == 0:
                print(f"Running query {i + 1}/100...")

            start_time = time.time()
            results = retriever.retrieve(query, k=5)
            query_time = time.time() - start_time
            query_times.append(query_time)

            if i < 5:  # Show details for first few queries
                print(f"\nQuery: '{query}'")
                print(f"Query time: {query_time:.4f} seconds")
                for j, doc in enumerate(results[:2]):  # Show top 2 results
                    print(f"Result {j + 1}: {doc.page_content[:100]}...")

        # Calculate and report statistics
        avg_time = sum(query_times) / len(query_times)
        min_time = min(query_times)
        max_time = max(query_times)
        p95_time = sorted(query_times)[int(len(query_times) * 0.95)]

        print("\nQuery benchmark results:")
        print(f"Average query time: {avg_time:.4f} seconds")
        print(f"Minimum query time: {min_time:.4f} seconds")
        print(f"Maximum query time: {max_time:.4f} seconds")
        print(f"95th percentile: {p95_time:.4f} seconds")
        print(f"Queries per second: {1 / avg_time:.2f}")

        return retriever, kg

    except Exception as e:
        print(f"Error during extreme-scale test: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print(f"Final memory usage: {monitor_memory():.2f} MB")


if __name__ == "__main__":
    # Test with different batch sizes to find the optimal one

    retriever, kg = run_extreme_scale_test()
