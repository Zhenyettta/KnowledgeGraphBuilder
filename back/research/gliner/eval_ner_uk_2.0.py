# GLINER_Ukrainian_NER_Evaluation.ipynb

# Cell 1: Import libraries
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from gliner import GLiNER
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Load the Ukrainian NER dataset
print("Loading dataset...")
dataset = load_dataset("Goader/ner-uk-2.0")
print(f"Dataset loaded with splits: {dataset.keys()}")
print(f"Train set size: {len(dataset['train'])}")
print(f"Validation set size: {len(dataset['validation'])}")
print(f"Test set size: {len(dataset['test'])}")

# Sample a record to understand the structure
print("\nSample record:")
sample = dataset['train'][0]
print(f"Tokens: {sample['tokens']}")
print(f"NER tags: {sample['ner_tags']}")

# Cell 3: Define label mappings
# Map the numeric labels to text labels
# Note: The actual mapping should be verified with the dataset documentation
# Here we use a simplified mapping assuming 1=Person, 2=Organization, 3=Location, 0=O (outside)
label_map = {
    0: "O",  # Outside (not an entity)
    1: "Person",
    2: "Organization",
    3: "Location"
}

# Define the GLINER label set
gliner_labels = ["Person", "Organization", "Location"]

print(f"Label mapping: {label_map}")
print(f"GLINER labels: {gliner_labels}")

# Cell 4: Load the GLINER model
print("Loading GLINER model...")
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
print("Model loaded")


# Cell 5: Function to convert tokens to text for GLINER
def tokens_to_text(tokens):
    """Convert a list of tokens to a single string."""
    # Simple joining - might need to be adapted for Ukrainian language specifics
    text = " ".join(tokens)
    return text


def convert_numeric_labels_to_entities(tokens, ner_tags):
    """Convert numeric NER tags to entity spans."""
    entities = []
    current_entity = None

    for i, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
        tag = label_map.get(tag_id, "O")

        if tag != "O":  # If it's an entity
            if current_entity and current_entity["label"] == tag:
                current_entity["end"] += len(token) + 1  # +1 for space
            else:
                # Start position calculation is simplified and might need adjustment
                if current_entity:
                    entities.append(current_entity)

                start_pos = sum(len(t) for t in tokens[:i]) + i  # Add spaces
                current_entity = {
                    "start": start_pos,
                    "end": start_pos + len(token),
                    "label": tag
                }
        elif current_entity:  # If current token is O but we were tracking an entity
            entities.append(current_entity)
            current_entity = None

    # Add the last entity if there is one
    if current_entity:
        entities.append(current_entity)

    return entities


# Cell 6: Evaluation function
def evaluate_ner(true_entities, pred_entities):
    """
    Evaluate NER predictions against ground truth.
    Returns precision, recall and F1 score.
    """
    # Convert entities to token-level labels for evaluation
    # This is a simplification - real implementation would need span matching

    # Then use sklearn metrics
    y_true = [entity["label"] for entity in true_entities]
    y_pred = [entity["label"] for entity in pred_entities]

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=gliner_labels
    )

    # Calculate aggregate metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=gliner_labels
    )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', labels=gliner_labels
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=gliner_labels
    )

    # Create results dict
    results = {
        'per_class': {
            'precision': {label: p for label, p in zip(gliner_labels, precision)},
            'recall': {label: r for label, r in zip(gliner_labels, recall)},
            'f1': {label: f for label, f in zip(gliner_labels, f1)},
            'support': {label: s for label, s in zip(gliner_labels, support)}
        },
        'macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'micro': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1},
        'weighted': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1}
    }

    return results


# Cell 7: Process dataset and run GLINER
print("Processing test set and running GLINER...")

# For demonstration, we'll use a smaller subset
test_subset = dataset['test'].select(range(100))  # Adjust the size as needed

results = []
for sample in tqdm(test_subset):
    tokens = sample['tokens']
    ner_tags = sample['ner_tags']

    # Convert tokens to text
    text = tokens_to_text(tokens)

    # Get ground truth entities
    true_entities = convert_numeric_labels_to_entities(tokens, ner_tags)

    # Run GLINER prediction
    predictions = model.predict_entities(text, gliner_labels)

    # Save results for later analysis
    results.append({
        'text': text,
        'true_entities': true_entities,
        'pred_entities': predictions
    })

# Cell 8: Compute evaluation metrics
print("Computing evaluation metrics...")

# Collect all predictions and ground truth
all_true_entities = []
all_pred_entities = []

for result in results:
    all_true_entities.extend(result['true_entities'])
    all_pred_entities.extend(result['pred_entities'])

# Run evaluation
metrics = evaluate_ner(all_true_entities, all_pred_entities)

# Display results
print("\nPer-class metrics:")
for label in gliner_labels:
    print(f"{label}:")
    print(f"  Precision: {metrics['per_class']['precision'][label]:.4f}")
    print(f"  Recall: {metrics['per_class']['recall'][label]:.4f}")
    print(f"  F1-score: {metrics['per_class']['f1'][label]:.4f}")
    print(f"  Support: {metrics['per_class']['support'][label]}")

print("\nAggregate metrics:")
print(
    f"Macro - Precision: {metrics['macro']['precision']:.4f}, Recall: {metrics['macro']['recall']:.4f}, F1: {metrics['macro']['f1']:.4f}")
print(
    f"Micro - Precision: {metrics['micro']['precision']:.4f}, Recall: {metrics['micro']['recall']:.4f}, F1: {metrics['micro']['f1']:.4f}")
print(
    f"Weighted - Precision: {metrics['weighted']['precision']:.4f}, Recall: {metrics['weighted']['recall']:.4f}, F1: {metrics['weighted']['f1']:.4f}")

# Cell 9: Visualize results
# Create a DataFrame for visualization
metrics_df = pd.DataFrame({
    'Entity Type': gliner_labels,
    'Precision': [metrics['per_class']['precision'][label] for label in gliner_labels],
    'Recall': [metrics['per_class']['recall'][label] for label in gliner_labels],
    'F1-Score': [metrics['per_class']['f1'][label] for label in gliner_labels]
})

# Melt the DataFrame for easier plotting
melted_df = pd.melt(metrics_df, id_vars=['Entity Type'],
                    value_vars=['Precision', 'Recall', 'F1-Score'],
                    var_name='Metric', value_name='Score')

# Create the visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Entity Type', y='Score', hue='Metric', data=melted_df)
plt.title('GLINER Performance on Ukrainian NER Dataset')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Also visualize aggregate metrics
agg_metrics_df = pd.DataFrame({
    'Aggregation': ['Macro', 'Micro', 'Weighted'],
    'Precision': [metrics['macro']['precision'], metrics['micro']['precision'], metrics['weighted']['precision']],
    'Recall': [metrics['macro']['recall'], metrics['micro']['recall'], metrics['weighted']['recall']],
    'F1-Score': [metrics['macro']['f1'], metrics['micro']['f1'], metrics['weighted']['f1']]
})

melted_agg_df = pd.melt(agg_metrics_df, id_vars=['Aggregation'],
                        value_vars=['Precision', 'Recall', 'F1-Score'],
                        var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(x='Aggregation', y='Score', hue='Metric', data=melted_agg_df)
plt.title('Aggregate Metrics for GLINER on Ukrainian NER')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
