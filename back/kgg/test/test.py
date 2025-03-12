from pprint import pprint
import tqdm
import json

import evaluate
from datasets import load_dataset
from gliner import GLiNER

# Load GLINER model
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1").to("cuda")

# GLINER_LABEL_NAMES = {
#     'ORG': 'Organization',
#     'PERS': 'Person',
#     'LOC': 'Location',
#     'MON': 'Money',
#     'PCT': 'Percent',
#     'DATE': 'Date',
#     'TIME': 'Time',
#     'PERIOD': 'Period',
#     'JOB': 'Job',
#     'DOC': 'Document',
#     'QUANT': 'Quantity',
#     'ART': 'Article',
#     'MISC': 'Miscellaneous'
# }

GLINER_LABEL_NAMES = {
    'ORG': 'Організація',
    'PERS': 'Особа',
    'LOC': 'Місце',
    'MON': 'Гроші',
    'PCT': 'Відсоток',
    'DATE': 'Дата',
    'TIME': 'Час',
    'PERIOD': 'Період',
    'JOB': 'Посада',
    'DOC': 'Документ',
    'QUANT': 'Кількість',
    'ART': 'Стаття',
    'MISC': 'Різне'
}


def convert_indices_to_labels(tag_indices):
    return [label_names[idx] if idx < len(label_names) else 'O' for idx in tag_indices]


def process_predictions(tokens, pred_entities):
    # Initialize tags as 'O' for all tokens
    pred_tags = ['O'] * len(tokens)

    # Create a mapping from character positions to token indices
    char_to_token = []

    for token_idx, token in enumerate(tokens):
        char_to_token.extend([token_idx] * len(token))
        if token_idx < len(tokens) - 1:
            char_to_token.append(token_idx)

    rev_gliner_labels = {v: k for k, v in GLINER_LABEL_NAMES.items()}

    # Fill in the predicted entities using span information
    for entity in pred_entities:
        start_char = entity['start']
        end_char = entity['end']
        entity_type = rev_gliner_labels.get(entity['label'])

        # Convert character positions to token positions
        # Find the token containing the start character
        start_token = char_to_token[start_char]
        # Find the token containing the character just before end_char
        # (end_char is exclusive in the GLiNER output)
        end_token = char_to_token[end_char - 1] + 1

        # Mark entity tokens with appropriate tags
        for i in range(start_token, end_token):
            pred_tags[i] = f"B-{entity_type}" if i == start_token else f"I-{entity_type}"

    return pred_tags


if __name__ == "__main__":
    # Initialize seqeval metric
    seqeval = evaluate.load("seqeval")

    dataset = load_dataset("Goader/ner-uk-2.0", split='test')

    # Process validation set
    all_true_tags = []
    all_pred_tags = []

    label_names = dataset.features['ner_tags'].feature.names

    for example in tqdm.tqdm(dataset.select(range(500)), desc="Processing examples"):
        # Get true tags
        true_tags = convert_indices_to_labels(example['ner_tags'])

        # Get text and predict entities
        tokens = example['tokens']
        text = ' '.join(tokens)

        # print('\n\n')
        # print(text)

        # Using predict_entities instead of predict
        pred_entities = model.predict_entities(
            text=text,
            labels=list(GLINER_LABEL_NAMES.values()),
            threshold=0.7,
        )

        # Convert predictions to BIO format
        pred_tags = process_predictions(tokens, pred_entities)

        # print('tags (true, pred):')
        # for i, (token, true_tag, pred_tag) in enumerate(zip(tokens, true_tags, pred_tags)):
        #     print(f"{token: >25}\t{true_tag: >15}\t{pred_tag}")

        all_true_tags.append(true_tags)
        all_pred_tags.append(pred_tags)

    # Calculate metrics
    results = seqeval.compute(predictions=all_pred_tags, references=all_true_tags)
    #
    # pprint(results)

    # Print results
    print("\nOverall metrics:")
    print(f"Micro F1: {results['overall_f1']:.2%}")
    print(f"Micro Precision: {results['overall_precision']:.2%}")
    print(f"Micro Recall: {results['overall_recall']:.2%}")

    print("\nPer entity type metrics:")
    for entity_type in GLINER_LABEL_NAMES.keys():
        if entity_type == 'O':
            continue

        if entity_type not in results:
            print(f"\n{entity_type}:")
            print("No predictions found")
            continue

        print(f"\n{entity_type}:")
        print(f"F1: {results[entity_type]['f1']:.2%}")
        print(f"Precision: {results[entity_type]['precision']:.2%}")
        print(f"Recall: {results[entity_type]['recall']:.2%}")
