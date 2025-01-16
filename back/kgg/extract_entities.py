import spacy
from gliner import GLiNER


def extract_entities(text: str, labels: list) -> list:
    """
    Extract entities from the given text using GLiNER and map character indices to token indices.

    :param text: The input text to process.
    :param labels: The list of labels to recognize in the text.
    :return: A list of entities in the format
             [[token_start_idx, token_end_idx, 'LABEL', 'ENTITY_TEXT'], ...]
    """
    # Initialize the GLiNER model
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    # Tokenize the text using SpaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Perform entity prediction with the specified labels
    entities = model.predict_entities(text, labels, threshold=0.5)

    # Map character indices to token indices
    extracted_entities = []
    for entity in entities:
        start_char_idx = entity["start"]
        end_char_idx = entity["end"]
        label = entity["label"]
        entity_text = text[start_char_idx:end_char_idx]

        # Find the corresponding token indices
        token_start_idx = None
        token_end_idx = None
        for i, token in enumerate(doc):
            if token.idx == start_char_idx:
                token_start_idx = i
            if token.idx + len(token.text) == end_char_idx:
                token_end_idx = i + 1  # Make end_idx exclusive
                break

        # Verify the match
        if token_start_idx is not None and token_end_idx is not None:
            extracted_entities.append([token_start_idx, token_end_idx, label, entity_text])
        else:
            print(f"WARNING: Could not find token indices for '{entity_text}'")

    return extracted_entities
