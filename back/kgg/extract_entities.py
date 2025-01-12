from gliner import GLiNER


def extract_entities(text: str, labels: list) -> list:
    """
    Extract entities from the given text using GLiNER based on provided labels.

    :param text: The input text to process.
    :param labels: The list of labels to recognize in the text.
    :return: A list of entities in the format
             [[start_idx, end_idx, 'LABEL', 'ENTITY_TEXT'], ...]
    """
    # Initialize the GLiNER model
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    # Perform entity prediction with the specified labels
    entities = model.predict_entities(text, labels, threshold=0.5)

    # Convert GLiNER output to the desired format
    extracted_entities = []
    for entity in entities:
        start_idx = entity["start"]
        end_idx = entity["end"]
        label = entity["label"]
        entity_text = text[start_idx:end_idx]
        extracted_entities.append([start_idx, end_idx, label, entity_text])

    return extracted_entities
