import spacy
from glirel import GLiREL

def extract_relations_with_auto_labels(text: str, ner: list) -> list:
    """
    Extract relations from the given text using GLiREL with automatically retrieved labels.

    :param text: The input text to process.
    :param ner: The list of entities in the format:
                [[start_idx, end_idx, 'LABEL', 'ENTITY_TEXT'], ...].
    :return: A list of relations in the format:
             [[subject_text, relation, object_text, score], ...]
    """
    # Initialize the GLiREL model
    model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
    print(ner)


    relation_labels = [
        "born in",
        "won award",
        "collaborated with",
        "founded",
        "located in",
        "mentored",
        "science"
    ]

    # Tokenize the text using SpaCy (required by GLiREL)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]
    print(tokens)

    relations = model.predict_relations(tokens, relation_labels, threshold=0.0, ner=ner, top_k=1)

    print('Number of relations:', len(relations))

    sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
    print("\nDescending Order by Score:")
    for item in sorted_data_desc:
        print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | score: {item['score']}")
    return sorted_data_desc
