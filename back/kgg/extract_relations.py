import spacy
from glirel import GLiREL

def extract_relations_with_auto_labels(text: str, ner: list) -> list:
    """
    Extract relations from the given text using GLiREL with debugging enhancements.
    """
    # Initialize the GLiREL model
    model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")


    # Define relation labels
    relation_labels = [
        "born in",
        "died in",
        "educated at",
        "married to",
        "parent of",
        "child of",
        "sibling of",
        "won award",
        "nominated for",
        "associated with",
        "worked at",
        "founded",
        "mentored",
        "collaborated with",
        "studied under",
        "discovered",
        "invented",
        "created",
        "developed",
        "contributed to",
        "inspired by",
        "affiliated with",
        "wrote",
        "directed",
        "acted in",
        "produced",
        "composed",
        "painted",
        "sculpted",
        "designed",
        "published",
        "taught at",
        "awarded by",
        "located in",
        "headquartered in",
        "partnered with",
        "succeeded by",
        "preceded by",
        "advocated for",
        "opposed by",
        "appointed as",
        "researched",
        "invested in",
        "employed by",
        "part of",
        "member of",
        "leadership role in",
        "represented by",
        "associated with movement",
        "discovered in",
        "known for",
        "pioneered",
        "specialized in",
        "patented",
        "organized",
        "participated in",
        "conferred title",
        "co-authored",
        "hosted",
        "performed in",
        "achieved",
        "visited",
        "presented at",
        "named after",
        "honored by",
        "built by",
        "supported by",
        "helped by",
        "linked to",
        "serves as",
        "contributor to",
        "advocated by",
        "studied at",
        "related to",
        "championed by",
        "discovered by",
        "attended by"
    ]


    # Tokenize the text using SpaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Debug: Print tokens
    print("Tokens:")
    for i, token in enumerate(tokens):
        print(f"{i}: {token}")

    # Predict relations with adjusted parameters
    relations = model.predict_relations(tokens, relation_labels, threshold=0.4, ner=ner, top_k=5)

    # Debug: Print raw relations
    print("Raw relations:")
    for relation in relations:
        print(relation)

    # Filter empty results
    filtered_relations = [
        r for r in relations if r["head_text"] and r["tail_text"]
    ]
    print(f"Filtered relations: {len(filtered_relations)}")

    # Sort by score
    sorted_data_desc = sorted(filtered_relations, key=lambda x: x['score'], reverse=True)
    print("\nDescending Order by Score:")
    for item in sorted_data_desc:
        print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | score: {item['score']}")

    return sorted_data_desc
