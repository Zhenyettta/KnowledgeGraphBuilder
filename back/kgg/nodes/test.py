import spacy
from glirel import GLiREL

model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")  # Load GLiREL model
nlp = spacy.load("en_core_web_lg")  # Load spaCy's large English NER model

text = "Amazon, founded by Jeff Bezos, is a leader in e-commerce and cloud computing. The company has also ventured into artificial intelligence and digital streaming."
doc = nlp(text)  # Run spaCy NER on the text

    # Convert extracted entities into GLiREL-compatible format
ner = [[ent.start_char, ent.end_char, ent.label_, ent.text] for ent in doc.ents]

print(ner)



