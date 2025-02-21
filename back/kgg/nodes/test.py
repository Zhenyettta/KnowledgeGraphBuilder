import spacy
import torch
from glirel import GLiREL
device = torch.device("cuda")
model = GLiREL.from_pretrained("jackboyla/glirel-large-v0").to(device)  # Load GLiREL model
nlp = spacy.load("en_core_web_lg")  # Load spaCy's large English NER model

text = "Bob has a dog"
doc = nlp(text)  # Run spaCy NER on the text
tokens = [token.text for token in doc]
# Convert extracted entities into GLiREL-compatible format
ner = [[0, 0, 'person', 'Bob'], [3, 3, 'animal', 'dog']]

labels = {"glirel_labels": {"has": {"allowed_head": ["person"], "allowed_tail": ["animal"]},
                             "owns": {"allowed_head": ["person"], "allowed_tail": ["animal"]},
                            }
          }

relations = model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=1, flat_ner=False)

print(relations)
