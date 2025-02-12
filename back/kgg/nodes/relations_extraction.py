from abc import abstractmethod
from typing import Optional, Any

import spacy
from glirel import GLiREL
from langchain_core.runnables import Runnable, RunnableConfig

from kgg.models import NERDocument, RelationDocument, Relation, Schema


class BaseRelationsGenerator(Runnable[dict[str, NERDocument | Schema], RelationDocument]):
    @abstractmethod
    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> RelationDocument:
        raise NotImplementedError()


class GLiRELRelationsGenerator(BaseRelationsGenerator):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")

    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> RelationDocument:
        ner_document = input["document"]
        text = ner_document.document.text
        ner = [[e.token_start_idx, e.token_end_idx, e.label, e.text] for e in ner_document.entities]
        relation_labels = input["schema"].labels
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        adjusted_ner = [list(span) for span in ner]


        relations_raw = self.model.predict_relations(tokens, relation_labels, threshold=0.5, ner=adjusted_ner,
                                                     top_k=5, flat_ner=False)
        filtered_relations = [r for r in relations_raw if r["head_text"] and r["tail_text"]]
        sorted_relations = sorted(filtered_relations, key=lambda r: r["score"], reverse=True)
        relations = [Relation(head_text=r["head_text"], tail_text=r["tail_text"], label=r["label"], score=r["score"])
                     for r in sorted_relations]

        return RelationDocument(document=ner_document.document, entities=ner_document.entities, relations=relations)
