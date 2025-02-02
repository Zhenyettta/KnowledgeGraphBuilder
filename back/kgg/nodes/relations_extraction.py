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
        self.model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
        self.nlp = spacy.load("en_core_web_sm")

    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> RelationDocument:

        text = input["document"].text
        ner = input["ner"]
        relation_labels = input["schema"].labels
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        adjusted_ner = [list(span) for span in ner]
        for span in adjusted_ner:
            span[1] -= 1
        relations_raw = self.model.predict_relations(tokens, relation_labels, threshold=0.0, ner=adjusted_ner,
                                                     top_k=1)
        filtered_relations = [r for r in relations_raw if r["head_text"] and r["tail_text"]]
        sorted_relations = sorted(filtered_relations, key=lambda r: r["score"], reverse=True)
        relations = [Relation(head_text=r["head_text"], tail_text=r["tail_text"], label=r["label"], score=r["score"])
                     for r in sorted_relations]
        return RelationDocument(relations=relations)
