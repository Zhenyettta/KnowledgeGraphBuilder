from abc import abstractmethod
from typing import Optional, Any

import spacy
from gliner import GLiNER
from langchain_core.runnables import Runnable, RunnableConfig

from kgg.models import Entity, Document, KnowledgeGraph


class BaseEntitiesGenerator(Runnable[dict[str, KnowledgeGraph], Document]):

    @abstractmethod
    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Document:
        raise NotImplementedError()


# TODO to use new object
class GLiNEREntitiesGenerator(BaseEntitiesGenerator):

    def __init__(self):
        self.model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
        self.nlp = spacy.load("en_core_web_lg")

    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Document:
        text = input["document"].text
        doc = self.nlp(text)
        entities = self.model.predict_entities(text, input["schema"].labels, threshold=0.2, multi_label=True)
        extracted_entities = []
        for entity in entities:
            start_char_idx = entity["start"]
            end_char_idx = entity["end"]
            label = entity["label"]
            entity_text = text[start_char_idx:end_char_idx]
            token_start_idx = None
            token_end_idx = None
            for i, token in enumerate(doc):
                if token.idx == start_char_idx:
                    token_start_idx = i
                if token.idx + len(token.text) == end_char_idx:
                    token_end_idx = i
                    break
            if token_start_idx is not None and token_end_idx is not None:
                extracted_entities.append(
                    Entity(
                        token_start_idx=token_start_idx,
                        token_end_idx=token_end_idx,
                        label=label,
                        text=entity_text
                    )
                )
        return Document(text=input["document"].text, entities=set(extracted_entities))
