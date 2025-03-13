import uuid

from gliner import GLiNER

from kgg.config import KGGConfig
from kgg.models import Entity, Document


class GLiNEREntitiesGenerator:

    def __init__(self, config: KGGConfig):
        self.model = GLiNER.from_pretrained(config.gliner_model)
        self.config = config

    def generate(self, documents: list[Document]) -> list[Document]:
        for document in documents:
            entities = self._extract_entities(document)
            document.entities.update(entities)

        return documents

    def _extract_entities(self, document: Document) -> list[Entity]:
        predicted_entities = self.model.predict_entities(
            document.text,
            self.config.ner_labels,
            threshold=self.config.ner_threshold,
            multi_label=True
        )

        extracted_entities = []
        for entity in predicted_entities:
            extracted_entities.append(
                Entity(
                    id=str(uuid.uuid4()),
                    start_idx=entity["start"],
                    end_idx=entity["end"],
                    label=entity["label"],
                    text=entity["text"],
                    document_id=document.id
                )
            )

        return extracted_entities
