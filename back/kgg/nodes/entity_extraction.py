import gc
import uuid
import re

import torch
from gliner import GLiNER

from kgg.config import KGGConfig
from kgg.models import Entity, Document


class GLiNEREntitiesGenerator:

    def __init__(self, config: KGGConfig):
        self.model = GLiNER.from_pretrained(config.gliner_model).to("cpu")
        self.config = config
        self.chunk_size = 100
        self.overlap = 20

    def generate(self, documents: list[Document]) -> list[Document]:
        for document in documents:
            entities = self._extract_entities(document)
            document.entities.update(entities)

        return documents

    def _extract_entities(self, document: Document) -> list[Entity]:
        chunks = self._split_into_chunks(document.text)
        all_entities = []
        total_chunks = len(chunks)

        print(f"Document split into {total_chunks} chunks")

        for i, (chunk_text, offset) in enumerate(chunks):

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            chunk_entities = self.model.predict_entities(
                chunk_text,
                self.config.ner_labels,
                threshold=self.config.ner_threshold,
                multi_label=True
            )

            for entity in chunk_entities:
                entity["start"] += offset
                entity["end"] += offset

            all_entities.extend(chunk_entities)

        unique_entities = self._remove_duplicate_entities(all_entities)

        extracted_entities = []
        for entity in unique_entities:
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

    def _split_into_chunks(self, text: str) -> list[tuple]:
        words = re.findall(r'\S+|\s+', text)

        chunks = []
        start_word = 0

        while start_word < len(words):
            end_word = min(start_word + self.chunk_size, len(words))

            char_start = sum(len(w) for w in words[:start_word])
            chunk_text = ''.join(words[start_word:end_word])

            chunks.append((chunk_text, char_start))

            start_word += self.chunk_size - self.overlap

        return chunks

    def _remove_duplicate_entities(self, entities: list[dict]) -> list[dict]:
        if not entities:
            return entities

        sorted_entities = sorted(entities, key=lambda e: (e["start"], e["end"]))
        unique_entities = []

        for entity in sorted_entities:
            is_duplicate = False

            for existing in unique_entities:
                if (entity["start"] == existing["start"] and
                        entity["end"] == existing["end"] and
                        entity["label"] == existing["label"]):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_entities.append(entity)

        return unique_entities
