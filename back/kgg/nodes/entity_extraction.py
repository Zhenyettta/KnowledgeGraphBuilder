import gc
import uuid

import torch
from gliner import GLiNER
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from kgg.config import KGGConfig
from kgg.models import Entity, Document


class GLiNEREntitiesGenerator:

    def __init__(self, config: KGGConfig):
        self.model = None
        self.config = config
        self.chunk_size = 100
        self.overlap = 15
        self.text_splitter = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is None:
            print("Loading GLiNER model to GPU...")
            self.model = GLiNER.from_pretrained(self.config.gliner_model).to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
            self.text_splitter = RecursiveCharacterTextSplitter(
                length_function=self.length_function,
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )

    def generate(self, documents: list[Document]) -> list[Document]:
        self._load_model()
        for document in documents:
            entities = self._extract_entities(document)
            document.entities.update(entities)
        self.unload_model()
        return documents

    def _extract_entities(self, document: Document) -> list[Entity]:
        chunks = self.text_splitter.split_text(document.text)
        all_entities = []
        total_chunks = len(chunks)
        print(f"Document split into {total_chunks} chunks")

        chunk_offset = 0
        for chunk_text in chunks:
            chunk_offset = document.text.find(chunk_text, chunk_offset)

            chunk_entities = self.model.predict_entities(
                chunk_text,
                self.config.ner_labels,
                threshold=self.config.ner_threshold,
                multi_label=True
            )
            for entity in chunk_entities:
                entity["start"] += chunk_offset
                entity["end"] += chunk_offset

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

    def unload_model(self):
        if hasattr(self, 'model') and self.model is not None:
            self.model = self.model.to('cpu')
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            print("GLiNER model unloaded from GPU")

    def _remove_duplicate_entities(self, entities: list[dict]) -> list[dict]:
        filtered_entities = []
        unique_entities = set()

        for entity in entities:
            key = (entity["start"], entity["end"], entity["label"])
            if key not in unique_entities:
                unique_entities.add(key)
                filtered_entities.append(entity)

        return filtered_entities

    def length_function(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))
