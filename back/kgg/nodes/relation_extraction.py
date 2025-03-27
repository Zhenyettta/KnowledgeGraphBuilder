import gc
import json
import re
import uuid

import torch
from langchain_core.messages import BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from kgg.config import KGGConfig
from kgg.models import Document, Relation, Entity
from kgg.prompts import GLINER_LLM_PROMPT
from kgg.utils import initialize_llm


class RelationsGenerator:
    def __init__(self, config: KGGConfig):
        self.config = config
        self.llm = None
        self.prompt = GLINER_LLM_PROMPT
        self.chunk_size = 300
        self.overlap = 30
        self.text_splitter = None
        self.tokenizer = None

    def generate_relations(self, documents: list[Document]) -> list[Document]:
        self._load_model()
        for document in documents:
            relations = self._extract_relations(document)
            document.relations.update(relations)
            document.relations = self.deduplicate_relations(document.relations)
        self.unload_model()
        return documents

    def _extract_relations(self, document: Document) -> set[Relation]:
        try:
            unique_relations = set()
            print(f"Extracting relations for document {document.id}")

            chunks = self.text_splitter.split_text(document.text)
            total_chunks = len(chunks)
            print(f"Document split into {total_chunks} chunks for relation extraction")

            chunk_offset = 0
            for chunk_text in chunks:
                chunk_offset = document.text.find(chunk_text, chunk_offset)

                chunk_entities = []
                for entity in document.entities:

                    if entity.start_idx >= chunk_offset and entity.end_idx < chunk_offset + len(chunk_text):
                        chunk_entities.append(entity)
                if len(chunk_entities) < 2:
                    continue

                formatted_entities = self._format_entities(set(chunk_entities))
                response = self.llm.invoke(self.prompt.format_prompt(
                    text=chunk_text,
                    entities=formatted_entities
                ))

                unique_relations.update(self._parse_response(response, document=document))

            return unique_relations
        except Exception as e:
            print(f"Error generating relations: {e}")
            return set()

    def _parse_response(self, response: BaseMessage, document: Document) -> set[Relation]:
        try:
            content = response.content.strip()
            label_text2entity = {(entity.label, entity.text): entity for entity in document.entities}
            if not content:
                return set()

            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if not json_match:
                return set()

            json_str = json_match.group(1)
            json_str = json_str.strip()

            data = json.loads(json_str)
            relations = set()

            for rel in data:
                head = label_text2entity.get((rel["head"]["label"], rel["head"]["text"]))
                tail = label_text2entity.get((rel["tail"]["label"], rel["tail"]["text"]))
                if not head or not tail:
                    continue

                relation = Relation(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    head=head,
                    relation=rel["relation"],
                    tail=tail,
                    description=rel["description"]
                )
                relations.add(relation)

            return relations

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            print("Problematic JSON:", json_str if 'json_str' in locals() else "Not extracted")
            return set()

    def _format_entities(self, entities: set[Entity]) -> str:
        formatted = []
        for entity in entities:
            formatted.append(
                f"- {entity.text} ({entity.label})"
            )
        return "\n" + "\n".join(formatted)

    def _load_model(self):
        if self.llm is None:
            print("Loading llm model to GPU...")
            self.llm = initialize_llm(self.config, num_ctx=6000)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
            self.text_splitter = RecursiveCharacterTextSplitter(
                length_function=self.length_function,
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )

    def unload_model(self):
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
            gc.collect()
            print("Model unloaded from GPU")

    def length_function(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def deduplicate_relations(self, relations):
        unique = {}
        for rel in relations:
            key = (rel.head.id, rel.relation, rel.tail.id)
            if key not in unique:
                unique[key] = rel
        return set(unique.values())

