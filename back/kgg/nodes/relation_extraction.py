import gc
import json
import re
import uuid

import torch
from json_repair import repair_json
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
            print(f"Processing document {document.id}")
            relations = self._extract_relations(document)
            print(f"Total relations extracted before deduplication: {len(relations)}")
            document.relations.update(self.deduplicate_relations(relations))
            print(f"Relations after deduplication: {len(document.relations)}")
        self.unload_model()
        return documents

    def _extract_relations(self, document: Document) -> set[Relation]:
        try:
            unique_relations: set[Relation] = set()
            print(f"Extracting relations for document {document.id}")

            chunks = self.text_splitter.split_text(document.text)
            print(f"Document split into {len(chunks)} chunks for relation extraction")

            chunk_offset = 0
            for idx, chunk_text in enumerate(chunks):
                print(f"Processing chunk {idx + 1}/{len(chunks)}")
                chunk_offset = document.text.find(chunk_text, chunk_offset)

                chunk_entities = [
                    e for e in document.entities
                    if chunk_offset <= e.start_idx < e.end_idx <= chunk_offset + len(chunk_text)
                ]
                print(f"Found {len(chunk_entities)} entities in this chunk")
                if len(chunk_entities) < 2:
                    continue

                formatted_entities = self._format_entities(set(chunk_entities))

                response = self._invoke_with_retry(
                    self.prompt.format_prompt(
                        text=chunk_text,
                        entities=formatted_entities
                    )
                )

                unique_relations.update(self._parse_response(response, document=document))

            return unique_relations
        except Exception as e:
            print(f"Error generating relations: {e}")
            return set()

    def _invoke_with_retry(self, prompt: str, max_attempts: int = 5) -> BaseMessage:
        attempt = 1
        while True:
            response: BaseMessage = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            print(f"LLM raw response length: {len(response_text)}")
            preview = response_text[:300].replace('\n', ' ')
            print(f"LLM raw response preview: {preview}")

            finish_reason = response.response_metadata.get("done_reason")

            if finish_reason == "length" and attempt < max_attempts:
                print(f"[warning] Truncated response (finish_reason='length') – retrying ({attempt}/{max_attempts})")
                attempt += 1
                continue
            return response

    def _parse_response(self, response: BaseMessage, document: Document) -> set[Relation]:
        try:
            content = response.content.strip()
            print(f"Parsing response, content length: {len(content)}")
            if not content:
                return set()

            if content.startswith("```json"):
                content = content.removeprefix("```json").strip()
            if content.endswith("```"):
                content = content.removesuffix("```").strip()

            print(f"Attempting to repair and parse JSON of length: {len(content)}")
            repaired_json = repair_json(content)

            data = json.loads(repaired_json)
            label_text2entity = {(entity.label, entity.text): entity for entity in document.entities}
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

            print("Relation length:", len(relations))
            return relations

        except Exception as e:
            print(f"[error] Failed to parse/recover JSON: {type(e).__name__}: {e}")
            print("Raw response for debugging:\n", content[:1000])
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
            self.llm = initialize_llm(self.config, num_ctx=7000)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
            self.text_splitter = RecursiveCharacterTextSplitter(
                length_function=self.length_function,
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap
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
