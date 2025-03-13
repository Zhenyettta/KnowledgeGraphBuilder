import json
import re
import uuid

from langchain_core.messages import BaseMessage

from kgg.config import KGGConfig
from kgg.models import Document, Relation, Entity
from kgg.prompts import GLINER_LLM_PROMPT
from kgg.utils import initialize_llm


class RelationsGenerator:
    def __init__(self, config: KGGConfig):
        self.config = config
        self.llm = initialize_llm(config)
        self.prompt = GLINER_LLM_PROMPT

    def generate_relations(self, documents: list[Document]) -> list[Document]:
        for document in documents:
            relations = self._extract_relations(document)
            document.relations.update(relations)
        return documents

    def _extract_relations(self, document: Document) -> set[Relation]:
        try:
            unique_relations = set()
            formatted_entities = self._format_entities(document.entities)
            response = self.llm.invoke(self.prompt.format_prompt(
                text=document.text,
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
