import json
import re

from langchain_core.messages import BaseMessage

from kgg.config import KGGConfig
from kgg.models import Document, Entity, Relation
from kgg.prompts import NER_PROMPT
from kgg.utils import initialize_llm


# TODO TRY TO AGGREGATE LABELS AGAIN WITH LLM
class NERLabelsGenerator:
    def __init__(self, config: KGGConfig):
        self.config = config
        self.llm = initialize_llm(config)
        self.prompt = NER_PROMPT


    def generate_labels(self, documents: list[Document]) -> list[str]:
        try:
            unique_labels = set()
            for document in documents:
                response = self.llm.invoke(self.prompt.format_prompt(
                    user_input=document.text,

                ))
                unique_labels.update(self._parse_response(response))

            return sorted(unique_labels)

        except Exception as e:
            print(f"Error generating schema: {e}")
            return []

    def _parse_response(self, response: BaseMessage) -> list[str]:
        generated_text = response.content.strip()
        if not generated_text:
            print("Warning: Empty response content.")
            return []

        match = re.search(r'\[.*?]', generated_text, re.DOTALL)

        if not match:
            return []

        try:
            json_str = match.group(0).replace("'", '"').replace("\n", "")
            labels = json.loads(json_str)
            return list({
                str(label).lower().strip().replace(" ", "_")
                for label in labels
                if isinstance(label, (str, int, float))
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON Parsing Error: {e}, Response: {generated_text}")
            return []

class RelationsGenerator:
    def __init__(self, config: KGGConfig):
        self.config = config
        self.llm = initialize_llm(config)
        self.prompt = NER_PROMPT

    def generate_relations(self, documents: list[Document]) -> list[str]:
        try:
            unique_labels = set()
            for document in documents:
                formatted_entities = self._format_entities(input.entities)
                response = self.llm.invoke(self.prompt.format_prompt(
                    text=input.text,
                    entities=formatted_entities
                ))

                relations = self._parse_response(response)

            return sorted(unique_labels)

        except Exception as e:
            print(f"Error generating schema: {e}")
            return []

    def _parse_response(self, response: BaseMessage) -> set[Relation]:
        try:
            content = response.content.strip()
            if not content:
                return set()

            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if not json_match:
                return set()

            json_str = json_match.group(1)
            json_str = json_str.strip()

            data = json.loads(json_str)
            relations = set()

            for rel in data.get("relations", []):
                relation = (
                    rel["head"]["text"],
                    rel["relation"],
                    rel["tail"]["text"]
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
        return "\n".join(formatted)

