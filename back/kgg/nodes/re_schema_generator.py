import json
import re
from abc import abstractmethod
from typing import Optional, Any

from kg_gen import KGGen
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from kgg.models import Document, Entity, KnowledgeGraph
from kgg.prompts import ER_PROMPT, GLINER_LLM_PROMPT


class BaseRelationsSchemaGenerator(Runnable[Document, Schema]):
    @abstractmethod
    def invoke(self, input: Document, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Schema:
        raise NotImplementedError()


class ConstantRelationsSchemaGenerator(BaseRelationsSchemaGenerator):
    def __init__(self, schema: list[str]):
        if isinstance(schema, list):
            schema = Schema(schema)
        self.schema = schema

    def invoke(self, *args: Any, **kwargs: Any) -> Schema:
        return self.schema

#TODO change to use new Object


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



class KGGenGenerator(BaseRelationsSchemaGenerator):
    def invoke(
            self,
            input: Document,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ) -> Document:
        try:
            kg = KGGen(
                model="ollama_chat/phi4:14b-q4_K_M",
                temperature=0.0,
            )
            graph_1 = kg.generate(input_data=input.text)
            print(graph_1)
            return Document(text=input.text, entities=graph_1.entities, relations=graph_1.relations)
        except Exception as e:
            print(f"Error generating schema: {e}")
        return Document(text=input.text, entities=[], relations=[])


class GLiNERRelationExtractor(Runnable[Document, Document]):
    def __init__(
            self,
            server_url: str = "http://localhost:11434/v1/",
            max_tokens: int = 2048,
            prompt: ChatPromptTemplate = GLINER_LLM_PROMPT,
    ):
        self.server_url = server_url
        self.max_tokens = max_tokens
        self.prompt = prompt

    def invoke(
            self,
            input: Document,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ) -> Document:
        try:
            llm = ChatOpenAI(
                base_url=self.server_url,
                temperature=0.0,
                max_tokens=self.max_tokens,
                timeout=300,
                max_retries=1,
                api_key=SecretStr("ollama"),
                model="deepseek-r1:14b",
            )

            formatted_entities = self._format_entities(input.entities)
            response = llm.invoke(self.prompt.format_prompt(
                text=input.text,
                entities=formatted_entities
            ))
            print(response)

            relations = self._parse_response(response)
            return Document(text=input.text, entities=input.entities, relations=relations)

        except Exception as e:
            print(f"Error extracting relations: {e}")
            return Document(text=input.text, entities=input.entities, relations=set())

    def _format_entities(self, entities: set[Entity]) -> str:
        formatted = []
        for entity in entities:
            formatted.append(
                f"- {entity.text} ({entity.label}) [{entity.token_start_idx}:{entity.token_end_idx}]"
            )
        return "\n".join(formatted)

    def _parse_response(self, response: BaseMessage) -> set[tuple[str, str, str]]:
        try:
            content = response.content.strip()
            if not content:
                return set()

            # Find JSON content between ```json and ```
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if not json_match:
                return set()

            # Clean up the JSON string
            json_str = json_match.group(1)
            json_str = json_str.strip()

            # Parse JSON
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

    def batch(
            self,
            inputs: list[Document],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ) -> list[Document]:
        return [self.invoke(doc, config, **kwargs) for doc in inputs]
