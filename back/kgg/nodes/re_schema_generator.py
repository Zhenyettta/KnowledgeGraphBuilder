import json
import re
from abc import abstractmethod
from typing import Optional, Any, Union

from kg_gen import KGGen
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from kgg.models import RawDocument, Schema, ER_PROMPT, ER_NEW_PROMPT


class BaseRelationsSchemaGenerator(Runnable[RawDocument, Schema]):
    @abstractmethod
    def invoke(self, input: RawDocument, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Schema:
        raise NotImplementedError()


class ConstantRelationsSchemaGenerator(BaseRelationsSchemaGenerator):
    def __init__(self, schema: list[str] | Schema):
        if isinstance(schema, list):
            schema = Schema(schema)
        self.schema = schema

    def invoke(self, *args: Any, **kwargs: Any) -> Schema:
        return self.schema


class HTTPServerRelationSchemaGenerator(BaseRelationsSchemaGenerator):
    def __init__(
            self,
            server_url: str = "http://0.0.0.0:11434/v1",
            max_tokens: int = 2048,
            prompt: ChatPromptTemplate = None,
    ):
        self.server_url = server_url
        self.max_tokens = max_tokens
        self.prompt = prompt or ER_PROMPT

    def invoke(
            self,
            input: RawDocument,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ) -> Schema:
        try:
            llm = ChatOpenAI(
                base_url=self.server_url,
                temperature=0.8,
                max_tokens=self.max_tokens,
                timeout=300,
                max_retries=1,
                api_key=SecretStr("ollama"),
                model="ajindal/llama3.1-storm:8b-Q8_0",

            )
            response = llm.invoke(self.prompt.format_prompt(
                user_input=input.text,
                entities=[x.text for x in input.entities]
            ))
            if "glirel_labels" not in response.content:
                labels = self._parse_response(response)
            else:
                labels = self._parse_dict_response(response)
            print(labels)
            return Schema(labels=labels)

        except Exception as e:
            print(f"Error generating schema: {e}")
            return Schema(labels=[])

    def _parse_response(self, response: BaseMessage) -> list[str]:
        generated_text = response.content.strip()
        if not generated_text:
            print("Warning: Empty response content.")
            return []

        match = re.search(r'\[.*?]', generated_text, re.DOTALL)

        if not match:
            return []

        try:
            json_str = match.group(0)
            json_str = json_str.replace("'", '"').replace("\n", "")

            labels = json.loads(json_str)
            return list({
                str(label).lower().strip().replace(" ", "_")
                for label in labels
                if isinstance(label, (str, int, float))
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON Parsing Error: {e}, Response: {generated_text}")
            return []

    def _parse_dict_response(self, response: BaseMessage) -> dict[str, dict[str, list[str]]]:
        generated_text = response.content.strip()
        if not generated_text:
            print("Warning: Empty response content.")
            return {}
        try:
            json_str = generated_text.replace("'", '"').replace("\n", "")
            data = json.loads(json_str)
            if not isinstance(data, dict):
                print("Warning: Parsed response is not a dictionary.")
                return {}
            return data
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}, Response: {generated_text}")
            return {}


class KGGenGenerator(BaseRelationsSchemaGenerator):
    def invoke(
                self,
                input: RawDocument,
                config: Optional[RunnableConfig] = None,
                **kwargs: Any
        ) -> RawDocument:
            try:
                kg = KGGen(
                    model="ollama_chat/ajindal/llama3.1-storm:8b-Q8_0",
                    temperature=0.0,
                )
                graph_1 = kg.generate(input_data=input.text)
                return RawDocument(text=input.text, entities=graph_1.entities, relations=graph_1.relations)
            except Exception as e:
                print(f"Error generating schema: {e}")
            return RawDocument(text=input.text, entities=[], relations=[]);

