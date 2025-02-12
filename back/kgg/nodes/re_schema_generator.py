import json
import re
from abc import abstractmethod
from typing import Optional, Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from kgg.models import RawDocument, Schema


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
            server_url: str = "http://localhost:8080",
            instruction: str = None,
            max_tokens: int = -1
    ):
        self.server_url = server_url
        self.instruction = instruction
        self.max_tokens = max_tokens

    def invoke(
            self,
            input: RawDocument,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ) -> Schema:
        try:
            prompt = self._construct_prompt(input.text)

            llm = ChatOpenAI(
                base_url=self.server_url,
                temperature=0,
                max_tokens=None,
                timeout=300,
                max_retries=1,
                api_key=SecretStr("aboba")

            )
            response = llm.invoke(prompt)
            labels = self._parse_response(response)

            return Schema(labels=labels)

        except Exception as e:
            print(f"Error generating schema: {e}")
            return Schema(labels=[])

    def _construct_prompt(self, text: str) -> list[dict[str, str]]:
        prompt = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": text}
        ]
        return prompt

    def _parse_response(self, response: AIMessage) -> list[str]:
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
            print(labels)
            return list({
                str(label).lower().strip().replace(" ", "_")
                for label in labels
                if isinstance(label, (str, int, float))
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON Parsing Error: {e}, Response: {generated_text}")
            return []
