import json
import re
from abc import abstractmethod
from typing import Optional, Any
from typing import Union, Iterator
from warnings import deprecated

from langchain_core.runnables import Runnable, RunnableConfig
from llama_cpp import CreateCompletionResponse
from llama_cpp import Llama

from kgg.models import RawDocument, Schema

@deprecated(reason="Use `re_schema` instead")
class BaseNERSchemaGenerator(Runnable[RawDocument, Schema]):

    @abstractmethod
    def invoke(self, input: RawDocument, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Schema:
        raise NotImplementedError()


class ConstantNERSchemaGenerator(BaseNERSchemaGenerator):

    def __init__(self, schema: list[str] | Schema):
        if isinstance(schema, list):
            schema = Schema(schema)

        self.schema = schema

    def invoke(self, *args: Any, **kwargs: Any) -> Schema:
        return self.schema


class LlamaNERSchemaGenerator(BaseNERSchemaGenerator):
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 45,
            instruction: str = None,
            max_tokens: int = 512
    ):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=8192,
            verbose=False
        )
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

            response = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
            )

            labels = self._parse_response(response)
            return Schema(labels=labels)

        except Exception as e:
            print(f"Error generating schema: {e}")
            return Schema(labels=[])

    def _construct_prompt(self, text: str) -> str:
        return (
            f"{self.instruction}\n\n"
            f"Text:\n{text}\n\n"
            "JSON array of entity labels:"
        )

    def _parse_response(
            self,
            response: Union[CreateCompletionResponse, Iterator[CreateCompletionResponse]]
    ) -> list[str]:

        if isinstance(response, Iterator):
            full_response = ""
            for chunk in response:
                if 'text' in chunk['choices'][0]:
                    full_response += chunk['choices'][0]['text']
            generated_text = full_response.strip()
        else:

            generated_text = response['choices'][0].get('text', '').strip()

        match = re.search(r'\[.*?]', generated_text, re.DOTALL)
        if not match:
            return []

        try:
            json_str = match.group(0)
            json_str = json_str.replace("'", '"').replace("\n", "")
            labels = json.loads(json_str)
            return list({
                str(label).lower().strip()
                for label in labels
                if isinstance(label, (str, int, float))
            })
        except (json.JSONDecodeError, KeyError):
            return []
