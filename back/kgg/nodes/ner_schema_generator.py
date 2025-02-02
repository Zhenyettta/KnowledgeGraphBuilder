from abc import abstractmethod
from typing import Optional, Any

from langchain_core.runnables import Runnable, RunnableConfig

from kgg.models import RawDocument, Schema


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
