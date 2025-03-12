from langchain_openai import ChatOpenAI

from kgg.config import KGGConfig


def initialize_llm(config: KGGConfig) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.server_url,
        temperature=0.0,
        max_tokens=config.max_tokens,
        timeout=300,
        max_retries=1,
        api_key=config.api_key,
        model=config.llm_model,
    )
