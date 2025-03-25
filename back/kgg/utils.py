

from kgg.config import KGGConfig


def initialize_llm(config: KGGConfig):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.llm_model,
        temperature=0.0,
        num_ctx=10000,  # Set context window size
    )

