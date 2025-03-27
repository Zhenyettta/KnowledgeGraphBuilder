from kgg.config import KGGConfig


def initialize_llm(config: KGGConfig, num_ctx: int = 15000):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.llm_model,
        temperature=0.0,
        num_ctx=num_ctx
    )



