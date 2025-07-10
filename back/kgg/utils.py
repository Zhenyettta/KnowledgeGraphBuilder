from kgg.config import KGGConfig


def initialize_llm(config: KGGConfig, num_ctx: int = 15000, num_predict: int = 10000) -> object:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.llm_model,
        temperature=0.0,
        num_ctx=num_ctx,
        num_predict=num_predict
    )



