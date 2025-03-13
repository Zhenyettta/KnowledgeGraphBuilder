from dataclasses import dataclass, field

from pydantic import SecretStr


@dataclass
class KGGConfig:
    # FIXME better attribute names
    gliner_model: str = 'urchade/gliner_large-v2.1'
    spacy_model: str = 'en_core_web_lg'
    llm_model: str = 'phi4:14b-q4_K_M'
    server_url: str = 'http://localhost:11434/v1/'
    api_key: SecretStr = SecretStr("ollama")
    encoder_model: str = 'my big dick'
    ner_threshold: float = 0.5
    synonym_threshold: float = 0.8
    max_tokens: int = 4096
    use_old_approach: bool = False
    ner_labels: list[str] = field(default_factory=list)
    sample_size_ner_labels: int = 2


