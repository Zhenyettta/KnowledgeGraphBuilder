from dataclasses import dataclass, field
from typing import Any


@dataclass
class Schema:
    labels: list[str]


@dataclass
class RawDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    token_start_idx: int
    token_end_idx: int
    label: str
    text: str


@dataclass
class NERDocument:
    document: RawDocument
    entities: list[Entity]


@dataclass
class Relation:
    head_text: str
    tail_text: str
    label: str
    score: float


@dataclass
class RelationDocument:
    document: RawDocument
    entities: list[Entity]
    relations: list[Relation]


NER_instruction = ("Extract the unique (no duplication) types of relationships present in the following text. "
                   "Return the results as a JSON array of strings, with each string representing a relationship type. "
                   "Use lowercase letters and underscores for multi-word relationship types. "
                   "If no relationships are found, return an empty JSON array []. "
                   "Do not include any explanations or additional text.\n\n"
                   "Examples:\n"
                   "Input: 'Elon Musk is the CEO of Tesla.'\n"
                   "Output: [\"person\", \"organization\", \"role\"]\n\n"
                   "Input: 'Google acquired YouTube in 2006.'\n"
                   "Output: [\"company\", \"acquisition\", \"year\"]\n\n"
                   "Input: 'Python is a programming language developed by Guido van Rossum.'\n"
                   "Output: [\"programming_language\", \"developer\"]\n\n"
                   "Input: 'This sentence contains no named entities.'\n"
                   "Output: []")


ER_instruction = """
### Instruction for Extracting Relation Labels

#### Objective  
Extract structured relation labels from unstructured text. Identify relationships, then return them in a structured list format.

---

### Guidelines for Relation Extraction

1. **Identify Relations**  
   - Determine meaningful relationships between entities, including explicit and implicit connections.
   - Consider **all contextual clues** that indicate a relationship, even if not directly stated.
   - Each relation should be represented as a **lowercase, snake_case label**.

2. **Return Format (List of Relation Labels)**  
   - The model should output a list of unique relation labels found in the text.
   - The output should be formatted as:

   ```json example
   ["...", "...", "...", "..."]
"""