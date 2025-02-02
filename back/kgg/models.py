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
    relations: list[Relation]
