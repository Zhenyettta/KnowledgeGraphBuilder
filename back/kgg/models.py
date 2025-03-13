from dataclasses import dataclass, field
from typing import Any, Tuple



@dataclass(frozen=True)
class Entity:
    id: str
    document_id: str
    start_idx: int
    end_idx: int
    label: str
    text: str


@dataclass(frozen=True)
class Relation:
    id: str
    document_id: str
    head: Entity
    tail: Entity
    relation: str
    description: str


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: set[Entity] = field(default_factory=set)
    relations: set[Relation] = field(default_factory=set)


@dataclass
class KnowledgeGraph:
    documents: list[Document]
    # TODO the rest of the data (nodes, edges)

