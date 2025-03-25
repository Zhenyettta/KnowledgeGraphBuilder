from dataclasses import dataclass, field
from typing import Any


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
class Node:
    id: str
    entities: list[Entity] = field(default_factory=list)

    @property
    def text(self):
        # Assumes all entities have the same text
        return self.entities[0].text


@dataclass
class Edge:
    id: str
    head: Node
    tail: Node
    relation: Relation

    @property
    def description(self):
        # Assumes all relation have the same description
        return self.relation.description


@dataclass
class KnowledgeGraph:
    documents: list[Document]
    nodes: list[Node]
    edges: list[Edge]

