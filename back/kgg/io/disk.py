import json
import os
from dataclasses import asdict
from typing import Any

from kgg.models import KnowledgeGraph, Entity, Relation, Document, Node, Edge


class KnowledgeGraphStorage:

    def read(self, file_path: str) -> KnowledgeGraph:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = []
        for doc_dict in data['documents']:
            entities = {Entity(**e) for e in doc_dict['entities']}
            relations = {Relation(**r) for r in doc_dict['relations']}
            doc = Document(
                id=doc_dict['id'],
                text=doc_dict['text'],
                metadata=doc_dict['metadata'],
                entities=entities,
                relations=relations
            )
            documents.append(doc)

        nodes = []
        for node_dict in data['nodes']:
            entities = [Entity(**e) for e in node_dict['entities']]
            node = Node(id=node_dict['id'], entities=entities)
            nodes.append(node)

        node_map = {node.id: node for node in nodes}

        edges = []
        for edge_dict in data['edges']:
            head = node_map[edge_dict['head']['id']]
            tail = node_map[edge_dict['tail']['id']]
            relation = Relation(**edge_dict['relation'])
            edge = Edge(id=edge_dict['id'], head=head, tail=tail, relation=relation)
            edges.append(edge)

        kg = KnowledgeGraph(documents=documents, nodes=nodes, edges=edges)
        return kg

    def write(self, knowledge_graph: KnowledgeGraph, file_path: str) -> None:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

            with open(file_path, 'w', encoding='utf-8') as file:
                kg_dict: dict[str, Any] = asdict(knowledge_graph)
                file.write(json.dumps(kg_dict, indent=2, ensure_ascii=False))
