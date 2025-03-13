import json
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict

from kgg.config import KGGConfig
from kgg.models import Document, KnowledgeGraph, Node, Edge, Relation, Entity
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator
from kgg.nodes.ner_labels_generator import NERLabelsGenerator
from kgg.nodes.relation_extraction import RelationsGenerator



class KnowledgeGraphGenerator:
    def __init__(self, config: KGGConfig):
        self.config = config
        self.ner_labels_generator = NERLabelsGenerator(config)
        self.ner_generator = GLiNEREntitiesGenerator(config)
        self.relation_generator = RelationsGenerator(config)

    def generate(self, documents: list[Document]) -> KnowledgeGraph:
        if not self.config.ner_labels:
            self.config.ner_labels = self.generate_labels(documents)

        documents = self.generate_entities(documents)
        from pprint import pprint
        pprint(self.cluster(self.generate_relations(documents)))

    def generate_labels(self, documents: list[Document]) -> list[str]:
        # TODO: log warning if sample size is bigger than the number of documents
        documents = random.sample(documents, k=self.config.sample_size_ner_labels)
        return self.ner_labels_generator.generate_labels(documents)

    def generate_entities(self, documents: list[Document]) -> list[Document]:
        return self.ner_generator.generate(documents)

    def generate_relations(self, documents: list[Document]) -> list[Document]:
        return self.relation_generator.generate_relations(documents)

    def cluster(self, documents: list[Document]) -> KnowledgeGraph:
        text2entities = defaultdict(list)
        for document in documents:
            for entity in document.entities:
                text2entities[entity.text].append(
                    entity)  # Зелебоба : [Entity(text=Зелебоба), Entity(text=Зелебоба), Entity(text=Зелебоба]


        nodes = []
        for text, entities in text2entities.items():
            node = Node(id=str(uuid.uuid4()), entities=entities)
            nodes.append(
                node)  # Node(entities=[Entity(text=Зелебоба), Entity(text=Зелебоба), Entity(text=Зелебоба)], text=Зелебоба)

        text2node = dict()
        for node in nodes:
            text2node[
                node.text] = node  # Зелебоба: Node(entities=[Entity(text=Зелебоба), Entity(text=Зелебоба), Entity(text=Зелебоба)], text=Зелебоба)


        edges = []
        for document in documents:
            for relation in document.relations:
                head_node = text2node[
                    relation.head.text]  # Node(entities=[Entity(text=Зелебоба), Entity(text=Зелебоба), Entity(text=Зелебоба)], text=Зелебоба)
                tail_node = text2node[
                    relation.tail.text]  # Node(entities=[Entity(text=України), Entity(text=України), Entity(text=України)], text=України)
                edge = Edge(id=str(uuid.uuid4()), head=head_node, tail=tail_node,
                            relation=relation)  # Edge(head=Node(entities=[Entity(text=Зелебоба), Entity(text=Зелебоба), Entity(text=Зелебоба)], text=Зелебоба), tail=Node(entities=[Entity(text=України), Entity(text=України), Entity(text=України)], text=України), relation=Relation(id='b1b3b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b1', document_id='doc_0', head=Entity(id='b1b3b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b1', document_id='doc_0', start_idx=0, end_idx=8, label='LOC', text='Зелебоба'), tail=Entity(id='b1b3b1b1-1b1b-1b1b-1b1b-1b1b1b1b1b1b1', document_id='doc_0', start_idx=9, end_idx=16, label='LOC', text='України'), relation='located_in', description='')]
                edges.append(edge)
        return KnowledgeGraph(nodes=nodes, edges=edges, documents=documents)

