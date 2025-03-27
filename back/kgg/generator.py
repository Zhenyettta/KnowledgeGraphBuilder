import random
import uuid
import json
import os
from collections import defaultdict
from pathlib import Path

from kgg.config import KGGConfig
from kgg.io.graph import Neo4j
from kgg.models import Document, KnowledgeGraph, Node, Edge, Relation, Entity
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator
from kgg.nodes.graph_answering import GraphAnswering
from kgg.nodes.ner_labels_generator import NERLabelsGenerator
from kgg.nodes.relation_extraction import RelationsGenerator
from kgg.retriever import KnowledgeGraphRetriever


class KnowledgeGraphGenerator:
    def __init__(self, config: KGGConfig):
        self.config = config
        self.ner_labels_generator = NERLabelsGenerator(config)
        self.ner_generator = GLiNEREntitiesGenerator(config)
        self.relation_generator = RelationsGenerator(config)
        self.graph_db = Neo4j()
        self.retriever = KnowledgeGraphRetriever(self.graph_db)
        self.graph_answering = GraphAnswering(config)
        self.cache_dir = Path("./cache")
        self.graph_cache_path = self.cache_dir / "graph.json"

    def generate(self,query: str, documents: list[Document] = None, use_cache: bool = False) -> str:
        """
        Complete pipeline to generate a knowledge graph and answer a query.

        Args:
            documents: List of documents to process
            query: The query to answer
            use_cache: If True, try to load graph from cache instead of initializing (default: False)

        Returns:
            str: The answer to the query
        """
        if use_cache:
            graph = self._load_graph_from_cache()
            if graph is None:
                graph = self.initialize_graph(documents)
                self._save_graph_to_cache(graph)
                self.graph_db.import_graph(graph)
            self.retriever.index(graph)
        else:
            graph = self.initialize_graph(documents)
            self._save_graph_to_cache(graph)
            self.graph_db.import_graph(graph)
            self.retriever.index(graph)

        return self.query(graph, query)

    def initialize_graph(self, documents: list[Document]) -> KnowledgeGraph:
        """
        Initialize and return a knowledge graph from documents.

        Args:
            documents: List of documents to process

        Returns:
            KnowledgeGraph: The generated knowledge graph
        """
        if not self.config.ner_labels:
            self.config.ner_labels = self.generate_labels(documents)

        documents_with_entities = self.generate_entities(documents)
        print("boba")
        documents_with_relations = self.generate_relations(documents_with_entities)
        graph = self.cluster(documents_with_relations)

        return graph

    def _save_graph_to_cache(self, graph: KnowledgeGraph) -> None:
        """
        Save the knowledge graph to a JSON file on disk.

        Args:
            graph: The knowledge graph to save
        """
        try:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

            serializable_graph = {
                "nodes": [self._serialize_node(node) for node in graph.nodes],
                "edges": [self._serialize_edge(edge) for edge in graph.edges],
                "documents": [self._serialize_document(doc) for doc in graph.documents]
            }

            with open(self.graph_cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_graph, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error saving graph to cache: {str(e)}")

    def _load_graph_from_cache(self) -> KnowledgeGraph or None:
        """
        Load a knowledge graph from the cache file if it exists.

        Returns:
            KnowledgeGraph if successful, None otherwise
        """
        if not self.graph_cache_path.exists():
            return None

        try:
            with open(self.graph_cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            nodes = [self._deserialize_node(node_data) for node_data in data["nodes"]]
            node_map = {node.id: node for node in nodes}

            edges = [self._deserialize_edge(edge_data, node_map) for edge_data in data["edges"]]
            documents = [self._deserialize_document(doc_data) for doc_data in data["documents"]]

            return KnowledgeGraph(nodes=nodes, edges=edges, documents=documents)

        except Exception as e:
            print(f"Error loading graph from cache: {str(e)}")
            return None

    def _serialize_node(self, node: Node) -> dict:
        """Convert a Node to a serializable dictionary"""
        return {
            "id": node.id,
            "entities": [self._serialize_entity(entity) for entity in node.entities]
        }

    def _serialize_edge(self, edge: Edge) -> dict:
        """Convert an Edge to a serializable dictionary"""
        return {
            "id": edge.id,
            "head_id": edge.head.id,
            "tail_id": edge.tail.id,
            "relation": self._serialize_relation(edge.relation)
        }

    def _serialize_entity(self, entity: Entity) -> dict:
        """Convert an Entity to a serializable dictionary"""
        return {
            "id": entity.id,
            "document_id": entity.document_id,
            "start_idx": entity.start_idx,
            "end_idx": entity.end_idx,
            "label": entity.label,
            "text": entity.text
        }

    def _serialize_relation(self, relation: Relation) -> dict:
        """Convert a Relation to a serializable dictionary"""
        return {
            "id": relation.id,
            "document_id": relation.document_id,
            "head": self._serialize_entity(relation.head),
            "tail": self._serialize_entity(relation.tail),
            "relation": relation.relation,
            "description": relation.description
        }

    def _serialize_document(self, document: Document) -> dict:
        """Convert a Document to a serializable dictionary"""
        return {
            "id": document.id,
            "text": document.text,
            "metadata": document.metadata,
            "entities": [self._serialize_entity(entity) for entity in document.entities],
            "relations": [self._serialize_relation(relation) for relation in document.relations]
        }

    def _deserialize_node(self, data: dict) -> Node:
        """Convert a dictionary to a Node"""
        entities = [self._deserialize_entity(entity_data) for entity_data in data["entities"]]
        return Node(id=data["id"], entities=entities)

    def _deserialize_edge(self, data: dict, node_map: dict) -> Edge:
        """
        Convert a dictionary to an Edge

        Args:
            data: Edge data
            node_map: Mapping from node ID to Node object
        """
        head_node = node_map[data["head_id"]]
        tail_node = node_map[data["tail_id"]]
        relation = self._deserialize_relation(data["relation"])
        return Edge(id=data["id"], head=head_node, tail=tail_node, relation=relation)

    def _deserialize_entity(self, data: dict) -> Entity:
        """Convert a dictionary to an Entity"""
        return Entity(
            id=data["id"],
            document_id=data["document_id"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            label=data["label"],
            text=data["text"]
        )

    def _deserialize_relation(self, data: dict) -> Relation:
        """Convert a dictionary to a Relation"""
        head = self._deserialize_entity(data["head"])
        tail = self._deserialize_entity(data["tail"])
        return Relation(
            id=data["id"],
            document_id=data["document_id"],
            head=head,
            tail=tail,
            relation=data["relation"],
            description=data["description"]
        )

    def _deserialize_document(self, data: dict) -> Document:
        """Convert a dictionary to a Document"""
        entities = [self._deserialize_entity(entity_data) for entity_data in data["entities"]]
        relations = [self._deserialize_relation(relation_data) for relation_data in data["relations"]]
        return Document(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"],
            entities=set(entities),
            relations=set(relations)
        )


    def query(self, graph: KnowledgeGraph, query: str) -> str:
        """
        Query the knowledge graph to get an answer.
        """
        print("boba2")
        retrieved_data = self.retriever.retrieve(query=query, knowledge_graph=graph)
        return self.graph_answering.generate_answer(query, retrieved_data)

    def is_graph_empty(self):
        """
        Check if the graph database is empty.
        """
        return self.graph_db.has_data()

    def generate_labels(self, documents: list[Document]) -> list[str]:
        """
        Generate NER labels from documents.
        """
        # TODO: log warning if sample size is bigger than the number of documents
        documents = random.sample(documents, k=min(self.config.sample_size_ner_labels, len(documents)))
        return self.ner_labels_generator.generate_labels(documents)

    def generate_entities(self, documents: list[Document]) -> list[Document]:
        """
        Generate entities for documents.
        """
        return self.ner_generator.generate(documents)

    def generate_relations(self, documents: list[Document]) -> list[Document]:
        """
        Generate relations between entities in documents.
        """
        print("boba1")
        return self.relation_generator.generate_relations(documents)

    def cluster(self, documents: list[Document]) -> KnowledgeGraph:
        """
        Cluster entities and relations into a knowledge graph.
        """
        text2entities = defaultdict(list)
        for document in documents:
            for entity in document.entities:
                text2entities[entity.text].append(entity)

        nodes = []
        for text, entities in text2entities.items():
            node = Node(id=str(uuid.uuid4()), entities=entities)
            nodes.append(node)

        text2node = dict()
        for node in nodes:
            text2node[node.text] = node

        edges = []
        for document in documents:
            for relation in document.relations:
                head_node = text2node[relation.head.text]
                tail_node = text2node[relation.tail.text]
                edge = Edge(id=relation.id, head=head_node, tail=tail_node, relation=relation)
                edges.append(edge)

        return KnowledgeGraph(nodes=nodes, edges=edges, documents=documents)
