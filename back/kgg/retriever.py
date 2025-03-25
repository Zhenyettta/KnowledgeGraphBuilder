from collections import defaultdict

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from kgg.io.graph import GraphDatabase
from kgg.models import KnowledgeGraph


class KnowledgeGraphRetriever:
    # FIXME it should rather instantiate databases on its own, not receive them as arguments
    def __init__(self, graph_database: GraphDatabase):  # TODO whatever else needed, config, so on..

        self.graph_database = graph_database

        bge_m3_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True, "batch_size": 100}
        )

        index = faiss.IndexFlatIP(1024)

        self.vector_store = FAISS(
            embedding_function=bge_m3_embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy=DistanceStrategy.COSINE,
        )

    def index(self, knowledge_graph: KnowledgeGraph):
        documents = []
        for edge in knowledge_graph.edges:
            description = edge.description
            documents.append(Document(page_content=description, metadata={"id": edge.id}))

        self.vector_store.add_documents(documents=documents)

    # def retrieve(self, query: str) -> list[Document]:
    #     # TODO this should be better documented, if we want to leave it as a docstring!
    #     """
    #     Steps:
    #
    #     1. Encode the query to the vector
    #     2. Retrieve N (should be in config) most similar documents from the vector database (it contains relation descriptions)
    #     3. Retrieve the edges from the graph database (it contains the entities - nodes)
    #     4. Aggregate weights on the entities from the similarities of the edges
    #         * Set all nodes to have 0 weight
    #         * For each edge, increment the weight of the nodes it connects by the similarity of the edge
    #     5. Run personalized pagerank on the graph
    #     6. Create a ranking of documents based on the weights of the nodes
    #         * For each document, find all nodes that mention it and set
    #           the weight of the document to the sum of their weights
    #     7. Sort the documents by their weights and return the top M (should be in config) documents
    #     """
    #     pass

    def retrieve(self, knowledge_graph: KnowledgeGraph, query: str, k=3) -> list[Document]:
        """Query the indexed knowledge graph"""
        if self.vector_store.index.ntotal == 0:
            raise ValueError("Knowledge graph not indexed. Call index() first.")

        retrieved = self.vector_store.similarity_search_with_score(query, k=k)
        edge_id2score = {doc.metadata["id"]: score for doc, score in retrieved}
        edge_id2edge = {edge.id: edge for edge in knowledge_graph.edges}

        node_ids = set()
        for edge_id in edge_id2score.keys():
            node_ids.add(edge_id2edge[edge_id].head.id)
            node_ids.add(edge_id2edge[edge_id].tail.id)

        node_ids = list(node_ids)
        ranked_node_ids = self.graph_database.personalized_pagerank(node_ids, edge_id2score)

        document_weights = {doc.id: 0 for doc in knowledge_graph.documents}

        for node in knowledge_graph.nodes:
            if node.id in ranked_node_ids:
                node_score = ranked_node_ids[node.id]['score']
                for entity in node.entities:
                    document_weights[entity.document_id] += node_score

        sorted_documents = sorted(
            knowledge_graph.documents,
            key=lambda doc: document_weights.get(doc.id, 0),
            reverse=True
        )

        # Return the top `k` documents
        return sorted_documents[:k]



        # fsociety00.dat
        # ---------- readme.txt ----------
        #
        #           LEAVE ME HERE
        #
        # --------------------------------
