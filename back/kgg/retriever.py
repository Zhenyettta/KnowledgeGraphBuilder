from kgg.models import Document, KnowledgeGraph
from kgg.io.graph import GraphDatabase, VectorDatabase


class KnowledgeGraphRetriever:
    # FIXME it should rather instantiate databases on its own, not receive them as arguments
    def __init__(self, ...):  # TODO whatever else needed, config, so on..
        pass
        # graph_db =
        # vector_db =

    def index(self, knowledge_graph: KnowledgeGraph):
        # TODO: it should upload the graph to the graph database, then encode the relation descriptions and upload their vectors to the vector database
        pass

    def retrieve(self, query: str) -> list[Document]:
        # TODO this should be better documented, if we want to leave it as a docstring!
        """
        Steps:

        1. Encode the query to the vector
        2. Retrieve N (should be in config) most similar documents from the vector database (it contains relation descriptions)
        3. Retrieve the edges from the graph database (it contains the entities - nodes)
        4. Aggregate weights on the entities from the similarities of the edges
            * Set all nodes to have 0 weight
            * For each edge, increment the weight of the nodes it connects by the similarity of the edge
        5. Run personalized pagerank on the graph
        6. Create a ranking of documents based on the weights of the nodes
            * For each document, find all nodes that mention it and set
              the weight of the document to the sum of their weights
        7. Sort the documents by their weights and return the top M (should be in config) documents
        """
        pass
