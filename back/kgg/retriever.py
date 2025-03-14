import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from kgg.models import KnowledgeGraph


class KnowledgeGraphRetriever:
    # FIXME it should rather instantiate databases on its own, not receive them as arguments
    def __init__(self):  # TODO whatever else needed, config, so on..
        pass

        bge_m3_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3"
        )

        index = faiss.IndexFlatL2(1024)

        self.vector_store = FAISS(
            embedding_function=bge_m3_embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def index(self, knowledge_graph: KnowledgeGraph):
        documents = []
        for edge in knowledge_graph.edges:
            description = edge.description
            documents.append(Document(page_content=description, metadata={"id": edge.id}))

        self.vector_store.add_documents(documents=documents)

        # Print information about the index
        print(f"Index size: {self.vector_store.index.ntotal}")
        print(f"Dimension: {self.vector_store.index.d}")
        print(f"Number of documents: {len(self.vector_store.docstore._dict)}")

        # You can also print the first few document IDs
        print(f"Document IDs: {list(self.vector_store.index_to_docstore_id.values())[:5]}")

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
