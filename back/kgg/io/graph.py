from kgg.models import KnowledgeGraph


# TODO write docstrings everywhere (use AI)
class GraphDatabase:
    # TODO handle all ids in the DB to make sure we can query it
    def import_graph(self, knowledge_graph: KnowledgeGraph):
        raise NotImplementedError()

    def get_nodes(self, ids: list[str]) -> ???:  # TODO it should return something with the identifier of the node from the DB
        raise NotImplementedError()

    def get_edges(self, ids: list[str]) -> ???:  # TODO same here
        raise NotImplementedError()

    def personalized_pagerank(self, node_weights: dict[str, float]) -> dict[str, float]:  # TODO str being a node_id (maybe change it to a Node object)
        raise NotImplementedError()


class Neo4j(GraphDatabase):
    def __init__(self, host: ..., port: ..., ...):
    # TODO implement
    pass
