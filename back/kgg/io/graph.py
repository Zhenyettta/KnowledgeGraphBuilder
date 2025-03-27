from typing import Optional, Dict, List, Any, cast

from neo4j import GraphDatabase as Neo4jDriver
from neo4j.exceptions import ServiceUnavailable
from neo4j.graph import Node as Neo4jNode
from typing_extensions import LiteralString

from kgg.models import KnowledgeGraph, Edge, Node


class GraphDatabase:
    def import_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        raise NotImplementedError()

    def get_nodes(self, ids: list[str]) -> list[Node]:
        raise NotImplementedError()

    def get_edges(self, ids: list[str]) -> list[Edge]:
        raise NotImplementedError()

    def personalized_pagerank(self, node_ids: list[str], edge_ids2weights: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError()


class Neo4j(GraphDatabase):
    def __init__(
            self,
            uri: str = "bolt://localhost:7687",
            user: str = "neo4j",
            password: str = "newPassword",
            database: Optional[str] = None,
            max_connection_lifetime: int = 3600,
            max_connection_pool_size: int = 100,
            connection_timeout: int = 30,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database

        self.driver = Neo4jDriver.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_timeout=connection_timeout
        )

        try:
            self.driver.verify_connectivity()
        except ServiceUnavailable:
            raise ConnectionError(f"Failed to connect to Neo4j at {uri}")

    def close(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()

    def import_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        # Create nodes
        for node in knowledge_graph.nodes:
            self.create_node(node.id, node.text)

        # Create edges
        for edge in knowledge_graph.edges:
            self.create_edge(edge.id, edge.head.id, edge.tail.id, edge.relation.relation,
                             edge.relation.id, edge.relation.description)

    def create_node(self, node_id: str, text: str) -> None:
        with self.driver.session(database=self.database) as session:
            query = cast(LiteralString,
                         "CREATE (n:Node {id: $id, text: $text}) "
                         "RETURN n"
                         )
            session.run(query, id=node_id, text=text)

    def create_edge(self, edge_id: str, head_id: str, tail_id: str, rel_type: str, relation_id: str,
                    description: str, weight: float = 0.2) -> None:
        with self.driver.session(database=self.database) as session:
            query = cast(LiteralString,
                         "MATCH (head:Node {id: $head_id}), (tail:Node {id: $tail_id}) "
                         "CREATE (head)-[r:Edge {id: $edge_id, relation_id: $relation_id, rel_type: $rel_type, "
                         "description: $description, weight: $weight}]->(tail) "
                         "RETURN r"
                         )
            session.run(query, head_id=head_id, tail_id=tail_id, edge_id=edge_id, relation_id=relation_id,
                        description=description, rel_type=rel_type, weight=weight)

    def update_edge_weights_by_id(self, target_ids: List[str], weight: float) -> None:
        with self.driver.session(database=self.database) as session:
            query = cast(LiteralString,
                         "MATCH ()-[r:Edge]->() "
                         "WHERE r.id IN $target_ids "
                         "SET r.weight = $weight"
                         )
            session.run(query, target_ids=target_ids, weight=weight)

    def get_nodes(self, ids: list[str]) -> list[Neo4jNode]:
        with self.driver.session(database=self.database) as session:
            query = cast(LiteralString, """
                MATCH (n:Node)
                WHERE n.id IN $ids
                RETURN n
            """)
            result = session.run(query, ids=ids)
            nodes = [record["n"] for record in result]
            return nodes

    def has_data(self) -> bool:
        with self.driver.session(database=self.database) as session:
            query = cast(LiteralString, "MATCH (n) RETURN count(n) as count")
            result = session.run(query)
            count = result.single()["count"]
            return count > 0

    def personalized_pagerank(self, node_ids: list[str], edge_ids2weights: dict[str, float]) -> Dict[
        str, Dict[str, Any]]:
        print(node_ids, edge_ids2weights)
        for id, weight in edge_ids2weights.items():
            self.update_edge_weights_by_id([id], weight)

        nodes = self.get_nodes(node_ids)
        user_info = {n['id']: {'internal_id': n.id, 'text': n['text']} for n in nodes}

        missing = set(node_ids) - user_info.keys()
        if missing:
            raise ValueError(f"Missing nodes: {missing}")

        internal_ids = [info['internal_id'] for info in user_info.values()]
        internal_scores = self.run_pagerank(internal_ids)

        return {
            uid: {
                'score': internal_scores[info['internal_id']],
                'text': info['text']
            } for uid, info in user_info.items()
        }

    def run_pagerank(self, source_ids: List[int]) -> Dict[int, float]:
        with self.driver.session(database=self.database) as session:
            graph_name = "myGraph"
            create_graph_query = cast(LiteralString, """
                MATCH (source:Node)-[r:Edge]->(target:Node)
                RETURN gds.graph.project(
                'myGraph',
                source,
                target,
                { relationshipProperties: r { .weight }},
                { undirectedRelationshipTypes: ['*']})
            """)
            session.run(create_graph_query)

            pagerank_query = cast(LiteralString, """
                CALL gds.pageRank.stream($name, {
                    sourceNodes: $source_ids,
                    maxIterations: 20,
                    dampingFactor: 0.85,
                    relationshipWeightProperty: 'weight'
                })
                YIELD nodeId, score
                RETURN nodeId, score
            """)
            result = session.run(pagerank_query, name=graph_name, source_ids=source_ids)

            scores = {record["nodeId"]: record["score"] for record in result}

            # drop_query = cast(LiteralString, "CALL gds.graph.drop($name)")
            # session.run(drop_query, name=graph_name)
            return scores
