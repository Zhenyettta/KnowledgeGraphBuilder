from typing import Optional, Any, Dict, Tuple, Set
from warnings import deprecated

from neo4j import GraphDatabase, Query
from langchain_core.runnables import Runnable, RunnableConfig
from kgg.models import Document

@deprecated
class BaseNeo4jLoader(Runnable[Document, Dict[str, Any]]):
    def __init__(
        self,
        bolt_uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "newPassword",
        database: Optional[str] = None
    ):
        self.driver = GraphDatabase.driver(bolt_uri, auth=(user, password))
        self.database = database

    def invoke(
        self,
        input: Document,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        raise NotImplementedError()

class Neo4jRelationsInserter:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "newPassword"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def invoke(
        self,
        input: Document
    ):
        with self.driver.session() as session:
            for entity_name in input.entities:
                session.run("MERGE (e:Entity {name: $name})", name=entity_name)
            for head, rel_type, tail in input.relations:
                rel_type = rel_type.replace(" ", "_").upper()
                rel_type = rel_type.replace(" ", "_").replace("-", "_").upper()
                query_str = f"""
                MATCH (h:Entity {{name: $head}})
                MATCH (t:Entity {{name: $tail}})
                MERGE (h)-[r:{rel_type}]->(t)
                RETURN r
                """
                query = Query(query_str)  # type: ignore
                session.run(query, head=head, tail=tail)

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self):
        self.driver.close()
