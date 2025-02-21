import uuid
from typing import Optional, Any, Dict
from neo4j import GraphDatabase
from langchain_core.runnables import Runnable, RunnableConfig
from kgg.models import RawDocument

class BaseNeo4jLoader(Runnable[RawDocument, Dict[str, Any]]):
    def invoke(self, input: RawDocument, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError()

class Neo4jRelationsInserter(BaseNeo4jLoader):
    def __init__(self, bolt_uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "newPassword", database: Optional[str] = None):
        self.driver = GraphDatabase.driver(bolt_uri, auth=(user, password))
        self.database = database

    def invoke(self, input: RawDocument, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            document_result = session.execute_write(self._create_document, input.text)
            document_id = document_result["document_id"]
            entities_count = 0
            for entity in input.entities:
                result = session.execute_write(self._create_entity, document_id, entity)
                if result["created"]:
                    entities_count += 1
            relations_count = 0
            for relation in input.relations:
                result = session.execute_write(self._create_relation, document_id, relation)
                if result["created"]:
                    relations_count += 1
        return {"document_id": document_id, "entities_created": entities_count, "relations_created": relations_count}

    def clear_database(self, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(self._clear_all)
        return result

    @staticmethod
    def _create_document(tx, text: str) -> Dict[str, Any]:
        doc_id = str(uuid.uuid4())
        query = "CREATE (d:Document {text: $text, doc_id: $doc_id}) RETURN d.doc_id AS document_id"
        result = tx.run(query, text=text, doc_id=doc_id)
        record = result.single()
        return {"document_id": record["document_id"]} if record else {"document_id": None}

    @staticmethod
    def _create_entity(tx, document_id: str, entity) -> Dict[str, Any]:
        query = (
            "MATCH (d:Document {doc_id: $document_id}) "
            "CREATE (e:Entity {text: $text, label: $label, token_start_idx: $token_start_idx, token_end_idx: $token_end_idx}) "
            "MERGE (d)-[:HAS_ENTITY]->(e) "
            "RETURN true AS created"
        )
        result = tx.run(query, document_id=document_id, text=entity.text, label=entity.label, token_start_idx=entity.token_start_idx, token_end_idx=entity.token_end_idx)
        record = result.single()
        return {"created": record["created"]} if record else {"created": False}

    @staticmethod
    def _create_relation(tx, document_id: str, relation) -> Dict[str, Any]:
        head_text = " ".join(relation.head_text) if isinstance(relation.head_text, list) else relation.head_text
        tail_text = " ".join(relation.tail_text) if isinstance(relation.tail_text, list) else relation.tail_text
        rel_type = relation.label
        query = (
                "MATCH (d:Document {doc_id: $document_id}) "
                "MATCH (d)-[:HAS_ENTITY]->(head:Entity {text: $head_text}) "
                "MATCH (d)-[:HAS_ENTITY]->(tail:Entity {text: $tail_text}) "
                "CREATE (head)-[r:" + rel_type + " {score: $score}]->(tail) "
                                                 "RETURN true AS created"
        )
        result = tx.run(query, document_id=document_id, head_text=head_text, tail_text=tail_text, score=relation.score)
        record = result.single()
        return {"created": record["created"]} if record else {"created": False}


    @staticmethod
    def _clear_all(tx) -> Dict[str, Any]:
        query = "MATCH (n) DETACH DELETE n RETURN 'database cleared' AS status"
        result = tx.run(query)
        record = result.single()
        return {"status": record["status"]} if record else {"status": "no operation"}
