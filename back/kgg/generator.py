import random

from kgg.config import KGGConfig
from kgg.models import Document, KnowledgeGraph
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator
from kgg.nodes.ner_labels_generator import NERLabelsGenerator


# TODO old pipeline based on config
class KnowledgeGraphGenerator:
    # TODO somehow pass the configuration or whatever
    def __init__(self, config: KGGConfig):
        # TODO here we should create a langchain chain based on the configuration

        self.config = config
        self.ner_labels_generator = NERLabelsGenerator(config)
        self.ner_generator = GLiNEREntitiesGenerator(config)

    def generate(self, documents: list[Document]) -> KnowledgeGraph:
        # TODO 1: if ner_labels not present, use your step to generate them ✅
        #               * sample N documents and generate all possible entity labels for them
        #               * create a unique set of labels aggregating all generated labels
        #               * sort them and store in a list!
        # TODO 2: use gliner and ner_labels to generate entities, then populate document objects with generated entities (pay attention to storing document ids and so on..)✅
        # TODO 3: use llm and generated entities to generate relations, then populate document objects with them (same here)
        # TODO 4: add clustering step (for now it will not merge entities, but it should create Node and Edge objects, they in turn should contain a list of Entity and Relation objects)
        # TODO 5: create a KnowledgeGraph object containing the list of documents and list of nodes and edges
        if not self.config.ner_labels:
            self.config.ner_labels = self.generate_labels(documents)
            print(self.generate_entities(documents))

    def generate_labels(self, documents: list[Document]) -> list[str]:
        # TODO: log warning if sample size is bigger than the number of documents
        documents = random.sample(documents, k=self.config.sample_size_ner_labels)
        return self.ner_labels_generator.generate_labels(documents)

    def generate_entities(self, documents: list[Document]) -> list[Document]:
        return self.ner_generator.generate(documents)

    # ner_schema_generator = HTTPServerRelationSchemaGenerator(prompt=NER_PROMPT)
    # ner_schema = ner_schema_generator.invoke(raw_doc)
    #
    # entities_generator = GLiNEREntitiesGenerator()
    # doc_with_entities = entities_generator.invoke({"document": raw_doc, "schema": ner_schema})
    #
    # relation_schema_generator = HTTPServerRelationSchemaGenerator()
    # relation_schema = relation_schema_generator.invoke(doc_with_entities)
    #
    # relations_generator = GLiRELRelationsGenerator()
    # processed_doc = relations_generator.invoke({"document": doc_with_entities, "schema": relation_schema})
