import jsonlines
import os
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from kgg import config
from kgg.generator import KnowledgeGraphGenerator
from kgg.models import Document, KnowledgeGraph
from kgg.nodes.neo4j_loader import Neo4jRelationsInserter
from kgg.prompts import NER_PROMPT
from kgg.config import KGGConfig



def setup_environment() -> str:
    if os.getenv('KGG_UPLOADS_CACHE_DIR'):
        uploads_dir = Path(os.getenv('KGG_UPLOADS_CACHE_DIR'))
    else:
        uploads_dir = Path.home() / '.cache' / 'kgg' / 'uploads'
    uploads_dir.mkdir(parents=True, exist_ok=True)

    return str(uploads_dir)

def initialize_session_state():
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if "raw_doc" not in st.session_state:
        st.session_state.raw_doc = None
    if "relation_document" not in st.session_state:
        st.session_state.relation_document = None
    if "config" not in st.session_state:
        st.session_state.config = KGGConfig()


def initialize_sidebar_config():
    with st.sidebar:
        st.header("Configuration Settings")

        st.session_state.config.gliner_model = st.text_input(
            "GLiNER Model",
            value=st.session_state.config.gliner_model
        )

        st.session_state.config.predefined_gliner_labels = st.text_input(
            "Predefined GLiNER Labels",
            value=st.session_state.config.ner_labels
        )

        st.session_state.config.sample_size_ner_labels = st.number_input(
            "Sample Size NER Labels",
            value=st.session_state.config.sample_size_ner_labels
        )

        st.session_state.config.encoder_model = st.text_input(
            "Encoder Model",
            value=st.session_state.config.encoder_model
        )

        st.session_state.config.ner_threshold = st.slider(
            "NER Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config.ner_threshold,
            step=0.05
        )

        st.session_state.config.synonym_threshold = st.slider(
            "Synonym Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config.synonym_threshold,
            step=0.05
        )

        st.session_state.config.use_old_approach = st.checkbox(
            "Use Old Approach",
            value=st.session_state.config.use_old_approach
        )



# def process_file_with_gliner_glirel(raw_doc):
#     ner_schema_generator = HTTPServerRelationSchemaGenerator(prompt=NER_PROMPT)
#     ner_schema = ner_schema_generator.invoke(raw_doc)
#
#     entities_generator = GLiNEREntitiesGenerator()
#     doc_with_entities = entities_generator.invoke({"document": raw_doc, "schema": ner_schema})
#
#     relation_schema_generator = HTTPServerRelationSchemaGenerator()
#     relation_schema = relation_schema_generator.invoke(doc_with_entities)
#
#     relations_generator = GLiRELRelationsGenerator()
#     processed_doc = relations_generator.invoke({"document": doc_with_entities, "schema": relation_schema})
#
#     return processed_doc
#
#
# def process_file_with_gliner_llm(raw_doc):
#     ner_schema_generator = HTTPServerRelationSchemaGenerator(prompt=NER_PROMPT)
#     ner_schema = ner_schema_generator.invoke(raw_doc)
#
#     entities_generator = GLiNEREntitiesGenerator()
#     doc_with_entities = entities_generator.invoke({"document": raw_doc, "schema": ner_schema})
#
#     relation_extractor = GLiNERRelationExtractor()
#     processed_doc = relation_extractor.invoke(doc_with_entities)
#
#     return processed_doc


def parse_jsonl_file(filepath: str) -> list[Document]:
    documents = []

    with jsonlines.open(filepath, 'r') as reader:
        for i, document in enumerate(reader):
            documents.append(Document(
                id=document.get('id', f"doc_{i}"),
                text=document['text'],
                metadata=document.get('metadata', dict()),
            ))

    return documents


def save_uploaded_file(uploaded_file, upload_dir):
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def parse_uploaded_file(filepath: str):
    if filepath.endswith('.jsonl'):
        return parse_jsonl_file(filepath)
    elif filepath.endswith('.txt'):
        with open(filepath, "r", encoding="utf-8") as file:
            return [Document(id="doc_0", text=file.read())]
    else:
        # TODO change to streamlit error, not a python one
        raise ValueError('unsupported file format')


def handle_neo4j_operations(document):
    neo4j_inserter = Neo4jRelationsInserter(user="neo4j", password="newPassword")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Очистити базу даних Neo4j"):
            clear_result = neo4j_inserter.clear()
            st.success(f"Результат очищення: {clear_result}")

    with col2:
        if st.button("Вставити дані у Neo4j"):
            insert_result = neo4j_inserter.invoke(document)
            st.success(f"Результати вставлення: {insert_result}")


def main():
    st.title("Автоматизація наповнення баз знань")
    st.write("Завантажте файл, щоб розпочати обробку.")

    uploads_dir = setup_environment()
    initialize_session_state()

    if not st.session_state.file_uploaded:
        initialize_sidebar_config()


    uploaded_file = st.file_uploader("Завантажте текстовий файл",
                                     type=["txt", "jsonl"])

    processing_option = st.radio(
        "Оберіть метод обробки:",
        ("KGGenGenerator", "GLiNER + GLiREL", "GLiNER + LLM")
    )

    if uploaded_file and not st.session_state.file_uploaded:
        save_uploaded_file(uploaded_file, uploads_dir)
        st.session_state.file_uploaded = True
        st.success(f"Файл '{uploaded_file.name}' успішно завантажено.")


        processed_doc = parse_uploaded_file(os.path.join(uploads_dir, uploaded_file.name))
        generator = KnowledgeGraphGenerator(st.session_state.config)
        generator.generate(documents=processed_doc)

        # TODO

        # TODO 1: Get configuration from streamlit sidebar (optional)
        # TODO 2: Create KnowledgeGraphGenerator using the configuration
        # TODO 3: Generate knowledge graph
        # TODO 4: Cache knowledge graph on the disk (optional)
        # TODO 5: fix app py
        # TODO 5: Change the state of the application to retrieval mode
        # TODO 6: Create KnowledgeGraphRetriever using the configuration (it should have connections to DBs)
        # TODO 7: Index the knowledge graph using KnowledgeGraphRetriever
        # TODO 8: Get question from the user and retrieve the relevant documents
        # TODO 9: Generate an answer based on the retrieved documents (optinal, but recommended)

        st.session_state.raw_doc = processed_doc
        st.session_state.relation_document = processed_doc

    if st.session_state.raw_doc:
        st.write("Результати обробки:")
        st.json(asdict(st.session_state.raw_doc))

        handle_neo4j_operations(st.session_state.relation_document)


if __name__ == "__main__":
    main()
