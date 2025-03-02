import os
from dataclasses import asdict
import streamlit as st
from kgg.models import RawDocument, ER_instruction, NER_instruction, EXAMPLE_DOCUMENT1, EXAMPLE_ENTITIES1, \
    EXAMPLE_RELATIONS1, NER_PROMPT
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator
from kgg.nodes.re_schema_generator import HTTPServerRelationSchemaGenerator, KGGenGenerator
from kgg.nodes.relations_extraction import GLiRELRelationsGenerator
from kgg.nodes.neo4j_loader import Neo4jRelationsInserter

st.title("Автоматизація наповнення баз знань")
st.write("Завантажте файл, щоб розпочати обробку.")

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "raw_doc" not in st.session_state:
    st.session_state.raw_doc = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "relation_document" not in st.session_state:
    st.session_state.relation_document = None

uploaded_file = st.file_uploader("Завантажте текстовий файл", type=["txt", "docx", "pdf"])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Select processing pipeline
processing_option = st.radio(
    "Оберіть метод обробки:",
    ("KGGenGenerator", "GLiNER + GLiREL")
)

if uploaded_file and not st.session_state.file_uploaded:
    upload_dir = os.path.join("front", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.file_uploaded = True
    st.success(f"Файл '{uploaded_file.name}' успішно завантажено.")

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    raw_doc = RawDocument(text=text)

    if processing_option == "KGGenGenerator":
        kggen = KGGenGenerator()
        processed_doc = kggen.invoke(raw_doc)
    else:
        ner_schema_generator = HTTPServerRelationSchemaGenerator(prompt=NER_PROMPT)
        ner_schema = ner_schema_generator.invoke(raw_doc)

        entities_generator = GLiNEREntitiesGenerator()
        raw_doc = entities_generator.invoke({"document": raw_doc, "schema": ner_schema})

        relation_schema_generator = HTTPServerRelationSchemaGenerator()
        relation_schema = relation_schema_generator.invoke(raw_doc)

        relations_generator = GLiRELRelationsGenerator()
        processed_doc = relations_generator.invoke({"document": raw_doc, "schema": relation_schema})

    st.session_state.raw_doc = processed_doc
    st.session_state.relation_document = processed_doc

if st.session_state.raw_doc:
    st.write("Результати обробки:")
    st.json(asdict(st.session_state.raw_doc))

    neo4j_inserter = Neo4jRelationsInserter(user="neo4j", password="newPassword")

    if st.button("Очистити базу даних Neo4j"):
        clear_result = neo4j_inserter.clear_database()
        st.success(f"Результат очищення: {clear_result}")

    if st.button("Вставити дані у Neo4j"):
        insert_result = neo4j_inserter.invoke(st.session_state.relation_document)
        st.success(f"Результати вставлення: {insert_result}")
