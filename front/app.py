import os
from dataclasses import asdict
import streamlit as st
from kgg.models import RawDocument, ER_instruction, NER_instruction
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator
from kgg.nodes.re_schema_generator import HTTPServerRelationSchemaGenerator
from kgg.nodes.relations_extraction import GLiRELRelationsGenerator
from kgg.nodes.neo4j_loader import Neo4jRelationsInserter

st.title("Автоматизація наповнення баз знань")
st.write("Завантажте файл, щоб розпочати обробку.")

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "relation_document" not in st.session_state:
    st.session_state.relation_document = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

uploaded_file = st.file_uploader("Завантажте текстовий файл", type=["txt", "docx", "pdf"])

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

    ner_schema_generator = HTTPServerRelationSchemaGenerator(instruction=NER_instruction)
    ner_schema = ner_schema_generator.invoke(raw_doc)

    entities_generator = GLiNEREntitiesGenerator()
    ner_document = entities_generator.invoke({"document": raw_doc, "schema": ner_schema})

    relation_schema_generator = HTTPServerRelationSchemaGenerator(instruction=ER_instruction)
    relation_schema = relation_schema_generator.invoke(raw_doc)

    relations_generator = GLiRELRelationsGenerator()
    relation_document = relations_generator.invoke({"document": ner_document, "schema": relation_schema})

    st.session_state.relation_document = relation_document

if st.session_state.relation_document:
    st.write("Результати витягування відношень (GLiREL):")
    st.json(asdict(st.session_state.relation_document))

    neo4j_inserter = Neo4jRelationsInserter(user="neo4j", password="newPassword")

    if st.button("Очистити базу даних Neo4j"):
        clear_result = neo4j_inserter.clear_database()
        st.success(f"Результат очищення: {clear_result}")

    if st.button("Вставити дані у Neo4j"):
        insert_result = neo4j_inserter.invoke(st.session_state.relation_document)
        st.success(f"Результати вставлення: {insert_result}")
