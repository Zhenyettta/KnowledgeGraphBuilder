import os
from dataclasses import asdict

import streamlit as st

from kgg.models import RawDocument
from kgg.nodes.entity_extraction import GLiNEREntitiesGenerator
from kgg.nodes.ner_schema_generator import ConstantNERSchemaGenerator
from kgg.nodes.re_schema_generator import ConstantRelationsSchemaGenerator
from kgg.nodes.relations_extraction import GLiRELRelationsGenerator

st.title("Автоматизація наповнення баз знань")
st.write("Завантажте файл, щоб розпочати обробку.")

uploaded_file = st.file_uploader("Завантажте текстовий файл", type=["txt", "docx", "pdf"])
labels_file = st.file_uploader("Завантажте файл із labels (формат .txt)", type=["txt"])
manual_labels = st.text_input("Або введіть labels вручну через кому (наприклад: Person, Award, Date):")

labels = []
if labels_file:
    labels = [line.decode("utf-8").strip() for line in labels_file.readlines()]
    st.success(f"Labels успішно завантажено: {', '.join(labels)}")
elif manual_labels:
    labels = [label.strip() for label in manual_labels.split(",")]

if labels:
    st.write("Обрані labels:")
    st.json(labels)
else:
    st.warning("Введіть або завантажте labels для обробки.")

if uploaded_file and labels:
    upload_dir = os.path.join("front", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Файл '{uploaded_file.name}' успішно завантажено.")
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    ner_schema_generator = ConstantNERSchemaGenerator(labels)
    ner_schema = ner_schema_generator.invoke(text)
    raw_doc = RawDocument(text=text)
    entities_generator = GLiNEREntitiesGenerator()
    ner_document = entities_generator.invoke({"document": raw_doc, "schema": ner_schema})
    st.write("Результати витягування сутностей (GLiNER):")
    st.json(asdict(ner_document))
    relation_labels = [
        "born in", "died in", "educated at", "invented by"
    ]
    relation_schema_generator = ConstantRelationsSchemaGenerator(relation_labels)
    relation_schema = relation_schema_generator.invoke(raw_doc)
    ner_spans = [[e.token_start_idx, e.token_end_idx, e.label, e.text] for e in ner_document.entities]
    relations_generator = GLiRELRelationsGenerator()
    relation_document = relations_generator.invoke({"document": raw_doc, "ner": ner_spans, "schema": relation_schema})
    st.write("Результати витягування відношень (GLiREL):")
    st.json(asdict(relation_document))
