import streamlit as st
import os

from kgg.extract_entities import extract_entities
from kgg.extract_relations import extract_relations_with_auto_labels

# Налаштування інтерфейсу
st.title("Автоматизація наповнення баз знань")
st.write("Завантажте файл, щоб розпочати обробку.")

# Секція для завантаження файлу
uploaded_file = st.file_uploader("Завантажте текстовий файл", type=["txt", "docx", "pdf"])
labels_file = st.file_uploader("Завантажте файл із labels (формат .txt)", type=["txt"])

# Введення labels вручну
manual_labels = st.text_input("Або введіть labels вручну через кому (наприклад: Person, Award, Date):")

# Зчитування labels із файлу або вручну
labels = []
# Read labels from file if uploaded
if labels_file:
    labels = [line.decode("utf-8").strip() for line in labels_file.readlines()]
    st.success(f"Labels успішно завантажено: {', '.join(labels)}")
elif manual_labels:
    labels = [label.strip() for label in manual_labels.split(",")]

# Перевірка завантаження labels
if labels:
    st.write("Обрані labels:")
    st.json(labels)
else:
    st.warning("Введіть або завантажте labels для обробки.")


if uploaded_file and labels:
    # Збереження завантаженого файлу
    upload_dir = os.path.join("front", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Файл '{uploaded_file.name}' успішно завантажено.")

    # Читання тексту з файлу
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Виклик функції GLiNER (extract_entities)
    st.write("Результати витягування сутностей (GLiNER):")
    try:
        entities = extract_entities(text, labels)
        st.json(entities)
        relations = extract_relations_with_auto_labels(text, entities)
        for relation in relations:
            st.json(relation)
    except Exception as e:
        st.error(f"Помилка при витягуванні сутностей: {e}")
