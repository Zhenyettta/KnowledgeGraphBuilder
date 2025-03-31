import os
from pathlib import Path

import jsonlines
import streamlit as st

from kgg.config import KGGConfig
from kgg.generator import KnowledgeGraphGenerator
from kgg.models import Document
from kgg.nodes.neo4j_loader import Neo4jRelationsInserter


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
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = None


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

        # st.session_state.config.encoder_model = st.text_input(
        #     "Encoder Model",
        #     value=st.session_state.config.encoder_model
        # )

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


def qa_tab():
    st.header("Question Answering")
    user_question = st.text_input("Enter your question:", key="qa_question")

    if st.button("Get Answer") and user_question:
        try:
            with st.spinner("Generating answer..."):
                generator = KnowledgeGraphGenerator(st.session_state.config)
                answer = generator.generate(query=user_question, use_cache=True)

            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")


def data_processing_tab():
    st.header("Document Processing")
    st.write("Upload a file to begin processing.")

    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "jsonl"])

    if uploaded_file and not st.session_state.file_uploaded:
        uploads_dir = setup_environment()
        save_uploaded_file(uploaded_file, uploads_dir)
        st.session_state.file_uploaded = True
        st.success(f"File '{uploaded_file.name}' successfully uploaded.")

        processed_doc = parse_uploaded_file(os.path.join(uploads_dir, uploaded_file.name))
        st.session_state.raw_doc = processed_doc
        st.session_state.relation_document = processed_doc
        st.session_state.knowledge_graph = None

    st.divider()

    if st.session_state.raw_doc:
        process_query = st.text_input("Enter an optional query for processing:", key="process_query",
                                      placeholder="Leave empty to just process the document")

        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    generator = KnowledgeGraphGenerator(st.session_state.config)
                    answer = generator.generate(
                        documents=st.session_state.raw_doc,
                        query=process_query,
                        use_cache=False
                    )

                    if process_query:
                        st.markdown("### Result")
                        st.write(answer)

                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        handle_neo4j_operations(st.session_state.relation_document)


def main():
    st.title("Knowledge Graph Generation and Question Answering")

    initialize_session_state()
    initialize_sidebar_config()

    tab1, tab2 = st.tabs(["Document Processing", "Question Answering"])

    with tab1:
        data_processing_tab()

    with tab2:
        qa_tab()


if __name__ == "__main__":
    main()
