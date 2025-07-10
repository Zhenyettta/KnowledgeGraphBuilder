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
    if "kgg_generator" not in st.session_state: # Initialize generator
        st.session_state.kgg_generator = KnowledgeGraphGenerator(st.session_state.config)

def initialize_sidebar_config():
    with st.sidebar:
        st.header("Configuration Settings")

        # Store previous config values to check for changes
        prev_gliner_model = st.session_state.config.gliner_model
        prev_ner_labels = st.session_state.config.ner_labels
        prev_sample_size_ner_labels = st.session_state.config.sample_size_ner_labels
        prev_ner_threshold = st.session_state.config.ner_threshold
        prev_synonym_threshold = st.session_state.config.synonym_threshold

        st.session_state.config.gliner_model = st.text_input(
            "GLiNER Model",
            value=st.session_state.config.gliner_model
        )

        ner_labels_input = st.text_input(
            "Predefined GLiNER Labels",
            value=",".join(st.session_state.config.ner_labels)
        )

        st.session_state.config.ner_labels = [label.strip() for label in ner_labels_input.split(',')] if ner_labels_input else []


        st.session_state.config.sample_size_ner_labels = st.number_input(
            "Sample Size NER Labels",
            value=st.session_state.config.sample_size_ner_labels
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

        # Check if any config value changed
        config_changed = (
            prev_gliner_model != st.session_state.config.gliner_model or
            prev_ner_labels != st.session_state.config.ner_labels or
            prev_sample_size_ner_labels != st.session_state.config.sample_size_ner_labels or
            prev_ner_threshold != st.session_state.config.ner_threshold or
            prev_synonym_threshold != st.session_state.config.synonym_threshold
        )

        if config_changed:
            # Re-initialize the generator if config changed
            st.session_state.kgg_generator = KnowledgeGraphGenerator(st.session_state.config)
            st.sidebar.info("Configuration updated. Generator re-initialized.")


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
    if filepath.endswith('.jsonl'): # Corrected from .json to .jsonl
        return parse_jsonl_file(filepath)
    elif filepath.endswith('.txt'):
        with open(filepath, "r", encoding="utf-8") as file:
            return [Document(id="doc_0", text=file.read())]
    else:
        # TODO change to streamlit error, not a python one
        raise ValueError('unsupported file format')

def handle_neo4j_operations(document): # document parameter seems unused here
    neo4j_inserter = Neo4jRelationsInserter(user="neo4j", password="newPassword") # Consider making credentials configurable

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Neo4j Database"):
            clear_result = neo4j_inserter.clear()
            st.success(f"Clearing result: {clear_result}")

    with col2:
        if st.button("Insert Data into Neo4j"):
            # Use the generator from session state
            generator = st.session_state.kgg_generator
            graph = generator._load_graph_from_cache() # Ensure this is the desired graph
            if graph:
                generator.graph_db.import_graph(graph)
                st.success("Data successfully inserted into Neo4j!")
            else:
                st.warning("No graph found in cache to insert into Neo4j.")


def qa_tab():
    st.header("Question Answering")
    user_question = st.text_input("Enter your question:", key="qa_question")

    if st.button("Get Answer") and user_question:
        try:
            with st.spinner("Generating answer..."):
                # Use the generator from session state
                generator = st.session_state.kgg_generator
                answer = generator.generate(query=user_question, use_cache=True)

            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")


def data_processing_tab():
    st.header("Document Processing")
    st.write("Upload a file to begin processing.")

    uploaded_file = st.file_uploader("Upload a text file", type=["txt", "jsonl", "json"])

    if uploaded_file and not st.session_state.file_uploaded:
        uploads_dir = setup_environment()
        file_path = save_uploaded_file(uploaded_file, uploads_dir) # Store file_path
        st.session_state.file_uploaded = True
        st.success(f"File '{uploaded_file.name}' successfully uploaded.")
        # Parse after saving and use the actual file_path
        processed_doc = parse_uploaded_file(file_path)
        st.session_state.raw_doc = processed_doc
        st.session_state.relation_document = processed_doc # Assuming this is intended
        st.session_state.knowledge_graph = None

    st.divider()


    process_query = st.text_input("Enter an optional query for processing:", key="process_query",
                                  placeholder="Leave empty to just process the document")

    if st.button("Process Document"):
        if st.session_state.raw_doc:
            with st.spinner("Processing document..."):
                try:
                    # Use the generator from session state
                    generator = st.session_state.kgg_generator
                    answer = generator.generate(
                        documents=st.session_state.raw_doc,
                        query=process_query if process_query else None, # Pass None if empty
                        use_cache=False # Assuming new processing shouldn't use cache for the whole graph
                    )

                    if process_query:
                        st.markdown("### Result")
                        st.write(answer)

                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        else:
            st.warning("Please upload a document first.")


    handle_neo4j_operations(st.session_state.relation_document)


def main():
    st.title("Knowledge Graph Generation and Question Answering")

    initialize_session_state() # Initialize state first
    initialize_sidebar_config() # Then config, which might re-init generator

    tab1, tab2 = st.tabs(["Document Processing", "Question Answering"])

    with tab1:
        data_processing_tab()

    with tab2:
        qa_tab()


if __name__ == "__main__":
    main()