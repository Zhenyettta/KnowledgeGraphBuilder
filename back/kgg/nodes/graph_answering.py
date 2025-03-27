from copy import deepcopy

from langchain_core.messages import BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from kgg.config import KGGConfig
from kgg.models import Document
from kgg.prompts import GRAPH_ANSWERING_PROMPT
from kgg.utils import initialize_llm


class GraphAnswering:
    def __init__(self, config: KGGConfig):
        self.config = config
        llm_config = deepcopy(config)
        self.llm = initialize_llm(llm_config, num_ctx=16000)
        self.prompt = GRAPH_ANSWERING_PROMPT

    def generate_answer(self, question: str, documents: list[Document]) -> str:
        try:
            texts = []
            for i, document in enumerate(documents, 1):
                texts.append(f"[Text {i}] {document.text}")

            context_texts = "\n\n".join(texts)
            response = self.llm.invoke(
                self.prompt.format_prompt(
                    question=question,
                    texts=context_texts
                )
            )
            print(response.content)
            print(question)

            return self._process_response(response)

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while trying to answer your question."

    def _process_response(self, response: BaseMessage) -> str:
        if not response or not response.content:
            return "No answer could be generated."

        content = response.content.strip()

        think_start = content.find("<think>")
        if think_start != -1:
            think_end = content.find("</think>", think_start)
            if think_end != -1:
                content = content[:think_start] + content[think_end + 8:]

        return content


