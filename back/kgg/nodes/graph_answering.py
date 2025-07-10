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
        llm_config.llm_model="deepseek-r1:14b"
        self.llm = initialize_llm(llm_config, num_ctx=15000)
        self.prompt = GRAPH_ANSWERING_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")


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

            return self._process_response(response)

        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while trying to answer your question."

    def _process_response(self, response: BaseMessage) -> str:
        if not response or not response.content:
            return "No answer could be generated."

        content = response.content.strip()

        # Remove thinking sections
        think_start = content.find("<think>")
        if think_start != -1:
            think_end = content.find("</think>", think_start)
            if think_end != -1:
                content = content[:think_start] + content[think_end + 8:]

        # Log the full response for analysis
        print("Full LLM response:")
        print(content)

        # Extract the direct answer from the response
        # First, try to get the first sentence or paragraph which is often the direct answer
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else ""

        # If the response starts with "Based on the provided texts" or similar, extract the actual answer
        if first_line.startswith(("Based on", "According to", "From the", "The ")):
            # Extract the first sentence which is likely the direct answer
            sentences = first_line.split('.')
            if sentences:
                direct_answer = sentences[0].strip()
                if direct_answer.startswith(("Based on the provided texts, ", "According to the texts, ")):
                    direct_answer = direct_answer.split(", ", 1)[1] if ", " in direct_answer else direct_answer

                # Log the extracted direct answer
                print("Extracted direct answer:")
                print(direct_answer)

                # If the direct answer is too short or generic, use the full response
                if len(direct_answer) < 5 or direct_answer in ["I cannot answer this question based on the provided information"]:
                    return content

                return direct_answer

        # If we couldn't extract a clear direct answer, return the full response
        return content


    def length_function(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))
