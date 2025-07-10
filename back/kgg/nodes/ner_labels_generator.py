import gc
import json
import re

import torch
from langchain_core.messages import BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from json_repair import repair_json

from kgg.config import KGGConfig
from kgg.models import Document
from kgg.prompts import NER_PROMPT
from kgg.utils import initialize_llm


# TODO TRY TO AGGREGATE LABELS AGAIN WITH LLM
class NERLabelsGenerator:
    def __init__(self, config: KGGConfig):
        self.config = config
        self.llm = None
        self.prompt = NER_PROMPT
        self.chunk_size = 300
        self.overlap = 30
        self.text_splitter = None
        self.tokenizer = None

    def generate_labels(self, documents: list[Document]) -> list[str]:
        try:
            self._load_model()
            unique_labels = set()

            for document in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(document.text)
                total_chunks = len(chunks)
                print(f"Document split into {total_chunks} chunks for label generation")

                for chunk_text in chunks:
                    response = self._invoke_with_retry(self.prompt.format_prompt(
                        user_input=chunk_text,
                    ))
                    unique_labels.update(self._parse_response(response))
                    print(len(unique_labels))

            print(f"Unique labels: {len(unique_labels)}")
            self.unload_model()
            return sorted(unique_labels)

        except Exception as e:
            print(f"Error generating schema: {e}")
            return []

    def _invoke_with_retry(self, prompt: str, max_attempts: int = 5) -> BaseMessage:
        """
        Invoke the LLM and automatically retry the same prompt if the model
        stopped because the context/window was exhausted (`finish_reason == "length"`).
        """
        attempt = 1
        while True:
            response: BaseMessage = self.llm.invoke(prompt)

            finish_reason = response.response_metadata.get("done_reason")

            if finish_reason == "length" and attempt < max_attempts:
                print(f"[warning] Truncated response (finish_reason='length') – retrying ({attempt}/{max_attempts})")
                attempt += 1
                continue  # resend the identical prompt
            return response

    def _parse_response(self, response: BaseMessage) -> list[str]:
        generated_text = response.content.strip()
        if not generated_text:
            print("Warning: Empty response content.")
            return []

        match = re.search(r'\[.*?]', generated_text, re.DOTALL)

        if match:
            try:
                json_str = match.group(0).replace("\n", " ")
                repaired_json = repair_json(json_str)
                labels = json.loads(repaired_json)

                return list({
                    str(label).lower().strip().replace(" ", "_")
                    for label in labels
                    if isinstance(label, (str, int, float))
                })
            except (json.JSONDecodeError, Exception) as e:
                print(f"JSON repair failed: {e}")
                content = match.group(0).strip('[]')
                items = re.findall(r"'[^']*'|\"[^\"]*\"|[^,]+", content)

                labels = []
                for item in items:
                    item = item.strip().strip("'\"").strip()
                    if item:
                        labels.append(item)

                return list({
                    str(label).lower().strip().replace(" ", "_")
                    for label in labels
                    if label
                })

        print("No list found in response.")
        return []

    def _load_model(self):
        if self.llm is None:
            print("Loading llm model to GPU...")
            self.llm = initialize_llm(self.config, num_ctx=4000, num_predict=1000)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
            self.text_splitter = RecursiveCharacterTextSplitter(
                length_function=self.length_function,
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
            )

    def unload_model(self):
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
            gc.collect()
            print("Model unloaded from GPU")

    def length_function(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))
