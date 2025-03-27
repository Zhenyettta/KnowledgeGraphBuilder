import gc
import json
import re

import torch
from langchain_core.messages import BaseMessage

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


    def generate_labels(self, documents: list[Document]) -> list[str]:
        try:
            self._load_model()
            unique_labels = set()
            for document in documents:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                response = self.llm.invoke(self.prompt.format_prompt(
                    user_input=document.text,
                ))
                unique_labels.update(self._parse_response(response))
            print(f"Unique labels: {len(unique_labels)}")
            self.unload_model()
            return sorted(unique_labels)

        except Exception as e:
            print(f"Error generating schema: {e}")
            return []

    def _parse_response(self, response: BaseMessage) -> list[str]:
        generated_text = response.content.strip()
        if not generated_text:
            print("Warning: Empty response content.")
            return []

        match = re.search(r'\[.*?]', generated_text, re.DOTALL)

        if not match:
            return []

        try:
            json_str = match.group(0).replace("'", '"').replace("\n", "")
            labels = json.loads(json_str)
            return list({
                str(label).lower().strip().replace(" ", "_")
                for label in labels
                if isinstance(label, (str, int, float))
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON Parsing Error: {e}, Response: {generated_text}")
            return []


    def _load_model(self):
        if self.llm is None:
            print("Loading llm model to GPU...")
            self.llm = initialize_llm(self.config, num_ctx=9000)

    def unload_model(self):
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
            gc.collect()
            print("Model unloaded from GPU")


