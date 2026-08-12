import logging
import numpy as np

logger = logging.getLogger(__name__)

from transformers import CLIPTokenizer

class Tokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
    
    def load_models(self):
        logger.debug(f"Load tokenizer from: {self.tokenizer_path}")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_path)

    def run(self, prompt: str, auto_mem_free=True, padding=49407):
        if self.tokenizer is None:
            self.load_models()

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np"
        )
        input_ids = text_inputs.input_ids.astype(np.int32)
        logger.debug(f"tokens length: {len(input_ids)}")
        logger.debug(f"tokens: {input_ids}")
        return input_ids
