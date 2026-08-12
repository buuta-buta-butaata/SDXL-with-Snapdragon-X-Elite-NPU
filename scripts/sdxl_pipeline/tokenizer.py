import gc
import logging
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self, tokenizer_path):
        self.model = None
        self.model_path = tokenizer_path
        self.session_options = ort.SessionOptions()
        self.session_options.register_custom_ops_library(r".\lib\ortextensions.dll")
        self.load_models()

    def load_models(self):
        logger.debug(f"Load tokenizer from: {self.model_path}")
        self.model = ort.InferenceSession(self.model_path, sess_options=self.session_options)

    def run(self, prompt: str, auto_mem_free=True, padding=49407):
        if self.model is None:
            self.load_models()
            
        input_ids = self.model.run(["input_ids", "attention_mask"],
                                   {"string_input": [prompt]})[0][0].astype(np.int32)
        if len(input_ids) > 77:
            logger.warning(f"tokens length: {len(input_ids)}")
            logger.warning(f"Your input was truncated because text_encoder can only handle sequences up to 77 tokens.")
            input_ids[76] = 49407
        logger.debug(f"tokens length: {len(input_ids)}")
        input_ids = np.pad(input_ids[:77], (0, 77 - len(input_ids[:77])), mode="constant", constant_values=(0, padding))
        logger.debug(f"tokens: {input_ids}")

        if auto_mem_free:
            self.free_memory()
            
        return [input_ids]

    def free_memory(self):
        del self.model
        gc.collect()


if __name__ == "__main__":
    pass
