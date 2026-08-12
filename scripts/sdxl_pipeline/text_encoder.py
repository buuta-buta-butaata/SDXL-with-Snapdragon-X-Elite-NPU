import logging
import gc
import os
import onnxruntime as ort

from utils import qnn_ep_helper as qnn

logger = logging.getLogger(__name__)

class TextEncoder:
    def __init__(self, text_encoder_dir, tokenizer_dir, use_torch=False):
        self.tokenizer = None
        self.model = None
        self.model_path = os.path.join(text_encoder_dir, "model.onnx")
        if use_torch:
            from .tokenizer_torch import Tokenizer
            self.tokenizer_path = tokenizer_dir
            self.Tokenizer = Tokenizer
        else:
            from .tokenizer import Tokenizer
            self.tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.onnx")
            self.Tokenizer = Tokenizer

    def load_models(self):
        logger.debug(f"Load text_encoder from: {self.model_path}")
        self.model = ort.InferenceSession(self.model_path, sess_options=qnn.session_options)
        self.tokenizer = self.Tokenizer(self.tokenizer_path)

    def get_text_embeddings(self, prompt: str, auto_mem_free=True):
        if self.tokenizer is None:
            self.tokenizer = self.Tokenizer(self.tokenizer_path)

        input_ids = self.tokenizer.run(prompt, auto_mem_free)
        inputs = {"input_ids": input_ids}

        if self.model is None:
            self.load_models()
        output_names = list(map(lambda x: x.name,  self.model.get_outputs()))
        output_list = self.model.run(output_names, inputs)
        
        prompt_embeds = output_list[1]
        
        if auto_mem_free:
            self.free_memory()
            
        return prompt_embeds

    def free_memory(self):
        del self.tokenizer, self.model
        gc.collect()

if __name__ == "__main__":
    # 単体テスト用
    te = TextEncoder()
    res = te.get_text_embeddings("A beautiful cyberpunk city, 8k resolution")
    logger.info(f"Text Encoder 1 出力形状: {res.shape}") # 想定: (1, 77, 768)
