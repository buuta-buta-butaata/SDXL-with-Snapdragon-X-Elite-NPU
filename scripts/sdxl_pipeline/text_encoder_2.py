import logging
import gc
import os
import numpy as np
import onnxruntime as ort

from utils import qnn_ep_helper as qnn

logger = logging.getLogger(__name__)

class TextEncoder2:
    def __init__(self, text_encoder_2_dir, tokenizer_2_dir, use_torch=False):
        self.tokenizer_2 = None
        self.model_2 = None
        self.model_path = os.path.join(text_encoder_2_dir, "model.onnx")
        if use_torch:
            from .tokenizer_torch import Tokenizer
            self.tokenizer_2_path = tokenizer_2_dir
            self.Tokenizer = Tokenizer
        else:
            from .tokenizer import Tokenizer
            self.tokenizer_2_path = os.path.join(tokenizer_2_dir, "tokenizer_2.onnx")
            self.Tokenizer = Tokenizer

    def load_models(self):
        logger.debug(f"Load text_encoder_2 from: {self.model_path}")
        self.model_2 = ort.InferenceSession(self.model_path, qnn.session_options)
        self.tokenizer_2 = self.Tokenizer(self.tokenizer_2_path)

    def get_text_embeddings_2(self, prompt: str, auto_mem_free=True):
        if self.tokenizer_2 is None:
            self.tokenizer_2 = self.Tokenizer(self.tokenizer_2_path)

        input_ids_2 = self.tokenizer_2.run(prompt, auto_mem_free, 0)
        inputs = {"input_ids": input_ids_2}

        if self.model_2 is None:
            self.load_models()

        output_names = list(map(lambda x: x.name,  self.model_2.get_outputs()))
        output_list = self.model_2.run(output_names, inputs)

        pooled_prompt_embeds = output_list[0].astype(np.float32)
        prompt_embeds_2 = output_list[1].astype(np.float32)
        
        if auto_mem_free:
            self.free_memory()
     
        return pooled_prompt_embeds, prompt_embeds_2

    def free_memory(self):
        del self.tokenizer_2, self.model_2
        gc.collect()

if __name__ == "__main__":
    te = TextEncoder2()
    res_dict = te.get_text_embeddings_2("A beautiful cyberpunk city, 8k resolution")
    for k, v in res_dict.items():
        logger.info(f"入力名: {k}, 出力形状: {v.shape}")
