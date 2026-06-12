import gc
import os
import numpy as np
import onnxruntime as ort
import qnn_ep_helper as qnn
from transformers import CLIPTokenizer

class TextEncoder2:
    def __init__(self, text_encoder_2_dir, tokenizer_2_dir):
        self.tokenizer_2 = None
        self.model_2 = None
        self.model_path = os.path.join(text_encoder_2_dir, "model.onnx")
        self.tokenizer_2_path = tokenizer_2_dir

    def get_text_embeddings_2(self, prompt: str, auto_mem_free=True):
        print("--- Text Encoder 2 処理開始 ---")
        
        # 1. ローカルフォルダからトークナイザー2をロード
        if self.tokenizer_2 is None:
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.tokenizer_2_path)
        
        # 2. トークンIDに変換
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="np"
        )
        # print(f"tokens_2: {text_inputs_2}")
        input_ids_2 = text_inputs_2.input_ids.astype(np.int32)
        
        # 3. NPUにロードして推論
        if self.model_2 is None:
            self.model_2 = ort.InferenceSession(self.model_path, qnn.session_options)
        output_names = list(map(lambda x: x.name,  self.model_2.get_outputs()))
        output_list = self.model_2.run(output_names, {"input_ids": input_ids_2})

        # 4. 出力データの取得
        pooled_prompt_embeds = output_list[0].astype(np.float32)
        prompt_embeds_2 = output_list[1].astype(np.float32)
        
        # 5. メモリの完全解放
        if auto_mem_free:
            self.free_memory()
     
        del text_inputs_2
        
        return pooled_prompt_embeds, prompt_embeds_2

    def free_memory(self):
        del self.tokenizer_2, self.model_2
        gc.collect()

if __name__ == "__main__":
    # 単体テスト用
    te = TextEncoder2()
    res_dict = te.get_text_embeddings_2("A beautiful cyberpunk city, 8k resolution")
    for k, v in res_dict.items():
        print(f"入力名: {k}, 出力形状: {v.shape}")
        # 想定される出力:
        # 1. prompt_embeds_2 相当 -> (1, 77, 1280)
        # 2. pooled_prompt_embeds (text_embeds) 相当 -> (1, 1280)
