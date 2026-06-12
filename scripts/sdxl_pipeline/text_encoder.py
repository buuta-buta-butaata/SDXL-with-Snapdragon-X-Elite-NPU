import gc
import os
import numpy as np
import onnxruntime as ort
import qnn_ep_helper as qnn
from transformers import CLIPTokenizer

class TextEncoder:
    def __init__(self, text_encoder_dir, tokenizer_dir):
        self.tokenizer = None
        self.model = None
        self.model_path = os.path.join(text_encoder_dir, "model.onnx")
        self.tokenizer_path = tokenizer_dir

    def get_text_embeddings(self, prompt: str, auto_mem_free=True):
        print("--- Text Encoder 1 処理開始 ---")
        
        # 1. ローカルフォルダからトークナイザーをロード
        if self.tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_path)
        
        # 2. プロンプトをトークンIDに変換 (最大長77に固定)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np"
        )
        # print(f"tokens: {text_inputs}")
        input_ids = text_inputs.input_ids.astype(np.int32)
        
        # 3. NPUにロードして推論
        if self.model is None:
            self.model = ort.InferenceSession(self.model_path, sess_options=qnn.session_options)
        output_names = list(map(lambda x: x.name,  self.model.get_outputs()))
        output_list = self.model.run(output_names, {"input_ids": input_ids})
        
        # 4. 出力データの取得
        prompt_embeds = output_list[1]
        
        # 5. メモリの完全解放
        if auto_mem_free:
            self.free_memory()
            
        del text_inputs
        
        return prompt_embeds

    def free_memory(self):
        # メモリの完全解放
        del self.tokenizer, self.model
        gc.collect()

if __name__ == "__main__":
    # 単体テスト用
    te = TextEncoder()
    res = te.get_text_embeddings("A beautiful cyberpunk city, 8k resolution")
    print(f"Text Encoder 1 出力形状: {res.shape}") # 想定: (1, 77, 768)
