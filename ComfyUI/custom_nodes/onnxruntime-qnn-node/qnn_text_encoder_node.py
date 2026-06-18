import os
import torch
import numpy as np
import onnxruntime as ort
import folder_paths

from . import qnn_ep_helper as qnn
from . import common_utils as utils

# =====================================================================
# 2. QNN Text Encoder (CLIP) Loader & ラッパー
# =====================================================================
class QNNTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        te_list = folder_paths.get_filename_list("text_encoders")
        folders = sorted(list(set([os.path.dirname(f) for f in te_list if os.path.dirname(f) != ""])))
        if not folders: folders = ["フォルダを作成してください"]
        return {"required": {"folder_name": (folders,)}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "QNN_Optimization"

    def load_text_encoder(self, folder_name):
        base_path = os.path.abspath(os.path.join(folder_paths.get_output_directory(), "..", "models", "text_encoders", folder_name))
        path_te1 = os.path.join(base_path, "model.onnx")
        path_te2 = os.path.join(base_path, "..", "text_encoder_2", "model.onnx")

        #print(f"[QNN CLIP] セッション初期化中...")
        #sess_te1 = ort.InferenceSession(path_te1, sess_options=qnn.session_options)
        #sess_te2 = ort.InferenceSession(path_te2, sess_options=qnn.session_options)

        from comfy.sdxl_clip import SDXLTokenizer
        
        class QNNCLIPWrapper:
            def __init__(self):
                self.cond_stage_model = self
                self.patcher = self
                self.tokenizer = SDXLTokenizer()

            def tokenize(self, text, return_word_ids=False):
                return self.tokenizer.tokenize_with_weights(text, return_word_ids)

            def clone(self): return self

            def encode_from_tokens_scheduled(self, tokens):
                cond, pooled = self.encode_token_ids(tokens)
                return [[cond, {"pooled_output": pooled}]]

            def encode_token_ids(self, token_ids):
                print(f"[QNN CLIP] セッション初期化中...")
                sess_te1 = ort.InferenceSession(path_te1, sess_options=qnn.session_options)
                sess_te2 = ort.InferenceSession(path_te2, sess_options=qnn.session_options)
                tokens_1 = token_ids.get("l", None)
                tokens_2 = token_ids.get("g", None)

                # TE1 (CLIP-L)
                if tokens_1 is not None:
                    t1_np = np.array([t[0] for t in tokens_1[0]], dtype=np.int32)
                    if len(t1_np.shape) == 1: t1_np = np.expand_dims(t1_np, axis=0)
                else: t1_np = np.zeros((1, 77), dtype=np.int32)

                # TE2 (CLIP-G)
                if tokens_2 is not None:
                    t2_np = np.array([t[0] for t in tokens_2[0]], dtype=np.int32)
                    if len(t2_np.shape) == 1: t2_np = np.expand_dims(t2_np, axis=0)
                else: t2_np = np.zeros((1, 77), dtype=np.int32)

                input_name_1 = sess_te1.get_inputs()[0].name
                input_name_2 = sess_te2.get_inputs()[0].name

                out_te1_list = sess_te1.run(None, {input_name_1: t1_np})
                out_te2_list = sess_te2.run(None, {input_name_2: t2_np})
                
                # 【CLIP Skip=2 (hidden_states[-2]) の出力を取得】
                out_te1 = out_te1_list[1] 
                
                # TE2の出力展開（インデックスの逆転対策）
                if len(out_te2_list) > 1:
                    pooled_output_g = out_te2_list[0]
                    hidden_states_g = out_te2_list[1]
                else:
                    hidden_states_g = out_te2_list[0]
                    pooled_output_g = out_te2_list[0][:, 0, :]

                # SDXL結合仕様
                cond_np = np.concatenate([out_te1, hidden_states_g], axis=-1)

                cond_tensor = torch.from_numpy(cond_np).float()
                pooled_tensor = torch.from_numpy(pooled_output_g).float()

                del t1_np, t2_np, out_te1_list, out_te2_list, cond_np, pooled_output_g, hidden_states_g, sess_te1, sess_te2
                utils.clean_memory()

                return cond_tensor, pooled_tensor

        return (QNNCLIPWrapper(),)
