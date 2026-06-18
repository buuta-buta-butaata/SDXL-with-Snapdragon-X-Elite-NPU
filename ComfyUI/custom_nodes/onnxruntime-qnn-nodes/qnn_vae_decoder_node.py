import os
import torch
import numpy as np
import onnxruntime as ort
import folder_paths

from . import qnn_ep_helper as qnn
from . import common_utils as utils
from .qnn_unet_node import QNNUNetLoader
from .qnn_base_path_node import QNNBasePathLoader

# =====================================================================
# 3. QNN VAE Decoder Loader & ラッパー
# =====================================================================
class QNNVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # パス指定ノードからルートパスを引き継ぎます
                "base_path": ("QNN_BASE_PATH",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "QNN_Optimization"

    def load_vae(self, base_path):
        # 引き渡されたベースパスから、HF構造のVAE Decoderへのフルパスを生成
        path_vae = os.path.join(base_path, "vae_decoder", "model.onnx")
        
        class QNNVAEWrapper:
            def decode(self, latent_tokens):
                # パート3：QNNVAELoader の一番最初に追加してRAMを確保
                print(f"[QNN VAE] 連動処理：UNetセッションの強制アンロードを実行します。")
                QNNUNetLoader.unload_all_unet_sessions()
                
                print(f"[QNN VAE] セッション初期化中...")
                sess_vae = ort.InferenceSession(path_vae, sess_options=qnn.session_options)
                
                latent_np = latent_tokens.cpu().numpy().astype(np.float16) / 0.13025
                
                input_name = sess_vae.get_inputs()[0].name
                out_vae_list = sess_vae.run(None, {input_name: latent_np})
                out_vae = out_vae_list[0]
                
                # 画像の形状整形 (NCHW -> NHWC)
                if len(out_vae.shape) == 4 and out_vae.shape[1] == 3:
                    out_vae = out_vae.transpose(0, 2, 3, 1)
                
                # 色空間のスケール正規化（一般的なVAEに合わせた調整）
                out_vae = (out_vae + 1.0) / 2.0
                out_vae = np.clip(out_vae, 0.0, 1.0)
                
                out_tensor = torch.from_numpy(out_vae).float().cpu()
                print(f"[QNN VAE] 画像のデコード完了。Shape: {out_tensor.shape}")
                
                # 最終出力直前で不要になった巨大データをすべて完全消去
                del latent_np, out_vae_list, out_vae, sess_vae
                utils.clean_memory()
                
                return out_tensor

        return (QNNVAEWrapper(),)
