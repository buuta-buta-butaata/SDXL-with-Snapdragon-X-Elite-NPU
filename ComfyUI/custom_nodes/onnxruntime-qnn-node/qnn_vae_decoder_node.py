import os
import torch
import numpy as np
import onnxruntime as ort
import folder_paths

from . import qnn_ep_helper as qnn
from . import common_utils as utils
from .qnn_unet_node import QNNUNetLoader

# =====================================================================
# 3. QNN VAE Decoder Loader & ラッパー
# =====================================================================
class QNNVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        vae_list = folder_paths.get_filename_list("vae")
        folders = sorted(list(set([os.path.dirname(f) for f in vae_list if os.path.dirname(f) != ""])))
        if not folders: folders = ["フォルダを作成してください"]
        return {"required": {"folder_name": (folders,)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "QNN_Optimization"

    def load_vae(self, folder_name):
        base_path = os.path.abspath(os.path.join(folder_paths.get_output_directory(), "..", "models", "vae", folder_name))
        path_vae = os.path.join(base_path, "model.onnx")

        class QNNVAEWrapper:
            def decode(self, latent_tokens):
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
                # パート3：QNNVAELoader の一番最後、clean_memory() の直前に追加
                print(f"[QNN VAE] 連動処理：UNetセッションの強制アンロードを実行します。")
                QNNUNetLoader.unload_all_unet_sessions()
                utils.clean_memory()
                
                return out_tensor

        return (QNNVAEWrapper(),)
