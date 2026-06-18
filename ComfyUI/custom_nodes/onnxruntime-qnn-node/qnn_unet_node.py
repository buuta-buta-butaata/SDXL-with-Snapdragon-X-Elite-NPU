import os
import torch
import numpy as np
import onnxruntime as ort

import folder_paths

from . import qnn_ep_helper as qnn
from . import common_utils as utils
# =====================================================================
# 1. 5-Part UNet Loader & 推論ロジック
# =====================================================================
class QNNUNetLoader:
    # 【ここを追加】生成されたラッパー（Wrapper）の参照も追跡できるようにします
    active_wrapper = None
    
    @classmethod
    def INPUT_TYPES(s):
        unet_list = folder_paths.get_filename_list("diffusion_models")
        folders = sorted(list(set([os.path.dirname(f) for f in unet_list if os.path.dirname(f) != ""])))
        if not folders: folders = ["フォルダを作成してください"]
        return {"required": {"folder_name": (folders,)}}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_qnn_model"
    CATEGORY = "QNN_Optimization"

    @classmethod
    def unload_all_unet_sessions(cls, *args, **kwargs):
        """
        【ここを大幅に強化】
        すべての参照元（Loaderクラス、Wrapperオブジェクト）の変数を
        完全に None で上書きし、ONNX Runtimeにメモリを強制返還させます。
        """
        # ラッパーオブジェクトが握っているセッションの参照をすべて強制切断（None化）
        if cls.active_wrapper is not None:
            print(f"[QNN UNet] ラッパーオブジェクト内のセッション参照を切断中...")
            for attr in ['sess_common', 'sess_down', 'sess_mid', 'sess_up1', 'sess_up2']:
                if hasattr(cls.active_wrapper, attr):
                    setattr(cls.active_wrapper, attr, None)

            # Pythonのゴミ箱とお掃除関数をフル稼働
            utils.clean_memory()
            
    def load_qnn_model(self, folder_name):
        base_path = os.path.join(folder_paths.get_output_directory(), "..", "models", "diffusion_models", folder_name)
        
        paths = {
            "part0": os.path.join(base_path, "model.onnx"),
            "part1": os.path.join(base_path, "..", "Part1", "unet_part1_1024x1024.onnx"),
            "part2": os.path.join(base_path, "..", "Part2", "unet_part2_1024x1024.onnx"),
            "part3": os.path.join(base_path, "..", "Part3", "unet_part3_1024x1024.onnx"),
            "part4": os.path.join(base_path, "..", "Part4", "unet_part4_1024x1024.onnx"),
        }
        
        for name, path in paths.items():
            if not os.path.exists(path): raise FileNotFoundError(f"Missing: {path}")

        from comfy.model_base import BaseModel, ModelType
        from comfy.model_patcher import ModelPatcher
        from comfy.latent_formats import SDXL
        
        class QNNUNetWrapper:
            def __init__(self):
                print(f"[QNN UNet] セッション初期化中...")
                self._load_models()
                self.out_names_common = [x.name for x in self.sess_common.get_outputs()]
                self.out_names_down = [x.name for x in self.sess_down.get_outputs()]
                self.out_names_mid = [x.name for x in self.sess_mid.get_outputs()]
                self.out_names_up1 = [x.name for x in self.sess_up1.get_outputs()]
                self.out_names_up2 = [x.name for x in self.sess_up2.get_outputs()]
                self.dtype = torch.float32

            def _load_models(self):
                self.sess_common = ort.InferenceSession(paths["part0"], sess_options=qnn.session_options)
                self.sess_down = ort.InferenceSession(paths["part1"], sess_options=qnn.session_options)
                self.sess_mid = ort.InferenceSession(paths["part2"], sess_options=qnn.session_options)
                self.sess_up1 = ort.InferenceSession(paths["part3"], sess_options=qnn.session_options)
                self.sess_up2 = ort.InferenceSession(paths["part4"], sess_options=qnn.session_options)

            def __call__(self, x, timesteps, context, *args, **kwargs):
                """ComfyUI KSamplerメインループ（バッチサイズ自動分割対応版）"""
                
                # 1. 2回目以降の実行で、セッションが消えていたら自動リロードする（あなたの修正を踏襲）
                if self.sess_common is None:
                    print("[QNN UNet] 🌟 2回目の生成：アンロード済みのセッションを再読み込みします。")
                    self._load_models()

                # 入力PyTorchテンソルの形状をチェック
                batch_size, channels, latent_h, latent_w = x.shape
                
                # ComfyUIがまとめて送ってきたバッチ（2など）を、1つずつ処理して結果を貯めるリスト
                output_tensors = []

                # バッチの数（0番目、1番目…）だけループを回して、1件ずつONNXに流します
                for b in range(batch_size):
                    # 各入力テンソルから、b番目のデータだけを切り出して「バッチサイズ1」の形にする
                    # [b:b+1] と書くことで、次元数を減らさずに [1, C, H, W] を維持できます
                    x_single = x[b:b+1]
                    
                    # timesteps は [2] のように送られてくることがあるため、形状に合わせて切り分けます
                    if timesteps.shape[0] > 1:
                        timesteps_single = timesteps[b:b+1]
                    else:
                        timesteps_single = timesteps

                    # context (encoder_hidden_states) もb番目を抽出
                    context_single = context[b:b+1]

                    # --- NumPy型変換（バッチサイズは常に1になります） ---
                    latents_np = x_single.cpu().numpy().astype(np.float16)
                    timestep_np = timesteps_single.cpu().numpy().astype(np.float32)
                    encoder_np = context_single.cpu().numpy().astype(np.float16)
                    
                    text_embeds_np = None
                    time_ids_np = None
                    
                    # SDXL追加の引数の抽出
                    if "y" in kwargs and kwargs["y"] is not None:
                        y_tensor = kwargs["y"]
                        # yのバッチサイズも入力に合わせて切り出す
                        y_single = y_tensor[b:b+1] if y_tensor.shape[0] > 1 else y_tensor
                        y_np = y_single.cpu().numpy().astype(np.float32)
                        
                        if y_np.shape[-1] > 1280:
                            text_embeds_np = y_np[..., :1280]
                            time_ids_np = y_np[..., 1280:]
                        else:
                            text_embeds_np = y_np

                    # 動的解像度/バッチサイズ1への安全なフォールバック
                    img_h = latent_h * 8
                    img_w = latent_w * 8
                    if text_embeds_np is None:
                        text_embeds_np = np.zeros((1, 1280), dtype=np.float32)
                    if time_ids_np is None:
                        time_ids_np = np.array([[float(img_h), float(img_w), 0.0, 0.0, float(img_h), float(img_w)]], dtype=np.float32)

                    base_inputs = {"text_embeds": text_embeds_np, "time_ids": time_ids_np}

                    # ---- 5分割 UNet 推論実行 (ここはバッチ1として完璧に回ります) ----
                    # 0. Part 0: Common
                    feed_common = {"timestep": timestep_np, **base_inputs}
                    out_list_common = self.sess_common.run(self.out_names_common, feed_common)
                    common_inputs = out_list_common[0].astype(np.float16)
                    
                    # 1. Part 1: Down
                    feed_down = {"silu_3": common_inputs, "sample": latents_np, "encoder_hidden_states": encoder_np}
                    out_list_down = self.sess_down.run(self.out_names_down, feed_down)
                    skip_connections = dict(zip(self.out_names_down, out_list_down))
                    skip_connections['add_88'] = skip_connections['add_88'].transpose(0, 3, 1, 2)
                    
                    # 2. Part 2: Mid
                    feed_mid = {"add_88": skip_connections["add_88"], "silu_3": common_inputs, "encoder_hidden_states": encoder_np}
                    out_list_mid = self.sess_mid.run(self.out_names_mid, feed_mid)
                    add_123 = out_list_mid[0]
                    
                    # 3. Part 3: Up 前半
                    feed_up1 = {"add_88": skip_connections["add_88"], "add_123": add_123, "silu_3": common_inputs, "encoder_hidden_states": encoder_np}
                    out_list_up1 = self.sess_up1.run(self.out_names_up1, feed_up1)
                    add_156 = out_list_up1[0].transpose(0, 3, 1, 2)
                    
                    # 4. Part 4: Up 後半 + 最終Output
                    part4_inputs = skip_connections.copy()
                    del part4_inputs["add_88"]
                    feed_up2 = {"add_156": add_156, **part4_inputs, "silu_3": common_inputs, "encoder_hidden_states": encoder_np}
                    out_list_up2 = self.sess_up2.run(self.out_names_up2, feed_up2)
                    noise_pred = out_list_up2[0]
                    
                    # ---- 出力整形（各件ごとにPyTorchテンソル化） ----
                    noise_pred_f32 = noise_pred.astype(np.float32)
                    out_single_tensor = torch.from_numpy(noise_pred_f32).to(x.device)
                    
                    if out_single_tensor.shape[-1] == channels:
                        out_single_tensor = out_single_tensor.permute(0, 3, 1, 2)
                    
                    # リストに結果を保存
                    output_tensors.append(out_single_tensor)
                    
                    # メモリの小まめなお掃除
                    del latents_np, timestep_np, encoder_np, common_inputs, skip_connections, add_123, add_156, noise_pred
                    utils.clean_memory()

                # ---- 最終合流処理 ----
                # ループで貯めた「バッチサイズ1のテンソル」たちを、
                # 元通りのバッチ方向（次元0）にガッチャンコして、バッチサイズ2(以上)の1つのテンソルに戻します
                final_output = torch.cat(output_tensors, dim=0)
                
                return final_output

        # ラッパーのインスタンスを生成し、クラス変数に参照を追跡させます
        wrapper_instance = QNNUNetWrapper()
        QNNUNetLoader.active_wrapper = wrapper_instance
        
        class DummyConfig:
            def __init__(self):
                self.unet_config = {}
                self.latent_format = SDXL()

        class QNNBaseModel(BaseModel):
            @classmethod
            def create_mock(cls):
                obj = cls.__new__(cls)
                obj.__dict__['model_config'] = DummyConfig()
                obj.__dict__['model_type'] = ModelType.EPS
                obj.__dict__['diffusion_model'] = wrapper_instance # 追跡中のインスタンスを設定
                obj.__dict__['latent_format'] = SDXL()
                
                from comfy.model_sampling import ModelSamplingDiscrete, EPS
                class QNNSDXLSampling(ModelSamplingDiscrete, EPS):
                    def __init__(self): super().__init__()
                obj.__dict__['model_sampling'] = QNNSDXLSampling()
                
                obj.__dict__['memory_usage_factor'] = 1.0
                obj.__dict__['memory_usage_factor_conds'] = ()
                obj.__dict__['memory_usage_shape_process'] = {}
                obj.__dict__['concat_keys'] = []
                obj.__dict__['current_patcher'] = obj
                obj.__dict__['manual_cast_dtype'] = None
                obj.__dict__['memory_management_limit'] = lambda *args, **kwargs: None
                
                # PyTorch管理フックのダミー
                for hook in ['_modules', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_state_dict_pre_hooks', '_load_state_dict_pre_hooks']:
                    obj.__dict__[hook] = {}
                obj.__dict__['_non_persistent_buffers_set'] = set()
                obj.__dict__['_is_full_backward_hook'] = None
                return obj

            def prepare_state(self, *args, **kwargs): pass
            def state_dict(self, *args, **kwargs): return {}
            def model_size(self): return 0
            def loaded_size(self): return 0
            def process_latent_in(self, latent): return latent
            def process_latent_out(self, latent): return latent

        base_model = QNNBaseModel.create_mock()
        patcher = ModelPatcher(base_model, load_device="cpu", offload_device="cpu")
        patcher.model_options = {"transformer_options": {"wrappers": {}}}
        patcher.load_device = torch.device("cpu")
        patcher.offload_device = torch.device("cpu")
        patcher.model_is_loaded = lambda *args, **kwargs: True
        patcher.wrappers = {}
        
        return (patcher,)
