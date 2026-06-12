import gc
import os
import numpy as np
import onnxruntime as ort
import qnn_ep_helper as qnn

class VAEDecoder:
    def __init__(self, config):
        self.model = None
        self.onnx_path = self.find_model_path(config.dirs.vae_decoder_dir,
                                              config.width, config.height)

    def find_model_path(self, model_dir, width, height):
        single_graph_model = os.path.join(model_dir, "model.onnx")
        if os.path.exists(single_graph_model):
            return single_graph_model
        model_path = os.path.join(model_dir, f"vae_decoder_{width}x{height}.onnx")
        if os.path.exists(model_path):
            return model_path
        return os.path.join(model_dir, f"{width}x{height}", "model.onnx")

    def decode_latents(self, latents: np.ndarray, auto_mem_free=True):
        print("--- VAE Decoder 処理開始 ---")
    
        # 1. 入力データの型を float32 に合わせる
        if latents.dtype != np.float16:
            latents = latents.astype(np.float16)
        
        # VAEの入力名は通常 'latent_sample' や 'sample' です。Netronの入力名に合わせて指定してください
        # ここでは仮に 'latent_sample' とします
        input_name = "latent_sample"
    
        # 2. NPUにロードして推論
        if self.model is None:
            model = ort.InferenceSession(self.onnx_path, sess_options=qnn.session_options)
        output_names = list(map(lambda x: x.name, model.get_outputs()))
    
        output_list = model.run(output_names, {input_name: latents})
    
        # 3. 出力データの取得 (通常、第1出力が画像array)
        image_array = output_list[0]
    
        # 4. メモリの完全解放
        if auto_mem_free:
            self.free_memory()

        return image_array.astype(np.float32)

    def free_memory(self):
        # メモリの完全解放
        del self.model
        gc.collect()

if __name__ == "__main__":
    # 単体テスト用
    # UNetの最終出力 (1, 4, 128, 128) を想定したダミー入力
    dummy_latents = np.zeros((1, 4, 128, 128), dtype=np.float32)
    deco = VAEDecoder()
    res_image = deco.decode_latents(dummy_latents)
    print(f"VAE 出力形状: {res_image.shape}") # 想定: (1, 3, 1024, 1024)
