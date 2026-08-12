import logging
import gc
import os
import numpy as np
import onnxruntime as ort

from utils import qnn_ep_helper as qnn

logger = logging.getLogger(__name__)

class VAEDecoder:
    def __init__(self, config):
        self.model = None
        self.onnx_path = self.find_model_path(config.dirs["vae_decoder_dir"],
                                              config.width, config.height)

    def find_model_path(self, model_dir, width, height):
        single_graph_model = os.path.join(model_dir, "model.onnx")
        if os.path.exists(single_graph_model):
            return single_graph_model
        model_path = os.path.join(model_dir, f"vae_decoder_{width}x{height}.onnx")
        if os.path.exists(model_path):
            return model_path
        return os.path.join(model_dir, f"{width}x{height}", "model.onnx")

    def decode(self, latents: np.ndarray, auto_mem_free=True):
        logger.info("Decoding with VAE...")
        if latents.dtype != np.float16:
            latents = latents.astype(np.float16)
        
        input_name = "latent_sample"
    
        if self.model is None:
            self.model = ort.InferenceSession(self.onnx_path, sess_options=qnn.session_options)

        output_names = list(map(lambda x: x.name, self.model.get_outputs()))
        image_array = self.model.run(output_names, {input_name: latents})[0]
    
        if auto_mem_free:
            self.free_memory()

        return image_array.astype(np.float32)

    def free_memory(self):
        del self.model
        gc.collect()

if __name__ == "__main__":
    # 単体テスト用
    dummy_latents = np.zeros((1, 4, 128, 128), dtype=np.float32)
    deco = VAEDecoder()
    res_image = deco.decode_latents(dummy_latents)
    logging.info(f"VAE 出力形状: {res_image.shape}")
