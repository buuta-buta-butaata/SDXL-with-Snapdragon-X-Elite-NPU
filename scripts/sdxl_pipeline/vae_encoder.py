import logging
import gc
import os
import numpy as np
import onnxruntime as ort

from utils import qnn_ep_helper as qnn

logger = logging.getLogger(__name__)

class VAEEncoder:
    def __init__(self, config):
        self.model = None
        self.onnx_path = self.find_model_path(config.dirs["vae_encoder_dir"],
                                              config.width, config.height)

    def find_model_path(self, model_dir, width, height):
        single_graph_model = os.path.join(model_dir, "model.onnx")
        if os.path.exists(single_graph_model):
            return single_graph_model
        model_path = os.path.join(model_dir, f"vae_encoder_{width}x{height}.onnx")
        if os.path.exists(model_path):
            return model_path
        return os.path.join(model_dir, f"{width}x{height}", "model.onnx")

    def load_model(self, config):
        self.model = ort.InferenceSession(self.onnx_path, sess_options=qnn.session_options)

    def encode(self, preprocessed_image: np.ndarray, auto_mem_free=True):
        logger.info("Encoding with VAE...")
        if preprocessed_image.dtype != np.float16:
            preprocessed_image = preprocessed_image.astype(np.float16)

        if self.model is None:
            self.model = ort.InferenceSession(self.onnx_path, sess_options=qnn.session_options)

        input_name = list(map(lambda x: x.name, self.model.get_inputs()))[0]
        output_names = list(map(lambda x: x.name, self.model.get_outputs()))
        latents = self.model.run(output_names, {input_name: preprocessed_image})[0]
    
        if auto_mem_free:
            self.free_memory()

        return latents.astype(np.float32)

    def free_memory(self):
        del self.model
        gc.collect()

if __name__ == "__main__":
    # 単体テスト用
    from types import SimpleNamespace
    conf = SimpleNamespace(prompt="A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed")
    conf.dirs = {}
    conf.dirs["vae_encoder_dir"] = r"compiled_models\dreamshaper-xl-lightning-for-Snapdragon-X-Elite\vae_encoder\1024x1024"
    conf.width = 1024
    conf.height = 1024
    dummy_image = np.zeros((1, 3, 1024, 1024), dtype=np.float16)
    enco = VAEEncoder(conf)
    latents = enco.encode(dummy_image)
    print(f"VAE 出力形状: {latents.shape}")
