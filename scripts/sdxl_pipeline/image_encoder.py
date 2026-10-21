import gc
import logging
import os

import numpy as np
import onnxruntime as ort

from utils import qnn_ep_helper as qnn
from .image import preprocess_tile_for_clip

logger = logging.getLogger(__name__)


class ImageEncoder:
    def __init__(self, config):
        pass
    
    def load_model(self, config):
        path = self.find_model_path(config.dirs["image_encoder_dir"])

        logger.debug(f"Loading Image Encoder from: {path}")
        self.session = ort.InferenceSession(path, sess_options=qnn.session_options)
        self.out_names = [o.name for o in self.session.get_outputs()]
        
    def find_model_path(self, model_dir):
        single_graph_model = os.path.join(model_dir, "model.onnx")
        if os.path.exists(single_graph_model):
            return single_graph_model
        return os.path.join(model_dir, f"image_encoder.onnx")
        
    def inference(self, config, raw_img):
        self.load_model(config)
        img = preprocess_tile_for_clip(raw_img)
        image_embeds = self.session.run(self.out_names, { "pixel_values": img })[0]
        return image_embeds

    def free_memory(self):
        del self.session
        gc.collect()
    
