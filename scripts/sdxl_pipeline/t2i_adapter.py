import gc
import logging
import os

import numpy as np
import onnxruntime as ort

from utils import qnn_ep_helper as qnn
from .image import preprocess_image

logger = logging.getLogger(__name__)

t2i_list = [
    "None",
    "canny",
    "sketch",
    "lineart",
    "depth_midas",
    "depth_zoe",
    "openpose",
]

class T2IAdapter:
    def __init__(self, config):
        pass
    
    def load_model(self, config):
        path = self.find_model_path(os.path.join(config.dirs["t2i_adapter_dir"], t2i_list[config.t2i_mode]))
        # path = r"Z:\Work\model_converter\t2i_adapter\t2i-adapter-lineart-sdxl-1.0.onnx"

        logger.debug(f"Loading T2I Adapter from: {path}")
        self.session = ort.InferenceSession(path, sess_options=qnn.session_options)
        # self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.out_names = [o.name for o in self.session.get_outputs()]
        
    def find_model_path(self, model_dir):
        single_graph_model = os.path.join(model_dir, "model.onnx")
        if os.path.exists(single_graph_model):
            return single_graph_model
        return os.path.join(model_dir, f"t2i_adapter.onnx")
        
    def inference(self, config, raw_img, strength):
        self.load_model(config)
        img = preprocess_image(raw_img, for_vae=False)
        outputs = self.session.run(self.out_names, { "pixel_values": img })
        return dict(zip(self.out_names, [x * strength for x in outputs]))

    def free_memory(self):
        del self.session
        gc.collect()
    
