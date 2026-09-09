import logging
import gc
import os
import numpy as np
import onnxruntime as ort

from utils import qnn_ep_helper as qnn

logger = logging.getLogger(__name__)


# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
# 6 -- tile
# 7 -- repaint
class ControlNet:
    def __init__(self, config):
        self.model = None
        self.onnx_path = self.find_model_path(config.dirs["controlnet_dir"])
        
    def find_model_path(self, model_dir):
        single_graph_model = os.path.join(model_dir, "model.onnx")
        if os.path.exists(single_graph_model):
            return single_graph_model
        else:
            raise FileNotFoundError(f"File does not exist: {single_graph_model}")

    def load_model(self, config):
        self.model = ort.InferenceSession(self.onnx_path, sess_options=qnn.session_options)
        
    def run(self, feed, auto_mem_free=True):
        logger.info("Running ControlNet...")
    
        if self.model is None:
            self.model = ort.InferenceSession(self.onnx_path, sess_options=qnn.session_options)

        output_names = list(map(lambda x: x.name, self.model.get_outputs()))
        results = self.model.run(output_names, feed)
    
        if auto_mem_free:
            self.free_memory()

        results = dict(zip(output_names, results))
        for k,v in results.items():
            results[k] = v.astype(np.float16)
        
        return results

    def free_memory(self):
        del self.model
        gc.collect()

if __name__ == "__main__":
    pass
