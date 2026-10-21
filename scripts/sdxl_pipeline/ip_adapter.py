import gc
import logging
import os

import numpy as np
import onnxruntime as ort

from utils import qnn_ep_helper as qnn

logger = logging.getLogger(__name__)


class IPAdapter:
    def __init__(self, config):
        self.sessions = []
        self.out_names = []
    
    def load_model(self, config, part_num):
        part_name = f"part{part_num}"
        path = self.find_model_path(config.dirs["ip_adapter_dir"], part_name)
        
        logger.debug(f"Loading IP Adapter part{part_num} from: {path}")
        self.sessions.append(ort.InferenceSession(path, sess_options=qnn.session_options))
        self.out_names.append([o.name for o in self.sessions[part_num].get_outputs()])

    def load_models(self, config):
        for i in range(0, 5):
            self.load_model(config, i)

    def find_model_path(self, ip_adapter_dir, part):
        model_dir = os.path.join(ip_adapter_dir, part)
        single_graph_model = os.path.join(model_dir, "model.onnx")
        if os.path.exists(single_graph_model):
            return single_graph_model
        return os.path.join(model_dir, f"ip_adapter_{part}.onnx")

    def inference(self, config, image_embeds, strength=1.0):
        self.load_models(config)
        out_common = self.sessions[0].run(self.out_names[0], { "image_embeds": image_embeds })[0]

        common_inputs = { self.out_names[0][0]: out_common }
        out_list_down = self.sessions[1].run(self.out_names[1], common_inputs)
        out_list_mid = self.sessions[2].run(self.out_names[2], common_inputs)
        out_list_up1 = self.sessions[3].run(self.out_names[3], common_inputs)
        out_list_up2 = self.sessions[4].run(self.out_names[4], common_inputs)

        out_list_down = [i * strength for i in out_list_down]
        out_list_mid = [i * strength for i in out_list_mid]
        out_list_up1 = [i * strength for i in out_list_up1]
        out_list_up2 = [i * strength for i in out_list_up2]

        return {
            "down": dict(zip(self.out_names[1], out_list_down)),
            "mid": dict(zip(self.out_names[2], out_list_mid)),
            "up1": dict(zip(self.out_names[3], out_list_up1)),
            "up2": dict(zip(self.out_names[4], out_list_up2)),
        }

    def free_memory(self):
        del self.sessions
        gc.collect()
    

class IPAdapterFace(IPAdapter):
    def load_model(self, config, part_num):
        part_name = f"part{part_num}"
        path = self.find_model_path(config.dirs["ip_adapter_face_dir"], part_name)
        
        logger.debug(f"Loading IP Adapter Face part{part_num} from: {path}")
        self.sessions.append(ort.InferenceSession(path, sess_options=qnn.session_options))
        self.out_names.append([o.name for o in self.sessions[part_num].get_outputs()])
        
if __name__ == "__main__":
    pass
