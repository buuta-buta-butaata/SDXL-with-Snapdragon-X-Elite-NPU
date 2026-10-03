import gc
import logging
import numpy as np
import os

from sdxl_pipeline import SDXLBatchPipeline
from sdxl_pipeline.unet_controlnet import UNetControlNet
from . import numpy_io as npio

logger = logging.getLogger(__name__)

def safe_convert_to_number(value):
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        if value == "":
            return None
        return value

class CalibrationDataCollectorForControlNet(SDXLBatchPipeline):
    def __init__(self, sdxl_config):
        super().__init__(sdxl_config)
        self.config = sdxl_config

        os.makedirs(os.path.join(self.config.output_dir, "controlnet"), exist_ok=True)
        # self.unet = UNetWrapper(self.config, self.config.output_dir, 0)
        # self.unet = UNetWrapper(self.config)
        self.config.collection_strategy = 0
        self.config.current_num = 0
        self.config_list = []

    def set_text_processing_config(self, config, line, lineno):
        prompt, etc = line.split("\",")
        if prompt[0] == "\"":
            prompt = prompt[1:]

        config.prompt = prompt
        config.prompt_2 = config.prompt
        self.config_list.append([prompt] + [safe_convert_to_number(x) for x in etc.split(",")])

    def set_unet_config(self, config, lineno):
        (_, config.collection_strategy, config.step, config.cfg, config.control_mode,
         config.control_image, config.control_guidance_start, config.control_guidance_end, config.control_scale,
         _, config.seed, config.input_image, config.mask_image)  = self.config_list[lineno-1]
        config.current_num = lineno
        if self.unet:
            del self.unet
            gc.collect()
        self.unet = UNetWrapper(config)
        # self.unet.init(config)

    def set_vae_decoder_config(self, config, lineno):
        (config.prompt, config.collection_strategy, config.step, config.cfg, config.control_mode,
         config.control_image, config.control_guidance_start, config.control_guidance_end, config.control_scale,
         _, config.seed, config.input_image, config.mask_image) = self.config_list[lineno-1]
        config.prompt_2 = config.prompt
    

class UNetWrapper(UNetControlNet):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.tile_counter = 0

    def is_collection_target(self, step):
        # return True

        if self.config.steps == 20:
            if self.config.collection_strategy == 1:
                if step == 0 or step == 3 or step == 6 or step == 10 or step == 13 or step == 16:
                    return True
            elif self.config.collection_strategy == 2:
                if step == 1 or step == 4 or step == 7 or step == 11 or step == 14 or step == 17:
                    return True
            elif self.config.collection_strategy == 3:
                if step == 2 or step == 5 or step == 8 or step == 12 or step == 15 or step == 18:
                    return True
            elif self.config.collection_strategy == 4:
                if step == 3 or step == 6 or step == 9 or step == 13 or step == 16 or step == 19:
                    return True
            elif self.config.collection_strategy == 0:
                if step == 0:
                    self.tile_counter += 1
                    
                if self.tile_counter == step + 1:
                    return True

        return False
    
    def save_step(self, input_dict, step, file_prefix=""):
        if not self.is_collection_target(step):
            return

        pad_step = str(step).zfill(3)
        if len(file_prefix) > 0:
            file_name = f"{file_prefix}_{pad_step}.npy"
        else:
            pad_current_num = str(self.config.current_num).zfill(2)
            file_name = f"{pad_current_num}_{pad_step}.npy"

        save_dir = os.path.join(self.config.output_dir, "controlnet")
        print(f"{self.config.seed}")
        npio.save(save_dir, file_name, input_dict)

    def forward(self, latents, timestep, base_inputs, encoder_hidden_states, step = 0, is_uncond = False):
        self.step = step
        return super().forward(latents, timestep, base_inputs, encoder_hidden_states, step, is_uncond)
        
    def run_controlnet(self, latents, timestep, base_inputs, encoder_hidden_states, is_uncond=False):
        super().run_controlnet(latents, timestep, base_inputs, encoder_hidden_states, is_uncond)
        if is_uncond:
            return
        cnet_inputs = {
            "sample": latents.astype(np.float32),
            "timestep": timestep.astype(np.float32),
            "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
            "controlnet_cond": self.controlnet_cond.astype(np.float32),
            "control_type": self.control_type,
            "control_type_idx": self.control_type_idx,
            **base_inputs
        }
        self.save_step(cnet_inputs, self.step, "controlnet")

if __name__ == "__main__":
    pass

    
