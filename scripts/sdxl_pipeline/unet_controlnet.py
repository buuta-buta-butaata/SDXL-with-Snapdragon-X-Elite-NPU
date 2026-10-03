import logging
import math
import numpy as np

from .unet import UNet
from .controlnet import ControlNet
from . import image

logger = logging.getLogger(__name__)


class UNetControlNet(UNet):
    def __init__(self, config):
        super().__init__(config)
        self.init(config)
        
    def init(self, config):
        self.controlnet = ControlNet(config)

        self.control_type_idx = np.array([config.control_mode], dtype=np.int32)
        control_type = [0] * 8
        control_type[config.control_mode] = 1
        self.control_type = np.array([control_type], dtype=np.float32)

        self.control_start_step = math.ceil(config.steps * config.control_guidance_start)
        self.control_end_step = math.floor(config.steps * config.control_guidance_end) - 1
        self.conditioning_scale = config.control_scale
        if config.control_mode == 7: # repaint
            self.add_noise = self.add_noise_for_repaint
        self.guess_mode = False
        self.controlnet_results = []
        self.controlnet_uncond_results = []

    def preprocess(self, config):
        img = config.control_image

        # if config.control_mode == 3: # Canny
        #     image_np = image.preprocess_canny(img)
        # else:
        #     image_np = image.preprocess_image(img, for_vae=False)
        image_np = image.preprocess_image(img, for_vae=False)
        self.controlnet_cond = image_np
    
    def run_controlnet(self, latents, timestep, base_inputs, encoder_hidden_states, is_uncond=False):
        cnet_inputs = {
            "sample": latents.astype(np.float32),
            "timestep": timestep.astype(np.float32),
            "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
            "controlnet_cond": self.controlnet_cond.astype(np.float32),
            "control_type": self.control_type,
            "control_type_idx": self.control_type_idx,
            **base_inputs
        }
        if not is_uncond:
            self.controlnet_results = self.controlnet.run(cnet_inputs, False)
        else:
            self.controlnet_uncond_results = self.controlnet.run(cnet_inputs, False)

    def forward(self, latents, timestep, base_inputs, encoder_hidden_states, step=0, is_uncond=False):
        # guess_modeの時は、uncondの分はControlNetの処理の対象外
        if not self.guess_mode or not is_uncond:
            self.is_active_controlnet = (self.control_start_step <= step <= self.control_end_step)
            if self.is_active_controlnet:
                self.run_controlnet(latents, timestep, base_inputs, encoder_hidden_states, is_uncond)
            
        noise_pred, is_uncond = super().forward(latents, timestep, base_inputs, encoder_hidden_states, step, is_uncond)
        return noise_pred, is_uncond

    adapter_dict = {
        "add_2": "down_block_res_0",
        "add_4": "down_block_res_1",
        "conv2d": "down_block_res_2",
        "conv2d_5": "down_block_res_3",
        "add_13": "down_block_res_4",
        "add_22": "down_block_res_5",
        "conv2d_11": "down_block_res_6",
        "add_55": "down_block_res_7",
        "add_88": "down_block_res_8",
    }
    def run_part1(self, feed_down, is_uncond):
        skip_connections = super().run_part1(feed_down, is_uncond)
        if (not self.guess_mode or not is_uncond) and self.is_active_controlnet:
            controlnet_results = self.controlnet_results if not is_uncond else self.controlnet_uncond_results
            for k,v in self.adapter_dict.items():
                skip_connections[k] = skip_connections[k] + controlnet_results[v] * self.conditioning_scale
                # skip_connections[k] = skip_connections[k] + self.controlnet_results[v] * self.conditioning_scale
    
        return skip_connections

    def run_part2(self, feed_mid, is_uncond):
        result = super().run_part2(feed_mid, is_uncond)
        if (not self.guess_mode or not is_uncond) and self.is_active_controlnet:
            controlnet_results = self.controlnet_results if not is_uncond else self.controlnet_uncond_results
            result = result + controlnet_results["mid_block_res"] * self.conditioning_scale
            # result = result + self.controlnet_results["mid_block_res"] * self.conditioning_scale
        return result
         
    def add_noise_for_repaint(self, init_latents, latents, mask, scheduler, timestep, config):
        current_noise = scheduler.generate_noise_latents(config)
        init_latents_proper_noise = scheduler.add_noise(init_latents, current_noise, np.array([timestep]))
    
        # マスクを使ってブレンド
        # mask が 1.0 (白) の部分は latents（新しい絵）
        # mask が 0.0 (黒) の部分は init_latents_proper_noise（元画像＋その時点のノイズ）
        latents = (1.0 - mask) * init_latents_proper_noise + mask * latents
        return latents
        
