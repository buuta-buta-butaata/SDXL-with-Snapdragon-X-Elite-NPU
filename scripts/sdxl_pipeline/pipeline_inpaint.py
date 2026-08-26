import logging
import numpy as np

from .vae_encoder import VAEEncoder
from . import image
from .pipeline_img2img import SDXLImg2ImgPipeline, UNetWrapper

logger = logging.getLogger(__name__)


class SDXLInpaintPipeline(SDXLImg2ImgPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.unet = UNetWrapperInpaint(config)


class UNetWrapperInpaint(UNetWrapper):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.mask = image.preprocess_mask(config.mask_file, (config.width // 8, config.height // 8), config.blur_radius)

    def get_init_latents(self, scheduler, config):
        self.init_latents = super().get_init_latents(scheduler, config)
        return self.init_latents

    def add_noise(self, latents, scheduler, timestep, config):
        if scheduler.timesteps[-1] == timestep:
            background_latents = self.init_latents
        else:
            current_noise = scheduler.generate_noise_latents(config)
            background_latents = scheduler.add_noise(self.init_latents, current_noise, np.array([timestep]))
    
        # マスクを使ってブレンド
        # mask が 1.0 (白) の部分は latents（新しい絵）
        # mask が 0.0 (黒) の部分は init_latents_proper_noise（元画像＋その時点のノイズ）
        latents = (1.0 - self.mask) * background_latents + self.mask * latents
        return latents
