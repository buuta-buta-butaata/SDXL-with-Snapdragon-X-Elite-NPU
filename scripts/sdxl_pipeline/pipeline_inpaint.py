import logging
import numpy as np

from .vae_encoder import VAEEncoder
from . import image
from .pipeline_img2img import SDXLImg2ImgPipeline
from .pipeline import SDXLPipeline
from .unet import UNet

logger = logging.getLogger(__name__)


class SDXLInpaintPipeline(SDXLImg2ImgPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.unet = UNetInpaint(config)

    def get_init_latents(self, scheduler, config):
        result, _ = super().get_init_latents(scheduler, config)
        mask = image.preprocess_mask(config.mask_image, (config.width // 8, config.height // 8), config.mask_blur)
        return result, mask


class UNetInpaint(UNet):
    def add_noise(self, init_latents, latents, mask, scheduler, timestep, config):
        current_noise = scheduler.generate_noise_latents(config)
        init_latents_proper_noise = scheduler.add_noise(init_latents, current_noise, np.array([timestep]))
    
        # マスクを使ってブレンド
        # mask が 1.0 (白) の部分は latents（新しい絵）
        # mask が 0.0 (黒) の部分は init_latents_proper_noise（元画像＋その時点のノイズ）
        latents = (1.0 - mask) * init_latents_proper_noise + mask * latents
        return latents
