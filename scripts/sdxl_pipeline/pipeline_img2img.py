import logging
import numpy as np

from .vae_encoder import VAEEncoder
from .unet import UNet
from .pipeline import SDXLPipeline, SCALING_FACTOR
from . import image

logger = logging.getLogger(__name__)


class SDXLImg2ImgPipeline(SDXLPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.vae_encoder = VAEEncoder(self.config)

    def get_init_latents(self, scheduler, config):
        preprocessed_image = image.preprocess_image(config.input_image)

        vae_output = self.vae_encoder.encode(preprocessed_image)
    
        init_latents = vae_output[:, :4, :, :]

        scaling_factor = SCALING_FACTOR
        init_latents = init_latents * scaling_factor
        
        return init_latents, None

    def set_timesteps(self, scheduler, config):
        scheduler.set_timesteps(config.steps)
    
        offset = max(int(config.steps * config.denoising_strength), 1)
        start_step = config.steps - offset
    
        timesteps = scheduler.timesteps[start_step:]
    
        return timesteps

    def prepare_latents(self, init_latents, scheduler, timesteps, config):
        start_timestep = [timesteps[0]]

        noise = scheduler.generate_noise_latents(config)
        noisy_latents = scheduler.add_noise(init_latents, noise, start_timestep)
    
        return noisy_latents


