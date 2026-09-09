import logging
import numpy as np

from .vae_encoder import VAEEncoder
from . import image
from .pipeline_inpaint import SDXLInpaintPipeline
from .pipeline import SDXLPipeline, SCALING_FACTOR

logger = logging.getLogger(__name__)


class SDXLOutpaintPipeline(SDXLInpaintPipeline):
    def __init__(self, config):
        super().__init__(config)

    def get_init_latents(self, scheduler, config):
        (self.orig_image,
         canvas_image,
         mask_image) = image.generate_outpaint_inputs(config.input_image,
                                                      direction=config.outpaint_direction,
                                                      target_size=(1024, 1024),
                                                      outpaint_ratio=self.config.outpaint_ratio)

        preprocessed_image = image.preprocess_image(canvas_image)
        vae_output = self.vae_encoder.encode(preprocessed_image)
    
        init_latents = vae_output[:, :4, :, :]

        init_latents = init_latents * SCALING_FACTOR

        mask_latents = image.preprocess_mask(mask_image, (config.width // 8, config.height // 8), config.mask_blur)
        
        return init_latents, mask_latents
    
    def output_image(self, image_np):
        img = image.generate_outpaint_outputs(self.orig_image, image_np, direction=self.config.outpaint_direction,
                                              target_size=(1024, 1024), outpaint_ratio=self.config.outpaint_ratio)
        image.save(img, **vars(self.config))
        
