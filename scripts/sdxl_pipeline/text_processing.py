import logging
import numpy as np
from .text_encoder import TextEncoder
from .text_encoder_2 import TextEncoder2

from profilers import simple_profiler as prof

logger = logging.getLogger(__name__)

class TextProcessing:
    def __init__(self, text_encoder_dir, tokenizer_dir, text_encoder_2_dir, tokenizer_2_dir, use_torch=False):
        self.text_encoder = TextEncoder(text_encoder_dir, tokenizer_dir, use_torch)
        self.text_encoder_2 = TextEncoder2(text_encoder_2_dir, tokenizer_2_dir, use_torch)
        return

    def _encode_text(self, prompt, prompt_2, is_uncond=False):
        profiler = prof.get("pipeline")
        profile_name = "text_encoder" if not is_uncond else "text_encoder_uncond"
        profiler.start_profile(profile_name)
        prompt_embeds_1 = self.text_encoder.get_text_embeddings(prompt, auto_mem_free=False)
        profiler.stop_profile(profile_name)
    
        profile_name = "text_encoder_2" if not is_uncond else "text_encoder_2_uncond"
        profiler.start_profile(profile_name)
        pooled_prompt_embeds, prompt_embeds_2 = self.text_encoder_2.get_text_embeddings_2(prompt_2, auto_mem_free=False)
        profiler.stop_profile(profile_name)
    
        prompt_embeds = np.concatenate([prompt_embeds_1, prompt_embeds_2], axis=-1)
    
        return prompt_embeds, pooled_prompt_embeds

    def encode_text(self, config, auto_mem_free=True):
        logger.info("Encoding text...")
        prompt_embeds, pooled_prompt_embeds = self._encode_text(
            config.prompt, config.prompt_2, is_uncond=False)
        
        if config.cfg != 1:
            uncond_embeds, uncond_pooled_embeds = self._encode_text(
                config.negative_prompt, config.negative_prompt_2, is_uncond=True)
        else:
            uncond_embeds, uncond_pooled_embeds = None, None

        if auto_mem_free:
            self.text_encoder.free_memory()
            self.text_encoder_2.free_memory()
        return prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled_embeds

    def free_memory(self):
        if self.text_encoder:
            self.text_encoder.free_memory()
        if self.text_encoder_2:
            self.text_encoder_2.free_memory()
