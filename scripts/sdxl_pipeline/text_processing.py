import gc
import numpy as np
from text_encoder import TextEncoder
from text_encoder_2 import TextEncoder2

class TextProcessing:
    def __init__(self, text_encoder_dir, tokenizer_dir, text_encoder_2_dir, tokenizer_2_dir):
        self.text_encoder = TextEncoder(text_encoder_dir, tokenizer_dir)
        self.text_encoder_2 = TextEncoder2(text_encoder_2_dir, tokenizer_2_dir)
        return

    def _encode_text(self, prompt, prompt_2):
        # Text Encoder 1 の処理 & 即時解放
        prompt_embeds_1 = self.text_encoder.get_text_embeddings(prompt, auto_mem_free=False)
    
        # Text Encoder 2 の処理 & 即時解放
        pooled_prompt_embeds, prompt_embeds_2 = self.text_encoder_2.get_text_embeddings_2(
            prompt_2, auto_mem_free=False)
    
        # SDXL仕様に合わせて2つのエンコーダーの出力をドッキング
        prompt_embeds = np.concatenate([prompt_embeds_1, prompt_embeds_2], axis=-1)
    
        # 不要になった中間変数を徹底的に掃除
        del prompt_embeds_1, prompt_embeds_2
        gc.collect()
        return prompt_embeds, pooled_prompt_embeds

    def encode_text(self, config):
        prompt_embeds, pooled_prompt_embeds = self._encode_text(
            config.prompt, config.prompt_2)
        
        if config.guidance_scale != 1:
            uncond_embeds, uncond_pooled_embeds = self._encode_text(
                config.negative_prompt, config.negative_prompt_2)
        else:
            uncond_embeds, uncond_pooled_embeds = None, None
        
        self.text_encoder.free_memory()
        self.text_encoder_2.free_memory()
        return prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled_embeds
