import logging
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor

from .text_processing import TextProcessing
from .unet import UNet
from .vae_decoder import VAEDecoder
from . import image

from .pipeline import SDXLPipeline, SCALING_FACTOR

logger = logging.getLogger(__name__)

class SDXLBatchPipeline(SDXLPipeline):
    def __init__(self, config):
        self.config = config
        self.text_processing = TextProcessing(self.config.dirs["text_encoder_dir"],
                                              self.config.dirs["tokenizer_dir"],
                                              self.config.dirs["text_encoder_2_dir"],
                                              self.config.dirs["tokenizer_2_dir"],
                                              self.config.use_torch)
        self.unet = UNet(self.config)
        self.vae_decoder = VAEDecoder(self.config)

    def set_text_processing_config(self, config, line, lineno):
        config.prompt = line
        config.prompt_2 = line

    def set_unet_config(self, config, lineno):
        pass

    def set_vae_decoder_config(self, config, lineno):
        pass
    
    def run(self):
        print("Running txt2img (batch mode)...")
        text_embeds_list = []
        if not os.path.isfile(self.config.input_file):
            text_embeds_list.append(self._run_text_processing())
        else:
            lineno = 1
            with open(self.config.input_file) as f:
                lines = f.readlines()
                print(f"\nLoaded {len(lines)} prompts from {self.config.input_file}:")
                for line in lines:
                    prompt = line.rstrip("\n")
                    print(f"  Prompt {lineno}: {prompt}")
                    self.set_text_processing_config(self.config, prompt, lineno)
                    text_embeds_list.append(self._run_text_processing())
                    lineno += 1
                print("")

        del self.text_processing

        with ThreadPoolExecutor(max_workers=2) as executor:
            self._run(text_embeds_list, self.config, executor)

    def _run_text_processing(self):
        return self.text_processing.encode_text(self.config.prompt, self.config.prompt_2,
                                                self.config.negative_prompt, self.config.negative_prompt_2,
                                                self.config, False)
        
    def _run(self, text_embeds_list, config, executor):
        random_seed = self.config.seed == -1
        latents_list = []
        seed_list = []

        # FIXME: Separate UNet and VAE processing loops to mitigate unexpected RAM spikes.
        # Combining them causes RAM usage to balloon significantly beyond theoretical limits.
        # 
        # TODO: Refactor loop and switch to the following progress message format once the memory issue is resolved:
        # 
        # [1/3] Processing: {prompt}...
        #   ➔ Denoising with UNet...
        # 100%|███████████████████████████████████████████████| 6/6 [00:11<00:00,  1.99s/it]
        #   ➔ Decoding VAE...
        # Image saved to: {filepath}
        
        all_count = len(text_embeds_list) * self.config.batch_count
        count = 1
        for lineno, (prompt_embeds, pooled_prompt_embeds,
                     uncond_embeds, uncond_pooled_embeds) in enumerate(text_embeds_list, 1):
            self.set_unet_config(self.config, lineno)
        
            for i in range(0, self.config.batch_count):
                if config.seed == -1:
                    config.seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
                
                scheduler = self.get_scheduler(config)
                timesteps = self.set_timesteps(scheduler, config)
                init_latents, mask_latents = self.get_init_latents(scheduler, config)
                latents = self.prepare_latents(init_latents, scheduler, timesteps, config)
                
                print(f"[{count}/{all_count}] Denoising with UNet...")
                latents = self.unet.inference(self.config, init_latents, latents, mask_latents, scheduler, timesteps,
                                              prompt_embeds, pooled_prompt_embeds,
                                              uncond_embeds, uncond_pooled_embeds, executor)
        
                latents = latents / SCALING_FACTOR
                # image_np = self.vae_decoder.decode(latents, auto_mem_free=False)
                # image.save(image_np, **vars(self.config))
                latents_list.append(latents)

                if random_seed:
                    seed_list.append(self.config.seed)
                    self.config.seed = -1

                count += 1

            del prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled_embeds

        print("")
        del self.unet

        for i, latents in enumerate(latents_list):
            print(f"[{i+1}/{all_count}] Decoding VAE & saving image...")
            self.config.seed = seed_list[i]
            self.set_vae_decoder_config(self.config, i+1);
            image_np = self.vae_decoder.decode(latents, auto_mem_free=False)
            image.save(image_np, **vars(self.config))
            
if __name__ == "__main__":
    from types import SimpleNamespace
    conf = SimpleNamespace(prompt = "A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed")
    main = SDXLPipeline(conf)
    main.run()

