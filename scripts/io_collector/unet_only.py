import glob
import logging
import numpy as np
import os
import time

from concurrent.futures import ThreadPoolExecutor

from profilers import simple_profiler as prof
from sdxl_pipeline import SDXLPipeline
from sdxl_pipeline.unet import UNet
from . import numpy_io as npio

logger = logging.getLogger(__name__)


class SDXLPipelineUNetOnly(SDXLPipeline):
    def __init__(self, sdxl_config):
        self.config = sdxl_config
        prof.available = False

    def _run(self, unet, file_path, output_dir, executor):
        config = self.config
        print(f"Denoising: {file_path}")
        data = npio.load(file_path).item()

        if self.config.seed == -1:
            self.config.seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)

        scheduler = self.get_scheduler(self.config)
        timesteps = self.set_timesteps(scheduler, config)
        init_latents, mask_latents = self.get_init_latents(scheduler, config)
        latents = self.prepare_latents(init_latents, scheduler, timesteps, config)
        
        # backup_scheduler_config = self.config.scheduler_config
        latents = unet.inference(self.config, init_latents, latents, mask_latents, scheduler, timesteps,
                                 data["pos_embeds"], data["pos_pooled"],
                                 data["neg_embeds"], data["neg_pooled"], executor)

        self.config.prompt = data["prompt"]
        self.config.prompt_2 = data["prompt"]
        self.config.negative_prompt  = data["negative_prompt"]
        self.config.negative_prompt_2  = data["negative_prompt"]

        outputs = {
            "config": self.config,
            "latents": latents,
        }
        npio.save(output_dir, "unet", outputs)
        print("  -> Done!")
        
    def run(self):
        unet = UNet(self.config)
        unet.load_models(self.config)
        all_files = sorted(glob.glob(os.path.join(self.config.input_dir, "text*.npy")))

        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            if os.path.isfile(self.config.input_file):
                self._run(unet, self.config.input_file, output_dir, executor)
                return
        
            for i, file_path in enumerate(all_files):
                self._run(unet, file_path, output_dir, executor)

                self.config.seed = -1
                # self.config.scheduler_config = backup_scheduler_config

                if i % 10 == 9:
                    time.sleep(10)
                

    
if __name__ == "__main__":
    from types import SimpleNamespace
    conf = SimpleNamespace(prompt = "A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed")
    main = SDXLPipelineUNetOnly(conf)
    main.run();

