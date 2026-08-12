import glob
import logging
import os

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from profilers import simple_profiler as prof
from sdxl_pipeline import image, BasePipeline
from sdxl_pipeline.vae_decoder import VAEDecoder

from . import numpy_io as npio

logger = logging.getLogger(__name__)


class SDXLPipelineVAEDecoderOnly(BasePipeline):
    def __init__(self, sdxl_config):
        self.config = sdxl_config
        prof.available = False

    def run(self):
        vae_decoder = VAEDecoder(self.config)
        
        if os.path.isfile(self.config.input_file):
            self.decode(vae_decoder, self.config.input_file)
            return
        
        all_files = sorted(glob.glob(os.path.join(self.config.input_dir, "unet*.npy")))

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self.decode, vae_decoder, file_path) for file_path in all_files
            ]
            done, not_done = wait(futures, return_when=ALL_COMPLETED)

    def decode(self, vae_decoder, file_path):
        print(f"Decoding: {file_path}")
        data = npio.load(file_path).item()
        latents = data["latents"] / 0.13025

        image_tensor = vae_decoder.decode(latents, auto_mem_free=False)
        config = data["config"]

        output_dir = self.config.output_dir # self.config!
        os.makedirs(output_dir, exist_ok=True)
        config.output_dir = output_dir
        
        image.output_image(image_tensor, **vars(config))
        # if i % 10 == 9:
        #     time.sleep(2)

            
if __name__ == "__main__":
    from types import SimpleNamespace
    conf = SimpleNamespace(prompt = "A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed")
    main = SDXLPipelineVAEDecoderOnly(conf)
    main.run();

