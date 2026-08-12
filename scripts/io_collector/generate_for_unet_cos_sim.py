import glob
import os

from concurrent.futures import ThreadPoolExecutor

from sdxl_pipeline import BasePipeline
from sdxl_pipeline.unet import UNet
from . import numpy_io as npio

FILE_PREFIX = "cos_sim_"

class UNetCosSim(BasePipeline):
    def __init__(self, sdxl_config):
        self.config = sdxl_config

    def clear_dir(self):
        target = os.path.join(self.config.output_dir, f"{FILE_PREFIX}*.npy")
        print(f"Cleaning directory: {target}")
        for file in glob.glob(target):
            if os.path.isfile(file):
                os.remove(file)
                print(f"Deleted: {file}")

    def run(self):
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.clear_dir()
        
        unet = UNetWrapper(self.config, output_dir)
        unet.load_models(self.config)
        file_path = self.config.input_file

        with ThreadPoolExecutor(max_workers=2) as executor:
            print(file_path)
            data = npio.load(file_path).item()

            latents = unet.inference(self.config, data["pos_embeds"], data["pos_pooled"],
                                     data["neg_embeds"], data["neg_pooled"], executor)
            npio.save(output_dir, f"{FILE_PREFIX}latents_{self.config.seed}_", latents)
            
                

class UNetWrapper(UNet):
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.config = config
        super().__init__(config)
        
    def forward(self, latents, timestep, base_inputs, encoder_hidden_states, step = 0, is_uncond = False):
        self.step = step
        return super().forward(latents, timestep, base_inputs, encoder_hidden_states, step, is_uncond)
    
    # def run_part0(self, feed_common, is_uncond):
    #     if not is_uncond:
    #         npio.save(output_dir, "unet", feed_common)
    #     return super().run_part0(feed_common, is_uncond)
    
    def run_part1(self, feed_down, is_uncond):
        if not is_uncond:
            npio.save(self.output_dir, f"{FILE_PREFIX}{self.step}_1_{self.config.seed}_", feed_down)
        return super().run_part1(feed_down, is_uncond)

    def run_part2(self, feed_mid, is_uncond):
        if not is_uncond:
            npio.save(self.output_dir, f"{FILE_PREFIX}{self.step}_2_{self.config.seed}_", feed_mid)
        return super().run_part2(feed_mid, is_uncond)
    
    def run_part3(self, feed_up1, is_uncond):
        if not is_uncond:
            npio.save(self.output_dir, f"{FILE_PREFIX}{self.step}_3_{self.config.seed}_", feed_up1)
        return super().run_part3(feed_up1, is_uncond)
    
    def run_part4(self, feed_up2, is_uncond):
        if not is_uncond:
            npio.save(self.output_dir, f"{FILE_PREFIX}{self.step}_4_{self.config.seed}_", feed_up2)

        result = super().run_part4(feed_up2, is_uncond)
        if not is_uncond:
            npio.save(self.output_dir, f"{FILE_PREFIX}{self.step}_5_{self.config.seed}_", result)

        return result

if __name__ == "__main__":
    pass

    
