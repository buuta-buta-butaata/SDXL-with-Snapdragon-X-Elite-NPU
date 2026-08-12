import logging
import os

from sdxl_pipeline import SDXLBatchPipeline
from sdxl_pipeline.unet import UNet
from . import numpy_io as npio

logger = logging.getLogger(__name__)


class CalibrationDataCollector(SDXLBatchPipeline):
    def __init__(self, sdxl_config):
        super().__init__(sdxl_config)
        self.config = sdxl_config

        for part in ["part0", "part1", "part2", "part3", "part4", "output"]:
            os.makedirs(os.path.join(self.config.output_dir, part), exist_ok=True)

        os.makedirs(os.path.join(self.config.output_dir, "vae_decoder"), exist_ok=True)
        self.unet = UNetWrapper(self.config, self.config.output_dir, 0)
        self.config.collection_strategy = 0
        self.config.current_num = 0
        self.config_list = []

    def set_text_processing_config(self, config, line, lineno):
        prompt, etc = line.split("\",")
        if prompt[0] == "\"":
            prompt = prompt[1:]

        config.prompt = prompt
        config.prompt_2 = config.prompt
        self.config_list.append([prompt] + [int(x) for x in etc.split(",")])

    def set_unet_config(self, config, lineno):
        _, config.step, config.scheduler_type, config.seed, config.collection_strategy = self.config_list[lineno-1]
        config.current_num = lineno

    def set_vae_decoder_config(self, config, lineno):
        (config.prompt, config.step, config.scheduler_type,
         config.seed, config.collection_strategy) = self.config_list[lineno-1]
        config.prompt_2 = config.prompt
    

class UNetWrapper(UNet):
    def __init__(self, config):
        self.config = config
        super().__init__(config)

    def is_collection_target(self, part_name, step):
        # return True

        if part_name == "part1":
            return True
        else:
            return False

        if self.config.steps == 20:
            if self.config.collection_strategy == 1:
                if step == 0 or step == 3 or step == 6 or step == 10 or step == 13 or step == 16:
                    return True
            elif self.config.collection_strategy == 2:
                if step == 1 or step == 4 or step == 7 or step == 11 or step == 14 or step == 17:
                    return True
            elif self.config.collection_strategy == 3:
                if step == 2 or step == 5 or step == 8 or step == 12 or step == 15 or step == 18:
                    return True
            elif self.config.collection_strategy == 4:
                if step == 3 or step == 6 or step == 9 or step == 13 or step == 16 or step == 19:
                    return True
        elif self.config.steps > 3 and self.config.steps < 9:
            if self.config.collection_strategy == 1:
                if part_name == "part2":
                    if step == 0:
                        return True
                elif part_name == "part3":
                    if step == 1 or step == 3:
                        return True
                return False
            elif self.config.collection_strategy == 2:
                if part_name == "part1":
                    if step == 0 or step == 5:
                        return True
                elif part_name == "part2":
                    if step == 0:
                        return True
                elif part_name == "part3":
                    if step == 5:
                        return True
                elif part_name == "part4":
                    if step == 3 or step == 4 or step == 5:
                        return True
                return False
            elif self.config.collection_strategy == 3:
                if part_name == "part1":
                    if step == 0:
                        return True
                elif part_name == "part2":
                    if step == 0:
                        return True
                return False
            return True

        # if self.collection_strategy == 1:
        #     if (step > 1 and step < 5) or (step > self.steps - 4 and step < self.steps):
        #         return True
        # elif self.collection_strategy == 2:
        #     if (step > 4 and step < 8) or (step > self.steps - 7 and step < self.steps - 3):
        #         return True
        # elif self.collection_strategy == 3:
        #     if step == 2 or step == 5 or step == 8 or step == 12 or step == 15 or step == 18:
        #         return True
        # elif self.collection_strategy == 4:
        #     if step == 3 or step == 6 or step == 9 or step == 13 or step == 16 or step == 19:
        #         return True
        return False
    
    def save_step(self, part_name, input_dict, step, is_uncond=False, file_prefix=""):
        if not self.is_collection_target(part_name, step):
            return

        pad_current_num = str(self.config.current_num).zfill(2)
        pad_step = str(step).zfill(3)
        if len(file_prefix) > 0:
            file_name = f"{file_prefix}_{pad_step}.npy"
        else:
            file_name = f"{pad_current_num}_{pad_step}_" if not is_uncond else f"{pad_current_num}_{pad_step}_uncond_"

        save_dir = os.path.join(self.config.output_dir, part_name)
        npio.save(save_dir, file_name, input_dict)
        
    def forward(self, latents, timestep, base_inputs, encoder_hidden_states, step = 0, is_uncond = False):
        self.step = step
        return super().forward(latents, timestep, base_inputs, encoder_hidden_states, step, is_uncond)
    
    def run_part0(self, feed_common, is_uncond):
        if not is_uncond:
            self.save_step("part0", feed_common, self.step, is_uncond)
        return super().run_part0(feed_common, is_uncond)
    
    def run_part1(self, feed_down, is_uncond):
        if not is_uncond:
            self.save_step("part1", feed_down, self.step, is_uncond)
        return super().run_part1(feed_down, is_uncond)

    def run_part2(self, feed_mid, is_uncond):
        if not is_uncond:
            self.save_step("part2", feed_mid, self.step, is_uncond)
        return super().run_part2(feed_mid, is_uncond)
    
    def run_part3(self, feed_up1, is_uncond):
        if not is_uncond:
            self.save_step("part3", feed_up1, self.step, is_uncond)
        return super().run_part3(feed_up1, is_uncond)
    
    def run_part4(self, feed_up2, is_uncond):
        if not is_uncond:
            self.save_step("part4", feed_up2, self.step, is_uncond)

        result = super().run_part4(feed_up2, is_uncond)
        if not is_uncond:
            self.save_step("output", result, self.step, is_uncond)

        return result

if __name__ == "__main__":
    pass

    
