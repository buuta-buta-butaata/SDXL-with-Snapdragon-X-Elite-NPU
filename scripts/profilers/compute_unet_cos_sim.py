import glob
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from io_collector import numpy_io as npio
from io_collector.generate_for_unet_cos_sim import FILE_PREFIX
from sdxl_pipeline import BasePipeline, image
from sdxl_pipeline.unet import UNet
from sdxl_pipeline.vae_decoder import VAEDecoder

class UNetCosSim(BasePipeline):
    def __init__(self, sdxl_config):
        self.config = sdxl_config

    def all_run(self):
        file_path = self.config.input_file
        input_dir = self.config.input_dir
        fp16_output_file = sorted(glob.glob(os.path.join(input_dir, f"{FILE_PREFIX}*.npy")))[-2]
        seed = int(fp16_output_file.split("_")[-2])
        self.config.seed = seed
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            data = npio.load(file_path).item()

            latents = self.unet.inference(self.config, data["pos_embeds"], data["pos_pooled"],
                                          data["neg_embeds"], data["neg_pooled"], executor)
            outputs = self.unet.result_part4

            fp16_outputs = npio.load(fp16_output_file)
            self.compute_cosine_similarity(outputs, fp16_outputs, "total")
            print(f"\ninput file: {file_path}")
            print("prompt: " + data["prompt"])

        del self.unet

        fp16_latents_file = sorted(glob.glob(os.path.join(input_dir, f"{FILE_PREFIX}latents*.npy")))[-1]
        fp16_latents = npio.load(fp16_latents_file)

        vae_decoder = VAEDecoder(self.config)
        fp16_latents = fp16_latents / 0.13025
        image_tensor = vae_decoder.decode(fp16_latents, auto_mem_free=False)
        self.config.output_prefix = "orig_output_"
        image.output_image(image_tensor, **vars(self.config))
        
        latents = latents / 0.13025
        image_tensor = vae_decoder.decode(latents, auto_mem_free=False)
        self.config.output_prefix = "quantized_output_"
        image.output_image(image_tensor, **vars(self.config))

    def write(self):
        filename = f"log_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}.txt"
        file_path = os.path.join(self.config.output_dir, filename)
        print(f"\nlog file: {file_path}")
        with open(file_path, "w") as file:
            file.writelines(self.log_lines)

    def compute_cosine_similarity(self, quantized_data, orig_data, name, step = 0, part_num = 0):
        a = quantized_data.flatten().astype(np.float32)
        b = orig_data.flatten().astype(np.float32)
        cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        print(f"{name.rjust(16)}: {cosine_similarity}")
        self.log_lines.append(f"{step},{part_num},{name},{cosine_similarity}\n")
        
    def run(self):
        input_dir = self.config.input_dir

        self.unet = UNetWrapper(self.config)
        self.unet.load_models(self.config)
        all_files = sorted(glob.glob(os.path.join(input_dir, f"{FILE_PREFIX}*.npy")))[:-1]

        func = [self.unet.run_part1, self.unet.run_part2, self.unet.run_part3, self.unet.run_part4]

        self.log_lines = []
        step = 1
        print(f"\nCosine similarity:")
        for i, file_path in enumerate(all_files):
            if i % 5 == 0:
                step = int(i / 5) + 1
                print("\n" + "=" * 15 + f" step {step} " + "=" * 15)

            if i % 5 == 4:
                continue

            part_num = (i+1) % 5
            print("-" * 15 + f" part {part_num} " + "-" * 15)
            inputs = npio.load(file_path).item()

            fp16_outputs = {}
            for j in range(1, 4 - (i % 5)):
                fp16_temp_outputs = npio.load(all_files[i+j]).item()
                fp16_outputs.update(fp16_temp_outputs)

            outputs = func[i % 5](inputs, True)

            if i % 5 == 1:
                self.compute_cosine_similarity(outputs, fp16_outputs["add_123"], "add_123", step, part_num)
                continue

            if i % 5 == 2:
                self.compute_cosine_similarity(outputs, fp16_outputs["add_156"], "add_156", step, part_num)
                continue

            if i % 5 == 3:
                fp16_outputs = npio.load(all_files[i+1])
                self.compute_cosine_similarity(outputs, fp16_outputs, "out_sample", step, part_num)
                continue

            for k, (name, output) in enumerate(outputs.items()):
                if len(output.shape) > 3 and output.shape[3] == 1280:
                    output = output.transpose(0, 3, 1, 2)
                if name in fp16_outputs:
                    fp16_output = fp16_outputs[name]
                else:
                    print("not found")
                    continue
                self.compute_cosine_similarity(output, fp16_output, name, step, part_num)

        self.all_run()
        self.write()

class UNetWrapper(UNet):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, latents, timestep, base_inputs, encoder_hidden_states, step = 0, is_uncond = False):
        return super().forward(latents, timestep, base_inputs, encoder_hidden_states, step, is_uncond)
    
    def run_part4(self, feed_up2, is_uncond):
        result = super().run_part4(feed_up2, is_uncond)
        self.result_part4 = result
        return result

if __name__ == "__main__":
    pass

    
