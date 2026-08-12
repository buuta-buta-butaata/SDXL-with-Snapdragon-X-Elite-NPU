import logging
import numpy as np
import os

from datetime import datetime
from PIL import Image, PngImagePlugin

logger = logging.getLogger(__name__)

# From: https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
WEIGHTS = [
    [60, 60, 60],
    [-60, -5, 10],
    [25, 15, 5],
    [-70, -50, -35]
]

BIASES = [[[150, 140, 130]]]

METADATA_IGNORE_LIST = ["input_file", "input_dir", "output_dir", "output_prefix", "part", "func", "profile",
                        "debug_mode", "log_level", "dirs", "calib_strategy", "batch_count", "collection_strategy",
                        "current_num"]

def latents_to_rgb(latents):
    rgb = latents[0].transpose(1, 2, 0) @ WEIGHTS + BIASES
    image_array = rgb.clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(image_array)

def preview_image(latents, output_dir, output_prefix):
    filename = f"{output_prefix}preview_{datetime.now().strftime("%Y%m%d%H%M%S")}.png"
    output_path = os.path.join(output_dir, filename)
    output_image = latents_to_rgb(latents)
    output_image.save(output_path)
    
def output_image(image_tensor, output_dir, output_prefix, **config):
    metadata = PngImagePlugin.PngInfo()
    # import sys
    # metadata.add_text("command", " ".join(['"' + x + '"' if " " in x else x for x in sys.argv]))
    for k, v in config.items():
        if k in METADATA_IGNORE_LIST:
            continue
        metadata.add_text(k, str(v))
    filename = f"{output_prefix}{datetime.now().strftime("%Y%m%d%H%M%S")}.png"
    output_path = os.path.join(output_dir, filename)

    image = (image_tensor / 2 + 0.5).clip(0, 1)
    image = image.squeeze(0).transpose(1, 2, 0)

    image_uint8 = (image * 255).astype(np.uint8)

    output_image = Image.fromarray(image_uint8)
    output_image.save(output_path, pnginfo=metadata)

    print(f"Image saved to: {output_path}, seed: {config['seed']}")
    logger.info(f"Image saved to: {output_path}, seed: {config['seed']}")

