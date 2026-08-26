import logging
import numpy as np
import os

from datetime import datetime
from PIL import Image, PngImagePlugin, ImageFilter

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

def preprocess_image(image_path, target_size=(1024, 1024)):
    # 1. 画像の読み込みとRGB変換、リサイズ
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
    
    # 2. numpy配列に変換 (0 ~ 255)
    image_np = np.array(image).astype(np.float16)
    
    # 3. 軸の入れ替え (Height, Width, Channel) -> (Channel, Height, Width)
    image_np = image_np.transpose(2, 0, 1)
    
    # 4. バッチ次元の追加 (1, Channel, Height, Width)
    image_np = np.expand_dims(image_np, axis=0)
    
    # 5. 正規化 (0~255 -> -1.0~1.0)
    # VAE Encoderは、ピクセル値が -1 から 1 の範囲であることを想定しています
    image_normalized = (image_np / 127.5) - 1.0
    
    return image_normalized

def preprocess_mask(mask_path, target_size=(128, 128), blur_radius=16):
    """
    マスク画像の前処理
    :param mask_path: マスク画像のパス
    :param target_size: UNetに渡すLatentのサイズ（通常は 128x128）
    :param blur_radius: ぼかしの強度（ピクセル半径。0を指定するとぼかしなし）
    """
    mask = Image.open(mask_path).convert("L")

    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    mask = mask.resize(target_size, resample=Image.Resampling.BILINEAR)

    mask_np = np.array(mask).astype(np.float16) / 255.0
    # mask_np = np.floor(mask_np)

    # 形状をUNet/Latentに合わせる (1, 1, 128, 128)
    mask_np = np.expand_dims(mask_np, axis=(0, 1))
    
    return mask_np
