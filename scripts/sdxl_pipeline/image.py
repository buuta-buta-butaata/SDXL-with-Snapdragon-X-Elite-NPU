# TODO: Refactor

import logging
import numpy as np
import os

from datetime import datetime
from PIL import Image, PngImagePlugin, ImageFilter, ImageDraw, ImageOps

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
                        "current_num", "mask_image", "input_image", "control_image", "output_name_format"]

def latents_to_rgb(latents):
    rgb = latents[0].transpose(1, 2, 0) @ WEIGHTS + BIASES
    image_array = rgb.clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(image_array)

def preview_image(latents, output_dir, output_prefix):
    filename = f"{output_prefix}preview_{datetime.now().strftime("%Y%m%d%H%M%S")}.png"
    output_path = os.path.join(output_dir, filename)
    output_image = latents_to_rgb(latents)
    output_image.save(output_path)

def array_to_image(image_np, for_vae=True):
    image = (image_np / 2 + 0.5).clip(0, 1) if for_vae else image_np
    image = image.squeeze(0).transpose(1, 2, 0)

    image_uint8 = (image * 255).astype(np.uint8)

    output_image = Image.fromarray(image_uint8)
    return output_image

def save(image, output_dir, output_prefix, **config):
    if isinstance(image, np.ndarray):
        image = array_to_image(image)
    
    metadata = PngImagePlugin.PngInfo()
    # import sys
    # metadata.add_text("command", " ".join(['"' + x + '"' if " " in x else x for x in sys.argv]))
    for k, v in config.items():
        if k in METADATA_IGNORE_LIST:
            continue
        metadata.add_text(k, str(v))

    from string import Template
    t = Template(config["output_name_format"])
    filename = t.safe_substitute(now=datetime.now().strftime("%Y%m%d%H%M%S"), prompt=config["prompt"][:15],
                                 seed=config["seed"], cfg=config["cfg"], steps=config["steps"],
                                 scheduler_type=config["scheduler_type"])
    filename = f"{filename}.png"
    # filename = f"{output_prefix}{datetime.now().strftime("%Y%m%d%H%M%S")}.png"
    output_path = os.path.join(output_dir, filename)

    image.save(output_path, pnginfo=metadata)

    print(f"Image saved to: {output_path}, seed: {config['seed']}")
    logger.info(f"Image saved to: {output_path}, seed: {config['seed']}")

def preprocess_image(image, target_size=(1024, 1024), for_vae=True):
    # 1. 画像の読み込みとRGB変換、リサイズ
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        
    image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
    
    # 2. numpy配列に変換 (0 ~ 255)
    image_np = np.array(image).astype(np.float16)
    
    # 3. 軸の入れ替え (Height, Width, Channel) -> (Channel, Height, Width)
    image_np = image_np.transpose(2, 0, 1)
    
    # 4. バッチ次元の追加 (1, Channel, Height, Width)
    image_np = np.expand_dims(image_np, axis=0)
    
    # 5. 正規化 (0~255 -> -1.0~1.0)
    # VAE Encoderは、ピクセル値が -1 から 1 の範囲であることを想定しています
    if for_vae:
        image_normalized = (image_np / 127.5) - 1.0
    else:
        image_normalized = (image_np / 255.0)
    
    return image_normalized

def preprocess_mask(mask, target_size=(128, 128), blur_radius=16):
    """
    マスク画像の前処理
    :param mask: マスク画像、またはパス
    :param target_size: UNetに渡すLatentのサイズ（通常は 128x128）
    :param blur_radius: ぼかしの強度（ピクセル半径。0を指定するとぼかしなし）
    """
    if isinstance(mask, str):
        mask = Image.open(mask).convert("L")

    # if blur_radius > 0:
    #     mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    mask = mask.resize(target_size, resample=Image.Resampling.BILINEAR)

    mask_np = np.array(mask)
    # mask_np = np.floor(mask_np)
    mask_np[mask_np < 255] = 0

    if blur_radius > 0:
        mask = Image.fromarray(mask_np)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask_np = np.array(mask).astype(np.float16)

    mask_np = mask_np / 255.0

    # 形状をUNet/Latentに合わせる (1, 1, 128, 128)
    mask_np = np.expand_dims(mask_np, axis=(0, 1))
    
    return mask_np

def generate_outpaint_outputs(image1, image2_np, direction="right", target_size=(1024, 1024), outpaint_ratio=0.5):
    # image1 = array_to_image(image1_np)
    image2 = array_to_image(image2_np)

    W, H = target_size
    src_w, src_h = image1.size
    if direction in ["right", "left"]:
        target_w = round(W * outpaint_ratio)
        target_h = H
        crop_w = W - target_w
        crop_h = H
        out_W = src_w + target_w
        out_H = H
    else:
        target_w = W
        target_h = round(H * outpaint_ratio)
        crop_w = W
        crop_h = H - target_h
        out_W = W
        out_H = src_h + target_h

    canvas_image = Image.new("RGB", (out_W, out_H), (0, 0, 0))

    output_w = out_W - W
    output_h = out_H - H
    if direction == "right":
        canvas_image.paste(image1, (0, 0))
        canvas_image.paste(image2, (output_w, 0))
    elif direction == "left":
        canvas_image.paste(image1, (output_w, 0))
        canvas_image.paste(image2, (0, 0))
    elif direction == "top":
        canvas_image.paste(image1, (0, output_h))
        canvas_image.paste(image2, (0, 0))
    elif direction == "bottom":
        canvas_image.paste(image1, (0, 0))
        canvas_image.paste(image2, (0, output_h))

    return canvas_image

def generate_outpaint_inputs(image_path, direction="right", target_size=(1024, 1024),
                             outpaint_ratio=0.5, overlap_pixels=16):
    """
    固定解像度内でOutpaintを行うための、入力画像とマスク画像を自動生成する関数
    
    :param image_path: 元画像のパス
    :param direction: 拡張したい方向 ("right" または "both")
    :param target_size: NPUが要求する固定サイズ (1024, 1024)
    :param outpaint_ratio: 元画像をどれくらい残すか
    :return: (canvas_image, mask_image) - ともにPIL.Imageオブジェクト
    """
    src_image = Image.open(image_path).convert("RGB")
    
    W, H = target_size
    if W < target_size[0] or H < target_size[1]:
        src_image = ImageOps.pad(src_image, target_size)
    
    src_w, src_h = src_image.size
    if direction in ["right", "left"]:
        target_w = round(W * outpaint_ratio)
        target_h = H
        crop_w = W - target_w
        crop_h = H
        output_w = src_w + target_w
        output_h = H
    else:
        target_w = W
        target_h = round(H * outpaint_ratio)
        crop_w = W
        crop_h = H - target_h
        output_w = W
        output_h = src_h + target_h
    
    canvas_image = Image.new("RGB", target_size, (0, 0, 0))
    mask_image = Image.new("L", target_size, 0)

    margin = overlap_pixels
    padding_len = 4
    blur_radius = 2
    if direction == "right":
        cropped_src = src_image.crop((output_w - W, 0, W, H))
        canvas_image.paste(cropped_src, (0, 0))
        # padding_image = src_image.crop((W - padding_len, 0, W, H))
        # padding_canvas = Image.new("RGB", (target_w, H), (128, 128, 128))
        # for x in range(0, target_w, padding_len):
        #     padding_image = padding_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        #     padding_canvas.paste(padding_image, (x, 0))
        # canvas_image.paste(padding_canvas, (W - target_w, 0))
        mask_image.paste(255, (crop_w-margin, 0, W, H))
        
    elif direction == "left":
        cropped_src = src_image.crop((0, 0, crop_w, H))
        canvas_image.paste(cropped_src, (target_w, 0))
        # padding_image = src_image.crop((0, 0, padding_len, H))
        # padding_canvas = Image.new("RGB", (target_w, H), (128, 128, 128))
        # for x in range(target_w - padding_len, -padding_len, -padding_len):
        #     padding_image = padding_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        #     padding_canvas.paste(padding_image, (x, 0))
        # canvas_image.paste(padding_canvas, (0, 0))
        mask_image.paste(255, (0, 0, target_w+margin, H))
    elif direction == "top":
        cropped_src = src_image.crop((0, 0, W, crop_h))
        canvas_image.paste(cropped_src, (0, target_h))
        # padding_image = src_image.crop((0, 0, W, padding_len))
        # padding_canvas = Image.new("RGB", (W, target_h), (128, 128, 128))
        # for x in range(target_h - padding_len, -padding_len, -padding_len):
        #     padding_image = padding_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        #     padding_canvas.paste(padding_image, (0, x))
        # canvas_image.paste(padding_canvas, (0, 0))
        mask_image.paste(255, (0, 0, W, target_h+margin))
    elif direction == "bottom":
        # cropped_src = src_image.crop((0, target_h, W, H))
        cropped_src = src_image.crop((0, output_h - H, W, output_h))
        canvas_image.paste(cropped_src, (0, 0))
        # padding_image = src_image.crop((0, H - padding_len, W, H))
        # padding_canvas = Image.new("RGB", (W, target_h), (128, 128, 128))
        # for x in range(0, target_h, padding_len):
        #     padding_image = padding_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        #     padding_canvas.paste(padding_image, (0, x))
        # canvas_image.paste(padding_canvas, (0, H - target_h))
        mask_image.paste(255, (0, crop_h-margin, W, H))
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return src_image, canvas_image, mask_image


def split_into_tiles(src_image, tile_size=(1024, 1024), overlap_pixels=64):
    """
    画像を、指定サイズ(1024x1024)でオーバーラップさせながらタイルに分割する関数
    """
    W, H = src_image.size
    if W < tile_size[0] and H < tile_size[1]:
        target_size = (tile_size[0], H * tile_size[0] // W) if W < H else (W * tile_size[1] // H, tile_size[1])
        src_image = src_image.resize(target_size, resample=Image.Resampling.LANCZOS)

    horizontal_tile_count = (W // (tile_size[0] - overlap_pixels)) + 1 if W != tile_size[0] else 1
    vertical_tile_count = (H // (tile_size[1] - overlap_pixels)) + 1 if H != tile_size[1] else 1

    tile_width = (W - tile_size[0]) // (horizontal_tile_count - 1) if horizontal_tile_count != 1 else tile_size[0]
    tile_height = (H - tile_size[1]) // (vertical_tile_count - 1) if vertical_tile_count != 1 else tile_size[1]

    # 最終出力には、画像の中央部を優先して採用されるように外側から順に処理する
    horizontal_list = list(process_from_outside(list(range(horizontal_tile_count))))
    vertical_list = list(process_from_outside(list(range(vertical_tile_count))))

    tiles_info = []
    positions = []
    
    for x in horizontal_list:
        tile_pos_x = x * tile_width if x < horizontal_tile_count - 1 else W - tile_size[0]
        for y in vertical_list:
            tile_pos_y = y * tile_height  if y < vertical_tile_count - 1 else H - tile_size[1]
            positions.append((tile_pos_x, tile_pos_y))
            
    for z_index, (x, y) in enumerate(positions):
        # 指定座標から 1024x1024 の領域を切り出す
        box = (x, y, x + tile_size[0], y + tile_size[1])
        tile_img = src_image.crop(box)
        
        # 画像オブジェクトと、元の位置（x, y座標）をペアにして保持
        tiles_info.append((z_index, tile_img, (x, y)))

    return tiles_info


def create_blend_mask(size=(1024, 1024), fade_pixels=64,
                      top_edge=False, bottom_edge=False, left_edge=False, right_edge=False):
    """
    タイルの4辺のフチをフワッとぼかすためのグラデーションマスクを生成する
    """
    # すべて白（不透明：255）のベースを作成
    mask = Image.new("L", size, 255)
    mask_np = np.array(mask).astype(np.float32)

    # 4つの端から fade_pixels 分だけ、外側に向かってグラデーション（0〜255）を作る
    for i in range(fade_pixels):
        # 外側（i=0）ほど 0（透明）に近く、内側（i=fade_pixels-1）ほど 255（不透明）に近づく値
        alpha = (i / fade_pixels) * 255
        
        # np.minimum を使い、現在の行/列の値と alpha を比較して、小さい方を採用する
        # 上下のフチ
        if not top_edge:
            mask_np[i, :] = np.minimum(mask_np[i, :], alpha)
        if not bottom_edge:
            mask_np[size[1] - 1 - i, :] = np.minimum(mask_np[size[1] - 1 - i, :], alpha)
        
        # 左右のフチ
        if not left_edge:
            mask_np[:, i] = np.minimum(mask_np[:, i], alpha)
        if not right_edge:
            mask_np[:, size[0] - 1 - i] = np.minimum(mask_np[:, size[0] - 1 - i], alpha)

    return Image.fromarray(mask_np.astype(np.uint8))


def merge_tiles(processed_tiles_info, canvas_size=(2048, 2048), tile_size=(1024, 1024)):
    """
    処理後のタイルを、位置情報を元にグラデーションブレンドしながら1枚の大きな画像に合成する関数
    
    :param processed_tiles_info: [(処理後Image, (x, y)), ...] のリスト
    :param canvas_size: 最終的な出力画像サイズ (2048, 2048)
    """
    # 最終出力用の巨大キャンバスを作成
    final_canvas = Image.new("RGB", canvas_size)
    
    for z_index, tile_img, (x, y) in sorted(processed_tiles_info, key=lambda x: x[0]):
        top_edge = y == 0
        bottom_edge = y == canvas_size[1] - tile_size[1]
        left_edge = x == 0
        right_edge = x == canvas_size[0] - tile_size[0]
        # 境界線をなだらかにするブレンドマスクを生成（フチの64ピクセルをぼかす）
        blend_mask = create_blend_mask(size=tile_size, fade_pixels=64,
                                       top_edge=top_edge, bottom_edge=bottom_edge,
                                       left_edge=left_edge, right_edge=right_edge)

        # final_canvasの指定座標 (x, y) に、tile_img を貼り付ける
        # 第3引数に blend_mask を渡すことで、重なるフチが綺麗にグラデーションで混ざり合います
        final_canvas.paste(tile_img, (x, y), mask=blend_mask)

    return final_canvas

def process_from_outside(lst):
    """
    リストの外側（両端）から中央に向かって順に要素を返すジェネレータ。
    空リストや要素数が奇数・偶数の場合も正しく処理します。
    """
    if not isinstance(lst, list):
        raise TypeError("引数は list 型である必要があります。")

    left, right = 0, len(lst) - 1
    while left <= right:
        if left == right:
            yield lst[left]  # 中央の1要素
        else:
            yield lst[left]  # 左端
            yield lst[right] # 右端
        left += 1
        right -= 1

def preprocess_tile_for_clip(tile_image, size=(224, 224)):
    """
    Hires.fixのタイル画像をCLIP Image Encoder (NPU) が受け取れる形式に高速変換する
    
    Parameters:
    - tile_image_np: NumPy配列、形状 (H, W, 3) などのRGB画像、値は 0〜255 (uint8)
    
    Returns:
    - input_tensor: NumPy配列、形状 (1, 3, width, height)、float32型 (NPU入力用)
    """
    W, H = size
    # 1. リサイズ
    resized = tile_image.resize((W, H), resample=Image.Resampling.BICUBIC)
    resized_np = np.array(resized)
    
    # 2. float32に変換しつつ、0.0 〜 1.0 にスケーリング
    img_float = resized_np.astype(np.float32) / 255.0
    
    # 3. CLIPの標準的なImageNet平均・標準偏差で正規化 (NumPyのブロードキャスト)
    # ※ 形状 (3,) の配列を用意しておけば、(W, H, 3) に対して一括で計算できます
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    
    normalized = (img_float - mean) / std
    
    # 4. 軸の入れ替え: (H, W, C) -> (C, H, W)
    # 5. バッチ次元の追加: (C, H, W) -> (1, C, H, W)
    input_tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)
    input_tensor = cvtColor(input_tensor)
    
    # NPUのモデルが float16 を期待している場合は、ここで .astype(np.float16) にしてください
    return input_tensor.astype(np.float16)

def sample_tile_tokens(img_features, num_tokens=40):
    """
    CLIP Visionの特徴量(1, 257, D)から、指定されたトークン数に均等に間引く
    
    Parameters:
    - img_features: NumPy配列 (1, 257, D)
    - num_tokens: 最終的に出力したい合計トークン数（最低2以上を想定）
    
    Returns:
    - sampled_features: NumPy配列 (1, num_tokens, D)
    """
    # 安全のため、バッチサイズ1を前提としてスクイーズ（またはスライスで処理）
    # 実際は (257, D) の2次元としてインデックスを計算します
    feat_2d = img_features[0] 
    total_dim = feat_2d.shape[1]
    
    # 1. 画像全体を表すCLSトークン（先頭）を確保
    cls_token = feat_2d[0:1, :] # shape: (1, D)
    
    # 2. 空間パッチトークン（2番目〜197番目までの196個）を切り出し
    # ※257トークンのうち、後ろの60個はパディングや不要な情報のケースが多いため、
    # 有効な196個のパッチからサンプリングします。
    patch_tokens = feat_2d[1:197, :]
    
    # 3. 残りの必要なパッチ数を計算 (合計数 - CLSトークン1個)
    needed_patches = num_tokens - 1
    # needed_patches = num_tokens
    
    # 4. 196個のパッチから、指定数(needed_patches)を均等に間引くインデックスを生成
    # np.linspaceを使うことで、端から端まで綺麗にカバーしたインデックスを作れます
    #（例：needed_patches=39なら、0から195の間を等間隔に割った39個の整数を生成）
    indices = np.linspace(0, len(patch_tokens) - 1, needed_patches, dtype=np.int32)
    
    # 5. インデックスに基づいてパッチを抽出
    sampled_patches = patch_tokens[indices, :] # shape: (needed_patches, D)
    
    # 6. CLSトークンとサンプリングしたパッチを結合し、バッチ次元(1)を戻す
    combined = np.vstack([cls_token, sampled_patches]) # shape: (num_tokens, D)
    # combined = sampled_patches # shape: (num_tokens, D)
    sampled_features = np.expand_dims(combined, axis=0) # shape: (1, num_tokens, D)
    
    return sampled_features


def cvtColor(input_np):
    """ Convert RGB <=> BGR """
    return input_np[:, :, ::-1]


def preprocess_canny(image_path_or_pil):
    """
    OpenCV (cv2) も PyTorch (torch) も一切使わず、
    Pillow と NumPy だけで SDXL ControlNet 用のテンソルを作成する
    """
    # 1. 画像の読み込み
    if isinstance(image_path_or_pil, str):
        pil_image = Image.open(image_path_or_pil)
    else:
        pil_image = image_path_or_pil

    # 2. 静的サイズ（1024x1024）にリサイズ
    pil_image = pil_image.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
    
    # 3. モノクロ（グレースケール）化
    gray_image = pil_image.convert("L")
    
    # 4. Pillow の輪郭抽出フィルタを適用 (Cannyの代わり)
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    
    # コントラストの自動調整（エッジの強調）
    edges = ImageOps.autocontrast(edges)
    # edges.save("outputs/debug.png")

    # 5. 3チャンネル (RGB) へ変換
    edges_rgb = edges.convert("RGB")

    # 6. NumPy配列への変換
    # この時点で形状は HWC (1024, 1024, 3)、データ型は uint8 (0〜255)
    img_np = np.array(edges_rgb)

    # 7. 軸の入れ替えと正規化、型変換
    # HWC (1024, 1024, 3) -> CHW (3, 1024, 1024)
    img_np = np.transpose(img_np, (2, 0, 1))
    
    # 0.0 〜 1.0 の float16 に変換
    img_np = img_np.astype(np.float16) / 255.0

    # 8. バッチ次元の追加 (1, 3, 1024, 1024)
    controlnet_cond = np.expand_dims(img_np, axis=0)

    return controlnet_cond

# --- 使い方イメージ ---
# controlnet_cond = preprocess_canny("input_image.png")
# print(controlnet_cond.shape)  # (1, 3, 1024, 1024)
# print(controlnet_cond.dtype)  # float16


if __name__ == "__main__":
    import conf
    tags_dict_path = rf"{conf.MODEL_ROOT_DIR}\helper\wd_vit_tagger_v3\selected_tags.csv"
    model_path = rf"{conf.MODEL_ROOT_DIR}\helper\wd_vit_tagger_v3\model.onnx"
    
    # img = create_blend_mask()
    # img.save("mask.png")

    # from PIL import Image
    # src_image = Image.open(rf"")
    # split_into_tiles(src_image, tile_size=1024, overlap_pixels=64):
    # 使用例
    data = [1, 2, 3, 4, 5, 6]
    print(list(process_from_outside(data)))
    # 出力: [1, 6, 2, 5, 3, 4]

    data2 = ["a", "b", "c", "d", "e"]
    print(list(process_from_outside(data2)))
    for x in process_from_outside(data2):
        print(x)
    # 出力: ['a', 'e', 'b', 'd', 'c']
    # white tiger
    image_path = rf"{conf.PROJECT_ROOT_DIR}\outputs\output_sdxl_npu_20260816172911.png"
    # dog
    image_path = rf"{conf.PROJECT_ROOT_DIR}\outputs\output_sdxl_npu_20260816174620.png"
    # castle
    image_path = rf"{conf.PROJECT_ROOT_DIR}\outputs\output_sdxl_npu_20260816172907.png"
    
    src_image = Image.open(image_path).convert("RGB")
    # tiles_info = split_into_tiles(src_image, tile_size=(512, 512), overlap_pixels=64)
    tiles_info = split_into_tiles(src_image, tile_size=(1024, 1024), overlap_pixels=64)

    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import qnn_ep_helper as qnn
    import onnxruntime as ort

    from .text_encoder import TextEncoder

    text_encoder_dir = rf"{conf.MODEL_ROOT_DIR}\dreamshaper-xl-lightning-for-Snapdragon-X-Elite\text_encoder"
    tokenizer_dir = rf"{conf.MODEL_ROOT_DIR}\dreamshaper-xl-lightning-for-Snapdragon-X-Elite\tokenizer"
    text_encoder = TextEncoder(text_encoder_dir, tokenizer_dir, True)
    text_embeds = text_encoder.get_text_embeddings("a photo of a white tiger")
    model_path = rf"{conf.MODEL_ROOT_DIR}\image_encoder\model.onnx"
    model = ort.InferenceSession(model_path, sess_options=qnn.session_options)
    for z_index, tile, (x, y) in tiles_info:
        inp = preprocess_tile_for_clip(tile)
        # print(inp)

        inputs = {"pixel_values": inp}

        output_names = list(map(lambda x: x.name, model.get_outputs()))
        output_list = model.run(output_names, inputs)
        print(f"CLIP-L ViT shape: {output_list[0].shape}")

        # result = sample_tile_tokens(output_list[0], 77)
        result = output_list[2][:, :77, :]
        print(f"-> {result.shape}")
        proj = np.load(rf"{conf.PROJECT_ROOT_DIR}\scripts\sdxl_pipeline\generated\visual_projection_l.npy")
        proj_l = np.dot(result, proj)

        t_proj = np.load(rf"{conf.PROJECT_ROOT_DIR}\scripts\sdxl_pipeline\generated\text_projection_l.npy")
        text_embeds = np.dot(text_embeds, t_proj)
        print(f"-> {proj_l.shape}")

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # print(proj_l @ text_embeds.T)
        cos = cosine_similarity(proj_l.flatten().astype(np.float32), text_embeds.flatten().astype(np.float32))
        print(cos)
        


