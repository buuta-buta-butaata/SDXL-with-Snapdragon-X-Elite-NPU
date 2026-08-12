import os
import argparse
import numpy as np
import qai_hub as hub
import time
import common_conf as conf
import json
from generated import unet_part0_specs, unet_part1_specs, unet_part2_specs, unet_part3_specs, unet_part4_specs

def parse_args():
    parser = argparse.ArgumentParser(description="5分割モデル用 AI Hub 連続量子化・コンパイル自動化ツール")
    parser.add_argument("--name", type=str, default="sdxl_model", help="モデルの名称、短い方が見やすい")
    parser.add_argument("--width", type=int, default=1024, help="対象モデルの幅")
    parser.add_argument("--height", type=int, default=1024, help="対象モデルの高さ")
    return parser.parse_args()

def run_ai_hub_hybrid_quantization(name, width, height):
    res_str = f"{width}x{height}"

    with open(f"generated/{name}_{res_str}.json") as f:
        d = json.load(f)
    
    # 1. 5分割モデルの正確なファイルidマッピング
    PART_FILES = {
        # "part0": d["unet_part0"],
        "part1": d["unet_part1"],
        "part2": d["unet_part2"],
        "part3": d["unet_part3"],
        "part4": d["unet_part4"],
    }

    PART_OUTPUTS = {
        # "part0": "silu_3",
        # "part1": "add_13,add_2,add_22,add_4,add_55,add_88,conv2d,conv2d_11,conv2d_5",
        # "part2": "add_123",
        # "part3": "add_156",
        # "part4": "out_sample"
        "part1": unet_part1_specs.output_names,
        "part2": unet_part2_specs.output_names,
        "part3": unet_part3_specs.output_names,
        "part4": unet_part4_specs.output_names,
    }
    
    print(f"\n==================================================")
    print(f" 🚀 AI Hub 連続ハイブリッド量子化開始 [解像度: {res_str}]")
    print(f"==================================================")

    # いったん削除


if __name__ == "__main__":
    args = parse_args()
    run_ai_hub_hybrid_quantization(args.name, args.width, args.height)
