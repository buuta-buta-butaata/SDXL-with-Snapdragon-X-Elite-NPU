import argparse
import numpy as np
import os
import random
import sys

from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

from pipeline import SDXLPipeline, SDXLConfig, SDXLDirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="prompt")
    parser.add_argument("--prompt_2", type=str, help="prompt_2")
    parser.add_argument("--steps", type=int, default=6, help="steps")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--negative_prompt_2", type=str, default="", help="negative prompt 2")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    # parser.add_argument("--layout", choices=["P", "L", "Portrait", "Landscape"], default="", help="画像の形状を指定する。指定がない場合、1024x1024)")
    parser.add_argument("--guidance_scale", type=float, default=2, help="height")
    parser.add_argument("--output_dir", type=str, default=None, help="画像出力先のディレクトリ")
    # parser.add_argument("--quantized_model", action="store_true", help="量子化されたモデルを使う")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.steps > 50:
        args.steps = 20
        print(f"ステップ数({args.steps})が大きすぎるため、20に修正")

    print(f"steps: {args.steps}")

    dirs = SDXLDirs(output_dir = args.output_dir)

    # dirs.text_encoder_dir = r"..\..\compiled_models\single\jnxl\fp16\text_encoder"
    # dirs.text_encoder_2_dir =  r"..\..\compiled_models\single\jnxl\fp16\text_encoder_2"
    # dirs.unet_dir = r"..\..\compiled_models\linked\jnxl\unet"
    # if args.quantized_model:
    #     dirs.unet_dir = r".\compiled_models\dreamshaper-xl-lightning-for-Snapdragon-X-Elite-Quantized\unet"
    #     dirs.vae_decoder_dir = r".\compiled_models\dreamshaper-xl-lightning-for-Snapdragon-X-Elite-Quantized\vae_decoder"
    
    scheduler_type = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler

    config = SDXLConfig(args.prompt,
                        prompt_2 = args.prompt_2,
                        negative_prompt = args.negative_prompt,
                        negative_prompt_2 = args.negative_prompt_2,
                        seed = args.seed,
                        steps = args.steps,
                        guidance_scale = args.guidance_scale,
                        scheduler_type = scheduler_type,
                        sdxl_dirs = dirs)

    # if args.layout == "P" or args.layout == "Portrait":
    #     config.width = 832
    #     config.height = 1216
    # elif args.layout == "L" or args.layout == "Landscape":
    #     config.width = 1344
    #     config.height = 768
    # else:
    #     config.width = 1024
    #     config.height = 1024

    # if not args.quantized_model:
    #     config.width = 1024
    #     config.height = 1024
    config.width = 1024
    config.height = 1024

    # config.calib_data_collector=collector
    # pipeline.negative_prompt = get_random_negative_prompt()
    pipe = SDXLPipeline(config)
    pipe.run()

if __name__ == "__main__":
    main()

