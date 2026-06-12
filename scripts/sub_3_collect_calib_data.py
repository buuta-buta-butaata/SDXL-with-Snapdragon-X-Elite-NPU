import argparse
import numpy as np
import os
import random
import sys
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

sys.path.append(os.path.join(os.path.dirname(__file__), 'sdxl_pipeline'))

from pipeline import SDXLPipeline, SDXLConfig, SDXLDirs
from calib_data_collector import CalibrationDataCollector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="prompt")
    parser.add_argument("--name", type=str, default="sdxl_model", help="モデルの名称、短い方が見やすい")
    parser.add_argument("--prompt_2", type=str, default=None, help="prompt_2")
    parser.add_argument("--steps", type=int, default=20, help="steps")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--negative_prompt_2", type=str, default="", help="negative prompt 2")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    parser.add_argument("--width", type=int, default=1024, help="width")
    parser.add_argument("--height", type=int, default=1024, help="height")
    parser.add_argument("--guidance_scale", type=float, default=5, help="height")
    parser.add_argument("--output_dir", type=str, default="", help="画像出力先のディレクトリ")
    # parser.add_argument("--calib_strategy", type=int, default=0, help="キャリブレーションデータ収集の方針[1-3], 1:3つ取得、2:2つ取得（最初と最後の方を優先）、3:2つ取得（中盤よりを優先）")
    parser.add_argument("--calib_strategy", type=int, default=0, help="キャリブレーションデータ収集の方針[1-3] まだ検討中, 1:6つ取得（最初と最後の方を優先）、2:6つ取得（中盤あたりを取得）")
    parser.add_argument("--calib_data_dir", type=str, default="../calibration_data", help="キャリブレーションデータ出力先のディレクトリ")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.steps > 60:
        args.steps = 20
        print(f"ステップ数({args.steps})が大きすぎるため、20に修正")

    print(f"steps: {args.steps}")

    dirs = SDXLDirs()
    dirs.text_encoder_dir = f"../compiled_models/single/{args.name}/fp16/text_encoder"
    dirs.text_encoder_2_dir = f"../compiled_models/single/{args.name}/fp16/text_encoder_2"
    dirs.unet_dir = f"../compiled_models/single/{args.name}/{args.width}x{args.height}_fp16/unet"

    config = SDXLConfig(args.prompt, prompt_2 = args.prompt_2, sdxl_dirs = dirs)
    config.seed = args.seed
    # if args.calib_strategy < 3:
    #     config.steps = args.steps
    # else:
    #     config.steps = 6
    config.steps = args.steps
    config.width = args.width
    config.height = args.height
    config.guidance_scale=args.guidance_scale
    # config.calib_data_collector=collector
    #config.scheduler_type = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler
    config.scheduler_type = KarrasDiffusionSchedulers.DPMSolverMultistepScheduler
    config.calib_strategy = args.calib_strategy
    config.calib_data_collector = CalibrationDataCollector(base_dir=args.calib_data_dir, steps=config.steps, calib_strategy=config.calib_strategy)
    # pipeline.negative_prompt = get_random_negative_prompt()
    pipe = SDXLPipeline(config)
    pipe.run()

if __name__ == "__main__":
    main()

