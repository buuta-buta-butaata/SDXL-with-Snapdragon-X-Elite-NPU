import argparse
import glob
import json
import os
import re
import qai_hub as hub
import common_conf as conf

def parse_args():
    parser = argparse.ArgumentParser(description="5分割モデル用 AI Hub 量子化済みDLCを結合するツール")
    parser.add_argument("--name", type=str, default="sdxl_model", help="モデルの名称、短い方が見やすい")
    return parser.parse_args()


def run_ai_hub_linking():
    args = parse_args()
    # 統合対象の3つの解像度
    # RESOLUTIONS = ["1024x1024", "1344x768", "832x1216"]

    # 処理する5つのPart
    # PARTS = ["part0", "part1", "part2", "part3", "part4"]
    PARTS = ["part1", "part2", "part3", "part4"]
    part_models = {
        "part1": [],
        "part2": [],
        "part3": [],
        "part4": [],
    }
    resolutions = []

    files = glob.glob(f"./generated/{args.name}*_dlc.json")
    for target_file in files:
        print(target_file)
        res_str = re.search(r"[0-9]+x[0-9]+", target_file)
        resolutions.append(res_str.group())
        with open(target_file) as f:
            d = json.load(f)
            for part in PARTS:
                part_models[part].append(d[part])

    print(f"処理対象の解像度: {resolutions}")
    print(f"処理対象のmodel_id: {part_models}")

    # 出力先ディレクトリ
    OUTPUT_DIR = f"../compiled_models/linked/{args.name}/unet"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub_dir_name in PARTS:
        sub_dir = os.path.join(OUTPUT_DIR, sub_dir_name)
        os.makedirs(sub_dir, exist_ok=True)
    
    client = hub.Client()
    
    print("\n==================================================")
    print(" 🔗 AI Hub マルチ解像度 リンク(Weight-sharing)ジョブ開始")
    print("==================================================")

    # いったん削除
    

if __name__ == "__main__":
    run_ai_hub_linking()
