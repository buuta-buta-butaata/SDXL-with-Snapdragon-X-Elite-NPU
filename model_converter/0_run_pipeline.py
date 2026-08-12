import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL UNet 分割パイプライン")
    parser.add_argument("--name", type=str, default="sdxl_model", help="モデルの名称、短い方が見やすい")
    parser.add_argument("--width", type=int, default=1024, help="生成画像の幅 (1024, 1344, 832など)")
    parser.add_argument("--height", type=int, default=1024, help="生成画像の高さ (1024, 768, 1216など)")
    parser.add_argument("--model_path", type=str, default="../safetensors/your_model.safetensors", help="元モデルのパス")
    return parser.parse_args()

def run_cmd(cmd_list):
    """外部スクリプトを安全に実行し、エラーがあれば即停止するヘルパー"""
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] スクリプトの実行に失敗しました。停止します。")
        sys.exit(1)

def build_resolution_assets(width, height, name, model_path):
    res_str = f"{width}x{height}"
    print(f"==================================================")
    # 現在のコンテキスト（西暦2026年）における最新の高速自動化パイプラインを回します
    print(f" 🚀 解像度 {res_str} のNPUアセット自動生成を開始します")
    print(f"==================================================")

    # Step 1: safetensorsから、ONNXエクスポート
    run_cmd([sys.executable, "1_export_raw_onnxes.py", "--model_path", model_path, "--width", str(width), "--height", str(height), "--name", name, "--unet"])

    # Step 2: 5分割処理 (リファクタリングした2_split_unet.py)
    run_cmd([sys.executable, "2_split_unet.py", "--width", str(width), "--height", str(height), "--name", name])

    # Step 3: FP16パイプラインを動かして生キャリブレーションデータを抽出
    # (※3_collect_calib.py 側も引数で解像度を受け取れるようにします)
    # run_cmd([sys.executable, "scripts/3_collect_calib.py", "--width", str(width), "--height", str(height), "--name", name])

    # Step 4: キャリブレーションデータの軽量化・ONNX順序固定化
    # run_cmd([sys.executable, "scripts/4_shrink_calib.py", "--width", str(width), "--height", str(height), "--name", name])

    # print(f"\n✅ 【全工程完了】解像度 {res_str} の量子化準備が整いました！")
    # print(f"    ../split_models/{res_str}/ にONNXが、")
    # print(f"    ../calibration_data/{res_str}/ にAI Hub用データが揃っています。")

if __name__ == "__main__":
    args = parse_args()
    # テストとして、まずは新しい解像度「1024x1024」を一撃で構築してみる場合
    build_resolution_assets(width=args.width, height=args.height, name=args.name, model_path=args.model_path)
    
    # 最終的にはループで全解像度を自動巡回可能です
    # for w, h in [(1024, 1024), (1344, 768), (832, 1216)]:
    #     build_resolution_assets(width=w, height=h)
