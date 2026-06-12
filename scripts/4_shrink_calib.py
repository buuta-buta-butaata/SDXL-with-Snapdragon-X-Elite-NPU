import os
import argparse
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="5分割・解像度別 キャリブレーションデータ軽量化ツール")
    parser.add_argument("--width", type=int, default=1024, help="対象モデルの幅")
    parser.add_argument("--height", type=int, default=1024, help="対象モデルの高さ")
    parser.add_argument("--samples", type=int, default=24, help="抽出するステップ数（デフォルト24）")
    return parser.parse_args()

def shrink_perfect_calib_data(width, height, target_samples=24):
    res_str = f"{width}x{height}"
    input_root_dir = f"../calibration_data/{res_str}/raw"
    output_root_dir = f"../calibration_data/{res_str}/shrunk"
    
    # 🌟 【最新仕様】Netronで確認していただいた厳格なONNXの入力順定義
    from generated import unet_part0_specs, unet_part1_specs, unet_part2_specs, unet_part3_specs, unet_part4_specs
    PART_EXACT_ORDER = {
        "part0": unet_part0_specs.input_names,
        "part1": unet_part1_specs.input_names,
        "part2": unet_part2_specs.input_names,
        "part3": unet_part3_specs.input_names,
        "part4": unet_part4_specs.input_names,
    }
    
    parts = ["part0", "part1", "part2", "part3", "part4"]
    
    print(f"\n==================================================")
    print(f" 📦 キャリブデータ軽量化・整形開始 [解像度: {res_str}]")
    print(f"==================================================")

    for part in parts:
        part_input_dir = os.path.join(input_root_dir, part)
        part_output_dir = os.path.join(output_root_dir, part)
        os.makedirs(part_output_dir, exist_ok=True)
        
        # ゼロパディング(step_000.npy)されているため時系列順に正しくソート
        all_files = sorted(glob.glob(os.path.join(part_input_dir, "*.npy")))
        # 偶数のみ
        # all_files = sorted(glob.glob(os.path.join(part_input_dir, "step_*.npy")))[::2]
        total_files = len(all_files)
        
        if total_files == 0:
            print(f" 【スキップ】{part_input_dir} に生データがありません。")
            continue
            
        print(f" -> [{part}] 総数 {total_files} から {target_samples} ステップをサンプリング中...")
        
        # 時系列から均等に抽出
        indices = np.linspace(0, total_files - 1, target_samples, dtype=int)
        selected_files = [all_files[i] for i in indices]
        
        # ONNXの入力順に基づき、順序が固定された空の辞書を初期化（AI Hub必須仕様）
        hub_formatted_dataset = {key: [] for key in PART_EXACT_ORDER[part]}
        
        for file_path in selected_files:
            print(file_path)
            # 1ステップ分の全入力辞書をロード
            step_data = np.load(file_path, allow_pickle=True).item()
            
            # text_embeds以外はfloat16
            for key in list(step_data.items()):
                if key[0] != "text_embeds":
                    step_data[key[0]] = step_data[key[0]].astype(np.float16)
                    
            # Netron準拠の正確な順番でリストへ詰め込み
            for input_name in PART_EXACT_ORDER[part]:
                if input_name in step_data:
                    hub_formatted_dataset[input_name].append(step_data[input_name])
                else:
                    raise KeyError(f"❌ エラー: 生データ内にONNXが要求する入力名 '{input_name}' が見つかりません。")

        # 保存
        output_file_path = os.path.join(part_output_dir, f"{part}_calib_dataset.npy")
        np.save(output_file_path, hub_formatted_dataset)
        
        new_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
        print(f"    [完了] -> {output_file_path} (サイズ: {new_size_mb:.2f} MB)")

if __name__ == "__main__":
    args = parse_args()
    shrink_perfect_calib_data(args.width, args.height, args.samples)
