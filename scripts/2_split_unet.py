import onnx
import os
import argparse
import our_onnx_utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL UNet 1本物ONNXからの自動5分割ツール")
    parser.add_argument("--name", type=str, default="sdxl_model", help="モデルの名称、短い方が見やすい")
    parser.add_argument("--width", type=int, default=1024, help="対象モデルの幅 (1024, 1344など)")
    parser.add_argument("--height", type=int, default=1024, help="対象モデルの高さ (1024, 768など)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 解像度に基づき、入力1本物モデルのパスと、5分割出力先フォルダを動的決定
    res_str = f"{args.width}x{args.height}"
    input_model_path = f"../raw_models/{args.name}/unet/unet_{res_str}.onnx"
    output_dir = f"../split_models/{args.name}/{res_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"エラー: 分割元のONNXファイルが存在しません: {input_model_path}")
        
    print(f" -> 5分割対象のONNXをロード中: {input_model_path}")
    model = onnx.load(input_model_path, load_external_data=False)
    down_end_tensor, mid_end_tensor, up_0_mid_tensor, skip_connections, common_outputs = utils.find_boundary(model, output_dir)
    utils.split_unet(model, input_model_path, output_dir, down_end_tensor, mid_end_tensor, up_0_mid_tensor, skip_connections, common_outputs)
    
if __name__ == "__main__":
    main()

