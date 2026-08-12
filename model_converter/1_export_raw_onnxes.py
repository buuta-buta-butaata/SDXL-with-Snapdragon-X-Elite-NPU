# ref. https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/utils.py
import os
import argparse
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL UNet 1本物ONNX自動エクスポート")
    parser.add_argument("--name", type=str, default="sdxl_model", help="モデルの名称、短い方が見やすい")
    parser.add_argument("--width", type=int, default=1024, help="生成画像の幅 (1024, 1344, 832など)")
    parser.add_argument("--height", type=int, default=1024, help="生成画像の高さ (1024, 768, 1216など)")
    parser.add_argument("--model_path", type=str, default="../safetensors/your_model.safetensors", help="元モデルのパス")
    parser.add_argument("--text_encoder", action="store_true", help="text_encoderをエクスポート")
    parser.add_argument("--text_encoder_2", action="store_true", help="text_encoder_2をエクスポート")
    parser.add_argument("--unet", action="store_true", help="unetをエクスポート")
    parser.add_argument("--vae_decoder", action="store_true", help="vae_decoderをエクスポート")
    parser.add_argument("--vae_encoder", action="store_true", help="vae_encoderをエクスポート")
    parser.add_argument("--opset", type=int, default=22, help="使用するopset、とりあえず22でうまくいってる")
    return parser.parse_args()

class TextEncoder(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.encoder = text_encoder

    def forward(self, input_ids):
        prompt_embeds = self.encoder(input_ids, output_hidden_states=True)
        return prompt_embeds[0], prompt_embeds.hidden_states[-2]

class UNetWrapper(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids
        }
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        ).sample

def export_text_encoder(pipe, output_dir, args):
    text_encoder = TextEncoder(pipe.text_encoder)
    text_encoder.eval()

    dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.int32)
    output_onnx_dir = os.path.join(output_dir, "text_encoder")
    os.makedirs(output_onnx_dir, exist_ok=True)
    output_onnx_path = os.path.join(output_onnx_dir, "text_encoder.onnx")

    print(f" -> Text Encoder ONNXエクスポート実行中... ")
    with torch.no_grad():
        torch.onnx.export(
            text_encoder,
            (dummy_input,),
            output_onnx_path,
            input_names=["input_ids"],
            output_names=['pooled_prompt_embed', 'prompt_embed'],
            opset_version=args.opset,
        )
    print(f"✅ 【成功】Text Encoder ONNXを保存しました: {output_onnx_path}")

def export_text_encoder_2(pipe, output_dir, args):
    text_encoder_2 = TextEncoder(pipe.text_encoder_2)
    text_encoder_2.eval()

    dummy_input = torch.randint(0, 49408, (1, 77), dtype=torch.int32)
    output_onnx_dir = os.path.join(output_dir, "text_encoder_2")
    os.makedirs(output_onnx_dir, exist_ok=True)
    output_onnx_path = os.path.join(output_onnx_dir, "text_encoder_2.onnx")

    print(f" -> Text Encoder 2 ONNXエクスポート実行中... ")
    with torch.no_grad():
        torch.onnx.export(
            text_encoder_2,
            (dummy_input,),
            output_onnx_path,
            input_names=["input_ids"],
            output_names=['pooled_prompt_embed', 'prompt_embed'],
            opset_version=args.opset,
        )    
    print(f"✅ 【成功】Text Encoder 2 ONNXを保存しました: {output_onnx_path}")

def export_unet(pipe, output_dir, args):
    unet = pipe.unet
    res_str = f"{args.width}x{args.height}"
    
    # 3. 解像度からダミー入力のサイズを自動計算 (VAEの圧縮率 1/8)
    latents_h = args.height // 8
    latents_w = args.width // 8
    print(f" -> 計算された内部Latentsシェイプ: (1, 4, {latents_h}, {latents_w})")

    output_onnx_dir = os.path.join(output_dir, "unet")
    os.makedirs(output_onnx_dir, exist_ok=True)
    output_onnx_path = os.path.join(output_onnx_dir, f"unet_{res_str}.onnx")
    
    # 動的解像度に基づいた正確なダミーテンソル生成
    sample = torch.randn(1, 4, latents_h, latents_w, dtype=torch.float16)
    timestep = torch.tensor([1], dtype=torch.float16)
    encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float16)
    
    # 前回発見した重要な仕様を完全維持
    text_embeds = torch.randn(1, 1280, dtype=torch.float32) # float32必須仕様
    time_ids = torch.randn(1, 6, dtype=torch.float16)       # time conditioning
    
    # ラッパーの構築
    w_unet = UNetWrapper(unet).to(dtype=torch.float16)
    w_unet.eval()
    
    # 4. ONNXエクスポート実行
    print(f" -> UNet ONNXエクスポート実行中... (大容量のため、RAM 16GB環境では数分かかります)")
    with torch.no_grad():
        torch.onnx.export(
            w_unet,
            (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
            output_onnx_path,
            input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
            output_names=["out_sample"],
            export_params=True,
            opset_version=args.opset,
        )
    print(f"✅ 【成功】UNet ONNXを保存しました: {output_onnx_path}")

def export_vae_decoder(pipe, output_dir, args):
    res_str = f"{args.width}x{args.height}"
    output_onnx_dir = os.path.join(output_dir, "vae_decoder")
    os.makedirs(output_onnx_dir, exist_ok=True)
    output_onnx_path = os.path.join(output_onnx_dir, f"vae_decoder_{res_str}.onnx")
    
    vae_decoder = copy.deepcopy(pipe.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_decoder.eval()

    latents_h = args.height // 8
    latents_w = args.width // 8
    sample = torch.randn(1, 4, latents_h, latents_w, dtype=torch.float16)

    print(f" -> VAE Decoder ONNXエクスポート実行中... ")
    with torch.no_grad():
        torch.onnx.export(
            vae_decoder,
            (sample,),
            output_onnx_path,
            input_names=["latent_sample"],
            output_names=["sample"],
            external_data=False,
            opset_version=args.opset,
        )

    print(f"✅ 【成功】VAE Decoder ONNXを保存しました: {output_onnx_path}")
    

def export_vae_encoder(pipe, output_dir, args):
    res_str = f"{args.width}x{args.height}"
    output_onnx_dir = os.path.join(output_dir, "vae_encoder")
    os.makedirs(output_onnx_dir, exist_ok=True)
    output_onnx_path = os.path.join(output_onnx_dir, f"vae_encoder_{res_str}.onnx")

    vae_encoder = copy.deepcopy(pipe.vae)
    vae_encoder.forward = lambda sample: {"latent_parameters": vae_encoder.encode(x=sample)["latent_dist"].parameters}
    vae_encoder.eval()
     
    sample = torch.randn(1, 3, args.height, args.width, dtype=torch.float16)
     
    print(f" -> VAE Encoder ONNXエクスポート実行中... ")
    with torch.no_grad():
        torch.onnx.export(
            vae_encoder,
            (sample,),
            output_onnx_path,
            input_names=["sample"],
            output_names=["latent"],
            external_data=False,
            opset_version=args.opset,
        )

    print(f"✅ 【成功】VAE Encoder ONNXを保存しました: {output_onnx_path}")
    
def main():
    args = parse_args()
    
    # 1. 解像度に応じたフォルダ名の自動構成
    res_str = f"{args.width}x{args.height}"
    output_dir = os.path.join("../raw_models", args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n==================================================")
    print(f" 🚀 UNet ONNX Export 開始 [ターゲット解像度: {res_str}]")
    print(f"==================================================")
    
    # 2. モデルのロード
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"エラー: 元モデルが指定パスに見つかりません: {args.model_path}")
        
    print(f" -> Safetensors をロード中: {args.model_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.model_path,
        torch_dtype=torch.float16,
    )

    if args.text_encoder:
        export_text_encoder(pipe, output_dir, args)

    if args.text_encoder_2:
        export_text_encoder_2(pipe, output_dir, args)

    if args.unet:
        export_unet(pipe, output_dir, args)

    if args.vae_decoder:
        export_vae_decoder(pipe, output_dir, args)
    
    if args.vae_encoder:
        export_vae_encoder(pipe, output_dir, args)

if __name__ == "__main__":
    main()
