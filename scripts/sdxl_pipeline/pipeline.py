import time

import gc
import os
import psutil
import torch

import numpy as np
from datetime import datetime
from PIL import Image, PngImagePlugin

from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

# 自作した各モジュールから関数・クラスをインポート
from text_processing  import TextProcessing
from unet import NpuUNetLoop
from vae_decoder import VAEDecoder
from calib_data_collector import CalibrationDataCollector

import default_sdxl_config as def_conf
import utils

class SDXLDirs:
    def __init__(
            self,
            output_dir = None,
            scheduler_dir = None,
            tokenizer_dir = None,
            tokenizer_2_dir = None,
            text_encoder_dir = None,
            text_encoder_2_dir = None,
            unet_dir = None,
            vae_decoder_dir = None,
            #vae_encoder_dir = None,
            ):
        self.output_dir = utils.value_or_default(output_dir,
                                                 def_conf.DEFAULT_OUTPUT_DIR)
        self.scheduler_dir = utils.value_or_default(scheduler_dir,
                                                    def_conf.SCHEDULER_DIR)
        self.tokenizer_dir = utils.value_or_default(tokenizer_dir,
                                                    def_conf.TOKENIZER_DIR)
        self.tokenizer_2_dir = utils.value_or_default(tokenizer_2_dir,
                                                      def_conf.TOKENIZER_2_DIR)
        self.text_encoder_dir = utils.value_or_default(text_encoder_dir,
                                                       def_conf.TEXT_ENCODER_DIR)
        self.text_encoder_2_dir = utils.value_or_default(text_encoder_2_dir,
                                                         def_conf.TEXT_ENCODER_2_DIR)
        self.unet_dir = utils.value_or_default(unet_dir,
                                               def_conf.UNET_DIR)
        self.vae_decoder_dir = utils.value_or_default(vae_decoder_dir,
                                                      def_conf.VAE_DECODER_DIR)

        
class SDXLConfig:
    def __init__(
            self,
            prompt,
            prompt_2 = None,
            negative_prompt = "",
            negative_prompt_2 = None,
            steps = 20,
            guidance_scale = 5.0,
            seed = -1,
            width = 1024,
            height = 1024,
            sdxl_dirs = None,
            scheduler_type = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler,
            calib_strategy = 0,
            calib_data_collector = None,
    ):
        self.prompt = prompt
        self.prompt_2 = utils.value_or_default(prompt_2,
                                               prompt)
        self.negative_prompt = negative_prompt
        self.negative_prompt_2 = utils.value_or_default(negative_prompt_2,
                                                        negative_prompt)
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.width = width
        self.height = height
        self.dirs = utils.value_or_default(sdxl_dirs,
                                                SDXLDirs())
        self.scheduler_type = scheduler_type
        self.calib_strategy = calib_strategy
        self.calib_data_collector = calib_data_collector
                                                

class SDXLPipeline:
    def __init__(self, sdxl_config):
        self.config = sdxl_config

    def run(self):
        print("処理時間計測開始")
        start_time = time.perf_counter()
        
        print("=========================================")
        print("🚀 SDXL NPU パイプライン 起動")
        print("=========================================")
        
        elapsed_time = time.perf_counter()
        print(f"\n総経過時間: {elapsed_time - start_time:.3f} 秒")
        
        text_processing = TextProcessing(self.config.dirs.text_encoder_dir,
                                         self.config.dirs.tokenizer_dir,
                                         self.config.dirs.text_encoder_2_dir,
                                         self.config.dirs.tokenizer_2_dir)

        prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled_embeds = text_processing.encode_text(self.config)
        
        elapsed_time = time.perf_counter()
        print(f"\n総経過時間: {elapsed_time - start_time:.3f} 秒")

        unet = NpuUNetLoop(self.config)
        latents_np = unet.inference(self.config, prompt_embeds, pooled_prompt_embeds,
                  uncond_embeds, uncond_pooled_embeds)
        
        # 用済みとなった常駐UNet、およびテキスト埋め込みをメモリから完全に抹殺
        del unet
        del prompt_embeds, pooled_prompt_embeds
        if self.config.guidance_scale != 1:
            del uncond_embeds, uncond_pooled_embeds

        # --------------------------------------------------
        # Step 5: VAE デコード ➔ 画像ファイル出力
        # --------------------------------------------------
        # VAEの標準的なスケーリングファクター (0.1305) で元に戻す
        latents_np = latents_np / 0.13025
        # self.output_image(latents_np)
            
        # VAEを実行してRGB画像テンソルを取得
        # vae_decoder = VAEDecoder()
        if self.config.calib_data_collector:
            self.config.calib_data_collector.save("vae_decoder", latents_np)

        elapsed_time = time.perf_counter()
        print(f"\n総経過時間: {elapsed_time - start_time:.3f} 秒")

        vae_decoder = VAEDecoder(self.config)
        image_tensor = vae_decoder.decode_latents(latents_np, auto_mem_free=False)
    
        elapsed_time = time.perf_counter()
        print(f"\n総経過時間: {elapsed_time - start_time:.3f} 秒")

        self.output_image(image_tensor)
        end_time = time.perf_counter()
        print(f"\n合計処理時間: {end_time - start_time:.3f} 秒")

        peak_mem = self.get_peak_memory_gb()
        print("==================================================")
        print(" 🛠️  MEMORY PROFILE REPORT")
        print("==================================================")
        print(f" 👑 推論中のピークRAM: {peak_mem:.2f} GB")
        print("==================================================")
        
    def output_image(self, image_tensor):
        print("\n--- [最終工程] 後処理 ＆ 画像保存 ---")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("prompt", self.config.prompt)
        metadata.add_text("negative_prompt", self.config.negative_prompt)
        metadata.add_text("prompt_2", self.config.prompt_2)
        metadata.add_text("negative_prompt_2", self.config.negative_prompt_2)
        metadata.add_text("steps", str(self.config.steps))
        metadata.add_text("guidance_scale", str(self.config.guidance_scale))
        metadata.add_text("seed", str(self.config.seed))
        filename = f"output_sdxl_npu{datetime.now().strftime("%Y%m%d%H%M%S")}.png"
        output_path = os.path.join(self.config.dirs.output_dir,
                                   filename)
        
        # テンソルを [0, 1] の範囲にクリップし、チャンネル順を (H, W, C) に変換
        image = (image_tensor / 2 + 0.5).clip(0, 1)
        image = image.squeeze(0).transpose(1, 2, 0) # (3, 1024, 1024) -> (1024, 1024, 3)
    
        # [0, 255] の uint8 (画像データ型) に変換
        image_uint8 = (image * 255).astype(np.uint8)
    
        # PILを使って画像オブジェクトに変換し、保存
        output_image = Image.fromarray(image_uint8)
        output_image.save(output_path, pnginfo=metadata)
        print(f"🎨 完了！ 『{filename}』 が正常に保存されました。")
        print(f"path: {output_path}")

    def get_peak_memory_gb(self):
        """
        現在のPythonプロセスが消費した最大物理メモリ（ピークRAM）をGB単位で返します。
        Windows環境専用の API (PeakWorkingSetSize) を安全に叩きます。
        """
        process = psutil.Process(os.getpid())
        
        if os.name == 'nt':  # Windows環境
            # info.PeakWorkingSetSize に最大消費バイト数が記録されています
            info = process.memory_info()
            # peak_bytes = getattr(info, 'PeakWorkingSetSize', info.rss)
            peak_bytes = info.peak_wset
        else:  # Linux / macOS 互換用の保険
            peak_bytes = process.memory_info().rss

        return peak_bytes / (1024 ** 3) # バイトからGBに変換

    
if __name__ == "__main__":
    conf = SDXLConfig("A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed")
    main = SDXLPipeline(conf)
    main.run();

