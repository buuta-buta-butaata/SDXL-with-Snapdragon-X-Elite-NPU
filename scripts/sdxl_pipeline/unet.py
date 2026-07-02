import copy
import gc
import os
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from tqdm.auto import tqdm

import qnn_ep_helper as qnn
from schedulers import Scheduler
from calib_data_collector import CalibrationDataCollector


class NpuUNetLoop:
    def __init__(self, config):
        self.collector = config.calib_data_collector
        
        common_path = self.find_model_path(config.dirs.unet_dir, 'part0', config.width, config.height)
        down_path = self.find_model_path(config.dirs.unet_dir, 'part1', config.width, config.height)
        mid_path = self.find_model_path(config.dirs.unet_dir, 'part2', config.width, config.height)
        up_1_path = self.find_model_path(config.dirs.unet_dir, 'part3', config.width, config.height)
        up_2_path = self.find_model_path(config.dirs.unet_dir, 'part4', config.width, config.height)

        print("--- UNet 5分割モデルをNPUにロード中 (常駐駆動) ---")
        # ループ開始前に1度だけロードしてRAMとNPU上に固定
        # ファイルサイズの大きいものから読み込み、RAMをうまく解放する
        self.sess_common = ort.InferenceSession(common_path, sess_options=qnn.session_options)
        self.sess_down = ort.InferenceSession(down_path, sess_options=qnn.session_options)
        self.sess_up2  = ort.InferenceSession(up_2_path, sess_options=qnn.session_options)
        self.sess_mid  = ort.InferenceSession(mid_path, sess_options=qnn.session_options)
        self.sess_up1  = ort.InferenceSession(up_1_path, sess_options=qnn.session_options)
        
        # 各セッションの出力を動的に自動取得
        self.out_names_common = [o.name for o in self.sess_common.get_outputs()]
        self.out_names_down = [o.name for o in self.sess_down.get_outputs()]
        self.out_names_mid  = [o.name for o in self.sess_mid.get_outputs()]
        self.out_names_up1  = [o.name for o in self.sess_up1.get_outputs()]
        self.out_names_up2  = [o.name for o in self.sess_up2.get_outputs()]

        print("  -> UNetすべてのロードが完了しました。")

    def find_model_path(self, unet_dir, part, width, height):
        model_dir = os.path.join(unet_dir, part)
        single_graph_model = os.path.join(model_dir, "model.onnx")
        # part0 はwidth、heightに依存しないので、model.onnx１つだけ
        if os.path.exists(single_graph_model):
            return single_graph_model
        return os.path.join(model_dir, f"unet_{part}_{width}x{height}.onnx")

    def inference(self, config, prompt_embeds, pooled_prompt_embeds,
                  uncond_embeds, uncond_pooled_embeds):
        # --------------------------------------------------
        # Step 1: スケジューラーの初期化
        # --------------------------------------------------
        # ローカルの設定ファイルを読み込んでスケジューラーを準備
        #scheduler = DPMSolverSinglestepScheduler.from_pretrained(
        #scheduler = DPMSolverMultistepScheduler.from_pretrained(
        #    config.dirs.scheduler_dir)
        #scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        #    config.dirs.scheduler_dir)
        scheduler = Scheduler.get(config.scheduler_type)
        scheduler.set_timesteps(config.steps)

        # --------------------------------------------------
        # Step 3: UNet デノイズループの準備
        # --------------------------------------------------
        # SDXLのデフォルトの条件付けメタデータ（time_ids: 1024x1024解像度等の固定値情報）
        # UNetに合わせてfloat32のNumPy配列として用意
        add_time_ids = np.array([[config.height, config.width,
                                  0, 0,
                                  config.height, config.width]], dtype=np.float32)

        # encoder_hidden_states = prompt_embeds
        # uncond_hidden_states = uncond_embeds
        base_inputs = {
            "text_embeds": pooled_prompt_embeds.astype(np.float32),
            "time_ids": add_time_ids
        }

        if config.guidance_scale != 1:
            uncond_base_inputs = {
                "text_embeds": uncond_pooled_embeds.astype(np.float32),
                "time_ids": add_time_ids
            }
    
        # 初期ノイズ (Latents) の生成
        # ※ 再現性を持たせるため、必要に応じて torch.manual_seed を設定してください
        if config.seed == -1:
            config.seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
        generator = torch.manual_seed(config.seed)
        print(f"seed: {config.seed}")
        
        latents = torch.randn(1, 4, config.height // 8, config.width // 8,
                              dtype=torch.float32, generator=generator)
        latents = latents * scheduler.init_noise_sigma
    
        # --------------------------------------------------
        # Step 4: デノイズ ループ実行
        # --------------------------------------------------
        print("\n--- デノイズループ開始 ---")
        # NumPy配列に変換してループに投入
        latents_np = latents.numpy()

        encoder_hidden_states = prompt_embeds.astype(np.float16)
        if config.guidance_scale != 1:
            uncond_hidden_states = uncond_embeds.astype(np.float16)
        latents_torch = torch.from_numpy(latents_np)
    
        # for i, t in enumerate(scheduler.timesteps):
        # for t in tqdm(scheduler.timesteps):
        for i, t in enumerate(tqdm(scheduler.timesteps)):
            # 現在の進捗を表示
            # print(f"🔄 Step {i+1}/{steps} (Timestep: {t.item():.1f})")
        
            scaled_latents_torch = scheduler.scale_model_input(latents_torch, t)
            scaled_latents_np = scaled_latents_torch.numpy().astype(np.float16)

            # タイムステップをUNetが要求する形状 [1] のfloat32配列にする
            timestep_np = np.array([t.item()], dtype=np.float32)
        
            # 常駐しているUNetのforwardを実行
            # 前回の修正（内部でのNHWC変換、float32統一）が施されたUNetが動きます
            if config.guidance_scale != 1:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(self.forward, scaled_latents_np, timestep_np,
                                        base_inputs,
                                        encoder_hidden_states,
                                        save_calibration_data=True,
                                        step=i, is_uncond=False),
                        executor.submit(self.forward, scaled_latents_np, timestep_np,
                                        uncond_base_inputs,
                                        uncond_hidden_states,
                                        save_calibration_data=True,
                                        step=i, is_uncond=True),
                    ]
                    done, not_done = wait(futures, return_when=ALL_COMPLETED)
                    for f in done:
                        if f.result()[1]:
                            noise_pred_uncond = f.result()[0]
                        else:
                            noise_pred_text = f.result()[0]
                
                noise_pred = noise_pred_uncond + config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.forward(scaled_latents_np, timestep_np,
                                          base_inputs, encoder_hidden_states).astype(np.float32)
            
            # スケジューラーを使ってノイズを除去し、次のlatentsを計算
            # ※ diffusersのschedulerはtorch.Tensorを期待するため、一時的に戻して処理
            latents_torch = scheduler.step(
                torch.from_numpy(noise_pred), 
                t, 
                torch.from_numpy(latents_np)
            ).prev_sample
        
            # 次のステップのためにNumPyに変換
            latents_np = latents_torch.numpy()
            """
            out_latents_np = latents_np / 0.13025
                
            out_image_tensor = vae_decoder.decode_latents(out_latents_np, auto_mem_free=False)
            output_image(out_image_tensor, f"{t.item():.1f}.png")
            """
        
        print("--- デノイズループ完了 ---")
        del base_inputs
        if config.guidance_scale != 1:
            del uncond_base_inputs

        return latents_np
    
    def forward(self, latents, timestep, base_inputs, encoder_hidden_states, save_calibration_data=False, step = 0, is_uncond = False):
        """1ステップ分のノイズ予測を4つのモデルを繋げて実行"""
        # 0. Part 0: Common
        feed_common = {"timestep": timestep, **base_inputs}
        if self.collector and save_calibration_data:
            self.collector.save_step("part0", feed_common, step)
        out_list_common = self.sess_common.run(self.out_names_common, feed_common)
        common_inputs = out_list_common[0].astype(np.float16)
        
        # 1. Part 1: Down
        feed_down = {"silu_3": common_inputs, "sample": latents, "encoder_hidden_states": encoder_hidden_states}
        if self.collector and save_calibration_data:
            self.collector.save_step("part1", feed_down, step)
        out_list_down = self.sess_down.run(self.out_names_down, feed_down)
        skip_connections = dict(zip(self.out_names_down, out_list_down))
        skip_connections['add_88'] = skip_connections['add_88'].transpose(0, 3, 1, 2)
        
        # 2. Part 2: Mid
        feed_mid = {"add_88": skip_connections["add_88"], "silu_3": common_inputs, "encoder_hidden_states": encoder_hidden_states}
        if self.collector and save_calibration_data:
            self.collector.save_step("part2", feed_mid, step)
        out_list_mid = self.sess_mid.run(self.out_names_mid, feed_mid)
        add_123 = out_list_mid[0] # add_123
        
        # 3. Part 3: Up 前半
        feed_up1 = {"add_88": skip_connections["add_88"], "add_123": add_123, "silu_3": common_inputs, "encoder_hidden_states": encoder_hidden_states}
        if self.collector and save_calibration_data:
            self.collector.save_step("part3", feed_up1, step)
        out_list_up1 = self.sess_up1.run(self.out_names_up1, feed_up1)
        add_156 = out_list_up1[0].transpose(0, 3, 1, 2) # add_7721
        
        # 4. Part 4: Up 後半 + 最終Output
        part4_inputs = skip_connections.copy()
        del part4_inputs["add_88"]
        feed_up2 = {"add_156": add_156, **part4_inputs, "silu_3": common_inputs, "encoder_hidden_states": encoder_hidden_states}
        if self.collector and save_calibration_data:
            self.collector.save_step("part4", feed_up2, step)
        out_list_up2 = self.sess_up2.run(self.out_names_up2, feed_up2)
        noise_pred = out_list_up2[0] # out_sample
        
        return noise_pred, is_uncond

if __name__ == "__main__":
    import torch
    # 単体テスト用
    unet = NpuUNetLoop()
    latents = torch.randn(1, 4, 128, 128, dtype=torch.float16).numpy()
    timestep = torch.randn(1, dtype=torch.float32).numpy()
    encoder_hidden_states = np.zeros((1, 77, 2048), dtype=np.float16)
    base_inputs = {
        "text_embeds": np.zeros((1, 1280), dtype=np.float32), # 実測通りのfloat32
        "time_ids": np.zeros((1, 6), dtype=np.float32)
    }
    res = unet.forward(latents, timestep, base_inputs, encoder_hidden_states)
    print(f"Unet 出力形状: {res.shape}") # 想定: (1, 77, 768)
