import logging
import os

import numpy as np
import onnxruntime as ort

from concurrent.futures import wait, ALL_COMPLETED
from tqdm.auto import tqdm

from utils import qnn_ep_helper as qnn
from profilers import simple_profiler as prof

logger = logging.getLogger(__name__)

class UNet:
    def __init__(self, config):
        if config.quantized_model:
            self.part0_type = np.float16
        else:
            self.part0_type = np.float32

        self.sessions = []
        self.out_names = []

    def load_model(self, config, part_num):
        part_name = f"part{part_num}"
        if part_name in config.unet_fp16:
            path = self.find_model_path(config.dirs["unet_dir_fp16"], part_name, config.width, config.height)
            if part_num == 0:
                self.part0_type = np.float32
        else:
            path = self.find_model_path(config.dirs["unet_dir"], part_name, config.width, config.height)
        
        logger.debug(f"Loading unet part{part_num} from: {path}")
        self.sessions.append(ort.InferenceSession(path, sess_options=qnn.session_options))
        self.out_names.append([o.name for o in self.sessions[part_num].get_outputs()])

    def load_models(self, config):
        for i in range(0, 5):
            self.load_model(config, i)

    def find_model_path(self, unet_dir, part, width, height):
        model_dir = os.path.join(unet_dir, part)
        single_graph_model = os.path.join(model_dir, "model.onnx")
        # part0 はwidth、heightに依存しないので、model.onnx１つだけ
        if os.path.exists(single_graph_model):
            return single_graph_model
        return os.path.join(model_dir, f"unet_{part}_{width}x{height}.onnx")

    def inference(self, config, prompt_embeds, pooled_prompt_embeds,
                  uncond_embeds, uncond_pooled_embeds, executor):

        prof.register("unet")

        if len(self.sessions) == 0:
            self.load_models(config)

        add_time_ids = np.array([[config.height, config.width,
                                  0, 0,
                                  config.height, config.width]], dtype=self.part0_type)

        base_inputs = {
            "text_embeds": pooled_prompt_embeds.astype(self.part0_type),
            "time_ids": add_time_ids
        }

        if config.cfg != 1:
            uncond_base_inputs = {
                "text_embeds": uncond_pooled_embeds.astype(self.part0_type),
                "time_ids": add_time_ids
            }
    
        if config.seed == -1:
            config.seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)
        logger.debug(f"seed: {config.seed}")
        
        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------
        if config.use_torch:
            from .scheduler_torch import Scheduler
        else:
            from .scheduler_numpy import Scheduler
        logger.debug(f"User input scheduler_config: {config.scheduler_config}")
        scheduler = Scheduler(config.scheduler_type, config.seed, **config.scheduler_config)
        scheduler.set_timesteps(config.steps)
        # 画像のmetadata用に保持しておく
        config.scheduler_config = vars(scheduler.config)
        logger.debug(f"Generated scheduler_config: {config.scheduler_config}")

        latents = scheduler.generate_noise_latents(config)
        latents = latents * scheduler.init_noise_sigma
        
        # --------------------------------------------------
        # Denoise
        # --------------------------------------------------
        logger.info("Denoising via UNet loop...")
        encoder_hidden_states = prompt_embeds.astype(np.float16)
        if config.cfg != 1:
            uncond_hidden_states = uncond_embeds.astype(np.float16)
    
        for i, t in enumerate(tqdm(scheduler.timesteps)):
            scaled_latents = scheduler.scale_model_input(latents, t).astype(np.float16)

            # タイムステップをUNetが要求する形状 [1] のfloat32配列にする
            timestep = np.array([t.item()], dtype=np.float32)
    
            # 常駐しているUNetのforwardを実行
            if config.cfg != 1:
                futures = [
                    executor.submit(self.forward, scaled_latents, timestep.astype(self.part0_type),
                                    base_inputs,
                                    encoder_hidden_states,
                                    step=i, is_uncond=False),
                    executor.submit(self.forward, scaled_latents, timestep.astype(self.part0_type),
                                    uncond_base_inputs,
                                    uncond_hidden_states,
                                    step=i, is_uncond=True),
                ]
                done, not_done = wait(futures, return_when=ALL_COMPLETED)
                for f in done:
                    if f.result()[1]:
                        noise_pred_uncond = f.result()[0].astype(np.float32)
                    else:
                        noise_pred_text = f.result()[0].astype(np.float32)
                        
                noise_pred = noise_pred_uncond + config.cfg * (
                    noise_pred_text - noise_pred_uncond
                ).astype(np.float32)
            else:
                noise_pred = self.forward(scaled_latents, timestep,
                                          base_inputs, encoder_hidden_states)[0].astype(np.float32)
        
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            prof.get("unet").destroy_profile_event()

        return latents

    def run_part0(self, feed_common, is_uncond):
        out_list_common = self.sessions[0].run(self.out_names[0], feed_common)
        return out_list_common[0].astype(np.float16)

    def run_part1(self, feed_down, is_uncond):
        out_list_down = self.sessions[1].run(self.out_names[1], feed_down)
        skip_connections = dict(zip(self.out_names[1], out_list_down))

        # NOTE: Prior to the June 2026 Qualcomm AI Hub update, the QNN compiler would 
        # inadvertently alter some model outputs from NCHW to NHWC format, requiring 
        # a manual CPU transpose overhead. While this issue has been resolved in newer 
        # toolchains, the code below is designed to dynamically handle both formats 
        # to ensure robust backward compatibility.
        if skip_connections['add_88'].shape[3] == 1280:
            skip_connections['add_88'] = skip_connections['add_88'].transpose(0, 3, 1, 2)

        return skip_connections

    def run_part2(self, feed_mid, is_uncond):
        out_list_mid = self.sessions[2].run(self.out_names[2], feed_mid)
        return out_list_mid[0] # add_123
         
    def run_part3(self, feed_up1, is_uncond):
        out_list_up1 = self.sessions[3].run(self.out_names[3], feed_up1)
        if out_list_up1[0].shape[3] == 1280:
            add_156 = out_list_up1[0].transpose(0, 3, 1, 2)
        else:
            add_156 = out_list_up1[0]
        return add_156
    
    def run_part4(self, feed_up2, is_uncond):
        return self.sessions[4].run(self.out_names[4], feed_up2)[0]

    def _set_profile_name(self, part_num, step, is_uncond):
        uncond_text = "_uncond" if is_uncond else ""
        return f"Part{part_num}{uncond_text} Step: {step+1}"

    def forward(self, latents, timestep, base_inputs, encoder_hidden_states, step = 0, is_uncond = False):
        """1ステップ分のノイズ予測を5つのモデルを繋げて実行"""
        profiler = prof.get("unet")
        
        profile_name = self._set_profile_name(0, step, is_uncond)
        profiler.start_profile(profile_name)
        feed_common = {"timestep": timestep, **base_inputs}
        common_inputs = self.run_part0(feed_common, is_uncond)
        profiler.stop_profile(profile_name)
        
        profile_name = self._set_profile_name(1, step, is_uncond)
        profiler.start_profile(profile_name)
        feed_down = {"silu_3": common_inputs, "sample": latents, "encoder_hidden_states": encoder_hidden_states}
        skip_connections = self.run_part1(feed_down, is_uncond)
        profiler.stop_profile(profile_name)

        profile_name = self._set_profile_name(2, step, is_uncond)
        profiler.start_profile(profile_name)
        feed_mid = {"add_88": skip_connections["add_88"], "silu_3": common_inputs,
                    "encoder_hidden_states": encoder_hidden_states}
        add_123 = self.run_part2(feed_mid, is_uncond)
        profiler.stop_profile(profile_name)
      
        profile_name = self._set_profile_name(3, step, is_uncond)
        profiler.start_profile(profile_name)
        feed_up1 = {"add_88": skip_connections["add_88"], "add_123": add_123, "silu_3": common_inputs,
                    "encoder_hidden_states": encoder_hidden_states}
        add_156 = self.run_part3(feed_up1, is_uncond)
        profiler.stop_profile(profile_name)

        profile_name = self._set_profile_name(4, step, is_uncond)
        profiler.start_profile(profile_name)
        part4_inputs = skip_connections.copy()
        del part4_inputs["add_88"]
        feed_up2 = {"add_156": add_156, **part4_inputs, "silu_3": common_inputs,
                    "encoder_hidden_states": encoder_hidden_states}
        noise_pred = self.run_part4(feed_up2, is_uncond)
        profiler.stop_profile(profile_name)

        return noise_pred, is_uncond

if __name__ == "__main__":
    import torch
    # 単体テスト用
    unet = UNet()
    latents = torch.randn(1, 4, 128, 128, dtype=torch.float16).numpy()
    timestep = torch.randn(1, dtype=torch.float32).numpy()
    encoder_hidden_states = np.zeros((1, 77, 2048), dtype=np.float16)
    base_inputs = {
        "text_embeds": np.zeros((1, 1280), dtype=np.float32), # 実測通りのfloat32
        "time_ids": np.zeros((1, 6), dtype=np.float32)
    }
    res = unet.forward(latents, timestep, base_inputs, encoder_hidden_states)
    logger.info(f"Unet 出力形状: {res.shape}") # 想定: (1, 77, 768)
