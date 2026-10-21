import gc
import logging
import os

import numpy as np
import onnxruntime as ort

from concurrent.futures import wait, ALL_COMPLETED
from tqdm.auto import tqdm

from utils import qnn_ep_helper as qnn
from profilers import simple_profiler as prof

from .image_encoder import ImageEncoder
from .ip_adapter import IPAdapter
from .t2i_adapter import T2IAdapter

from . import ip_adapter_helper as ip_helper
from . import t2i_adapter_helper as t2i_helper

logger = logging.getLogger(__name__)
        

class UNet:
    def __init__(self, config):
        if config.quantized_model:
            self.part0_type = np.float16
        else:
            self.part0_type = np.float32

        self.sessions = []
        self.out_names = []

        if config.use_legacy_model:
            self.use_legacy_model()
        else:
            self.part1_main_output = "add_123"
            self.part2_main_output = "add_168"
            self.part3_main_output = "add_212"

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

    def run_t2i_adapter(self, config):
        return t2i_helper.get_uncond_data(), t2i_helper.get_uncond_data()

    def run_ip_adapter(self, config):
        return ip_helper.get_uncond_data(), ip_helper.get_uncond_data()

    def run_adapters(self, config):
        ip_outputs, uncond_ip_outputs = self.run_ip_adapter(config)
        t2i_outputs, uncond_t2i_outputs = self.run_t2i_adapter(config)

        self.adapter_outputs = ip_outputs
        self.uncond_adapter_outputs = uncond_ip_outputs

        for x in t2i_outputs.keys():
            self.adapter_outputs[x].update(t2i_outputs[x])
            self.uncond_adapter_outputs[x].update(uncond_t2i_outputs[x])

    def use_legacy_model(self):
        self.part1_main_output = "add_88"
        self.part2_main_output = "add_123"
        self.part3_main_output = "add_156"
        self.run_t2i_adapter = self.run_t2i_adapter_for_legacy_model
        self.run_ip_adapter = self.run_ip_adapter_for_legacy_model
        self.run_adapters = self.run_adapters_for_legacy_model

    # def validate_model_version(self, config):
    #     for part_name, output_name in [("part1", self.part1_main_output), ("part2", self.part2_main_output), ("part3", self.part3_main_output)]:
    #         path = self.find_model_path(config.dirs["unet_dir"], part_name, config.width, config.height)

    #         with open(path, "r", encoding="utf-8", errors="replace") as f:
    #             data = f.read()
    #             if not output_name in data:
    #                 logger.warning("Running on an older model. Please run 'sdxlite-cli setup' to update.")
    #                 self.part1_main_output = "add_88"
    #                 self.part2_main_output = "add_123"
    #                 self.part3_main_output = "add_156"
    #                 self.run_t2i_adapter = self.run_t2i_adapter_for_legacy_model
    #                 self.run_ip_adapter = self.run_ip_adapter_for_legacy_model
    #                 self.run_adapters = self.run_adapters_for_legacy_model
    #                 break

    def run_t2i_adapter_for_legacy_model(self, config):
        return {}, {}

    def run_ip_adapter_for_legacy_model(self, config):
        return {}, {}

    def run_adapters_for_legacy_model(self, config):
        self.adapter_outputs = {
            "down": {}, "mid": {}, "up1": {}, "up2": {}
        }
        self.uncond_adapter_outputs = {
            "down": {}, "mid": {}, "up1": {}, "up2": {}
        }

    def inference(self, config, init_latents, latents, mask_latents, scheduler, timesteps,
                  prompt_embeds, pooled_prompt_embeds,
                  uncond_embeds, uncond_pooled_embeds, executor,
                  original_size=None,
                  crops_coords=(0, 0),
                  target_size=None):

        prof.register("unet")

        if original_size is None:
            original_size = (config.height, config.width)
        if target_size is None:
            target_size = (config.height, config.width)

        # self.validate_model_version(config)
        self.run_adapters(config)

        # --------------------------------------------------
        # Denoise
        # --------------------------------------------------
        logger.info("Denoising via UNet loop...")
        encoder_hidden_states = prompt_embeds.astype(np.float16)
        if config.cfg != 1:
            uncond_hidden_states = uncond_embeds.astype(np.float16)
    
        if len(self.sessions) == 0:
            self.load_models(config)

        add_time_ids = self.set_add_time_ids(original_size, crops_coords, target_size, self.part0_type)

        base_inputs = {
            "text_embeds": pooled_prompt_embeds.astype(self.part0_type),
            "time_ids": add_time_ids
        }

        if config.cfg != 1:
            uncond_base_inputs = {
                "text_embeds": uncond_pooled_embeds.astype(self.part0_type),
                "time_ids": add_time_ids
            }

        self.preprocess(config)

        for i, t in enumerate(tqdm(timesteps)):
            # タイムステップをUNetが要求する形状 [1] のfloat32配列にする
            timestep = np.array([t.item()], dtype=np.float32)
            scaled_latents = scheduler.scale_model_input(latents, timestep).astype(np.float16)

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
                try:
                    for f in done:
                        if f.result()[1]:
                            noise_pred_uncond = f.result()[0].astype(np.float32)
                        else:
                            noise_pred_text = f.result()[0].astype(np.float32)
                except ort.capi.onnxruntime_pybind11_state.InvalidArgument as e:
                    if "INVALID_ARGUMENT : Invalid input name: transpose_69" in e.__str__():
                        error_message = ("Running on an older model. Please run 'sdxlite-cli setup' to update"
                                         " or run with '--use_legacy_model'."
                                         )
                        logger.error(error_message)
                        raise Exception(error_message)
                    raise
                        
                noise_pred = noise_pred_uncond + config.cfg * (
                    noise_pred_text - noise_pred_uncond
                ).astype(np.float32)
            else:
                noise_pred = self.forward(scaled_latents, timestep,
                                          base_inputs, encoder_hidden_states)[0].astype(np.float32)
        
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample
            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                latents = self.add_noise(init_latents, latents, mask_latents, scheduler, noise_timestep, config)

        prof.get("unet").destroy_profile_event()

        return latents

    def preprocess(self, config):
        pass

    def set_add_time_ids(self, original_size, crops_coords, target_size, dtype):
        add_time_ids = np.array([list(original_size + crops_coords + target_size)], dtype=dtype)
        return add_time_ids

    def add_noise(self, init_latents, latents, mask_latents, scheduler, timestep, config):
        return latents

    def sanitize_outputs(self, outputs, key, shape_3_size):
        if outputs[key].shape[3] == shape_3_size:
            outputs[key] = outputs[key].transpose(0, 3, 1, 2)
        if f"{key}_permute" in outputs:
            del outputs[f"{key}_permute"]

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
        self.sanitize_outputs(skip_connections, self.part1_main_output, 1280)

        return skip_connections

    def run_part2(self, feed_mid, is_uncond):
        out_list_mid = self.sessions[2].run(self.out_names[2], feed_mid)
        return out_list_mid[0]
         
    def run_part3(self, feed_up1, is_uncond):
        out_list_up1 = self.sessions[3].run(self.out_names[3], feed_up1)
        outputs = dict(zip(self.out_names[3], out_list_up1))
        self.sanitize_outputs(outputs, self.part3_main_output, 1280)
        
        return outputs[self.part3_main_output]
    
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
        adapter_down_input = self.adapter_outputs["down"] if not is_uncond else self.uncond_adapter_outputs["down"]
        feed_down = {"silu_3": common_inputs, "sample": latents, "encoder_hidden_states": encoder_hidden_states,
                     **adapter_down_input}
        skip_connections = self.run_part1(feed_down, is_uncond)
        profiler.stop_profile(profile_name)

        profile_name = self._set_profile_name(2, step, is_uncond)
        profiler.start_profile(profile_name)
        adapter_mid_input = self.adapter_outputs["mid"] if not is_uncond else self.uncond_adapter_outputs["mid"]
        feed_mid = {self.part1_main_output: skip_connections[self.part1_main_output], "silu_3": common_inputs,
                    "encoder_hidden_states": encoder_hidden_states, **adapter_mid_input}
        part2_outputs = self.run_part2(feed_mid, is_uncond)
        profiler.stop_profile(profile_name)
      
        profile_name = self._set_profile_name(3, step, is_uncond)
        profiler.start_profile(profile_name)
        adapter_up1_input = self.adapter_outputs["up1"] if not is_uncond else self.uncond_adapter_outputs["up1"]
        feed_up1 = {self.part1_main_output: skip_connections[self.part1_main_output],
                    self.part2_main_output: part2_outputs, "silu_3": common_inputs,
                    "encoder_hidden_states": encoder_hidden_states, **adapter_up1_input}
        part3_outputs = self.run_part3(feed_up1, is_uncond)
        profiler.stop_profile(profile_name)

        profile_name = self._set_profile_name(4, step, is_uncond)
        profiler.start_profile(profile_name)
        part4_inputs = skip_connections.copy()
        del part4_inputs[self.part1_main_output]
        adapter_up2_input = self.adapter_outputs["up2"] if not is_uncond else self.uncond_adapter_outputs["up2"]
        feed_up2 = {self.part3_main_output: part3_outputs, **part4_inputs, "silu_3": common_inputs,
                    "encoder_hidden_states": encoder_hidden_states, **adapter_up2_input}
        noise_pred = self.run_part4(feed_up2, is_uncond)
        profiler.stop_profile(profile_name)

        return noise_pred, is_uncond

    def free_memory(self):
        del self.sessions
        gc.collect()

if __name__ == "__main__":
    from types import SimpleNamespace
    config = SimpleNamespace(dirs = {})
    config.dirs["image_encoder_dir"] = r"compiled_models\single\fp16\image_encoder"
    config.dirs["ip_adapter_dir"] = r"compiled_models\single\dsxl\1024x1024_fp16\ip_adapter"
    config.cfg = 2.0
    config.width = 1024
    config.height = 1024
    config.quantized_model = False
    
    import torch
    # 単体テスト用
    unet = UNet(config)
    latents = torch.randn(1, 4, 128, 128, dtype=torch.float16).numpy()
    timestep = torch.randn(1, dtype=torch.float32).numpy()
    encoder_hidden_states = np.zeros((1, 77, 2048), dtype=np.float16)
    base_inputs = {
        "text_embeds": np.zeros((1, 1280), dtype=np.float32), # 実測通りのfloat32
        "time_ids": np.zeros((1, 6), dtype=np.float32)
    }
    res = unet.forward(latents, timestep, base_inputs, encoder_hidden_states)
    logger.info(f"Unet 出力形状: {res.shape}") # 想定: (1, 77, 768)
