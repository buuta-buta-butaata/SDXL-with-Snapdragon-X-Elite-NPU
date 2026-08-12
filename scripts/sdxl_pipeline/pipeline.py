import logging

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from .text_processing import TextProcessing
from .unet import UNet
from .vae_decoder import VAEDecoder
from . import image

from profilers import simple_profiler as prof

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError()

class SDXLPipeline(BasePipeline):
    def __init__(self, config):
        logger.debug(vars(config))
        self.config = config
        self.text_processing = None
        self.unet = None
        self.vae_decoder = None

    def run(self):
        logger.info("Running SDXL pipeline on NPU...")
        print(f"Prompt: {self.config.prompt}")
        # print(f"prompt_2: {self.config.prompt_2}")
        # print(f"negative prompt: {self.config.negative_prompt}")
        # print(f"negative prompt_2: {self.config.negative_prompt_2}")
        profiler = prof.register("pipeline")

        if self.text_processing is None:
            self.text_processing = TextProcessing(self.config.dirs["text_encoder_dir"],
                                                  self.config.dirs["tokenizer_dir"],
                                                  self.config.dirs["text_encoder_2_dir"],
                                                  self.config.dirs["tokenizer_2_dir"],
                                                  self.config.use_torch)

        (prompt_embeds, pooled_prompt_embeds,
         uncond_embeds, uncond_pooled_embeds) = self.text_processing.encode_text(self.config, False)
        
        del self.text_processing

        self.print_current_used_memory()

        if self.unet is None:
            logger.info("*" * 43)
            logger.info("Loading UNet models...")
            self.unet = UNet(self.config)
            self.unet.load_models(self.config)
            logger.info("  -> Loaded")
            logger.info("*" * 43)

        self.print_current_used_memory()
        
        profiler.start_profile("unet")
        with ThreadPoolExecutor(max_workers=2) as executor:
            latents = self.unet.inference(self.config, prompt_embeds, pooled_prompt_embeds,
                                          uncond_embeds, uncond_pooled_embeds, executor)
        
        profiler.stop_profile("unet")
        del self.unet
        del prompt_embeds, pooled_prompt_embeds
        if self.config.cfg != 1:
            del uncond_embeds, uncond_pooled_embeds

        if self.config.debug_mode:
            image.preview_image(latents, self.config.dirs["output_dir"], self.config.output_prefix)

        # --------------------------------------------------
        # VAE decoder
        # --------------------------------------------------
        latents = latents / 0.13025
            
        profiler.start_profile("vae_decoder")
            
        if self.vae_decoder is None:
            self.vae_decoder = VAEDecoder(self.config)
        image_tensor = self.vae_decoder.decode(latents, auto_mem_free=False)
    
        profiler.stop_profile("vae_decoder")

        profiler.start_profile("pil")
        image.output_image(image_tensor, **vars(self.config))
        profiler.stop_profile("pil")

        profiler.destroy_profile_event()

        if self.config.profile:
            self.print_summary()

        if self.config.debug_mode:
            from utils import check_torch_imported
            check_torch_imported()

    def print_current_used_memory(self, output_func=logger.info):
        total, used, available, percent = prof.get_current_memory_info()
        width = 43
        col_1 = 23
        col_2 = width - col_1 - 10
        output_func("=" * width)
        output_func(" Current RAM Info")
        output_func("=" * width)
        output_func(" System RAM Info")
        output_func("-" * width)
        output_func("| {:^{}} | {:>{}} GB |".format("Total RAM", col_1, total, col_2))
        output_func("| {:^{}} | {:>{}} GB |".format("Used RAM", col_1, used, col_2))
        output_func("| {:^{}} | {:>{}} GB |".format("Available RAM", col_1, available, col_2))
        output_func("| {:^{}} | {:>{}}  % |".format("RAM Usage", col_1, percent, col_2))
        output_func("-" * width)
        
        peak_mem = prof.get_peak_memory_gb()
        output_func(" Process RAM Info")
        output_func("-" * width)
        output_func("| {:^{}} | {:>{}.2f} GB |".format("Peak RAM", col_1, peak_mem, col_2))
        output_func("-" * width)

    def print_latency(self, prof_name, width, col_1, col_2, sort=False, output_func=logger.info):
        profiler = prof.get(prof_name)
        output_func(f" {prof_name} Latency Info")
        output_func("-" * width)
        output_func("| {:^{}} | {:^{}} |".format("Module", col_1, "Latency", col_2))
        output_func("| {:^{}} | {:^{}} |".format("-" * col_1, col_1, "-" * col_2, col_2))
        if sort:
            events = profiler.events
            events.sort()
        else:
            events = profiler.events
        for event in events:
            if event == "all":
                continue
            output_func("| {:^{}} | {:>{}.3f} sec |".format(event, col_1, profiler.event_elapsed_time(event),
                                                             col_2 - 4))
        output_func("| {:^{}} | {:^{}} |".format("-" * col_1, col_1, "-" * col_2, col_2))
        output_func("| {:^{}} | {:>{}.3f} sec |".format("all", col_1, profiler.event_elapsed_time("all"), col_2 - 4))
        output_func("-" * width)
        
    def print_summary(self, output_func=logger.info):
        width = 43
        col_1 = 23
        col_2 = width - col_1 - 7
        output_func("=" * width)
        output_func(" 🛠️  PROFILE REPORT SUMMARY")
        output_func("=" * width)

        self.print_latency("pipeline", width, col_1, col_2, output_func)
        if self.config.debug_mode:
            self.print_latency("unet", width, col_1, col_2, True, output_func=logger.debug)

        peak_mem = prof.get_peak_memory_gb()
        output_func(" Process RAM Info")
        output_func("-" * width)
        output_func("| {:^{}} | {:>{}.2f} GB |".format("Peak RAM", col_1, peak_mem, col_2 - 3))
        output_func("-" * width)

    
if __name__ == "__main__":
    from types import SimpleNamespace
    conf = SimpleNamespace(prompt="A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed")
    main = SDXLPipeline(conf)
    main.run()

