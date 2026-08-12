import logging
import os

from profilers import simple_profiler as prof
from sdxl_pipeline import BasePipeline
from sdxl_pipeline.text_processing import TextProcessing
from . import numpy_io as npio

logger = logging.getLogger(__name__)


class SDXLPipelineTextOnly(BasePipeline):
    def __init__(self, sdxl_config):
        self.config = sdxl_config
        prof.available = False

    def _run(self, text_processing, prompt, output_dir):
        print(f"Encoding: {prompt}")
        self.config.prompt = prompt
        self.config.prompt_2 = prompt
        (prompt_embeds, pooled_prompt_embeds,
         uncond_embeds, uncond_pooled_embeds) = text_processing.encode_text(self.config, auto_mem_free=False)

        outputs = {"prompt":prompt,
                   "negative_prompt": self.config.negative_prompt,
                   "pos_embeds":prompt_embeds,
                   "pos_pooled":pooled_prompt_embeds,
                   "neg_embeds":uncond_embeds,
                   "neg_pooled":uncond_pooled_embeds,
                   }
                
        npio.save(output_dir, "text", outputs)
        print("  -> Done!")
        

    def run(self):
        text_processing = TextProcessing(self.config.dirs["text_encoder_dir"],
                                         self.config.dirs["tokenizer_dir"],
                                         self.config.dirs["text_encoder_2_dir"],
                                         self.config.dirs["tokenizer_2_dir"])

        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isfile(self.config.input_file):
            self._run(text_processing, self.config.prompt, output_dir)
            return
        
        with open(self.config.input_file) as f:
            line = f.readline()
            while line:
                prompt = line.rstrip("\n")
                self._run(text_processing, prompt, output_dir)
                line = f.readline()
                # time.sleep(1)

    
if __name__ == "__main__":
    from types import SimpleNamespace
    conf = SimpleNamespace(prompt = "A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed")
    main = SDXLPipelineTextOnly(conf)
    main.run();

