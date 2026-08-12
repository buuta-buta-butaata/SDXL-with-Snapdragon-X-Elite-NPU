import glob
import logging
import json
import os

from argparse import ArgumentParser, Namespace

from . import BaseCLICommand, MyHelpFormatter, int_range, format_dict_text
from sdxl_pipeline import SDXLPipeline, Scheduler
from conf import MODEL_ROOT_DIR

logger = logging.getLogger(__name__)

def txt2img_command_factory(args: Namespace):
    return Txt2ImgCommand(args)

class Txt2ImgCommand(BaseCLICommand):

    available_schedulers = Scheduler.get_available()
    available_models = dict([(i, os.path.basename(d))
                             for i, d in enumerate(glob.glob(os.path.join(MODEL_ROOT_DIR, "*"))) if os.path.isdir(d)])

    @staticmethod
    def register_subcommand(parser: ArgumentParser) -> None:

        txt2img_parser = parser.add_parser(
            "txt2img",
            formatter_class=MyHelpFormatter,
            help="Generate images from text prompts.",
            usage="\n sdxlite-cli txt2img [options]"
        )

        logger.debug(Txt2ImgCommand.available_schedulers)
        logger.debug(f"available_schedulers: {Txt2ImgCommand.available_schedulers}")

        logger.debug(Txt2ImgCommand.available_models)
        if len(Txt2ImgCommand.available_models) == 0:
            logger.critical("No available models found.\nPlease run 'sdxlite-cli setup' to download them.")
            exit()

        schedulers_text = format_dict_text(Txt2ImgCommand.available_schedulers)
        models_text = format_dict_text(Txt2ImgCommand.available_models)
        submodules = ["text_encoder", "unet", "vae_decoder", "generate_for_unet_cos_sim", "compute_unet_cos_sim",
                      "collect_calib_data", ""]

        # Prompt options
        prompt_group = txt2img_parser.add_argument_group("Prompt options")
        prompt_group.add_argument(
            "--prompt",
            type=str,
            metavar="STRING",
            help="Main text prompt for generation. Required unless --input_file is specified."
        )
        prompt_group.add_argument(
            "--prompt_2",
            type=str,
            default=None,
            metavar="STRING",
            help="Secondary text prompt for SDXL. Automatically copies --prompt if omitted."
        )
        prompt_group.add_argument(
            "--negative_prompt",
            type=str,
            default="",
            metavar="STRING",
            help="Negative prompt to guide generation away from unwanted elements."
        )
        prompt_group.add_argument(
            "--negative_prompt_2",
            type=str,
            default=None,
            metavar="STRING",
            help="Secondary negative prompt for SDXL. Automatically copies --negative_prompt if omitted."
        )

        # SDXL Options
        generation_group = txt2img_parser.add_argument_group("Generation options")
        generation_group.add_argument(
            "--steps",
            type=int_range(1, 50),
            default=6,
            metavar="INT(choose from 1-50)",
            help="Number of denoising steps for image generation."
        )
        generation_group.add_argument(
            "--seed",
            type=int,
            default=-1,
            metavar="INT",
            help="Random seed for generation. Set to -1 for a random seed."
        )
        generation_group.add_argument(
            "--cfg",
            "--guidance_scale",
            type=float,
            default=2,
            metavar="FLOAT",
            help="Classifier-Free Guidance (CFG) scale to control prompt adherence."
        )
        generation_group.add_argument(
            "--scheduler_type", "--scheduler",
            type=int,
            choices=Txt2ImgCommand.available_schedulers.keys(),
            default=0,
            metavar="SCHEDULER_ID",
            help=f"Select a noise scheduler by SCHEDULER_ID:\n{schedulers_text}"
        )
        generation_group.add_argument(
            "--scheduler_config",
            type=json.loads,
            default={},
            metavar="JSON",
            help=(
                "Scheduler custom parameters in JSON string"
                "(e.g., '{\\\"beta_schedule\\\": \\\"linear\\\"}')."
            )
        )
        generation_group.add_argument(
            "--layout",
            choices=["P", "L", "S", "Portrait", "Landscape", "Square"],
            default="S",
            help=(
                "[Experimental] Image aspect ratio.\n"
                "  P/Portrait: 832x1216\n"
                "  L/Landscape: 1344x768\n"
                "  S/Square: 1024x1024"
            )
        )
        generation_group.add_argument(
            "--batch_count",
            type=int_range(1, 100),
            default=1,
            metavar="INT(choose from 1-100)",
            help="Number of images to generate per prompt."
        )

        # Model options
        model_group = txt2img_parser.add_argument_group("Model options")
        model_group.add_argument(
            "--model", type=int,
            choices=Txt2ImgCommand.available_models.keys(), default=0,
            metavar="MODEL_ID",
            help=f"Select a base model by ID:\n{models_text}"
        )
        model_group.add_argument(
            "--quantized_model",
            action="store_true",
            help="Use the quantized version of the model for faster/low-memory inference."
        )
        model_group.add_argument(
            "--weight_shared_model",
            action="store_true",
            help="[Experimental] Use the weight-shared version of the quantized model."
        )
        model_group.add_argument(
            "--use_torch",
            action="store_true",
            help="Force PyTorch for tokenizer and scheduler (defaults to lightweight ONNX/NumPy processing)."
        )

        # Debug options
        debug_group = txt2img_parser.add_argument_group("Debug options")
        debug_group.add_argument(
            "--profile",
            action="store_true",
            help="Enable a simple profiler to measure execution time."
        )
        debug_group.add_argument(
            "--log_level",
            type=str,
            choices=[level for level in logging.getLevelNamesMapping().keys()],
            default="WARNING",
            help="Set the logging threshold level."
        )
        debug_group.add_argument(
            "--submodule",
            type=str,
            choices=submodules,
            default="",
            metavar="MODULE",
            help="Execute a specific pipeline submodule only, or compute UNet cosine similarity."
        )
        debug_group.add_argument(
            "--fp16_part",
            type=int_range(0, 4),
            nargs='+',
            default=[],
            metavar="BLOCK_ID",
            help=(
                "Substitute specific quantized UNet blocks with FP16 precision for evaluation.\n"
                "Available Block IDs:\n"
                "  0: Common\n"
                "  1: Down\n"
                "  2: Mid\n"
                "  3: Up1\n"
                "  4: Up2"
            )
        )

        # File I/O Options
        io_group = txt2img_parser.add_argument_group("I/O options")
        io_group.add_argument(
            "--input_file", "--prompt_list_file",
            type=str,
            default="",
            metavar="FILE",
            help="Text file containing one prompt per line for batch processing (see 'sample_prompts.txt')."
        )
        io_group.add_argument(
            "--input_dir",
            type=str,
            default="",
            metavar="DIR",
            help="Directory containing '*.npy' files to be loaded and processed sequentially."
        )
        io_group.add_argument(
            "--output_dir",
            type=str,
            default=r"outputs",
            metavar="DIR",
            help="Directory where generated images will be saved."
        )
        io_group.add_argument(
            "--output_prefix",
            type=str,
            default="output_sdxl_npu_",
            metavar="PREFIX",
            help="Prefix for the saved image filenames."
        )
        txt2img_parser.set_defaults(func=txt2img_command_factory)

    def _validate_prompts(self, config):
        if not config.prompt_2:
            config.prompt_2 = config.prompt
        if not config.negative_prompt_2:
            config.negative_prompt_2 = config.negative_prompt

        return config

    def _validate_model_options(self, config):
        if config.weight_shared_model:
            if not config.quantized_model:
                logger.warning("Warning: Weight-shared models currently consume about twice the model file size in RAM,"
                               "so only quantized versions are provided.\n"
                               "Please run with '--quantized_model'. Falling back to the quantized model.")
                config.quantized_model = True

        return config
        
    def _validate_layout(self, config):
        if config.layout == "P" or config.layout == "Portrait":
            config.width = 832
            config.height = 1216
            if not config.quantized_model:
                logger.warning("Warning: Portrait dimensions are currently only supported by weight-shared models.\n"
                               "Please run with '--quantized_model --weight_shared_model'."
                               "Falling back to the quantized weight-shared model.")
            config.quantized_model = True
            config.weight_shared_model = True
            config.fp16_part = [0]
        elif config.layout == "L" or config.layout == "Landscape":
            config.width = 1344
            config.height = 768
            if not config.quantized_model:
                logger.warning("Warning: Landscape dimensions are currently only supported by weight-shared models.\n"
                               "Please run with '--quantized_model --weight_shared_model'."
                               "Falling back to the quantized weight-shared model.")
            config.quantized_model = True
            config.weight_shared_model = True
            config.fp16_part = [0]
        else:
            config.width = 1024
            config.height = 1024

        return config, f"{config.width}x{config.height}"

    def _validate_dirs(self, config, res_str):
        name = config.model_name
        dirs = {}

        dirs["vae_decoder_dir"] = rf"{MODEL_ROOT_DIR}\{name}\vae_decoder\{res_str}"
        dirs["vae_encoder_dir"] = rf"{MODEL_ROOT_DIR}\{name}\vae_encoder\{res_str}"
        dirs["vae_encoder_dir"] = rf"{MODEL_ROOT_DIR}\{name}\vae_encoder"

        dirs["text_encoder_dir"] = rf"{MODEL_ROOT_DIR}\{name}\text_encoder"
        dirs["text_encoder_2_dir"] =  rf"{MODEL_ROOT_DIR}\{name}\text_encoder_2"
        dirs["tokenizer_dir"] = rf"{MODEL_ROOT_DIR}\{name}\tokenizer"
        dirs["tokenizer_2_dir"] = rf"{MODEL_ROOT_DIR}\{name}\tokenizer_2"

        if config.quantized_model:
            dirs["unet_dir"] = rf"{MODEL_ROOT_DIR}\{name}\unet_w8a16_quantized"
            if config.weight_shared_model:
                dirs["unet_dir"] = rf"{MODEL_ROOT_DIR}\{name}\unet_w8a16_weight_shared"
            dirs["unet_dir_fp16"] = rf"{MODEL_ROOT_DIR}\{name}\unet"
            config.unet_fp16 = ["part0"]
        else:
            dirs["unet_dir"] = rf"{MODEL_ROOT_DIR}\{name}\unet"
            dirs["unet_dir_fp16"] = rf"{MODEL_ROOT_DIR}\{name}\unet"
            config.unet_fp16 = ["part0"]

        for part in config.fp16_part:
            if part > 0 and part < 5:
                config.unet_fp16.append(f"part{part}")
            if part == 5:
                dirs["text_encoder_dir"] = rf"{MODEL_ROOT_DIR}\{name}\text_encoder"
            elif part == 6:
                dirs["text_encoder_2_dir"] = rf"{MODEL_ROOT_DIR}\{name}\text_encoder_2"
            elif part == 7:
                dirs["vae_decoder_dir"] = rf"{MODEL_ROOT_DIR}\{name}\vae_decoder"

        # check model dirs
        if not all([os.path.isdir(d) for d in dirs.values()]):
            logger.critical("Some required model components are missing.\n"
                            "Please run 'sdxlite-cli setup' to repair the installation.")
            exit()
                
        dirs["output_dir"] = config.output_dir
        os.makedirs(config.output_dir, exist_ok=True)

        config.dirs = dirs
        return config, dirs

    def _validate_torch(self, config, dirs):
        if not config.use_torch:
            from utils import required_modules as required

            if not required.is_onnxextensions_lib_available(r".\lib\ortextensions.dll"):
                logger.warning("ortextensions.dll not found.\n"
                               "Please run 'sdxlite-cli setup' or pass '--use_torch' to enable PyTorch.\n"
                               "Falling back to PyTorch for this run.")
                config.use_torch = True

            if not (required.is_onnx_tokenizer_available(os.path.join(dirs["tokenizer_dir"], "tokenizer.onnx")) and
                    required.is_onnx_tokenizer_available(os.path.join(dirs["tokenizer_2_dir"], "tokenizer_2.onnx"))):
                logger.warning("tokenizer.onnx not found.\n"
                               "Please run 'sdxlite-cli setup' or pass '--use_torch' to enable PyTorch.\n"
                               "Falling back to PyTorch for this run.")
                config.use_torch = True

        return config

    def __init__(self, args):
        config = args
        config.scheduler_name = Txt2ImgCommand.available_schedulers[config.scheduler_type]
        config.model_name = Txt2ImgCommand.available_models[config.model]
        config.log_level = logging.getLevelNamesMapping()[config.log_level]
        config.calib_strategy = 0
        config.debug_mode = config.log_level <= logging.DEBUG

        from profilers import simple_profiler as profiler
        profiler.available = config.profile

        config = self._validate_prompts(config)
        config = self._validate_model_options(config)
        config, res_str = self._validate_layout(config)
        config, dirs = self._validate_dirs(config, res_str)
        config = self._validate_torch(config, dirs)
        
        self.config = config
        
    def run(self):
        logger.info("Running txt2img...")
        if self.config.submodule == "":
            if self.config.input_file != "" or self.config.batch_count > 1:
                from sdxl_pipeline.batch_pipeline import SDXLBatchPipeline
                pipe = SDXLBatchPipeline(self.config)
            else:
                pipe = SDXLPipeline(self.config)
        elif self.config.submodule == "text_encoder":
            from io_collector.text_encode_only import SDXLPipelineTextOnly
            pipe = SDXLPipelineTextOnly(self.config)
        elif self.config.submodule == "unet":
            from io_collector.unet_only import SDXLPipelineUNetOnly
            pipe = SDXLPipelineUNetOnly(self.config)
        elif self.config.submodule == "vae_decoder":
            from io_collector.vae_only import SDXLPipelineVAEDecoderOnly
            pipe = SDXLPipelineVAEDecoderOnly(self.config)
        elif self.config.submodule == "generate_for_unet_cos_sim":
            from io_collector.generate_for_unet_cos_sim import UNetCosSim
            pipe = UNetCosSim(self.config)
        elif self.config.submodule == "compute_unet_cos_sim":
            from profilers.compute_unet_cos_sim import UNetCosSim
            pipe = UNetCosSim(self.config)
        elif self.config.submodule == "collect_calib_data":
            from io_collector.calib_data_collector import CalibrationDataCollector
            pipe = CalibrationDataCollector(self.config)
        pipe.run()
        
        

