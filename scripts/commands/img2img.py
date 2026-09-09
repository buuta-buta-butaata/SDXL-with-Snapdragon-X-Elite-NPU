import logging

from argparse import ArgumentParser, Namespace

from . import (BaseCLICommand, MyHelpFormatter, int_range, float_range,
               format_dict_text, modify_parser_action, remove_parser_action)
from .txt2img import Txt2ImgCommand


logger = logging.getLogger(__name__)

def img2img_command_factory(args: Namespace):
    return Img2ImgCommand(args)

class Img2ImgCommand(Txt2ImgCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser) -> None:

        img2img_parser = parser.add_parser(
            "img2img",
            formatter_class=MyHelpFormatter,
            help="[Experimental] Run SDXL img2img pipeline on NPU.",
            usage="\n sdxlite-cli img2img [options]"
        )

        Img2ImgCommand._register(img2img_parser)

        modify_parser_action(img2img_parser, "--prompt", help="The prompt to guide image generation / inpainting.",
                             required=True)

        modify_parser_action(img2img_parser, "--layout", help="Not supported")
        modify_parser_action(img2img_parser, "--weight_shared_model", help="Not supported")
        modify_parser_action(img2img_parser, "--submodule", help="Not supported")
        remove_parser_action(img2img_parser, "--batch_count")
        # remove_parser_action(img2img_parser, "--layout")
        # remove_parser_action(img2img_parser, "--weight_shared_model")
        # remove_parser_action(img2img_parser, "--submodule")

        generation_group = img2img_parser.add_argument_group("Img2Img generation options")
        generation_group.add_argument(
            "--input_image", type=str, required=True, 
            metavar="PATH",
            help="Path to the input source image."
        )
        generation_group.add_argument(
            "--denoising_strength", type=float_range(0, 1), default=0.6, 
            metavar="FLOAT",
            help="Denoising strength (0.0 to 1.0). Controls how much to transform the input image."
        )
        generation_group.add_argument(
            "--mask_image", type=str, default=None, 
            metavar="PATH",
            help=("Path to the boundary mask image for inpainting (Optional). "
                  "If not provided, normal img2img is executed.")
        )
        generation_group.add_argument(
            "--mask_blur", type=int, default=0,
            metavar="INT",
            help="Blur radius for the mask edge to blend smoothly. Set 0 for no blur."
        )
        generation_group.add_argument(
            "--outpaint_direction", type=str, default=None,
            choices=["right", "left", "top", "bottom"],
            help="どちらの方向に背景を広げるか"
        )
        generation_group.add_argument(
            "--outpaint_ratio", type=float_range(0, 0.5), default=0.2,
            metavar="FLOAT",
            help="Outpaintで、どのくらい拡大するか"
        )
        generation_group.add_argument(
            "--hires_fix",
            action="store_true",
            help="有効にすると、タイル式アップスケールを自動実行する"
        )
        generation_group.add_argument(
            "--upscale_factor", type=float_range(1, 4), default=2,
            metavar="FLOAT",
            help="何倍に拡大するか（defaultは2倍）"
        )

        img2img_parser.set_defaults(func=img2img_command_factory)

    def run(self):
        logger.info("Running img2img...")

        if self.config.width != 1024 or self.config.height != 1024:
            logger.warning("img2img only support 1024x1024, resized input image")
        
        if self.config.mask_image is not None:
            from sdxl_pipeline.pipeline_inpaint import SDXLInpaintPipeline
            pipe = SDXLInpaintPipeline(self.config)
        elif self.config.outpaint_direction is not None:
            from sdxl_pipeline.pipeline_outpaint import SDXLOutpaintPipeline
            pipe = SDXLOutpaintPipeline(self.config)
        elif self.config.hires_fix or self.config.control_mode == 6 or self.config.control_mode == 7:
            from sdxl_pipeline.pipeline_tiled_img2img import SDXLTiledImg2ImgPipeline
            pipe = SDXLTiledImg2ImgPipeline(self.config)
        else:
            from sdxl_pipeline.pipeline_img2img import SDXLImg2ImgPipeline
            pipe = SDXLImg2ImgPipeline(self.config)

        if self.config.use_controlnet:
            from sdxl_pipeline.unet_controlnet import UNetControlNet
            pipe.unet = UNetControlNet(self.config)
        pipe.run()
        
