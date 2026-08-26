import logging

from argparse import ArgumentParser, Namespace

from . import BaseCLICommand, MyHelpFormatter, int_range, format_dict_text
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
            help="Generate images from images and text prompts.",
            usage="\n sdxlite-cli img2img [options]"
        )

        Img2ImgCommand._register(img2img_parser)

        generation_group = img2img_parser.add_argument_group("Img2Img generation options")
        generation_group.add_argument(
            "--denoising_strength",
            type=float,
            default=0.8,
            metavar="FLOAT",
            help="Denoising strength"
        )
        generation_group.add_argument(
            "--mask_file",
            type=str,
            default=None,
            metavar="PATH",
            help="Mask for inpaint"
        )
        generation_group.add_argument(
            "--blur_radius",
            type=int,
            default=0,
            metavar="INT",
            help="Blur radius"
        )

        img2img_parser.set_defaults(func=img2img_command_factory)

    def run(self):
        logger.info("Running img2img...")

        if self.config.mask_file is not None:
            from sdxl_pipeline.pipeline_inpaint import SDXLInpaintPipeline
            pipe = SDXLInpaintPipeline(self.config)
        else:
            from sdxl_pipeline.pipeline_img2img import SDXLImg2ImgPipeline
            pipe = SDXLImg2ImgPipeline(self.config)
        pipe.run()
        
