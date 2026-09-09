import os

from argparse import ArgumentParser, Namespace
from huggingface_hub import snapshot_download

from . import BaseCLICommand
from conf import PROJECT_ROOT_DIR, MODEL_ROOT_DIR, HELPERS_DIR
from utils import required_modules as required
from utils.console import get_yes_no


def setup_command_factory(args: Namespace):
    return SetupCommand(args)

class SetupCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser) -> None:
        setup_parser = parser.add_parser(
            "setup",
            help="Download required modules, including DLLs and models.",
            usage="\n sdxlite-cli setup"
        )
        setup_parser.add_argument(
            "--controlnet",
            action="store_true",
            help="Download ControlNet Union"
        )
        setup_parser.set_defaults(func=setup_command_factory)

    def __init__(self, args):
        self.config = args

    def run(self):
        if self.config.controlnet:
            self.download_controlnet()
            return

        self.download_ortextensions()
        self.download_model()
        self.download_controlnet()

    def download_ortextensions(self):
        if not required.is_onnxextensions_lib_available(rf"{PROJECT_ROOT_DIR}\lib\ortextensions.dll"):
            print(f"\nDownloading ortextensions.dll...\n"
                  f"  From: {required.onnxextensions['url']}\n"
                  f"  To:   {required.onnxextensions['target_dir']}")

            if get_yes_no("\nDo you want to continue? [y/N]: "):
                required.ensure_onnxextensions_lib()
                print("  -> ✅ Done!")
            else:
                print("  -> Canceled.")
        

    def download_model(self):
        repo_id = "Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite"
        dir_name = repo_id.split("/")[-1]
        model_dir = os.path.join(MODEL_ROOT_DIR, dir_name)
        print(f"\nDownloading model...\n"
              f"  From: https://huggingface.co/{repo_id}\n"
              f"  To:   {model_dir}")

        if get_yes_no("\nDo you want to continue? [y/N]: "):
            snapshot_download(repo_id=repo_id,
                              allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt", "*.csv"],
                              ignore_patterns="*.md", cache_dir=model_dir, local_dir=model_dir)
            print("  -> ✅ Done!")
        else:
            print("  -> Canceled")

    def download_controlnet(self):
        repo_id = "Buuta/controlnet-union-sdxl-for-Snapdragon-X-Elite"
        dir_name = repo_id.split("/")[-1]
        model_dir = os.path.join(HELPERS_DIR, dir_name)
        print(f"\nDownloading ControlNet model...\n"
              f"  From: https://huggingface.co/{repo_id}\n"
              f"  To:   {model_dir}")

        if get_yes_no("\nDo you want to continue? [y/N]: "):
            snapshot_download(repo_id=repo_id,
                              allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt", "*.csv"],
                              ignore_patterns="*.md", cache_dir=model_dir, local_dir=model_dir)
            print("  -> ✅ Done!")
        else:
            print("  -> Canceled")
        
