from argparse import ArgumentParser, Namespace

from . import BaseCLICommand
from conf import MODEL_ROOT_DIR
from utils import get_project_root


def setup_command_factory(args: Namespace):
    return SetupCommand()

class SetupCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser) -> None:
        setup_parser = parser.add_parser(
            "setup",
            help="Download required modules, including DLLs and models.",
            usage="\n sdxlite-cli setup"
        )
        setup_parser.set_defaults(func=setup_command_factory)

    def run(self):
        from utils import required_modules as required
        from utils.console import get_yes_no

        proj_root = get_project_root()
        if not required.is_onnxextensions_lib_available(rf"{proj_root}\lib\ortextensions.dll"):
            print(f"\nDownloading ortextensions.dll...\n"
                  f"  From: {required.onnxextensions['url']}\n"
                  f"  To:   {required.onnxextensions['target_dir']}")

            if get_yes_no("\nDo you want to continue? [y/N]: "):
                required.ensure_onnxextensions_lib()
                print("  -> ✅ Done!")
            else:
                print("  -> Canceled.")

        import os
        repo_id = "Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite"
        dir_name = repo_id.split("/")[-1]
        model_dir = os.path.join(MODEL_ROOT_DIR, dir_name)
        print(f"\nDownloading model...\n"
              f"  From: https://huggingface.co/{repo_id}\n"
              f"  To:   {model_dir}")

        if get_yes_no("\nDo you want to continue? [y/N]: "):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id,
                              allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt"],
                              ignore_patterns="*.md", cache_dir=model_dir, local_dir=model_dir)
            print("  -> ✅ Done!")
        else:
            print("  -> Canceled")
