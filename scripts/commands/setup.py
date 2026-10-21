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
            help="Download ControlNet Union model"
        )
        setup_parser.add_argument(
            "--ip_adapter",
            action="store_true",
            help="Download IP Adapter models"
        )
        setup_parser.add_argument(
            "--t2i_adapter",
            action="store_true",
            help="Download T2I Adapter models"
        )
        setup_parser.set_defaults(func=setup_command_factory)

    def __init__(self, args):
        self.config = args

    def run(self):
        all_download_dict = {
            "DreamShaper XL": { "repo_id": "Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite",
                                "model_root_dir": MODEL_ROOT_DIR},
            "ControlNet": { "repo_id": "Buuta/controlnet-union-sdxl-for-Snapdragon-X-Elite",
                            "model_root_dir": HELPERS_DIR},
            "IP Adapter": { "repo_id": "Buuta/ip-adapter-sdxl-for-Snapdragon-X-Elite",
                            "model_root_dir": HELPERS_DIR},
            "T2I Adapter": { "repo_id": "Buuta/t2i-adapter-sdxl-for-Snapdragon-X-Elite",
                            "model_root_dir": HELPERS_DIR},
        }

        download_dict = {}
        
        if self.config.controlnet:
            download_dict["ControlNet"] = all_download_dict["ControlNet"]
        if self.config.ip_adapter:
            download_dict["IP Adapter"] = all_download_dict["IP Adapter"]
        if self.config.t2i_adapter:
            download_dict["T2I Adapter"] = all_download_dict["T2I Adapter"]

        if len(download_dict) == 0:
            download_dict = all_download_dict
        
        self.download_ortextensions()
        self.download_models(download_dict)
        
        # self.download_model()
        # self.download_controlnet()
        # self.download_ip_adapter()
        # self.download_t2i_adapter()

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

    def download_models(self, download_dict):
        for k, v in download_dict.items():
            dir_name = v["repo_id"].split("/")[-1]
            model_dir = os.path.join(v["model_root_dir"], dir_name)
            print(f"\nDownloading {k} model...\n"
                  f"  From: https://huggingface.co/{v['repo_id']}\n"
                  f"  To:   {model_dir}")

        if get_yes_no("\nDo you want to continue? [y/N]: "):
            for k, v in download_dict.items():
                self.download_model(name=k, **v)
        else:
            print("  -> Canceled")
        
    def download_model(self, name, repo_id,
                       allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt", "*.csv"],
                       ignore_patterns="*.md",
                       model_root_dir=MODEL_ROOT_DIR):
        dir_name = repo_id.split("/")[-1]
        model_dir = os.path.join(model_root_dir, dir_name)
        print(f"\nDownloading {name} model...\n"
              f"  From: https://huggingface.co/{repo_id}\n"
              f"  To:   {model_dir}")

        snapshot_download(repo_id=repo_id,
                          allow_patterns=allow_patterns,
                          ignore_patterns=ignore_patterns, cache_dir=model_dir, local_dir=model_dir)
        print("  -> ✅ Done!")

    # def download_model(self):
    #     repo_id = "Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite"
    #     dir_name = repo_id.split("/")[-1]
    #     model_dir = os.path.join(MODEL_ROOT_DIR, dir_name)
    #     print(f"\nDownloading model...\n"
    #           f"  From: https://huggingface.co/{repo_id}\n"
    #           f"  To:   {model_dir}")

    #     if get_yes_no("\nDo you want to continue? [y/N]: "):
    #         snapshot_download(repo_id=repo_id,
    #                           allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt", "*.csv"],
    #                           ignore_patterns="*.md", cache_dir=model_dir, local_dir=model_dir)
    #         print("  -> ✅ Done!")
    #     else:
    #         print("  -> Canceled")

    # def download_controlnet(self):
    #     repo_id = "Buuta/controlnet-union-sdxl-for-Snapdragon-X-Elite"
    #     dir_name = repo_id.split("/")[-1]
    #     model_dir = os.path.join(HELPERS_DIR, dir_name)
    #     print(f"\nDownloading ControlNet model...\n"
    #           f"  From: https://huggingface.co/{repo_id}\n"
    #           f"  To:   {model_dir}")

    #     if get_yes_no("\nDo you want to continue? [y/N]: "):
    #         snapshot_download(repo_id=repo_id,
    #                           allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt", "*.csv"],
    #                           ignore_patterns="*.md", cache_dir=model_dir, local_dir=model_dir)
    #         print("  -> ✅ Done!")
    #     else:
    #         print("  -> Canceled")
        
    # def download_ip_adapter(self):
    #     repo_id = "Buuta/ip-adapter-sdxl-for-Snapdragon-X-Elite"
    #     dir_name = repo_id.split("/")[-1]
    #     model_dir = os.path.join(HELPERS_DIR, dir_name)
    #     print(f"\nDownloading IP Adapter models...\n"
    #           f"  From: https://huggingface.co/{repo_id}\n"
    #           f"  To:   {model_dir}")

    #     if get_yes_no("\nDo you want to continue? [y/N]: "):
    #         snapshot_download(repo_id=repo_id,
    #                           allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt", "*.csv"],
    #                           ignore_patterns="*.md", cache_dir=model_dir, local_dir=model_dir)
    #         print("  -> ✅ Done!")
    #     else:
    #         print("  -> Canceled")
        
    # def download_t2i_adapter(self):
    #     repo_id = "Buuta/t2i-adapter-sdxl-for-Snapdragon-X-Elite"
    #     dir_name = repo_id.split("/")[-1]
    #     model_dir = os.path.join(HELPERS_DIR, dir_name)
    #     print(f"\nDownloading T2I Adapter models...\n"
    #           f"  From: https://huggingface.co/{repo_id}\n"
    #           f"  To:   {model_dir}")

    #     if get_yes_no("\nDo you want to continue? [y/N]: "):
    #         snapshot_download(repo_id=repo_id,
    #                           allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt", "*.csv"],
    #                           ignore_patterns="*.md", cache_dir=model_dir, local_dir=model_dir)
    #         print("  -> ✅ Done!")
    #     else:
    #         print("  -> Canceled")
        
