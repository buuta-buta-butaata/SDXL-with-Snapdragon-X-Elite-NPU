from huggingface_hub import snapshot_download
import os

current_dir = os.path.join(os.path.dirname(__file__))
snapshot_download(repo_id="Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite",
                  allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt"],
                  ignore_patterns="*.md", cache_dir=current_dir, local_dir=current_dir)
