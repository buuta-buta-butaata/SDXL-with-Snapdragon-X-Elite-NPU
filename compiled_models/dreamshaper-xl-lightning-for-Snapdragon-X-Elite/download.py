from huggingface_hub import snapshot_download

snapshot_download(repo_id="Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite", allow_patterns=["*.onnx", "*.bin", "*.json", "*.txt"], ignore_patterns="*.md", cache_dir="./", local_dir="./")
