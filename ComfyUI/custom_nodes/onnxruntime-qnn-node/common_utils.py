import gc
import torch

def clean_memory():
    """Pythonの未割り当てメモリとキャッシュを強制解放する共通関数"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

