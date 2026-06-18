import gc

def clean_memory():
    """Pythonの未割り当てメモリとキャッシュを強制解放する共通関数"""
    gc.collect()
