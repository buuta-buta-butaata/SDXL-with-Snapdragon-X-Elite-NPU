import os

# =====================================================================
# 0. 【新設】QNN Base Path Loader（パスを一元管理するノード）
# =====================================================================
class QNNBasePathLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # HuggingFaceからダウンロードしたルートフォルダの絶対パスをユーザーに入力させます
                "base_path": ("STRING", {"default": "C:\\AI\\dreamshaper-xl-lightning-for-Snapdragon-X-Elite"}),
            }
        }
    
    # 次のローダーノードたちに「QNN_BASE_PATH」という型でパスを渡します
    RETURN_TYPES = ("QNN_BASE_PATH",)
    RETURN_NAMES = ("base_path",)
    FUNCTION = "get_path"
    CATEGORY = "QNN_Optimization"

    def get_path(self, base_path):
        # 左右の余計な空白やクォーテーションを綺麗にして絶対パス化
        clean_path = base_path.strip().strip('"').strip("'")
        if not os.path.exists(clean_path):
            print(f"[QNN Warning] 指定されたベースパスが見つかりません: {clean_path}")
        return (clean_path,)
