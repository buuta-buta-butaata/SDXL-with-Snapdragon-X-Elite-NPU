import torch

# =====================================================================
# 0-B. 【新設】QNN 1024x1024 Latent Generator（サイズ固定Latent生成ノード）
# =====================================================================
class QNNFixedLatentGenerator:
    @classmethod
    def INPUT_TYPES(s):
        # 将来的に有効化したい解像度のリストを作成します
        # ユーザーに分かりやすいように「横x縦」の文字列にします
        resolution_options = [
            "1024 x 1024 (SDXL Square)",
            # 【後々の拡張用】必要になったら以下のコメントアウトを外すだけでメニューに追加されます！
            #"1344 x 768 (SDXL Landscape)",
            #"832 x 1216 (SDXL Portrait)",
        ]

        return {
            "required": {
                # 解像度を選択するドロップダウンメニュー
                "resolution": (resolution_options, {"default": "1024 x 1024 (SDXL Square)"}),
                # バッチサイズ
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1, "step": 1}),
            }
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "generate_latent"
    CATEGORY = "QNN_Optimization"

    def generate_latent(self, resolution, batch_size):
        # SDXLのチャンネル数は常に4
        channels = 4
        
        # 選択された文字列に応じて、内部で保持するLatentサイズ（解像度を8で割った値）を決定します
        if resolution == "1024 x 1024 (SDXL Square)":
            width = 128   # 1024 / 8
            height = 128  # 1024 / 8
        elif resolution == "1344 x 768 (SDXL Landscape)":
            width = 168   # 1344 / 8
            height = 96   # 768 / 8
        elif resolution == "832 x 1216 (SDXL Portrait)":
            width = 104   # 832 / 8
            height = 152  # 1216 / 8
        else:
            # 万が一のフォールバック
            width = 128
            height = 128
        
        # ComfyUIのKSamplerが喜ぶ、すべて0で初期化された標準テンソルを作成
        #（KSamplerが内部でこのテンソルに指定したシード値のノイズを自動でブレンドします）
        latent_tensor = torch.zeros([batch_size, channels, height, width], dtype=torch.float32, device="cpu")
        
        print(f"[QNN Latent] 1024x1024固定のLatent空間を生成しました。形状: {latent_tensor.shape}")
        
        # ComfyUIの標準フォーマットである辞書型に包んで返します
        return ({"samples": latent_tensor},)
