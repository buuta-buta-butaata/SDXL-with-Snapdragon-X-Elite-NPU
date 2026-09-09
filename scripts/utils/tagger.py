import csv
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image

from utils import qnn_ep_helper as qnn

logger = logging.getLogger(__name__)


class Category:
    GENERAL = 0
    CHARACTER =  4
    RATING = 9

    
class WD14Tagger:
    def __init__(self, model_path="wd14_tagger.onnx", csv_path="selected_tags.csv", categories=[Category.GENERAL]):
        # 1. ONNXセッションの初期化
        self.session = ort.InferenceSession(model_path, sess_options=qnn.session_options)
        
        # 2. CSVからタグ名リストを読み込む
        self.tag_list = []
        self.load_tags(csv_path, categories)

    def load_tags(self, csv_path, categories=[Category.GENERAL]):
        """CSVを読み込み、ONNXの出力インデックスと一致するインデックスにタグ名を格納する"""
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # ヘッダー(tag_id,name,category,count)をスキップ
            
            for row in reader:
                if not row:
                    continue
                tag_id = int(row[0])
                tag_name = row[1]
                category = int(row[2])
                
                # アンダースコアをスペースに置換（Stable Diffusionのプロンプト形式に合わせる）
                # 例: "long_hair" -> "long hair"
                tag_name_clean = tag_name.replace('_', ' ')
                
                # カテゴリが 0 (一般) のものだけを有効にする
                # キャラクター(4)、レーティング(9)などは無視するため、無効な場合は空文字を入れ、インデックスのズレを防ぐ
                if category in categories:
                    self.tag_list.append(tag_name_clean)
                else:
                    self.tag_list.append("") # インデックス維持のためのプレースホルダー

    def get_prompt(self, image_np, base_modifiers="", threshold=0.5):
        """
        画像からプロンプトを自動生成するメイン関数
        """
        # [A] 前処理: 画像を 448x448 にリサイズして、モデルが求める形状にする
        # ※WD14 Taggerは基本的に 448x448 のRGB入力を期待します
        resized = image_np
        
        # RGB配列の float32化 (Binds to ONNX input)
        # 形状を (1, 448, 448, 3) にする（モデルによっては (1, 3, 448, 448) の場合もあります）
        # ※Netronで見定めた入力形状に合わせて軸を調整してください。大半のWD14は (1, 448, 448, 3) です
        input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)
        
        # [B] 推論実行
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # 出力配列（各タグの確率群）を取得。通常は1つ目の出力に入っています。
        # shape: (1, タグ数) -> スクイーズして (タグ数,) の1次元NumPy配列にする
        probs = np.squeeze(outputs[0])
        
        # [C] NumPyの高速インデックス抽出
        # 閾値（例: 0.35）を超えた確率のインデックスを一覧で取得
        high_prob_indices = np.where(probs > threshold)[0]
        
        # 確率が高い順にソート（お好みで）
        high_prob_indices = high_prob_indices[np.argsort(probs[high_prob_indices])[::-1]]
        
        # [D] インデックスをタグ名に変換してプロンプト化
        detected_tags = []
        for idx in high_prob_indices:
            tag = self.tag_list[idx]
            if tag: # 空文字（レーティング等）でなければ採用
                # タイルプロンプトに悪影響を及ぼしそうな不要な単語（例: 背景一色時の「no humans」など）
                # をここで除外（ブラックリスト化）することも可能です
                if tag in ["no humans", "scenery"]:
                    continue
                detected_tags.append(tag)
        
        # 元の共通修飾語と合体させる
        if detected_tags:
            tags = ", ".join(detected_tags)
            prompt = tags if len(base_modifiers) == 0 else base_modifiers + ", " + tags
        else:
            prompt = base_modifiers
            
        return prompt

    def preprocess_image(self, image, size=(448, 448)):
        """
        画像をTaggerが受け取れる形式に変換する
        
        Parameters:
        - image: NumPy配列、形状 (H, W, 3) などのRGB画像、値は 0〜255 (uint8)
    
        Returns:
        - input_tensor: NumPy配列、形状 (1, 3, width, height)、float32型 (NPU入力用) BGR形式
        """
        W, H = size
        resized = image.resize((W, H), resample=Image.Resampling.BICUBIC)
        resized_np = np.array(resized)
        return resized_np.astype(np.float32)[:, :, ::-1]


if __name__ == "__main__":
    import conf
    tags_dict_path = rf"{conf.MODEL_ROOT_DIR}\helper\wd_vit_tagger_v3\selected_tags.csv"
    model_path = rf"{conf.MODEL_ROOT_DIR}\helper\wd_vit_tagger_v3\model.onnx"
    files = [
        "output_sdxl_npu_20260816172907",
        "output_sdxl_npu_20260816172911",
        "output_sdxl_npu_20260816174620",
        "output_sdxl_npu_20260816234100",
        "output_sdxl_npu_20260819021156",
        "output_sdxl_npu_20260826154700",
        "output_sdxl_npu_20260826172511",
        "output_sdxl_npu_20260826234458",
        "output_sdxl_npu_20260827184719",
        "output_sdxl_npu_20260827190401",
        "output_sdxl_npu_20260828021712",
        "output_sdxl_npu_20260828130339",
        "output_sdxl_npu_20260828142955",
        "output_sdxl_npu_20260828175746",
        "output_sdxl_npu_20260902014201",
        "output_sdxl_npu20260712015326",
    ]
    from . import safety_checker
    checker = safety_checker.RatingChecker(model_path, tags_dict_path)
    for file_prefix in files:
        img = Image.open(rf"{conf.PROJECT_ROOT_DIR}/outputs/{file_prefix}.png")
        
        tagger = WD14Tagger(model_path, tags_dict_path)
        tagger_input = tagger.preprocess_image(img, (448, 448))
        result = tagger.get_prompt(tagger_input, threshold=0.5)

        print("-" * 80)
        print(f"file: {file_prefix}.png")
        print(result)
        has_nsfw_concept, rating = checker.check_rating(img, threshold=0.5)
        if has_nsfw_concept:
            print(f"Has NSFW concept. Rating: {rating}")
