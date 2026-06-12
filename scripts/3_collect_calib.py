import argparse
import gc
import glob
import numpy as np
import os
import random
import subprocess
import sys

def run_cmd(cmd_list):
    """外部スクリプトを安全に実行し、エラーがあれば即停止するヘルパー"""
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] スクリプトの実行に失敗しました。停止します。")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="解像度別 キャリブレーションデータ収集ツール")
    parser.add_argument("--name", type=str, default="sdxl_model", help="モデルの名称、短い方が見やすい")
    parser.add_argument("--width", type=int, default=1024, help="生成画像の幅 (1024, 1344など)")
    parser.add_argument("--height", type=int, default=1024, help="生成画像の高さ (1024, 768など)")
    parser.add_argument("--steps_min", type=int, default=4, help="ステップ数")
    parser.add_argument("--steps_max", type=int, default=8, help="ステップ数")
    parser.add_argument("--guidance_scale", type=float, default=5, help="ガイダンススケールの指定")
    # parser.add_argument("--calib_strategy", type=int, default=0, help="キャリブレーションデータ収集の方針[1-3], 1:3つ取得、2:2つ取得（最初と最後の方を優先）、3:2つ取得（中盤よりを優先）")
    return parser.parse_args()

# キャリブレーション用のプロンプト
# ref. https://github.com/yuyun2000/sd15-to-ax8850-deploy/blob/main/prepare_data.py

PROMPTS = [
    ['Upper body anime illustration, a girl sitting by the window, detailed sparkling eyes, ornate lace dress, braiding her hair.', 1],
    ['1girl, dynamic action pose, jumping, holding a glowing magic wand, barefoot, high quality hand, on street, masterpiece', 3],
    ['A beautiful cyberpunk city, high resolution, 8k, neon lights, highly detailed', 6],
    ['(((gold, silver, glimmer)), faerie), limited palette, contrast, phenomenal aesthetic, best quality, sumptuous artwork', 2],
    ['woman, flower dress, colorful, darl background,flower armor,green theme,exposure blend, medium shot, bokeh, (hdr:1.4), high contrast, (cinematic, teal and orange:0.85), (muted colors, dim colors, soothing tones:1.3), low saturation,', 4],
    ['A girl sitting on a giant ice cream, which is adorned with vibrant colors, delightful frosting, and rainbow sprinkles. She holds an oversized waffle cone,shimmering candies. Floating around her are some small balloons, each tied to a petite candy. The scene is filled with sweetness and joy, showcasing the girls happiness and the enchanting fusion of her imagination and a fantastical world,fantasy, high contrast, ink strokes, explosions, over exposure, purple and red tone impression , abstract, ((watercolor painting by John Berkey and Jeremy Mann )) brush strokes, negative space,', 5],
    ['blonde, Curly hair,(hi-top fade:1.3), dark theme, soothing tones, muted colors, high contrast, (natural skin texture, hyperrealism, soft light, sharp),exposure blend, medium shot, bokeh, (hdr:1.4), high contrast, (cinematic, teal and orange:0.85), (muted colors, dim colors, soothing tones:1.3), low saturation, (hyperdetailed:1.2), (noir:0.4)', 5],
    ['a 20 yo woman, blonde, (hi-top fade:1.3), dark theme, soothing tones, muted colors, high contrast, (natural skin texture, hyperrealism, soft light, sharp)', 8],
    ['1girl, upper body, (huge Laughing),sweety,sun glare, bokeh, depth of field, blurry background, light particles, strong wind,head tilt,simple background, red background', 5],
    ['A woman sitting on a wooden chair, holding a coffee mug with both hands, looking side, intricate details', 2],
    # ['Close-up photo of a cyberpunk cybernetic eye, intricate metallic wires, glowing blue iris, pores on skin, looking directly at camera.', 1],
    #'a 30 yo woman,(hi-top fade:1.3),long hair,dark theme, soothing tones, muted colors, high contrast, (natural skin texture, hyperrealism, soft light, sharp),',
    ['modern living room, minimalist design, large window, sunlight, indoor plants, An elegant interior design photograph of a cozy apartment, warm sunshine spilling across the wooden floor, architectural digest feature.', 6],
    ['majestic lion, close-up face, golden mane, intense eyes, savanna background. A detailed wildlife photograph of a lion resting in the tall grass of the Serengeti, shallow depth of field, sharp focus.', 6],
    #['chibi, cute cat boy, cat ears, oversized hoodie, sitting on a giant pancake. Adorable pastel color illustration, pop art style, stickers, clean vectors, simple shading.', 7],
    #['A futuristic city street at night. Cyberpunk alleyway, neon signs reflection on wet asphalt, glowing holograms, rainy night, cinematic lighting, photorealistic, intricate details.', 8],
]


# Geminiおすすめプロンプト(写実系モデル向け)
"""
PROMPTS = [
    # 実写向け
    # パターン1: ポートレート（女性）
    ['close-up portrait of a woman, blue eyes, smiling, highly detailed skin',
     'A professional studio photograph of a young woman with natural makeup, soft lighting, 8k resolution, photorealistic.'],
    # パターン2: 風景（大自然）
    ['majestic mountains, foggy morning, pine trees, crystal clear lake, sunrise',
     'A breathtaking landscape photograph of the Swiss Alps at dawn, golden hour light, cinematic composition, national geographic style.'],
    # パターン3: サイバーパンクな街並み
    ['cyberpunk city street, neon signs, rainy night, puddles, glowing reflections',
     'A moody, realistic street photo of a futuristic Tokyo alley after rain, hyper-detailed, intricate environment design.'],
    # パターン4: 動物・ネイチャー
    ['majestic lion, close-up face, golden mane, intense eyes, savanna background',
     'A detailed wildlife photograph of a lion resting in the tall grass of the Serengeti, shallow depth of field, sharp focus.'],
    # パターン5: 室内・インテリア
    ['modern living room, minimalist design, large window, sunlight, indoor plants',
     'An elegant interior design photograph of a cozy apartment, warm sunshine spilling across the wooden floor, architectural digest feature.'],
    # パターン6: 女性のポートレート（定番の画質テスト）
    # 注目ポイント: 量子化で肌の質感が不自然に滑らか（プラスチックっぽく）になっていないか、瞳のハイライトが濁っていないか。
    ['A close-up portrait of a woman',
     'A professional studio photograph of a young woman with natural skin texture, soft side lighting, sharp focus on eyes, detailed hair, 8k resolution.'],
    # パターン7: サイバーパンクな街並み（複雑な光と文字のテスト）
    # 注目ポイント: 暗い部分のノイズ（ブロックノイズなど）や、濡れた地面の反射のディテールが維持されているか。
    ['A futuristic city street at night',
     'Cyberpunk alleyway, neon signs reflection on wet asphalt, glowing holograms, rainy night, cinematic lighting, photorealistic, intricate details.'],
    # パターン8: 美味しそうな料理（質感とディテールのテスト）
    # 注目ポイント: ステーキの表面の焦げ目や、お肉の「ジューシーな質感」が保たれているか。
    ['A plate of delicious food',
     'A gourmet beef steak with rosemary on a wooden table, steam rising, glossy sauce, macro photography, shallow depth of field, warm morning light.'],
    # パターン9: 大自然の風景（細部と遠景の潰れテスト）
    # 注目ポイント: 奥の木の葉っぱや、遠くの山の斜面がぼやけて潰れていないか。
    ['A beautiful mountain lake',
     'Majestic snow-capped mountains reflected in a crystal-clear lake, pine forest under a dramatic sunset sky, wide-angle lens, highly detailed landscape.'],
    # パターン10: アンティークな部屋（光と影、物体の形状テスト）
    # 注目ポイント: 窓から差し込む光のグラデーションや、空気中の塵（ダスト）の表現が綺麗に出るか。
    ['An old library room',
     'Sunlight streaming through a window into a dusty classic library, leather chairs, wooden bookshelves filled with ancient books, nostalgic atmosphere.'],
]
"""
# Geminiおすすめプロンプト(イラスト、アニメ系モデル向け
PROMPTS_ANIME = [
    # イラスト、アニメ向け
    # パターン6: 王道アニメの女の子
    ['1girl, solo, anime girl, pink hair, twintails, school uniform, cheerful smile, holding a book',
     'Beautiful anime illustration, vibrant colors, clean lineart, Makoto Shinkai style background, high quality.'],
    # パターン7: ファンタジー（魔法使い）
    ['1boy, mage, holding a glowing staff, casting magic, blue robes, wizard hat, fantasy world',
     'Genshin Impact style game illustration, epic fantasy art, digital painting, colorful magical particle effects, dynamic pose.'],
    # パターン8: レトロ・シティポップ風
    ['1girl, 1980s anime style, sunglasses, city pop aesthetic, night city view, driving a car',
     'Retro anime art, vintage aesthetic, neon color palette, VHS grain effect, nostalgic mood.'],
    # パターン9: ちびキャラ（デフォルメ）
    ['chibi, cute cat boy, cat ears, oversized hoodie, sitting on a giant pancake',
     'Adorable pastel color illustration, pop art style, stickers, clean vectors, simple shading.'],
    # パターン10: メカ・SFイラスト
    ['giant robot, mecha, sci-fi, glowing thrusters, launching from a futuristic hangar, sparks',
     'Detailed mecha design artwork, cyberpunk anime aesthetic, Studio Trigger style, high-energy action shot, dramatic lighting.'],
    # パターン6: 王道のアニメ美少女（顔と線のテスト）
    # 注目ポイント: アニメ調の「線」が途切れたりギザギザになったりしていないか。目が崩れていないか。
    ['1girl, anime style.',
     'A cute anime girl with long blue hair and purple eyes, smiling, wearing a school uniform, cherry blossom petals flying, sunny day, vibrant colors, masterwork.'],
    # パターン7: ファンタジー世界の戦士（衣装と装飾のテスト）
    # 注目ポイント: 鎧の細かい装飾や、光る剣のエフェクトが綺麗に描かれているか。
    ['A fantasy knight.',
     'A cool anime boy knight holding a glowing sword, intricate silver armor, blue cape flowing in the wind, dramatic sky, fantasy illustration, sharp lines.'],
    # パターン8: ちびキャラ・デフォルメ（スタイルの維持テスト）
    # 注目ポイント: 量子化によってデフォルメのバランスが崩れ、リアル寄りの不気味な見た目になっていないか。
    ['A chibi character.',
     'Super deformed chibi wizard girl, holding a magic wand, big expressive eyes, cute witch hat, colorful floating sparkles, pop art style, clean lineart.'],
    # パターン9: 水彩画・レトロイラスト風（独特なタッチのテスト）
    # 注目ポイント: 水彩特有の「にじみ」や「かすれ」、淡い色合いがちゃんと表現できているか。
    ['A girl with an umbrella.',
     'Beautiful watercolor illustration, a girl walking under a clear umbrella in the gentle rain, soft pastel colors, textured paper effect, artistic.'],
    # パターン10: 近未来SFアニメ（ポーズとエフェクトのテスト）
    # 注目ポイント: 複雑なコックピットの背景や、画面に散りばめられたUI（画面表示）の文字っぽさが保たれているか。
    ['A mecha pilot girl.',
     'Anime girl in a futuristic sci-fi bodysuit, inside a high-tech cockpit, glowing UI screens around her, dynamic action pose, neon lighting, epic anime screenshot.'],
]

NEGATIVE_PROMPT_LIB = [
    "sketch, duplicate, ugly, huge eyes, text, logo, worst face",
    "(bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0)",
    "horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy",
    "(interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4)",
    "(deformed fingers:1.2), (long fingers:1.2), bad-artist-anime, bad-artist, bad hand",
    "extra legs, nipples, nsfw, monochrome, greyscale, topless male",
    "(low quality, worst quality:1.4), (FastNegativeEmbedding:0.9)",
    "EasyNegativeV2, (bad anatomy), (mutated limbs:1.2), (mutated hands:1.2)",
    "(text:1.5), simple background, paintings, sketches,(bad-hands-5:1)",
    "lowres, blurry, floating limbs, extra limb, malformed limbs, long neck",
    "cross-eyed, bad body, ugly, disgusting, bad feet, bad leg",
    "missing limb, disconnected limbs, extra legs, missing legs, extra foot",
    "bad eyes, acnes, skin blemishes, signature, watermark, username, duplicate",
    "(2girl), feminine, feminine posture, normal quality:1.31, worst quality:1.33",
]

def get_random_negative_prompt() -> str:
    selected = random.sample(NEGATIVE_PROMPT_LIB, k=random.randint(2,4))
    return ", ".join(selected).replace(",,", ",")

def clean(target_dir):
    for p in glob.glob(f"{target_dir}/**/*.npy"):
        if os.path.isfile(p):
            os.remove(p)

def main():
    args = parse_args()
    
    # 💡 引数から解像度を渡して、コレクターを初期化
    #collector = CalibrationDataCollector(width=args.width, height=args.height)
    
    # -------------------------------------------------------------------------
    # 💡 ここに先駆者のプロンプト（計84組）を回す推論ループを配置
    # -------------------------------------------------------------------------
    # 5分割されたFP16のONNXセッション群をロードするパスも、動的に構成します：
    # model_dir = f"../split_models/{args.width}x{args.height}"
    # common_session = ort.InferenceSession(os.path.join(model_dir, "unet_part0_common.onnx"), ...)
    
    # 推論ループ内での呼び出し例：
    # collector.save_step("part0", current_step, {"sample": sample, ...})
    # 再現性を保持するため
    random.seed(77)
    res_str = f"{args.width}x{args.height}"
    calib_data_dir = "../calibration_data"
    output_root_dir = os.path.join(calib_data_dir, res_str, "raw") # raw（生データ）であることを明示
    os.makedirs(output_root_dir, exist_ok=True)
    clean(output_root_dir)
    print(f" -> キャリブデータ保存先を初期化しました: {output_root_dir}")
        
    #for p_idx, prompt in enumerate(PROMPTS_ANIME):
    for p_idx, prompt in enumerate(PROMPTS):
        seed = random.randint(0, np.iinfo(np.uint32).max)
        steps = random.randint(args.steps_min, args.steps_max)
        run_cmd([sys.executable, "sub_3_collect_calib_data.py",
                 # prompt[0], "--prompt_2", prompt[1],
                 # プロンプトを共通化して使う
                 #prompt[0] + "." + prompt[1],
                 f'"{prompt[0]}"',
                 "--negative_prompt", f'"{get_random_negative_prompt()}"',
                 "--name", args.name,
                 "--width", str(args.width), "--height", str(args.height),
                 "--seed", str(seed),  "--steps", str(steps),
                 "--guidance_scale", str(args.guidance_scale),
                 "--calib_strategy", str(prompt[1]),
                 "--calib_data_dir", output_root_dir])
       

if __name__ == "__main__":
    main()
