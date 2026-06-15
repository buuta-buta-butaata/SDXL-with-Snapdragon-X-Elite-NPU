# 本プロジェクトの目的

Snapdragon X EliteのNPUを用いて、SDXL系のモデルを動かすこと(達成済み)

## 2026/06/12時点の状況
- fp16のモデルは動いた(参考記録 1024x1024の画像1枚を20stepsで5.48s/it、生成にかかった時間138.028秒、RAM7.72GB使用)
- 量子化はまだ途中(今のところ1.6s/it、RAMの使用量が4GB程度に改善可能なことがわかった、もうちょいいけるかも)
- スクリプトのできは雑
- (もう動いちゃったので、やる気がなくなってきた)

建前としては、NPUで動かせることの概念実証がメインなので、動きゃいいんだよの精神で進めた結果がこれです。
本音としては、途中から面倒になっただけです。

# モデルの動かし方

## 前提条件

- Snapdragon X EliteでRAM16GB以上
- Windows11

RAMは実行時点で空きが8GBあればうれしいですが、仮想メモリが十分確保できれば動きます。
少しスクリプトを修正すればLinuxでも動くとは思いますが、現状はWindowsだけ。

申し訳ないのですが、以降はpythonをある程度動かせる人向けに説明します。

簡単に動かすスクリプトを用意しました。
とりあえず動く、雑なものです。
(雑に見えないような部分はGeminiが作ってくれました)

## 準備
### 1.ここにあるスクリプトのダウンロード
gitを使うとか、zipなどでまとめてダウンロードしてください。

### 2.モデルのダウンロード
[dreamshaper-xl-lightning-for-Snapdragon-X-Elite](https://huggingface.co/Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/tree/main)

↑にあるモデルをcompiled_models\dreamshaper-xl-lightning-for-Snapdragon-X-Eliteディレクトリに入れてください。

あるいは、ダウンロード用のスクリプトを用意してあるので、そちらを実行してもよいです。

`compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite`ディレクトリ内で、以下を実行してください。

```
pip install -r requirements.txt
python download.py

```

### 3.pythonの準備

バージョンは3.13.X(arm64)なら動くと思います。
requirements.txtを用意したので、ご利用ください。
```
pip install -r requirements.txt
```

## 実行

準備が完了したら、image_gen.batを実行します。
引数でプロンプトを与えればOKです。
lightning系のモデルなので、stepsが6、guidance_scaleが2で動くようにしています。
だいたい1分ちょっとで画像が生成されるはずです。

実行例1:
```
image_gen.bat "A beautiful cyberpunk city, neon lights, high resolution, 8k, highly detailed"
```

結果1:
![A beautiful cyberpunk city, neon lights, high resolution, 8k, highly detailed](/output_sdxl_npu20260612234103.png)

実行例2:
```
image_gen.bat "lion"
```

結果2:
![lion](/output_sdxl_npu20260613005411.png)

ちなみに
```
image_gen.bat --help
```
で、やる気のないオプションの説明が表示されます。

動作確認した環境:
- Snapdragon X Elite (RAM 16GB)
- Windows11
- python 3.13.3(arm64)

# どうやって実現したか

シンプルにUNetを分割しただけです。(5分割)
あとは、モデルをプリコンパイルすることでRAMの使用量や処理速度を改善しています。
今回の成果はだいたいQualcomm AI Hub WorkbenchとGoogle Geminiのおかげです。

UNetを分割するスクリプトもついでに置いてあるので、試してみたいという奇特な人がいればぜひ。

分割するスクリプトの使い方:
1. まずは分割したいモデル(fp16のsafetensorsファイル)を用意
1. **scriptsディレクトリ内**で、0_run_pipeline.pyを引数を指定して実行

実行例:
```
python 0_run_pipeline.py --name dreamshaper --model_path ..\safetensors\dreamshaperXL_lightningDPMSDE.safetensors
```
`--name`はモデルの識別用なので、好きな名前をつけてよいです。`--model_path`は適切に設定してください。 
ざっくりこれで動きます。split_modelsディレクトリが作られるので、そちら中に分割されたUNetのonnxファイルが出力されます。

プリコンパイルする部分のスクリプトは、Qualcommが用意したクラウド上にリクエストを投げてプリコンパイルしているので、サーバーに負荷をかけないようにコメントアウトや削除をしています。


