# 本プロジェクトの目的

Snapdragon X EliteのNPUを用いて、SDXL系のモデルを動かすこと(達成済み)

## 2026/06/12時点の状況
- fp16のモデルは動いた(参考記録 1024x1024の画像1枚を20stepsで5.48s/it、生成にかかった時間138.028秒、RAM7.72GB使用)
- 量子化はまだ途中(今のところ1.6s/it、RAMの使用量が4GB程度に改善可能なことがわかった、もうちょいいけるかも)
- スクリプトのできは雑(たまにあるきれいな部分はGoogle Geminiが書いてくれた部分)
- (もう動いちゃったので、やる気がなくなってきた)

建前としては、NPUで動かせることの概念実証がメインなので、動きゃいいんだよの精神で進めた結果がこれです。  
本音としては、途中から面倒になっただけです。

# モデルの動かし方

## 前提条件

### 必要な環境

- Snapdragon X EliteでRAM16GB以上
- Windows11
- Python 3.13.3(Arm64)

Snapdragon X Elite用に最適化しているため、SoCは必須条件です。  
RAMは実行時点で空きが8GBあればうれしいですが、仮想メモリが十分(8GB以上)確保できれば動きます。  
少しスクリプトを修正すればLinuxでも動くとは思いますが、現状はWindowsだけ。  
Pythonのバージョンは3.13.X(Arm64)なら動くと思います。

### 必要な知識・技能

- Pythonをwindows11上で動かせること

申し訳ないのですが、以降はPythonをある程度動かせる人向けに説明します。

簡単に動かすスクリプトを用意しました。  
とりあえず動く、雑なものです。

## 準備
### 1.ここにあるスクリプトのダウンロード
gitを使うなどしてダウンロードしてください。

コマンド例:
`
git clone https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU.git
`

### 2.モデルのダウンロード
ブラウザかPythonのスクリプトを利用してダウンロードしてください。

#### ブラウザを使う場合

[dreamshaper-xl-lightning-for-Snapdragon-X-Elite](https://huggingface.co/Buuta/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/tree/main)

上記のモデルを`compiled_models\dreamshaper-xl-lightning-for-Snapdragon-X-Elite`ディレクトリに入れてください。

#### Pythonを使う場合

ダウンロード用のスクリプトを用意してあるので、そちらを実行してください。

`compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite`ディレクトリ内で、以下を実行してください。

```
pip install -r requirements.txt
python download.py
```

### 3.Pythonの準備

`requirements.txt`を用意したので、ご利用ください。
```
pip install -r requirements.txt
```

## 実行

準備が完了したら、`image_gen.bat`を実行します。  
引数でプロンプトを与えればOKです。  
lightning系のモデルなので、stepsが6、guidance_scaleが2で動くようにしています。  
だいたい1分ちょっとで画像が生成されるはずです。

### 注意！

SafetyCheckerが実装されていないので、NSFWな画像が出力される可能性があります。  
取り扱いにはご注意ください。

## 実行例

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

# 問題点・課題

## NPU使用率が100%に達していない

NPU使用率が80%程度におさまってしまっている状態。  
RAM使用効率のよい量子化済みのモデルを動かすと、ちゃんと100%使えているので、たぶんRAMがボトルネック。

## 一部でCPUでのtranspose処理が必要

プリコンパイル時に、なぜか出力の一部がNCHWからNHWCに変化。  
そのため、NCHWに戻すためにCPUでtranspose処理を行っているが、これは本来余分な処理のはず。  
どうやってプリコンパイル時の変化をなくすのかは謎。

## 画像の解像度が固定

1024x1024のように画像のサイズを固定した状態にしないとプリコンパイルができないから、対処不能。

ただし、Qualcomm AI Hub Workbenchが提供しているweight-sharingの仕組みを使えば、  
固定した複数の画像サイズ(1024x1024、832x1216、1344x768など)で出力することは可能。  
手間はかかるけども。

# どうやって実現したか

モデルをプリコンパイルすることでRAMの使用量や処理速度を改善しています。  
また、UNetのファイルサイズが大きすぎてプリコンパイル不可能な状態だったので、
シンプルにUNetを分割して、それらをつなげて動くようにしただけです。 

## 分割の仕方

UNetの構造を利用して分割しています。  
UNetはおおまかに次の3つに分かれています。  

- down_blocks
- mid_block
- up_blocks

これで3分割ですが、up_blocksが2GBを超えている状態だったので、さらに分割しており、
また、各ブロックで共通部分があったので、それを抽出してCommonとして次のように5分割しています。

- Part0(Common)
- Part1(Down)
- Part2(Mid)
- Part3(Up1)
- Part4(Up2)

UNetを分割するスクリプトもついでに置いてあるので、試してみたいという奇特な方がいればぜひ。

分割するスクリプトの使い方:
1. まずは分割したいモデル(fp16のsafetensorsファイル)を用意
1. **scriptsディレクトリ内**で、`0_run_pipeline.py`を引数を指定して実行

実行例:
```
python 0_run_pipeline.py --name dreamshaper --model_path ..\safetensors\dreamshaperXL_lightningDPMSDE.safetensors
```
`--name`はモデルの識別用なので、好きな名前をつけてよいです。`--model_path`は適切に設定してください。  
ざっくりこれで動きます。split_modelsディレクトリが作られるので、その中に分割されたUNetのonnxファイルが出力されます。  
この分割されたonnxをプリコンパイルしたものが、今回用意したモデルです。

なお、Qualcommが用意したサーバー上にリクエストを投げてプリコンパイルしているので、プリコンパイルする部分のスクリプトはサーバーに負荷をかけないようにコメントアウトや削除をしています。

# 所感

## UNetを分割した副作用が面白い

### 量子化

意味のある単位(down_blocks, mid_block, up_blocks)で分割したため、
どういったプロンプトを入力すれば、各ブロックの量子化がうまくいくかが見えやすくなっています。

例えば、Qualcomm AI Hub Workbenchでは量子化した際にPSNRの値を確認することができて、
その値からどのブロックの量子化がうまくいっていないかを確認しやすいです。

### 省RAMでの動作

現状では、5分割したものをすべてRAM上に読み込んだ状態で動かしていますが、
使うものを1つずつRAMにおいて、使ったらすぐに解放するようにすれば、UNetをRAM5GB未満の使用で動かせます。  
この仕組みを利用すれば、もしかしたらRAMが8GBの環境でも動かせるように調整できるかもしれません。  
ただしストレージへの負担は莫大になるし、30s/itくらいの速度になっていたので、
20stepsでの画像生成で10分ちかくかかる計算になります。

### 他のNPUでもいける？

Snapdragon X EliteのNPU上で動かしましたが、今回の分割を応用すれば、
他社製のNPUでも同様に動かせる可能性があります。
他のNPUの仕様は知らないため明言できないですが。

# FAQ

## Q. Snapdragon X Plusなどでは動かないの？

A. Qualcomm AI Hub Workbenchで対応しているモデルであれば、
対象のSoC用にモデルをプリコンパイルしなおすことで動かせます。  
動かしたい方はぜひチャレンジしてみてください。  
(私はすでにSDXLモデルをNPU上で動かすことに興味をなくしつつあるので、他のSoC対応はやりません。  
SD3.5-mediumなど他のものを動かしてみたい)   
ただ、X PlusならX Elite用のモデルでも動くかもしれません。  
(Hexagon Tensor ProcessorのVersionやModelIDがX Eliteと一致しているため)

## Q. SDXL-baseを最適化の対象にしなかったのはなぜ？

fp16で動かすとVAE Decoderの出力が安定しないという情報を見たためです。  
人気のモデルならその辺も修正されているやろくらいの安直な理由でモデルを選定しました。

## Q. 本プロジェクトの意義とは？Nvidiaのグラボで動かせばよくない？

A. それは本当にそう(´・ω・｀)  
画像生成にはグラボを使えばいいと思います。  
というか、ChatGPTやらGeminiなどにお願いしちゃうのが一番楽で、質も良いと思います。  
意義を無理やりひねり出すと、NPUである程度大規模なモデルを動かせることの証明にはなったはず。  
あとは、GPUを使わないので、動画見ながら画像生成することができるくらいが利点でしょうか？  
ゲームの描画でGPU使いつつ、ゲーム内のAIでNPU使うみたいなことは実現できそうだねぇくらいの実感はできるかもしれません。  
他に何の役に立つかはわからない(´・ω・｀)   
自己満足。

# 謝辞

今回の成果はQualcomm AI Hub WorkbenchとGoogle Geminiのおかげです。  
すばらしいサービスを提供する両社に感謝を。
