# ComfyUI用カスタムノード OnnxRuntime-QNN-Nodes

まだ途中なので、内容は信用しないでください。
文章もひどいです。
あとで手順を確認しつつ、文章を直していきます。
(今は雑に情報を書きなぐっているだけ)

## これは何？

OnnxRuntime経由でQualcomm製NPUで動くプリコンパイル済みのモデルを実行する、ComfyUI用のカスタムノードです。
用意したモデルはSnapdragon X Elite用ですが、モデルさえ用意すれば、別のQualcomm製SoCでもこのカスタムノードを利用して実行可能です。

## ComfyUIを自力で実行できる方向けの説明

本リポジトリにあるComfyUI/custom_nodes内のディレクトリonnxruntime-qnn-nodesを、ComfyUI本体のcustom_nodesディレクトリに配置してください。
ワークフロー(OnnxRuntime-QNN-Workflow.json)を用意したので、ComfyUIの画面にドラッグアンドドロップすれば使えるようになります。
あとは、📁 QNN Model Base Path Specifierノードにダウンロードしたモデルのディレクトリへのフルパスを与えてください。
text_encoder、text_encoder_2、UNetなどのディレクトリがあるところを指定すればOKです。

## 使い方

### モデルの用意

[readme.md](readme.md)を参考にしてください

### ComfyUIの用意

```
git clone https://github.com/Comfy-Org/ComfyUI --depth 1
cd ComfyUI
pip install setuptools wheel
pip install onnxruntime-qnn==2.1.1
pip install onnxruntime==1.24.4
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

(バージョン指定した方がいいかも、カスタムノード用にrequirements.txt用意してもいいかも、そもそもComfyUIのインストール方法は省略しちゃってもいいのかな)

### カスタムノードの配置

```
git clone https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU.git
```

ComfyUI/custom_nodesの中にあるonnxruntime-qnn-nodesディレクトリを ComfyUI/custom_nodes/ に配置してください。

### 実行の仕方

ComfyUIの実行:
```
python main.py --cpu --auto-launch --disable-api-nodes
```

自動でブラウザが立ち上がるので、そうしたらOnnxRuntime-QNN-Workflow.jsonをComfyUIの画面にドラッグアンドドロップするのが楽です。
あとは、📁 QNN Model Base Path Specifierノードにダウンロードしたモデルのディレクトリへのフルパスを与えてください。
text_encoder、text_encoder_2などのディレクトリがあるところ

(バッチファイル用意して実行してもらった方が楽？)

#### 手動でノードをいじる場合

ComfyUIを開き、📁 QNN Model Base Path Specifier を配置し、ダウンロードしたモデルのディレクトリのフルパスを入力します。
そのノードの出力を、UNet / Text Encoder / VAE Loaderのそれぞれの base_path 入力に繋ぎましょう。

(スクリーンショットがあるとよいかな)

### 問題点

#### UNetのモデル解放をVAE Decoderのノードで行ってしまっている

間違いなくUNet側のノードでモデル解放の処理は実施した方がよいのだけど、
hookの仕方がよくわからない(´・ω・｀)

#### 処理速度が安定しない

例えば初回5.90s/it、2回目以降は8.08s/itになって悪化する。
全体的に処理速度が遅くなっている、
KサンプラーでCPUをめっちゃ使ってるのが原因っぽい

#### 既存のComfyUIノードのほとんどが利用できないっぽい？

CUDAというかtorchを使うことを前提みたいになっているので、
偽装してonnxruntimeを内部で利用している本カスタムノードと相性がよくない。
無理にComfyUIを使う理由ってあるんかな？
ComfyUIが抱える資産で流用できるものってあれば面白いのだけど
