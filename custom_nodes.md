# ComfyUI用カスタムノード OnnxRuntime-QNN-Nodes

![Custom nodes screenshot](/custom_nodes_screenshot.png)

## これは何？

Qualcomm製SoC向けに作成したComfyUI用のカスタムノードです。
現状は、用意したモデルをSnapdragon X EliteのNPU向けにプリコンパイルしているため
- Snapdragon X Elite
- NPU
- SDXLモデル
という条件で動きます。

### memo

用意したモデルの都合上、NPUで動かすように実装していますが、
ONNX Runtime QNN Execution Providerを利用しているため、少しコードを書き換えればGPU、CPUで動かすことも可能です。
カスタムノード自体はSoCに依存した形式になっていないため、モデルを用意すれば別のQualcomm製SoCでもこのカスタムノードを利用することは可能です。

## ComfyUIを自力で実行できる方向けの説明

本リポジトリにあるComfyUI/custom_nodes内のディレクトリonnxruntime-qnn-nodesを、ComfyUI本体のcustom_nodesディレクトリに配置してください。
ワークフロー(OnnxRuntime-QNN-Workflow.json)を用意したので、ComfyUIの画面にドラッグアンドドロップすれば使えるようになります。
あとは、📁 QNN Model Base Path Specifierノードにダウンロードしたモデルのディレクトリへのフルパスを与えてください。
text_encoder、text_encoder_2、UNetなどのディレクトリがあるところを指定すればOKです。

## 使い方

### モデルのダウンロード

[readme 2-download-the-model](readme.md#2-download-the-model)を参考にしてください

### ComfyUIの用意

動作確認済みのバージョンは
- python 3.13.3
- ComfyUI 0.25.0

このタイミングで、本カスタムノードの利用に必要な依存関係(onnxruntime-qnn==2.1.1、onnxruntime==1.24.4)をインストールしてしまいます。
また、事前にCPU向けのPyTorchをインストールしておくとComfyUIの依存関係(requirements.txt)をスムーズにインストールできます。

```
git clone --branch v0.25.0 --depth 1  https://github.com/Comfy-Org/ComfyUI
cd ComfyUI
pip install setuptools wheel
pip install onnxruntime-qnn==2.1.1
pip install onnxruntime==1.24.4
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### カスタムノードの配置

本リポジトリにあるComfyUI/custom_nodes内のディレクトリonnxruntime-qnn-nodesを、ComfyUI本体のcustom_nodesディレクトリに配置してください。

```
git clone https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU.git
```

### 実行の仕方

ComfyUIの実行:
```
python main.py --cpu --auto-launch --disable-api-nodes
```

自動でブラウザが立ち上がってComfyUIの画面が表示されるはずなので、そうしたらOnnxRuntime-QNN-Workflow.jsonをComfyUIの画面にドラッグアンドドロップするのが楽です。
あとは、📁 QNN Model Base Path Specifierノードにダウンロードしたモデルのディレクトリへのフルパスを与えてください。
(text_encoder、text_encoder_2などのディレクトリがあるところ)

## 問題点

#### UNetのモデル解放をVAE Decoderのノードで行ってしまっている

間違いなくUNet側のノードでモデル解放の処理は実施した方がよいのだけど、
hookの仕方がよくわからない(´・ω・｀)

#### 処理速度が安定しない

例えば初回5.90s/it、2回目以降は8.08s/itになって悪化する。
全体的に処理速度が遅くなっている、
KサンプラーでCPUをめっちゃ使ってるのが原因っぽい

#### 既存のComfyUIノードのほとんどが利用できないっぽい？

ComfyUIはCUDAというかtorchを使うことが前提みたいになっているので、
偽装してonnxruntimeを内部で利用している本カスタムノードと相性がよくない。
無理にComfyUIを使う理由ってあるんかな？
ComfyUIが抱える資産で流用できるものってあれば面白いのだけど
