# ComfyUI Custom Nodes: OnnxRuntime-QNN-Nodes

![Custom nodes screenshot](/custom_nodes_screenshot.png)

## What is this?

This is a set of ComfyUI custom nodes specifically engineered to leverage Qualcomm SoCs for hardware-accelerated image generation. 

Currently, the pipeline is fully pre-compiled and configured out-of-the-box for the following specific environment:
* **SoC**: Snapdragon X Elite
* **Processor**: Hexagon NPU
* **Model**: SDXL Models

### 💡 Tech Note
While the provided models are strictly optimized for NPU execution, these nodes are built on top of the **ONNX Runtime QNN Execution Provider**. With minor code modifications, the underlying backend can be pointed to the GPU or CPU. Furthermore, the custom nodes themselves are architecture-agnostic; as long as you compile and provide the compatible weights, this extension can be utilized across various Qualcomm SoCs.

---

## Installation & Setup (For Experienced ComfyUI Users)

This guide assumes you already have a working, standalone ComfyUI environment.

### 1. Install the Custom Node
Copy the `onnxruntime-qnn-nodes` directory located inside this repository's `ComfyUI/custom_nodes/` path, and paste it directly into your own ComfyUI's native `custom_nodes` folder:
`Your_ComfyUI_Path/custom_nodes/onnxruntime-qnn-nodes`

### 2. Load the Workflow
We have included a pre-configured workflow file (`OnnxRuntime-QNN-Workflow.json`) in this repository. Simply drag and drop this JSON file straight into your ComfyUI browser interface to load the pipeline.

### 3. Configure the Model Path
Locate the **📁 QNN Model Base Path Specifier** node within the loaded workflow. In the text field, input the **absolute full path** pointing to your downloaded model folder. 
*(Make sure the specified directory contains the subfolders for `text_encoder`, `text_encoder_2`, and the split `UNet` models).*


## How to Use & Step-by-Step Guide

### 1. Download the Pre-compiled Models
Please refer to the main repository guide: [readme 2-download-the-model](readme.md#2-download-the-model)

### 2. Set Up the ComfyUI Environment
The environment version verified to work with these custom nodes is:
* **Python**: 3.13.3
* **ComfyUI**: v0.25.0

At this stage, we will pre-install the explicit dependencies required for the QNN nodes (`onnxruntime-qnn==2.1.1` and `onnxruntime==1.24.4`). 
*Tip: Installing the CPU-targeted PyTorch wheel beforehand ensures a smoother installation of ComfyUI's core dependencies.*

```bash
# Clone the verified version of ComfyUI
git clone --branch v0.25.0 --depth 1 https://github.com/Comfy-Org/ComfyUI
cd ComfyUI

# Install pre-requisites and QNN dependencies
pip install setuptools wheel
pip install onnxruntime-qnn==2.1.1
pip install onnxruntime==1.24.4

# Install CPU-targeted PyTorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install default ComfyUI dependencies
pip install -r requirements.txt
```

### 3. Deploy the Custom Nodes
Clone this repository and move the `onnxruntime-qnn-nodes` folder into your ComfyUI's `custom_nodes` directory.

```bash
# Clone this repository
git clone https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU.git

# (Move / Copy the ComfyUI/custom_nodes/onnxruntime-qnn-nodes directory into your local ComfyUI installation)
```

### 4. Running ComfyUI
Launch ComfyUI using the CPU flags to prevent initialization errors:
```bash
python main.py --cpu --auto-launch --disable-api-nodes
```

Once the web interface automatically opens in your browser, drag and drop the `OnnxRuntime-QNN-Workflow.json` file straight into the ComfyUI grid. 

Lastly, find the **📁 QNN Model Base Path Specifier** node and input the **absolute full path** to your model root directory (the parent folder containing `text_encoder`, `text_encoder_2`, and the split `UNet` folders).

### 📊 Workflow Node Architecture & Hardware Allocation

To help understand how the pipeline is orchestrated, here is a breakdown of the custom nodes versus native ComfyUI nodes used in `OnnxRuntime-QNN-Workflow.json`, along with their actual hardware execution targets.

| Node Name | Type | Hardware Target | Technical Detail |
| :--- | :--- | :--- | :--- |
| **📁 QNN Model Base Path Specifier** | Custom | — (Path Config) | Sets the absolute path for the compiled model directory. |
| **🌟 QNN 5-Part UNet Loader** | Custom | **NPU** (QNN EP) | Sequentially loads the 5 UNet fragments. |
| **🌟 QNN Text Encoder Loader** | Custom | **NPU** (QNN EP) | Loads the Text Encoder models into the NPU. |
| **🌟 QNN VAE Decoder Loader** | Custom | **NPU** (QNN EP) | Loads the VAE Decoder model into the NPU. |
| **🌟 QNN SDXL Latent Generator** | Custom | **CPU** | Generates the initial latent noise. Built as a custom node to hardcode the resolution, preventing dynamic resizing accidents that crash the QNN pipeline. |
| **CLIP Text Encode (Prompt)** | Native | **NPU** (QNN EP) | *Note: While it is a native ComfyUI node, the actual text encoding backend is internally intercepted and routed to the NPU.* |
| **VAE Decode** | Native | **NPU** (QNN EP) | *Note: This node serves as a structural bridge to connect the pipeline to the preview node; the heavy VAE decoding itself runs natively on the NPU.* |
| **KSampler** | Native | **CPU & NPU** | **(Inference Bottleneck)** The heavy UNet computation itself is successfully routed to the **NPU**. However, the KSampler orchestrates the loops entirely on the **CPU**, causing massive data transfer overhead during the denoising steps. |
| **Preview Image** | Native | **CPU** | Displays the final rendered image. |

*(By reviewing this table, you can see why the **KSampler** introduces a significant CPU bottleneck, as explained in the Known Issues section below).*

---

## 🧠 Developer's Reflection: First Impressions of ComfyUI

This project marked my very first time diving into the ComfyUI ecosystem. To my astonishment, I was able to build and spin up a working set of custom nodes in **just a few hours**. 

Granted, the core UNet splitting logic was already fully written, and I had immense assistance from **Google Gemini** to glue everything together. Even so, the sheer extensibility and developer-friendliness of ComfyUI's architecture is absolutely phenomenal.

Through this experimentation, I truly came to appreciate the core advantages of ComfyUI:
* **Instant Visual Diagnostics**: It allows you to track the execution pipeline visually. For example, it became immediately obvious which specific node was dragging down performance and eating up processing time.
* **A Massive Global Ecosystem**: The sheer volume of community-driven nodes and pre-built assets available is incredible.

While the current QNN integration faces severe compatibility bottlenecks with native modules, the ability to visually dissect and understand the execution flow proves that building atop ComfyUI has immense future potential.

---

## Known Issues & Current Limitations

#### 🔄 UNet Memory Flush Is Triggered Inside the VAE Decoder Node
The memory cleanup/model flushing for the UNet fragments should ideally be handled inside the UNet execution nodes. However, due to complications with how ComfyUI handles custom node hooks, it is currently triggered awkwardly inside the VAE Decoder node instead. If anyone knows a cleaner way to implement this hook, contributions are very welcome!

#### 📉 Unstable Inference Speeds (Degradation on Subsequent Runs)
Inference speeds are inconsistent. For example, the first generation hits around ~5.90 s/it, but the second run drops to ~8.08 s/it. The overall processing speed is slower than the native Python pipeline script, seemingly caused by heavy CPU overhead spikes inside the KSampler execution block.

#### 🧩 Heavy Incompatibility with Native ComfyUI Ecosystem
ComfyUI is deeply integrated assuming `torch` acceleration backends. Because these custom nodes are bypassing the standard torch pipeline to drive the ONNX Runtime under the hood, they are incompatible with almost all standard native nodes (ControlNet, IP-Adapter, LoRAs, etc.). 

* *Developer's Note: Honestly, I’m questioning whether there's an actual benefit to using ComfyUI under these constraints.(´・ω・｀) However, it would be extremely interesting if we could find a way to hijack and repurpose ComfyUI's massive library of existing assets/nodes for this NPU environment.*
