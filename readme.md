# SDXL with Snapdragon X Elite NPU

## Project Objective
The initial goal of this project was to run Stable Diffusion XL (SDXL) models natively on the Snapdragon X Elite NPU using the QNN Execution Provider (**Achieved**).

Having successfully proven the core concept, this project has transitioned from a mere Proof of Concept (PoC) to a production-focused utility. Current development shifts toward maximizing practicality—focusing on memory efficiency, workflow optimizations, and delivering a fast, standalone experience for daily image generation on ARM64 Windows hardware.


## Current Status (As of August 2026)

### 🚀 Running FP16 Models (Fully Operational)
* **Performance (6 steps)**: ~3.92 s/it (Total time: ~34.7 seconds)
* **Performance (20 steps)**: ~3.88 s/it (Total time: ~88.5 seconds)
* **Memory Usage**: Peak RAM optimized down to **5.18 GB**

### 💎 Quantized Models (Fully Operational & Enhanced)
* **Performance (6 steps)**: ~2.00 s/it (Total time: ~20.1 seconds)
* **Performance (20 steps)**: ~1.99 s/it (Total time: ~49.5 seconds)
* **Memory Usage**: Drastically reduced to a peak of **2.78 GB**
* **Supported Resolutions**: Fully supports standard **1024x1024 (1:1)**, as well as **832x1216 (Portrait)** and **1344x768 (Landscape)** outputs via the experimental weight-shared layout options.
* **Quality Improvement**: Quantization accuracy has been vastly improved, with PSNR leaping from the previous 20–40 dB up to **57–70 dB** for near-lossless generation.

### 🧩 ComfyUI Custom Nodes Support (Experimental)
* **Status**: Functional but currently facing optimization challenges (such as inconsistent inference speeds and high CPU overhead). 
* **Note on Development**: Since our current development heavily prioritizes a completely torch-less, standalone CLI architecture to maximize NPU efficiency, updating these ComfyUI nodes is currently a lower priority.
* **Details**: For the implementation guide, setup script, and known limitations, please check out the dedicated documentation here: **[ComfyUI Custom Nodes: OnnxRuntime-QNN-Nodes](custom_nodes.md)**


## Getting Started

### Prerequisites

#### System Requirements
* **SoC**: Snapdragon X Elite (Strictly required, as the project is optimized specifically for this architecture).
* **OS**: Windows 11 (ARM64).
* **Python**: Python 3.13.x (ARM64 Native).
  * *Note: While it should theoretically run on Linux with minor script modifications, this repository currently only supports Windows.*
* **RAM**: 16 GB or higher. 
  * *Required Free RAM at Runtime:*
    * **Quantized Model**: Requires at least **4 GB** of free RAM.
    * **FP16 Model**: Requires at least **7 GB** of free RAM.
    * **Weight-Shared Model**: Requires at least **7 GB** of free RAM (even when quantized).
  * *Note: Having sufficient Virtual Memory (Pagefile) allocated (8 GB or more) is highly recommended to ensure stability.*
* **Storage**: At least **30 GB of free disk space** is highly recommended. 
  * *Reason: The total size of the model files has increased to around 13 GB due to the addition of quantized and weight-shared variants. Additional space is also required for the Windows Pagefile.*

#### Required Skills
* Basic knowledge of running Python scripts on Windows 11.
  * *Note: The following guide assumes you have a baseline understanding of Python environments.*

---

### Setup Instructions

We provide a dedicated CLI tool (`sdxlite-cli`) to automate the environment setup and model management, making it easier than ever to get started.

#### 1. Clone the Repository
Clone this repository to your local machine using Git:
```bash
git clone https://github.com/buuta-buta-butaata/SDXL-with-Snapdragon-X-Elite-NPU.git
cd SDXL-with-Snapdragon-X-Elite-NPU
```

#### 2. Install Dependencies
Ensure all required Python packages are installed before downloading the models:
```bash
pip install -r requirements.txt
```

#### 3. Download the Model & Required Components
Run the built-in setup command to automatically download and configure the required DLLs and pre-compiled model files:

```bash
sdxlite-cli setup
```

*⚠️ **Important Note on Downloading:** The total file size has increased to around **13 GB** due to the newly added model variants. Depending on your network speed, the download may take significant time (expect **10+ minutes**). Please ensure your connection is stable and be patient while the script fetches the assets.*

##### What happens under the hood:
The script will prompt you to confirm the installation and automatically place all downloaded files into their correct directories (e.g., placing `ortextensions.dll` into the `lib` folder and managing the pre-compiled models).

**Example Console Output:**
```text
(qai_hub) Z:\Temp>sdxlite-cli setup

Downloading ortextensions.dll...
  From: https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Extensions/0.14.0
  To:   Z:\Temp\lib

Do you want to continue? [y/N]: y
```

---

### Running the Text-to-Image Generation

Once the setup is complete, you can generate images by running `sdxlite-cli.bat` and executing the `txt2img` subcommand with your prompt.

Since this project utilizes an SDXL Lightning-based model, the script is pre-configured by default to run with **6 steps** and a **Guidance Scale (CFG) of 2.0**. Thanks to recent performance optimizations, image generation is now blazing fast—taking roughly **40 seconds** with the FP16 model and only **20 seconds** with the Quantized model.

💡 **Pro-Tip for Maximum Speed:**
Setting the Guidance Scale to `1.0` can speed up inference even further, though it may slightly degrade image quality depending on the prompt. To help you experiment with different setups, we provide various CLI arguments. Run `sdxlite-cli txt2img --help` to see the full list of available options. We also list some highly recommended configuration examples in the next section.

⚠️ **Important Notice:**
This repository **does not** include a Safety Checker (NSFW filter). The model may generate Not Safe For Work (NSFW) content depending on the prompt. Please use it responsibly.


#### Execution Examples

You can copy and paste the examples below into your terminal to test different models and configurations. 

##### 1. Running the Default FP16 Model
By default, the script loads the standard FP16 model. This configuration balances high image quality with native precision.
```bash
sdxlite-cli txt2img --prompt "lion"
```
**Output:**
![Lion](/output_sdxl_npu_20260816212957.png)

##### 2. Running the Quantized Model (Ultra-Low RAM)
Pass the `--quantized_model` flag to use the optimized version, which drastically cuts down memory usage to just 2.78 GB. 
*Note: In the example below, we lock the `--seed` to match the exact noise configuration of the FP16 example above, allowing you to easily compare visual fidelity and see how near-lossless the generation is.*
```bash
sdxlite-cli txt2img --prompt "lion" --quantized_model --seed 3238569144
```
**Output:**
![Quantized lion](/output_sdxl_npu_20260816213036.png)

##### 3. Generating Different Aspect Ratios (Experimental Weight-Shared Model)
Standard models in this repository are locked to a 1:1 square aspect ratio (1024x1024). This limitation exists because **the model graphs must be pre-compiled (into hardware-specific formats) to achieve optimal acceleration on the Snapdragon X Elite NPU.** 

To circumvent this constraint and allow flexible dimensions like Portrait or Landscape without duplicating massive model weights, this project introduces a **Weight-Shared Model** setup. Instead of cloning a full 2.4 GB+ package for every layout, we use separate resolution-specific ONNX graphs that all link back to a single, shared weight binary (`.bin`). This keeps the overall repository footprint remarkably lightweight.

*⚠️ **Technical Limitation & RAM Warning:** Due to a current known issue, weight-shared models consume roughly 1.5× to 2× more RAM during execution. To prevent out-of-memory errors on 16 GB RAM systems, **you must pass both `--quantized_model` and `--weight_shared_model` together.** Running this feature in FP16 mode is currently not supported on standard hardware.*

* **Portrait Mode (832x1216):**
```bash
sdxlite-cli txt2img --prompt "epic fantasy art, 1boy, dynamic pose, mage, casting magic, blue robes, wizard hat." --quantized_model --weight_shared_model --layout Portrait
```
**Output:**
![epic fantasy art](/output_sdxl_npu_20260816221504.png)

* **Landscape Mode (1344x768):**
```bash
sdxlite-cli txt2img --prompt "A beautiful cyberpunk city, neon lights, high resolution, 8k, highly detailed" --quantized_model --weight_shared_model --layout Landscape
```
**Output:**
![A beautiful cyberpunk city](/output_sdxl_npu_20260816234058.png)

##### 4. Batch Generation (Multiple Images or Multiple Prompts)
You can automate the generation of multiple images or process an entire list of different prompts sequentially.

* **Generate Multiple Images from a Single Prompt:**
  Use the `--batch_count` argument to specify how many images you want to generate.
```bash
sdxlite-cli txt2img --prompt "A very soft and faded pastel watercolor illustration of a dreamy cloud castle, pale pinks, light mint, faded lavender, low contrast, misty and washed out." --quantized_model --batch_count 4
```

* **Process Multiple Prompts from a Text File:**
  Pass a text file via `--input_file`. The file should contain one prompt per line (see [sample_prompts.txt](sample_prompts.txt) for reference).
```bash
sdxlite-cli txt2img --input_file "sample_prompts.txt" --quantized_model
```

##### 5. Ultra-Fast Generation (CFG 1.0 Tuning)
Setting the Guidance Scale to `--cfg 1` minimizes inference overhead for maximum speed. However, with default settings, CFG 1.0 often results in undersaturated, washed-out images (even if you explicitly include keywords like "vivid colors" in your prompt). 

To counteract this and maintain vibrant image quality at ultra-high speeds, we recommend either of the following advanced configurations:

* **Option A: Change Beta Schedule to Linear**
  Keep the default scheduler but explicitly switch its internal beta schedule to `linear` via JSON-formatted argument strings:
```bash
sdxlite-cli txt2img --prompt "anime illustration, robot, vivid color" --quantized_model --cfg 1 --scheduler 0 --scheduler_config "{\"beta_schedule\": \"linear\"}"
```

* **Option B: Switch to DPM-Solver Single-Step Scheduler**
  Alternatively, you can switch the entire scheduler to `DPMSolverSinglestepScheduler` (index `2`), which inherently handles low-CFG generations much better:
```bash
sdxlite-cli txt2img --prompt "anime illustration, robot, vivid color" --quantized_model --cfg 1 --scheduler 2
```


## Technical Architecture: How It Works

To achieve optimal inference speed and manage memory footprints on the Snapdragon X Elite NPU, the models must be pre-compiled via Qualcomm AI Hub tools. 

#### The Evolution of Our Architecture: Why UNet is Subdivided
Originally, the massive SDXL UNet graph could not be compiled in one piece due to a hard 2 GB Google Protobuf limitation in earlier toolchains. To circumvent this, **this project pioneered dividing the UNet into 5 smaller sub-models** and sequential bridging during runtime.

While a June 2026 Qualcomm AI Hub update resolved the strict 2 GB Protobuf constraint—allowing default FP16 models to compile as a single, undivided file—the multi-chunk approach remains highly crucial for the following architectural benefits:

1. **Circumventing Quantization Timeouts:** 
   Attempting to post-training quantize the entire, undivided UNet graph frequently hits a 1.5-hour execution timeout or crashes due to the sheer scale of the calculations. Sub-modelling acts as a robust workaround to compile enhanced quantized models reliably.
2. **Memory Segment Optimizations:**
   By evaluating the UNet block-by-block, we can selectively manage hardware memory allocations on the NPU and RAM more granularly.

### 📐 The 5-Split UNet Strategy
Standard UNet structures are broadly composed of three main components: `down_blocks`, `mid_block`, and `up_blocks`. 

While splitting into 3 parts seemed obvious, the `up_blocks` alone still exceeded the 2 GB threshold. To solve this, we further divided the network and extracted a "Common" base layer shared across the blocks, resulting in the following 5 parts:

1. **Part 0 (Common)**: Shared structural base layer.
2. **Part 1 (Down)**: The down-sampling blocks.
3. **Part 2 (Mid)**: The bottleneck middle block.
4. **Part 3 (Up1)**: First half of the up-sampling blocks.
5. **Part 4 (Up2)**: Second half of the up-sampling blocks.

By breaking the model down this way, every single ONNX file stays safely under 2 GB, allowing the QNN compiler to process them successfully.

---

### ✂️ How to Split the Model Yourself
If you want to replicate this process or experiment with other SDXL models, the splitting scripts are included in this repository.

#### Step-by-Step Splitting Guide:
1. Prepare your target model (an SDXL `fp16` model in `.safetensors` format).
2. Navigate to the **`model_converter` directory** and execute `0_run_pipeline.py` with the appropriate arguments.

#### Example Command:
```bash
python 0_run_pipeline.py --name dreamshaper --model_path ..\safetensors\dreamshaperXL_lightningDPMSDE.safetensors
```
* `--name`: A unique identifier name for your model output (feel free to name it anything).
* `--model_path`: The path pointing to your local `.safetensors` file.

Once execution finishes, a `split_models` directory will be generated containing the 5 split UNet `.onnx` files. The pre-compiled models provided in this project were generated exactly through this method.

⚠️ *Note: The final pre-compilation step requires uploading the models to Qualcomm's cloud compilation servers. To prevent unnecessary server load and respect API guidelines, the automated compilation hooks in these scripts have been safely commented out or removed.*


## Known Issues

### 🚨 Memory (RAM) Spike with Experimental Weight-Shared Models
When running the newly implemented weight-shared models, the hardware system consumes an abnormally large amount of RAM—roughly 1.5× to 2× the model file size—specifically during the UNet inference loop.

#### The Phenomenon & Data Breakdown
As visualized in the task manager's memory footprint, there is a massive disparity between a standard model run and a weight-shared model run, even under identical generation configurations (1024x1024, same seed, and prompt):

* **Left Side (Standard Model):** Normal, expected memory allocations for the Text Encoder, UNet, and VAE.
* **Right Side (Weight-Shared Model):** An abnormal, massive RAM peak during the UNet processing loop (the center peak on the right).

![task manager's memory footprint](/task_manager_memory_footprint.png)

#### Technical Inconsistency (Process-Level vs. System-Level)
Curiously, our internal process profiler reports that process-specific memory usage remains well within acceptable limits:
* **Standard Model Peak RAM (Process-Level):** `2.78 GB`
* **Weight-Shared Model Peak RAM (Process-Level):** `3.20 GB`

While the python process tracking shows a mere **0.42 GB difference**, the overall Windows system memory usage spikes violently. This strongly indicates that the Qualcomm QNN Execution Provider (EP), the NPU driver, or the internal runtime library is allocating unmanaged memory outside of the python process bounds—possibly due to dynamic graph resizing or linked binary buffer allocations. 

* **Impact:** Because of this system-level overhead, running weight-shared models in native FP16 is currently unfeasible on standard 16 GB RAM hardware. **Using the quantized version (`--quantized_model`) is strictly required to act as a buffer and maintain stability.**

---

## Technical Insights & Discussion

Splitting the UNet into 5 parts was originally a workaround for file-size limitations, but it unexpectedly revealed several fascinating architectural advantages:

### 🔬 Granular Quantization Analysis
Because the UNet is segmented into meaningful logical blocks (`down`, `mid`, `up`), it is much easier to isolate which specific parts of the network struggle during quantization.
* Using the Qualcomm AI Hub Workbench, we can monitor Peak Signal-to-Noise Ratio (PSNR) values on a per-block basis. This allows us to see exactly how different prompt characteristics impact the quantization quality of individual sections.

### 💾 Extreme Ultra-Low RAM Execution & 8 GB RAM Compatibility
In earlier versions, achieving ultra-low memory consumption required a hypothetical "Load-on-Demand" sequential pipeline (loading one block, executing it, flushing it from memory, and loading the next). While this approach could theoretically lower the UNet RAM footprint to under 2 GB (potentially allowing execution on extremely constrained 4 GB RAM systems), the constant disk I/O degrades performance down to roughly ~30 s/it—putting immense stress on storage drives.

**The Game Changer:** Thanks to our newly enhanced quantization process, the entire pipeline's peak RAM usage has dropped significantly to just **2.78 GB**. This means that **without the performance-destroying overhead of Load-on-Demand, the quantized model can now run natively and smoothly on standard 8 GB RAM machines**, as long as the OS background overhead is kept reasonably lightweight. 

### 🌐 Portability to Other NPUs
While this project was built and validated specifically for the Snapdragon X Elite, the core methodology—structurally partitioning massive model files to bypass hardware compiler constraints—could likely be applied to NPUs from other manufacturers (Intel, AMD, Apple, etc.), assuming they face similar file-size or memory ceiling limitations.


## FAQ (Frequently Asked Questions)

### Q. Will this run on other SoCs, like the Snapdragon X Plus?
**A.** If the model is supported by the Qualcomm AI Hub Workbench, it should theoretically work if you re-compile it targeting your specific SoC. Feel free to try it out! 

*(An update on development: I have actually successfully run **SD 3.5 Medium** in native FP16 on the NPU—you can check out that repository here: [SD35-with-Snapdragon-X-Elite-NPU](https://github.com/buuta-buta-butaata/SD35-with-Snapdragon-X-Elite-NPU). However, the generation speed for SD 3.5 is currently too slow—taking around 144 seconds for just 8 steps even with the Turbo model, making it unfeasible for daily use on the Snapdragon X Elite NPU. Because of this, my focus has shifted back to unlocking the full potential and daily practicality of SDXL in this repository).*

Additionally, since the Snapdragon X Plus shares the exact same Hexagon Tensor Processor architecture and Model ID as the X Elite, the pre-compiled models provided here *might* actually run on X Plus hardware without any modification.

### Q. Why didn't you target the standard SDXL-base model?
**A.** I came across reports stating that the VAE Decoder outputs become unstable when running the standard SDXL-base in `fp16`. I chose this specific model under the casual assumption that popular, community-tuned models would likely have those stability issues already ironed out.

### Q. What is the actual point of this project? Shouldn't we just use NVIDIA GPUs?
**A.** Honestly? You are still completely right. (´・ω・｀) 
If your goal is just to get high-quality images quickly and effortlessly, buying an NVIDIA GPU or using cloud services like ChatGPT, Gemini, or Midjourney remains the superior and easiest method.

However, the core purpose of this project has evolved. While it started as a mere Proof of Concept (PoC) to see *if* it could run, it has now shifted into an aggressive exploration of **NPU limits and real-world efficiency**. We want to answer one question: *"How far can we push a standalone consumer NPU?"*

With our latest optimized quantized models, the project has achieved genuine everyday practicality. Here is why this setup is uniquely fascinating:
1. **Ultra-Low Resource Footprint:** Image generation now requires a peak of just **2.78 GB of RAM**. As long as you can spare that tiny sliver of memory, the entire process runs completely hidden on the NPU.
2. **True Multitasking via Background Batching:** Because the GPU and CPU compute resources are left virtually untouched (with CPU usage averaging around 1% just for basic scheduling and file saving), you can use our newly added **Batch Mode** to quietly generate hundreds of images in the background while heavy-gaming, watching 4K videos, or coding without a single frame drop or stutter.

It gives us a tangible glimpse into a future where your hardware's rendering and AI processing workloads are seamlessly partitioned. But beyond that? It is still driven by pure technical curiosity and the joy of hardware hacking.

---

## Project Contributors & Context
As you might have noticed from the phrasing throughout this document, the "We" behind this project consists entirely of:
* **The Developer** (Human)
* **Google Gemini** (AI Collaborator)

This entire repository was built, debugged, and documented through a tag-team effort between one human developer pushing the hardware to its limits and an AI assistant helping to glue the architecture together. 

This project originally started as an experimental exploration into edge AI on ARM64 Windows hardware. As the application evolved from a simple Proof of Concept into a fully-featured, production-focused CLI tool, its codebase was iteratively refactored to achieve higher maintainability, performance, and memory efficiency.

---

## Acknowledgements
* **Google Gemini:** Generously assisted throughout the lifecycle of this project—providing critical pair-programming support, structuring complex pipeline components, and helping refine all console messages and English documentation into a polished, professional format.
* **Qualcomm AI Hub:** For providing the compiler tools, optimization workbench, and runtime execution providers that made native NPU inference possible on the Snapdragon X Elite.
