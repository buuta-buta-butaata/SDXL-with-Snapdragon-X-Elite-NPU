# [WIP] ControlNet

[Experimental] Run SDXL pipeline with ControlNet on NPU.

[https://huggingface.co/Buuta/controlnet-union-sdxl-for-Snapdragon-X-Elite](https://huggingface.co/Buuta/controlnet-union-sdxl-for-Snapdragon-X-Elite)

## Running the image generation with ControlNet

### Mode

0: openpose
1: depth
2: hed/pidi/scribble/ted
3: canny/lineart/anime_lineart/mlsd
4: normal
5: segment
6: tile
7: repaint

### Execution Examples

#### mode 0(OpenPose)
**Input:**  
<img src="images/jumping_mode0.png" width="50%" />
```bash
sdxlite-cli.bat txt2img --quantized_model --prompt "ninja" --control_image "jumping_mode0.png" --control_mode 0 --control_guidance_end 0.3 --steps 5 --control_scale 1 --use_controlnet
```
**Output:**  
<img src="images/output_sdxl_npu_20260905190729.png" width="50%" />

##### Key parameters
```
--control_image "pose image"
--control_mode 0
--control_guidance_end 0.3-0.5
--control_scale 0.8-1.0
--use_controlnet
```

---

#### mode 2(Scribble)
**Input:**  
<img src="images/pose_mode2.png" width="50%" />
```bash
sdxlite-cli.bat txt2img --quantized_model --prompt "ninja" --control_image "pose_mode2.png" --control_mode 2 --control_guidance_end 1 --steps 5 --control_scale 0.4 --use_controlnet
```
**Output:**  
<img src="images/output_sdxl_npu_20260907190003.png" width="50%" />

##### Key parameters
```
--control_image "scribble image"
--control_mode 2
--control_guidance_end 0.6-1.0
--control_scale 0.2-0.4
--use_controlnet
```

---

#### mode 3(LineArt)
**Input:**  
<img src="images/robot_mode3.png" width="50%" />
```bash
sdxlite-cli.bat txt2img --quantized_model --prompt "red robot" --control_image "robot_lineart.png" --control_mode 3 --control_guidance_end 1 --steps 5 --control_scale 0.4 --use_controlnet
```
**Output:**  
<img src="images/output_sdxl_npu_20260907222558.png" width="50%" />

##### Key parameters
```
--control_image "lineart image"
--control_mode 3
--control_guidance_end 0.8-1.0
--control_scale 0.2-0.4
--use_controlnet
```

---

#### mode 6(Tile)
**Input:**  
<img src="images/output_sdxl_npu_20260907222558.png" width="50%" />
```bash
sdxlite-cli.bat img2img --quantized_model --prompt "red robot" --input_image "output_sdxl_npu_20260907222558.png" --control_mode 6 --control_guidance_end 0.8 --steps 10 --denoising_strength 0.5 --control_scale 0.2 --use_controlnet
```
**Output:**  
<img src="images/output_sdxl_npu_20260908180958.png" width="100%" />

##### Key parameters
```
--control_mode 6
--control_guidance_end 0.6-0.8
--denoising_strength 0.2-0.5
--control_scale 0.2-0.4
--use_controlnet
```
