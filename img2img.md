# [WIP] Image-to-Image

[Experimental] Run SDXL img2img pipeline on NPU.

## Running the Image-to-Image Generation

### Execution Examples

**Input:**
![White tiger](/white_tiger.png)
```bash
sdxlite-cli.bat img2img --quantized_model --prompt "lion" --input_image "white_tiger.png" --denoising_strength 0.7 --steps 8
```
**Output:**
![White tiger to Lion](/output_sdxl_npu_20260907230507.png)

##### Key parameters
```
--denoising_strength 0.1-0.9
```

#### Inpaint

**Input:**
![White tiger](/white_tiger.png)
![Mask](/mask.png)
```bash
sdxlite-cli.bat img2img --quantized_model --input_image "white_tiger.png" --mask_image "mask.png" --prompt "red eyes" --denoising_strength 0.8 --steps 8
```
**Output:**
![Red eyes white tiger](/output_sdxl_npu_20260907225138.png)

##### Key parameters
```
--denoising_strength 0.8-0.99
```

#### Outpaint

**Input:**
![White tiger](/white_tiger.png)
```bash
sdxlite-cli.bat img2img --quantized_model --input_image "white_tiger.png" --prompt "white tiger" --steps 10 --denoising_strength 0.9 --outpaint_direction bottom --outpaint_ratio 0.2
```
**Output:**
![Outpaint white tiger](/output_sdxl_npu_20260908004414.png)

##### Key parameters
```
--outpaint_direction bottom or top or left or right
--denoising_strength 0.8-0.99
--outpaint_ratio 0.1-0.2
```

#### hires_fix
**Input:**
![White tiger](/white_tiger.png)
```bash
sdxlite-cli.bat img2img --quantized_model --prompt "white tiger" --input_image "white_tiger.png" --hires_fix
```
**Output:**
![Hires. white tiger](/output_sdxl_npu_20260907231224.png)

##### Key parameters
```
--hires_fix
```
