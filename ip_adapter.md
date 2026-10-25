# [WIP] IP Adapter

[Experimental] Run SDXL pipeline with IP Adapter on NPU.

[https://huggingface.co/Buuta/ip-adapter-sdxl-for-Snapdragon-X-Elite](https://huggingface.co/Buuta/ip-adapter-sdxl-for-Snapdragon-X-Elite)

## Running the image generation with IP Adapter

### Setup

#### Download the Model

```bash
sdxlite-cli.bat setup --ip_adapter
```

### Execution Examples

#### IP Adapter Plus

**Input:**  
<img src="images/ip_adapter/buuta512.png" width="50%" />

```bash
sdxlite-cli.bat txt2img --prompt "cat" --quantized_model --ip_adapter_image "buuta512.png" --ip_adapter_weight 0.65
```
**Output(Cat):**  
<img src="images/ip_adapter/Cat.png" width="50%" />

```bash
sdxlite-cli.bat txt2img --prompt "dog" --quantized_model --ip_adapter_image "buuta512.png" --ip_adapter_weight 0.65
```
**Output(Dog):**  
<img src="images/ip_adapter/Dog.png" width="50%" />

##### Key parameters
```
--ip_adapter_image "buuta512.png"
--ip_adapter_weight 0.65
```

---

#### IP Adapter Plus (Face)

**Input:**  
<img src="images/ip_adapter/nobunaga_face.png" width="50%" />
```bash
sdxlite-cli.bat txt2img --prompt "A photo of a chef" --quantized_model --ip_adapter_face_image "nobunaga_face.png" --ip_adapter_face_weight 1
```
**Output(Chef Nobunaga):**  
<img src="images/ip_adapter/ChefNobunaga.png" width="50%" />

##### Key parameters
```
--ip_adapter_face_image "nobunaga_face.png"
--ip_adapter_face_weight 1
```

---

**Input:**  
<img src="images/ip_adapter/Johann_Sebastian_Bach.png" width="50%" />
```bash
sdxlite-cli.bat txt2img --prompt "A photo of a chef" --quantized_model --ip_adapter_face_image "Johann_Sebastian_Bach.png" --ip_adapter_face_weight 0.8
```
**Output(Chef Bach):**  
<img src="images/ip_adapter/ChefBach.png" width="50%" />

##### Key parameters
```
--ip_adapter_face_image "Johann_Sebastian_Bach.png"
--ip_adapter_face_weight 0.8
```

---

