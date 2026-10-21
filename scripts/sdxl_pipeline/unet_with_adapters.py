import logging

import numpy as np
from PIL import Image

from .image_encoder import ImageEncoder
from .ip_adapter import IPAdapter, IPAdapterFace
from .t2i_adapter import T2IAdapter
from . import t2i_adapter_helper as t2i_helper
from . import ip_adapter_helper as ip_helper

from .unet import UNet

logger = logging.getLogger(__name__)
        

def set_adapters(unet, config):
    if config.t2i_adapter_image is not None:
        unet.run_t2i_adapter = run_t2i_adapter
    if config.ip_adapter_image is not None:
        unet.run_ip_adapter = run_ip_adapter

def run_t2i_adapter(config):
    src_image = Image.open(config.t2i_adapter_image).convert("RGB")

    t2i_adapter = T2IAdapter(config)
    t2i_outputs = t2i_adapter.inference(config, src_image, config.t2i_adapter_weight)
    t2i_adapter.free_memory()
    del t2i_adapter

    t2i_adapter_outputs = {
        "down":
        {
            "t2i_adapter_state_1": t2i_outputs["t2i_adapter_state_1"],
            "t2i_adapter_state_2": t2i_outputs["t2i_adapter_state_2"],
            "t2i_adapter_state_3": t2i_outputs["t2i_adapter_state_3"],
        },
        "up1":
        {
            "t2i_adapter_state_4": t2i_outputs["t2i_adapter_state_4"],
        }
    }
    return t2i_adapter_outputs, t2i_helper.get_uncond_data()

def _run_ip_adapter(img_enc, ip_adapter, config, image_path, weight):
    # Image Encoder
    src_image = Image.open(image_path).convert("RGB")

    image_embeds = img_enc.inference(config, src_image)
    img_enc.free_memory()

    # IP Adapter
    ip_outputs = ip_adapter.inference(config, image_embeds, weight)
    ip_adapter.free_memory()
    del ip_adapter

    return ip_outputs

def run_ip_adapter(config):
    img_enc = ImageEncoder(config)

    outputs = ip_helper.get_uncond_data()
    if config.ip_adapter_image is not None:
        ip_adapter = IPAdapter(config)
        ip_outputs = _run_ip_adapter(img_enc, ip_adapter, config, config.ip_adapter_image, config.ip_adapter_weight)
        for x in outputs.keys():
            outputs[x] = ip_outputs[x]

    if config.ip_adapter_face_image is not None:
        ip_adapter_face = IPAdapterFace(config)
        ip_outputs = _run_ip_adapter(img_enc, ip_adapter_face, config, config.ip_adapter_face_image, config.ip_adapter_face_weight)
        for x in outputs.keys():
            for y in outputs[x].keys():
                outputs[x][y] += ip_outputs[x][y]

    del img_enc

    return outputs, ip_helper.get_uncond_data()
