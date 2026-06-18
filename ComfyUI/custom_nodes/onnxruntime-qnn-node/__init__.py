from .qnn_base_path_node import QNNBasePathLoader
from .qnn_unet_node import QNNUNetLoader
from .qnn_text_encoder_node import QNNTextEncoderLoader
from .qnn_vae_decoder_node import QNNVAELoader

NODE_CLASS_MAPPINGS = {
    "QNNBasePathLoader": QNNBasePathLoader,
    "QNNUNetLoader": QNNUNetLoader,
    "QNNTextEncoderLoader": QNNTextEncoderLoader,
    "QNNVAELoader": QNNVAELoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QNNBasePathLoader": "🌟 QNN Model Base Path Loader",
    "QNNUNetLoader": "🌟 QNN 5-Part UNet Loader",
    "QNNTextEncoderLoader": "🌟 QNN Text Encoder Loader",
    "QNNVAELoader": "🌟 QNN VAE Decoder Loader"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
