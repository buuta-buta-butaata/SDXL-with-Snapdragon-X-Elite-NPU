from utils import get_project_root

MODEL_ROOT_DIR = rf"{get_project_root()}/compiled_models"
TEXT_ENCODER_DIR = r"./compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/text_encoder"
TEXT_ENCODER_2_DIR = r"./compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/text_encoder_2"

UNET_DIR = r"./compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/unet"

VAE_DECODER_DIR = r"./compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/vae_decoder"
VAE_ENCODER_DIR = r"./compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/vae_encoder"

SCHEDULER_DIR = r"./schedulers_config/EulerAncestralDiscreteScheduler"
TOKENIZER_DIR = r"./compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/tokenizer"
TOKENIZER_2_DIR = r"./compiled_models/dreamshaper-xl-lightning-for-Snapdragon-X-Elite/tokenizer_2"
OUTPUT_DIR = r"./outputs"
