import logging
import numpy as np

from .vae_encoder import VAEEncoder
from .unet import UNet
from .pipeline import SDXLPipeline, SCALING_FACTOR
from . import image

logger = logging.getLogger(__name__)


class SDXLImg2ImgPipeline(SDXLPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.vae_encoder = None
        self.unet = UNetWrapper(config)


class UNetWrapper(UNet):
    def __init__(self, config):
        self.config = config
        super().__init__(config)

        self.vae_encoder = VAEEncoder(self.config)

    def get_init_latents(self, scheduler, config):
        preprocessed_image = image.preprocess_image(config.input_file)

        # 1. NPUでVAE Encoderを実行
        vae_output = self.vae_encoder.encode(preprocessed_image)
    
        # 2. 前半の4チャンネル（平均値 mean）のみをスライスして取り出す
        init_latents = vae_output[:, :4, :, :]
    
        # 3. SDXLのVAE固有のスケーリング係数を掛ける
        scaling_factor = SCALING_FACTOR
        init_latents = init_latents * scaling_factor
        
        return init_latents

    def prepare_latents(self, init_latents, scheduler, config):
        # 1. スケジューラ（DPMSolverMultiStepなど）に総ステップ数をセット
        scheduler.set_timesteps(config.steps)
    
        # 2. Denoising Strength に応じて、開始するステップの位置（オフセット）を計算
        # 例: 30ステップ * 0.6 = 18ステップ分のノイズを混ぜる
        offset = int(config.steps * config.denoising_strength)
        start_step = config.steps - offset
    
        # 逆拡散に使用するタイムステップの配列を切り出す
        timesteps = scheduler.timesteps[start_step:]
    
        # 3. 開始地点のタイムステップ（一番ノイズが濃い状態）を取得
        # 例: timesteps[0] が 600 とか 700 などの大きな数値になります
        start_timestep = [timesteps[0]]
    
        # 4. 元画像のLatentと同じ形状の「ランダムノイズ」を生成
        noise = scheduler.generate_noise_latents(config)

        # 5. スケジューラを使って、元画像のLatentにノイズを混ぜる（前方拡散）
        noisy_latents = scheduler.add_noise(init_latents, noise, start_timestep)
    
        # 戻り値: ノイズが混ざったLatentと、ループで使うタイムステップのリスト
        return noisy_latents, timesteps


