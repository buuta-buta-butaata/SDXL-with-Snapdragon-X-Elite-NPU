import logging
import numpy as np

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

from .vae_decoder import VAEDecoder
from .vae_encoder import VAEEncoder
from .unet import UNet
from .text_processing import TextProcessing
from .pipeline import SDXLPipeline, SCALING_FACTOR
from .pipeline_img2img import SDXLImg2ImgPipeline, SCALING_FACTOR
from . import image

# from utils.tagger import WD14Tagger, Category

logger = logging.getLogger(__name__)


class SDXLTiledImg2ImgPipeline(SDXLImg2ImgPipeline):
    def __init__(self, config):
        self.config = config
        self.text_processing = TextProcessing(self.config.dirs["text_encoder_dir"],
                                              self.config.dirs["tokenizer_dir"],
                                              self.config.dirs["text_encoder_2_dir"],
                                              self.config.dirs["tokenizer_2_dir"],
                                              self.config.use_torch)
        self.unet = UNet(config)
        self.vae_decoder = VAEDecoder(self.config)
        self.vae_encoder = VAEEncoder(self.config)

        # tags_dict_path = rf"{self.config.dirs['tagger_dir']}/selected_tags.csv"
        # model_path = rf"{self.config.dirs['tagger_dir']}/model.onnx"
        # self.tagger = WD14Tagger(model_path, tags_dict_path, [Category.GENERAL])

    def run(self):
        if self.config.seed == -1:
            self.config.seed = np.random.randint(np.iinfo(np.uint32).max, dtype=np.uint32)

        (prompt_embeds,
         pooled_prompt_embeds,
         uncond_embeds,
         uncond_pooled_embeds) = self.text_processing.encode_text(self.config.prompt,
                                                                  self.config.prompt_2,
                                                                  self.config.negative_prompt,
                                                                  self.config.negative_prompt_2,
                                                                  self.config, auto_mem_free=False)

        # del self.text_processing

        with ThreadPoolExecutor(max_workers=2) as executor:
            self._run(self.config, prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled_embeds, executor)

    def _run(self, config, prompt_embeds, pooled_prompt_embeds, uncond_embeds, uncond_pooled_embeds, executor):
        scheduler = self.get_scheduler(config)
        base_image = Image.open(config.input_image).convert("RGB")
        large_image = base_image.resize((int(base_image.size[0] * config.upscale_factor),
                                         int(base_image.size[1] * config.upscale_factor)),
                                        resample=Image.Resampling.LANCZOS)

        original_size = (large_image.height, large_image.width)

        tiles_info = image.split_into_tiles(large_image, tile_size=(config.width, config.height))

        all_count = len(tiles_info)
        tiled_latents_info = []

        self.vae_encoder.load_model(config)
        futures = [
            executor.submit(self.encode, scheduler, config, i, all_count, z_index,
                            tile_img, x, y) for i, (z_index, tile_img, (x, y)) in enumerate(tiles_info, 1)
        ]
        tiled_latents_info = self.run_cancelable_thread(futures, executor)

        print("")
        del self.vae_encoder
        del self.text_processing
            
        self.unet.load_models(self.config)

        processed_latents_info = []
        for i, (z_index, tiled_latents, mask_latents, image_embeds, (x, y), tile_img) in enumerate(tiled_latents_info, 1):
            timesteps = self.set_timesteps(scheduler, config)
            latents = self.prepare_latents(tiled_latents, scheduler, timesteps, config)
            self.config.control_image = tile_img

            print(f"[{i}/{all_count}] Denoising with UNet...")

            combined_embeds = prompt_embeds.copy()
            # TODO: 画像からの情報も使う？
            if image_embeds is not None:
                alpha = 0.7
                if config.cfg != 1:
                    combined_embeds = (image_embeds - uncond_embeds) * alpha + uncond_embeds
                    # combined_embeds = (1 - alpha) * prompt_embeds + alpha * image_embeds
                else:
                    combined_embeds = (1 - alpha) * combined_embeds + alpha * image_embeds
            latents = self.unet.inference(self.config, tiled_latents, latents, mask_latents, scheduler, timesteps,
                                          # prompt_embeds, pooled_prompt_embeds,
                                          combined_embeds, pooled_prompt_embeds,
                                          uncond_embeds, uncond_pooled_embeds, executor,
                                          original_size=original_size,
                                          crops_coords=(y, x),
                                          target_size=(config.height, config.width),
                                          )
        
            latents = latents / SCALING_FACTOR
            processed_latents_info.append((z_index, latents, (x, y)))

        print("")
        del self.unet

        self.vae_decoder.load_model(config)
        futures = [
            executor.submit(self.decode, self.vae_decoder, i, all_count, z_index,
                            latents, x, y) for i, (z_index, latents, (x, y)) in enumerate(processed_latents_info, 1)
        ]
        processed_tiles_info = self.run_cancelable_thread(futures, executor)

        high_res_image = image.merge_tiles(processed_tiles_info, canvas_size=large_image.size)

        # TODO: rating
        # from . import safety_checker
        # tags_dict_path = rf"{self.config.dirs['tagger_dir']}/selected_tags.csv"
        # model_path = rf"{self.config.dirs['tagger_dir']}/model.onnx"
        # checker = safety_checker.RatingChecker(model_path, tags_dict_path)

        # has_nsfw_concept, rating = checker.check_rating(high_res_image, threshold=0.5)
        # if has_nsfw_concept:
        #     print(f"Has NSFW concept. Rating: {rating}")
        
        image.save(high_res_image, **vars(self.config))
    
    def encode(self, scheduler, config, i, all_count, z_index, tile_img, x, y):
        print(f"[{i}/{all_count}] Encoding VAE...")
            
        image_embeds = None
        init_latents, mask_latents = self.get_init_latents(scheduler, config, tile_img)

        # tagger_input = self.tagger.preprocess_image(tile_img, (448, 448))
        # result = self.tagger.get_prompt(tagger_input, threshold=0.7)
        # print(result)
        
        # (prompt_embeds, pooled_prompt_embeds,
        #  _, _) = self.text_processing.encode_text(result, result,
        #                                           self.config.negative_prompt, self.config.negative_prompt_2,
        #                                           self.config,
        #                                           auto_mem_free=False,
        #                                           return_uncond=False)
        # image_embeds = prompt_embeds
        return z_index, init_latents, mask_latents, image_embeds, (x, y), tile_img
        
    def decode(self, vae_decoder, i, all_count, z_index, latents, x, y):
        print(f"[{i}/{all_count}] Decoding VAE...")

        image_np = vae_decoder.decode(latents, auto_mem_free=False)
        tile_img = image.array_to_image(image_np)
        # tile_img.save(f"outputs/{x}_{y}.png")
        return z_index, tile_img, (x, y)
            

    def get_init_latents(self, scheduler, config, img):
        preprocessed_image = image.preprocess_image(img)

        vae_output = self.vae_encoder.encode(preprocessed_image, False)
    
        init_latents = vae_output[:, :4, :, :]

        scaling_factor = SCALING_FACTOR
        init_latents = init_latents * scaling_factor

        return init_latents, None

