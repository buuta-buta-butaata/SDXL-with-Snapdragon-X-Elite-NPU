import os
import numpy as np

from .schedulers_numpy import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)

class Scheduler:
    scheduler_dict = {
        0: (r".\schedulers_config\EulerAncestralDiscreteScheduler", EulerAncestralDiscreteScheduler),
        1: (r".\schedulers_config\EulerDiscreteScheduler", EulerDiscreteScheduler),
        2: (r".\schedulers_config\DPMSolverSinglestepScheduler", DPMSolverSinglestepScheduler),
        3: (r".\schedulers_config\DPMSolverMultistepScheduler", DPMSolverMultistepScheduler),
        4: (r".\schedulers_config\DDIMScheduler", DDIMScheduler)
    }

    @staticmethod
    def get_available():
        return dict([(k, v[1].__name__) for k, v in Scheduler.scheduler_dict.items()])
    
    def __init__(self, scheduler_type, seed, **kwargs):
        dir_path = os.path.dirname(os.path.abspath(__file__))

        config_path = os.path.join(dir_path, self.scheduler_dict[scheduler_type][0], "scheduler_config.json")
        self.scheduler = self.scheduler_dict[scheduler_type][1].from_pretrained(config_path, **kwargs)
        self.generator = np.random.default_rng(seed)
        # self.generator = np.random.Generator(np.random.MT19937(seed))

    def generate_noise_latents(self, config):
        return self.generator.standard_normal((1, 4, config.height // 8, config.width // 8), dtype=np.float32)

    @property
    def config(self):
        return self.scheduler.config
    
    @property
    def init_noise_sigma(self):
        return self.scheduler.init_noise_sigma

    @property
    def timesteps(self):
        return self.scheduler.timesteps
        
    def set_timesteps(self, *args, **kwargs):
        self.scheduler.set_timesteps(*args, **kwargs)

    def scale_model_input(self, sample, *args, **kwargs):
        return self.scheduler.scale_model_input(sample, *args, **kwargs)

    def step(
        self,
        model_output,
        timestep,
        sample,
        generator=None,
        return_dict=True,
    ):
        result = self.scheduler.step(
            model_output,
            timestep,
            sample,
            generator=self.generator,
            return_dict=return_dict,
        )
        return result

