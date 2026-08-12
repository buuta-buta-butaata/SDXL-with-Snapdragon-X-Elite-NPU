import os
import torch
from pathlib import Path

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    )

from . import scheduler_numpy

class Scheduler(scheduler_numpy.Scheduler):
    scheduler_dict = {
        0: (r".\schedulers_config\EulerAncestralDiscreteScheduler", EulerAncestralDiscreteScheduler),
        1: (r".\schedulers_config\EulerDiscreteScheduler", EulerDiscreteScheduler),
        2: (r".\schedulers_config\DPMSolverSinglestepScheduler", DPMSolverSinglestepScheduler),
        3: (r".\schedulers_config\DPMSolverMultistepScheduler", DPMSolverMultistepScheduler),
        4: (r".\schedulers_config\DDIMScheduler", DDIMScheduler)
    }
    
    def __init__(self, scheduler_type, seed, **kwargs):
        dir_path = os.path.dirname(os.path.abspath(__file__))

        config_path = os.path.join(dir_path, self.scheduler_dict[scheduler_type][0], "scheduler_config.json")
        self.scheduler = self.scheduler_dict[scheduler_type][1].from_pretrained(Path(config_path), **kwargs)
        self.generator = torch.manual_seed(seed)

    def generate_noise_latents(self, config):
        return torch.randn(1, 4, config.height // 8, config.width // 8,
                           dtype=torch.float32, generator=self.generator).numpy()

    @property
    def init_noise_sigma(self):
        if type(self.scheduler.init_noise_sigma) == float:
            return self.scheduler.init_noise_sigma
        return self.scheduler.init_noise_sigma.numpy()

    def scale_model_input(self, sample, timestep, *args, **kwargs):
        return self.scheduler.scale_model_input(torch.from_numpy(sample), timestep, *args, **kwargs).numpy()

    def step(
        self,
        model_output,
        timestep,
        sample,
        generator=None,
        return_dict=True,
    ):
        result = self.scheduler.step(
            torch.from_numpy(model_output),
            timestep,
            torch.from_numpy(sample),
            generator=self.generator,
            return_dict=return_dict,
        )
        result.prev_sample = result.prev_sample.numpy()
        return result

