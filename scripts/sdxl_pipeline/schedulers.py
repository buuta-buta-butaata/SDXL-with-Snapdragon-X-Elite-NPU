import os

from diffusers import EulerAncestralDiscreteScheduler
from diffusers import DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers

class Scheduler:
    path_dict = {
        KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler: r".\schedulers\EulerAncestralDiscreteScheduler",
        KarrasDiffusionSchedulers.DPMSolverMultistepScheduler: r".\schedulers\DPMSolverMultistepScheduler",
        KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler: r".\schedulers\DPMSolverSinglestepScheduler",
        KarrasDiffusionSchedulers.DDIMScheduler: r".\schedulers\DDIMScheduler",
    }
    
    @classmethod
    def get(cls, scheduler_type):
        dir_path = os.path.dirname(os.path.abspath(__file__))

        if scheduler_type == KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler:
            return EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(dir_path, cls.path_dict[scheduler_type]))
        elif scheduler_type == KarrasDiffusionSchedulers.DPMSolverMultistepScheduler:
            return DPMSolverMultistepScheduler.from_pretrained(os.path.join(dir_path, cls.path_dict[scheduler_type]))
        elif scheduler_type == KarrasDiffusionSchedulers.DPMSolverSinglestepScheduler:
            return DPMSolverSinglestepScheduler.from_pretrained(os.path.join(dir_path, cls.path_dict[scheduler_type]))
        else:
            return EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(dir_path, cls.path_dict[scheduler_type]))
            
        
