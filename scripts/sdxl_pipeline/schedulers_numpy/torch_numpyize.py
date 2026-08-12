import numpy as np
import numpy as torch
import warnings

# PyTorch suppresses 'divide by zero encountered in log' warnings; silencing here to match behavior.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*scheduling_dpmsolver_.*",
                        message = ".*divide by zero encountered in log")
"""
>>> import torch
>>> zero = torch.tensor(0)
>>> torch.log(zero)    # no warning
tensor(-inf)
>>>
>>> import numpy as np
>>> np.log(0)          # warning
<python-input-13>:1: RuntimeWarning: divide by zero encountered in log
np.float64(-inf)
>>>

torch version: 2.10.0+cpu
numpy version: 2.4.4
"""

torch.Tensor = np.ndarray
torch.tensor = np.array
torch.clamp = np.clip
torch.Generator = np.random._generator.Generator
torch.device = str
torch.IntTensor = np.int32
torch.LongTensor = np.int64
torch.from_numpy = lambda x: x

default_cumprod = np.cumprod
torch.cumprod = lambda input, dim, dtype=None, out=None: default_cumprod(input, axis=dim, dtype=dtype, out=out)
torch.cat = np.concatenate

class BaseOutput():
    prev_sample: torch.Tensor

class SchedulerOutput():
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
    
from enum import Enum
class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
    DPMSolverSDEScheduler = 14
    EDMEulerScheduler = 15

def randn_tensor(
    shape: tuple | list,
    generator: list[torch.Generator] | torch.Generator | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    # layout: torch.layout | None = None,
):
    latents = generator.standard_normal(shape, dtype=dtype)
    return latents

