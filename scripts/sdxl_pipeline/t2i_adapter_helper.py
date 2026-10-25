import copy
import numpy as np

def get_uncond_data():
    return copy.deepcopy({
        "down":
        {
            "t2i_adapter_state_1": np.zeros((1, 320, 64, 64), np.float16),
            "t2i_adapter_state_2": np.zeros((1, 640, 64, 64), np.float16),
            "t2i_adapter_state_3": np.zeros((1, 1280, 32, 32), np.float16),
        },
        "up1":
        {
            "t2i_adapter_state_4": np.zeros((1, 1280, 32, 32), np.float16),
        }
    })
