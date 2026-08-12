import os
import numpy as np
from datetime import datetime

def save(dir_path, file_prefix, data):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S%f")[:-3]
    file_name = file_prefix + formatted_time + ".npy"
    save_path = os.path.join(dir_path, file_name)
    np.save(save_path, data)

def load(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data
