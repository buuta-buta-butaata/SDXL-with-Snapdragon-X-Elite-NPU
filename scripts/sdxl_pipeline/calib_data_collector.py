import os
import numpy as np
from datetime import datetime


class CalibrationDataCollector:
    def __init__(self, base_dir="../calibration_data", steps=20, calib_strategy=0):
        self.output_root_dir = base_dir
        self.steps = steps
        self.collection_strategy = calib_strategy
        # 5分割（part0〜part4）に対応したフォルダを事前に作成
        for part in ["part0", "part1", "part2", "part3", "part4"]:
            os.makedirs(os.path.join(self.output_root_dir, part), exist_ok=True)

        os.makedirs(os.path.join(self.output_root_dir, "vae_decoder"), exist_ok=True)

    def is_collection_target(self, part_name, step):
        return True
        # return part_name == "part1"
        #if step > 0 and step < 7:
        #    return True
        """
        if self.collection_strategy == 1:
            if step == 0 or step == 3 or step == 6 or step == 10 or step == 13 or step == 16:
                return True
        elif self.collection_strategy == 2:
            if step == 3 or step == 6 or step == 9 or step == 13 or step == 16 or step == 19:
                return True
        elif self.collection_strategy == 3:
            if step == 1:
                return True
        elif self.collection_strategy == 4:
            if step == 2:
                return True
        elif self.collection_strategy == 5:
            if step == 4:
                return True
        elif self.collection_strategy == 6:
            if step == 5:
                return True
        elif self.collection_strategy == 7:
            if step == 0:
                return True
        elif self.collection_strategy == 8:
            if step == 3:
                return True
        """
        # if self.collection_strategy == 1:
        #     if step == 0 or step == 3 or step == 6 or step == 10 or step == 13 or step == 16:
        #         return True
        # elif self.collection_strategy == 2:
        #     if step == 3 or step == 6 or step == 9 or step == 13 or step == 16 or step == 19:
        #         return True
        # elif self.collection_strategy == 3:
        #     if step == 1:
        #         return True
        # elif self.collection_strategy == 4:
        #     if step == 2:
        #         return True
        # elif self.collection_strategy == 5:
        #     if step == 4:
        #         return True
        # elif self.collection_strategy == 6:
        #     if step == 5:
        #         return True

        # if self.collection_strategy == 1:
        #     if step == 0 or step == 3 or step == 6 or step == 10 or step == 13 or step == 16:
        #         return True
        # elif self.collection_strategy == 2:
        #     if step == 1 or step == 4 or step == 7 or step == 11 or step == 14 or step == 17:
        #         return True
        # elif self.collection_strategy == 3:
        #     if step == 2 or step == 5 or step == 8 or step == 12 or step == 15 or step == 18:
        #         return True
        # elif self.collection_strategy == 4:
        #     if step == 3 or step == 6 or step == 9 or step == 13 or step == 16 or step == 19:
        #         return True

        # if self.collection_strategy == 1:
        #     if (step > 1 and step < 5) or (step > self.steps - 4 and step < self.steps):
        #         return True
        # elif self.collection_strategy == 2:
        #     if (step > 4 and step < 8) or (step > self.steps - 7 and step < self.steps - 3):
        #         return True
        # elif self.collection_strategy == 3:
        #     if step == 2 or step == 5 or step == 8 or step == 12 or step == 15 or step == 18:
        #         return True
        # elif self.collection_strategy == 4:
        #     if step == 3 or step == 6 or step == 9 or step == 13 or step == 16 or step == 19:
        #         return True
        return False
    
    def save_step(self, part_name, input_dict, step):
        """
        1ステップ分の入力を1つの辞書として丸ごと1ファイルに保存する（前回確定仕様）。
        """
        if not self.is_collection_target(part_name, step):
            return
        # ファイル名を "step_000.npy" のようにゼロパディング
        file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.npy"
        save_path = os.path.join(self.output_root_dir, part_name, file_name)

        """
        ready_dict = {}
        for k, v in input_dict.items():
            if hasattr(v, "detach"): # PyTorchテンソルの保険
                v = v.detach().cpu().numpy()
            ready_dict[k] = np.array(v, dtype=np.float32)
            
        np.save(save_path, ready_dict)
        """
        np.save(save_path, input_dict)

    def save(self, dir_name, input_dict):
        file_name = datetime.now().strftime("%Y%m%d%H%M%S") + ".npy"
        save_path = os.path.join(self.output_root_dir, dir_name, file_name)
        np.save(save_path, input_dict)
