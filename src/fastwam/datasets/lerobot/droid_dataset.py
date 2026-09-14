"""DROID (DreamZero-DROID-Data, LeRobot v2.0) dataset classes for FastWAM.

Two deliberate differences from the generic `RobotVideoDataset`:
  * `presample_images = True`: only the stride-subsampled video frames (t = 0, r, 2r, ...) are decoded
    from the mp4s instead of all `num_frames` observations (same mode the lerobot3 base uses).
  * language: DROID stores three instruction annotations per frame
    (`annotation.language.language_instruction{,_2,_3}` -> meta/tasks.jsonl indices). During training one
    of the available ones is picked uniformly at random (parity with our gr00t `droid_config.py`);
    outside training the first one is used. Text-embedding caches must therefore cover ALL tasks.jsonl
    rows, which `scripts/precompute_text_embeds.py` already does.
"""
import numpy as np
import torch

from .base_lerobot_dataset import BaseLerobotDataset
from .robot_video_dataset import RobotVideoDataset

DROID_INSTRUCTION_KEYS = (
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
)


class DroidLerobotDataset(BaseLerobotDataset):
    presample_images = True

    def _get_additional_data(self, sample, lerobot_sample):
        ds_idx = lerobot_sample.get("dataset_index", 0)
        ds_idx = int(ds_idx.item()) if isinstance(ds_idx, torch.Tensor) else int(ds_idx)
        tasks = self.multi_dataset._datasets[ds_idx].meta.tasks
        candidates = []
        for key in DROID_INSTRUCTION_KEYS:
            if key not in lerobot_sample:
                continue
            task_idx = lerobot_sample[key]
            task_idx = int(task_idx.item()) if isinstance(task_idx, torch.Tensor) else int(task_idx)
            if task_idx in tasks:
                candidates.append(tasks[task_idx])
        if candidates:
            sample["task"] = candidates[np.random.randint(len(candidates))] if self.is_training_set else candidates[0]
        return sample


class DroidRobotVideoDataset(RobotVideoDataset):
    base_dataset_cls = DroidLerobotDataset
