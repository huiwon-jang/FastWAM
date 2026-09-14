from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self.resume_sample_offset = None  # takes precedence over resume_batch_offset when set

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)
        self.resume_sample_offset = None

    def set_resume_sample_offset(self, sample_offset: int):
        """Resume position in SAMPLES — independent of batch_size / num_processes, so a run can be resumed
        with a different per-device batch (e.g. after an OOM fallback) without shifting the data position."""
        self.resume_sample_offset = int(sample_offset)
        self.resume_batch_offset = 0

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0
        self.resume_sample_offset = None

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.resume_sample_offset is not None:
            sample_offset = self.resume_sample_offset
        else:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
        if self.epoch == 0 and sample_offset > 0:
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
