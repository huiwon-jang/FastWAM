from math import ceil
from typing import Iterator, Sequence, Sized

import numpy as np
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


class BalancedGroupEpochSampler(Sampler[int]):
    """ResumableEpochSampler with an EXACT per-batch group composition.

    Every consecutive `batch_size` chunk of the index stream holds `round(frac_g * batch_size)` samples of
    group g (in group order, e.g. [robot x8 | human x8]); the DataLoader's default BatchSampler and accelerate's
    BatchSamplerShard (whole-batch round-robin over processes) keep those chunks intact, so every per-GPU
    micro-batch has the same composition. One epoch = as many batches as the largest group needs to be seen
    once; smaller groups are re-permuted (fresh permutation per pass) and cycled. Resume semantics are those of
    ResumableEpochSampler (sample offset must be a multiple of batch_size, which `samples_in_epoch` guarantees
    for every OOM plate since the effective batch is a multiple of every per-device batch).
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int,
                 group_ids: Sequence[int], group_fractions: Sequence[float]):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self.resume_sample_offset = None
        group_ids = np.asarray(group_ids)
        if group_ids.shape[0] != len(dataset):
            raise ValueError(f"group_ids has {group_ids.shape[0]} entries, dataset has {len(dataset)}")
        self.groups = sorted(int(g) for g in np.unique(group_ids))
        if len(self.groups) != len(group_fractions):
            raise ValueError(f"{len(self.groups)} groups present {self.groups} but {len(group_fractions)} fractions given")
        self.per_batch = [int(round(float(f) * self.batch_size)) for f in group_fractions]
        if sum(self.per_batch) != self.batch_size or min(self.per_batch) <= 0:
            raise ValueError(f"group fractions {list(group_fractions)} x batch_size {self.batch_size} -> {self.per_batch}: "
                             "must be positive integers summing to batch_size")
        self.group_indices = {g: np.flatnonzero(group_ids == g) for g in self.groups}
        self.num_batches = max(ceil(len(self.group_indices[g]) / n) for g, n in zip(self.groups, self.per_batch))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)
        self.resume_sample_offset = None

    def set_resume_sample_offset(self, sample_offset: int):
        self.resume_sample_offset = int(sample_offset)
        self.resume_batch_offset = 0

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0
        self.resume_sample_offset = None

    def epoch_indices(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + int(epoch) + self.epoch_offset)
        cols = []
        for g, n in zip(self.groups, self.per_batch):
            need = self.num_batches * n
            pool = self.group_indices[g]
            chunks, have = [], 0
            while have < need:  # fresh permutation per pass over the group
                perm = rng.permutation(pool)
                chunks.append(perm)
                have += len(perm)
            cols.append(np.concatenate(chunks)[:need].reshape(self.num_batches, n))
        return np.concatenate(cols, axis=1).reshape(-1)

    def __iter__(self) -> Iterator[int]:
        indices = self.epoch_indices(self.epoch)
        if self.resume_sample_offset is not None:
            sample_offset = self.resume_sample_offset
        else:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
        if self.epoch == 0 and sample_offset > 0:
            if sample_offset % self.batch_size != 0:
                raise ValueError(f"resume sample offset {sample_offset} is not a multiple of batch_size {self.batch_size}; "
                                 "the balanced batch layout would be broken")
            indices = indices[sample_offset:]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_batches * self.batch_size
