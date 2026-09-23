"""'Moderate' image augmentation of our OpenArm WAM runs (WAM_IMAGE_AUG_MODERATE + ALLVIEW), for FastWAM.

The DROID FastWAM port trains WITHOUT image augmentation (FastWAM has none), so this is a new transform that
reproduces the gr00t recipe: per VIEW (independent params), identical for all frames of the clip:
  1. rotate by U(-rotate_deg, +rotate_deg) degrees (bilinear, black fill),
  2. random crop of `crop_area` of the image area (side factor sqrt(crop_area) = 0.949 for 0.9) at a random
     position, resized back to the original HxW (bilinear, antialias) — the eval counterpart in gr00t is a
     center crop of the same size,
  3. photometric jitter: brightness / contrast / saturation factors ~ U(1 - jitter, 1 + jitter) (0.8..1.2; no hue).
Runs inside the processor's `train_transforms` on the [T, C, H, W] float tensor in [0, 1] of ONE view (before the
two views are stacked vertically), so the synthesized black view of a right-only human subset stays black.
"""
import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class OpenArmModerateAug(nn.Module):
    NAME = "moderate"

    def __init__(self, crop_area: float = 0.9, rotate_deg: float = 5.0, jitter: float = 0.2, enabled: bool = True):
        super().__init__()
        self.crop_area = float(crop_area)
        self.rotate_deg = float(rotate_deg)
        self.jitter = float(jitter)
        self.enabled = bool(enabled)
        if not (0.0 < self.crop_area <= 1.0):
            raise ValueError(f"crop_area must be in (0, 1], got {crop_area}")

    def extra_repr(self) -> str:
        return f"crop_area={self.crop_area}, rotate_deg={self.rotate_deg}, jitter={self.jitter}, enabled={self.enabled}"

    def sample_params(self, h: int, w: int, rng: Optional[random.Random] = None) -> dict:
        rng = rng or random
        side = math.sqrt(self.crop_area)
        ch, cw = max(int(round(h * side)), 1), max(int(round(w * side)), 1)
        return {
            "angle": rng.uniform(-self.rotate_deg, self.rotate_deg),
            "top": rng.randint(0, h - ch), "left": rng.randint(0, w - cw), "ch": ch, "cw": cw,
            "brightness": rng.uniform(1 - self.jitter, 1 + self.jitter),
            "contrast": rng.uniform(1 - self.jitter, 1 + self.jitter),
            "saturation": rng.uniform(1 - self.jitter, 1 + self.jitter),
        }

    def apply(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.rotate_deg > 0:
            x = TF.rotate(x, p["angle"], interpolation=TF.InterpolationMode.BILINEAR, expand=False, fill=0.0)
        if self.crop_area < 1.0:
            x = TF.resized_crop(x, p["top"], p["left"], p["ch"], p["cw"], [h, w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        if self.jitter > 0:
            x = TF.adjust_brightness(x, p["brightness"])
            x = TF.adjust_contrast(x, p["contrast"])
            x = TF.adjust_saturation(x, p["saturation"])
        return x.clamp_(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.dtype == torch.float32, f"expected float [T, C, H, W] in [0,1], got {x.shape} {x.dtype}"
        if not self.enabled:
            return x
        return self.apply(x, self.sample_params(x.shape[-2], x.shape[-1]))
