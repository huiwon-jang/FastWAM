"""openarm_wam_v1 (LeRobot v2.1, 28-dim OpenArm V3 layout) -> FastWAM action/state adapter.

Raw layout (meta/modality.json of every subset):
  action = state = [neck(2) | left_arm(7) | right_arm(7) | left_hand(6) | right_hand(6)]  (28)

Output (mirrors our WAM/gr00t `openarm_all_rel_ah24_config.py` recipe and the DROID FastWAM port):
  action (28) = arm joints [2:16] RELATIVE to the window's first state (t0), neck [0:2] + hands [16:28] ABSOLUTE
  state  (28) = absolute (the model consumes state[t0] only, as the proprio token)

Relative anchor = `state[..., 0, :]` exactly like `DroidJointGripperTransform` / FastWAM's `RelativeJointTransform`.
Padded chunk steps (i + j >= episode end; loss-masked by `action_is_pad`) get their relative dims zeroed here,
because FastWAM's `delta_action_dim_mask` cannot express "relative arms only" on the un-split 28-dim vector
before the transform runs (it can, but doing it here keeps parity with the DROID transform).
Human subsets carry zero-filled state/action: relative(0 - 0) = 0, and every action row of a human sample is
excluded from the action loss upstream (`has_action=False`), so nothing here depends on human values.
"""
from typing import Dict, Sequence, Tuple

import torch

OPENARM_DIM = 28
ARM_DIMS: Tuple[int, int] = (2, 16)  # left_arm(7) + right_arm(7)


class OpenArmRelativeArmTransform:
    def __init__(
        self,
        keys: Sequence[str] = ("default",),
        relative_dims: Sequence[int] = ARM_DIMS,
        zero_padded_relative: bool = True,
    ):
        self.keys = list(keys)
        self.rel = slice(int(relative_dims[0]), int(relative_dims[1]))
        self.zero_padded_relative = bool(zero_padded_relative)

    def forward(self, batch: Dict) -> Dict:
        for k in self.keys:
            state = batch["state"][k]
            assert state.shape[-1] == OPENARM_DIM, f"OpenArm state must be 28-dim, got {tuple(state.shape)}"
            if "action" not in batch:
                continue
            action = batch["action"][k]
            assert action.shape[-1] == OPENARM_DIM, f"OpenArm action must be 28-dim, got {tuple(action.shape)}"
            anchor = state[..., :1, self.rel]  # first (current) observation, arms only
            action = action.clone()
            action[..., self.rel] = action[..., self.rel] - anchor
            pad = batch.get("action_is_pad", None)
            if self.zero_padded_relative and pad is not None:
                pad = torch.as_tensor(pad, dtype=torch.bool, device=action.device)
                if pad.ndim == 1 and action.ndim == 2 and pad.shape[0] == action.shape[0]:
                    action[pad, self.rel] = 0.0
            batch["action"][k] = action
        return batch

    def backward(self, batch: Dict) -> Dict:
        if "action" not in batch:
            return batch
        for k in self.keys:
            anchor = batch["state"][k][..., :1, self.rel]
            action = batch["action"][k].clone()
            action[..., self.rel] = action[..., self.rel] + anchor
            batch["action"][k] = action
        return batch
