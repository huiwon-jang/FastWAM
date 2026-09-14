"""DreamZero-DROID-Data (LeRobot v2.0) -> FastWAM action/state adapter.

Raw layout (meta/modality.json of /data/shared_dataset/DreamZero-DROID-Data):
  action (28) = cartesian_position[0:6] cartesian_velocity[6:12] gripper_position[12:13]
                gripper_velocity[13:14] joint_position[14:21] joint_velocity[21:28]
  state  (14) = cartesian_position[0:6] gripper_position[6:7] joint_position[7:14]

Output (matches our WAM/gr00t `droid_config.py` and the DreamZero DROID recipe):
  action (8) = joint_position[7] (RELATIVE to the window's first state when `relative_joint`)
               + gripper_position[1] (absolute)
  state  (8) = joint_position[7] + gripper_position[1] (absolute)

The relative anchor is `state[..., 0, :]` (the current observation, t0) exactly like FastWAM's own
`RelativeJointTransform` and DreamZero's `relative_stats_dreamzero.json` convention.
`backward` re-adds the anchor but returns the 8-dim vector (the 28-dim raw vector is not reconstructible).
"""
from typing import Dict, Sequence

import torch

ACTION_GRIPPER = slice(12, 13)
ACTION_JOINT = slice(14, 21)
STATE_GRIPPER = slice(6, 7)
STATE_JOINT = slice(7, 14)


class DroidJointGripperTransform:
    def __init__(
        self,
        keys: Sequence[str] = ("default",),
        relative_joint: bool = True,
        zero_padded_relative: bool = True,
    ):
        self.keys = list(keys)
        self.relative_joint = bool(relative_joint)
        # FastWAM's `delta_action_dim_mask` zeroes delta dims at padded steps but is applied on the RAW
        # vector (before this transform), so it cannot be used with raw_shape != shape; do it here instead.
        self.zero_padded_relative = bool(zero_padded_relative)

    @staticmethod
    def _state_to_8(state: torch.Tensor) -> torch.Tensor:
        assert state.shape[-1] == 14, f"DROID state must be 14-dim, got {tuple(state.shape)}"
        return torch.cat([state[..., STATE_JOINT], state[..., STATE_GRIPPER]], dim=-1)

    @staticmethod
    def _action_to_8(action: torch.Tensor) -> torch.Tensor:
        assert action.shape[-1] == 28, f"DROID action must be 28-dim, got {tuple(action.shape)}"
        return torch.cat([action[..., ACTION_JOINT], action[..., ACTION_GRIPPER]], dim=-1)

    def forward(self, batch: Dict) -> Dict:
        for k in self.keys:
            state8 = self._state_to_8(batch["state"][k])  # [..., T_obs, 8]
            batch["state"][k] = state8
            if "action" not in batch:
                continue
            action8 = self._action_to_8(batch["action"][k])  # [..., T_act, 8]
            if self.relative_joint:
                anchor = state8[..., :1, :7]  # first (current) observation
                action8 = action8.clone()
                action8[..., :7] = action8[..., :7] - anchor
                pad = batch.get("action_is_pad", None)
                if self.zero_padded_relative and pad is not None:
                    pad = torch.as_tensor(pad, dtype=torch.bool, device=action8.device)
                    if pad.ndim == 1 and action8.ndim == 2 and pad.shape[0] == action8.shape[0]:
                        action8[pad, :7] = 0.0
            batch["action"][k] = action8
        return batch

    def backward(self, batch: Dict) -> Dict:
        if not self.relative_joint or "action" not in batch:
            return batch
        for k in self.keys:
            anchor = batch["state"][k][..., :1, :7]
            action = batch["action"][k].clone()
            action[..., :7] = action[..., :7] + anchor
            batch["action"][k] = action
        return batch
