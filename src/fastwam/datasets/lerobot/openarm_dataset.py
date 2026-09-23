"""openarm_wam_v1 (robot + human co-training, LeRobot v2.1) dataset classes for FastWAM.

Root: /data/huiwon/data/openarm_wam_v1 — 11 subsets, all with 28-dim `observation.state`/`action` (OpenArm V3
layout: neck 2 | left arm 7 | right arm 7 | left hand 6 | right hand 6), 192x256 videos, fps 20 or 30.
  robot (6): openarm_ego_jungwook, openarm_teleop_v3/{bottle,cup,doll,snack}, banana_v21_openarm28
  human (5): rlwrld_human_lerobot, openarm_validation_v2_junhyeong/{close_air_fryer,left_hand_box_white_container,
             open_air_fryer}, anyh2r   (meta/wam_human.json marker; state/action are zero placeholders)

What differs from the generic / DROID classes and why:
  * per-dataset FRAME-index deltas instead of one seconds-based `delta_timestamps`: the subsets mix 20 and 30 fps
    and our other OpenArm arms use a fixed frame stride regardless of fps (video stride 3, action horizon 24 at
    native fps). `LeRobotDataset` converts seconds -> indices with each dataset's own fps, which would give
    different strides per subset; here every subset gets the same integer deltas.
  * per-dataset video-key aliasing: banana stores `observation.image.ego_{left,right}`, the others
    `observation.images.camera_ego_{left,right}`; four human subsets have the RIGHT view only. The canonical
    keys are `observation.images.camera_ego_left/right`; a missing view is synthesized as BLACK frames (and its
    `_is_pad` copied from the present view). `MultiLeRobotDataset` would instead DROP every key that is not
    common to all subsets, i.e. the left view everywhere, so the multi-dataset is rebuilt here without that rule.
  * FRAME-INDEX video decoding (not timestamps): every subset stores exactly one video frame per parquet row,
    but rlwrld_human_lerobot's mp4s are tagged 30 fps while the dataset is 20 fps, so LeRobot's timestamp
    lookup (`round(ts * video_fps)`) would fetch the wrong frames (1.5x) and trip the tolerance check. Frames
    are fetched by within-episode row index through torchcodec `get_frames_at` (pyav fallback).
  * tolerant metadata: banana and the human subsets carry no `episodes_stats.jsonl`; the stock loader would then
    try to `snapshot_download` from the Hub. FastWAM never uses LeRobot's stats (it normalizes with its own
    stats json), so metadata is loaded from info/tasks/episodes(/stats) only.
  * per-sample loss flags (consumed by `FastWAM.training_loss`):
      has_action      bool      False for human subsets -> the sample contributes NOTHING to the action loss
      action_dim_mask bool[28]  False for the action groups a subset does not supply (banana: only right arm +
                                right hand are real, meta/wam_action_groups.json); those dims are zero-filled and
                                excluded from the per-dim MSE
      is_human        bool      informational (== not has_action)
  * language: single `task_index` -> meta/tasks.jsonl string (already cleaned in openarm_wam_v1), wrapped in
    FastWAM's DEFAULT_PROMPT exactly like DROID (also for human clips, i.e. "recorded from a robot's point of
    view" — same text convention our other OpenArm arms use for both groups).
  * batch composition: `batch_group_ids` (0 = robot, 1 = human) + `batch_group_fractions` let the trainer build
    a `BalancedGroupEpochSampler` so EVERY per-GPU batch is exactly 50 % robot / 50 % human.
  * text-embedding cache is held in process memory (72 prompts x 1 MB) instead of a torch.load per sample.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fastwam.utils.logging_config import get_logger

from .base_lerobot_dataset import BaseLerobotDataset
from .lerobot.datasets.utils import get_episode_data_index, load_episodes, load_info, load_stats, load_tasks
import importlib
import warnings

from .lerobot.datasets.video_utils import get_safe_default_codec
from .lerobot.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
from .robot_video_dataset import RobotVideoDataset

logger = get_logger(__name__)

OPENARM_DIM = 28
OPENARM_ACTION_GROUPS = {
    "neck_joints": (0, 2),
    "left_arm_joints": (2, 9),
    "right_arm_joints": (9, 16),
    "left_hand_joints": (16, 22),
    "right_hand_joints": (22, 28),
}
CANONICAL_VIEW_KEYS = ("observation.images.camera_ego_left", "observation.images.camera_ego_right")
VIEW_ALIASES = {
    "observation.images.camera_ego_left": ("observation.images.camera_ego_left", "observation.image.ego_left"),
    "observation.images.camera_ego_right": ("observation.images.camera_ego_right", "observation.image.ego_right"),
}
GROUP_ROBOT, GROUP_HUMAN = 0, 1
STATS_MARKER = "huiwon_true_global_quantiles"


def read_human_marker(root: Path) -> bool:
    p = root / "meta" / "wam_human.json"
    if not p.is_file():
        return False
    with open(p) as f:
        return bool(json.load(f).get("human", False))


def read_action_group_mask(root: Path) -> torch.Tensor:
    """bool[28]: True where the subset supplies a real action dim (meta/wam_action_groups.json, default all)."""
    mask = torch.ones(OPENARM_DIM, dtype=torch.bool)
    p = root / "meta" / "wam_action_groups.json"
    if not p.is_file():
        return mask
    with open(p) as f:
        groups = json.load(f).get("valid_groups", None)
    if groups is None:
        return mask
    mask[:] = False
    for g in groups:
        if g not in OPENARM_ACTION_GROUPS:
            raise ValueError(f"{p}: unknown action group {g!r} (known: {sorted(OPENARM_ACTION_GROUPS)})")
        a, b = OPENARM_ACTION_GROUPS[g]
        mask[a:b] = True
    return mask


class OpenArmLeRobotDatasetMetadata(LeRobotDatasetMetadata):
    """Local-only metadata: info/tasks/episodes(/stats.json). Never pulls from the Hub, tolerates a missing
    episodes_stats.jsonl (banana + human subsets). Adds the OpenArm markers."""

    def __init__(self, repo_id: str, root):
        self.repo_id = repo_id
        self.revision = CODEBASE_VERSION
        self.root = Path(root)
        self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)  # informational only; FastWAM normalizes with its own stats file
        self.episodes_stats = dict.fromkeys(self.episodes, self.stats)
        self.is_human = read_human_marker(self.root)
        self.action_dim_mask = read_action_group_mask(self.root)
        self.view_key_map: Dict[str, Optional[str]] = {}
        for canonical in CANONICAL_VIEW_KEYS:
            actual = [k for k in VIEW_ALIASES[canonical] if k in self.video_keys]
            self.view_key_map[canonical] = actual[0] if actual else None
        if self.view_key_map["observation.images.camera_ego_right"] is None:
            raise ValueError(f"{self.root}: no right view among video keys {self.video_keys}")
        for k in ("observation.state", "action"):
            if tuple(self.features[k]["shape"]) != (OPENARM_DIM,):
                raise ValueError(f"{self.root}: {k} shape {self.features[k]['shape']} != ({OPENARM_DIM},)")


class OpenArmLeRobotDataset(LeRobotDataset):
    """LeRobotDataset over one openarm_wam_v1 subset with explicit frame-index deltas (no Hub access)."""

    def __init__(
        self,
        repo_id: str,
        root,
        episodes: Optional[List[int]],
        delta_indices: Dict[str, List[int]],
        tolerance_s: float,
        video_backend: Optional[str] = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root)
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.video_codec = "h264"
        self.is_compute_episode_stats_image = False
        self.during_training = True
        self.image_writer = None
        self.episode_buffer = None
        self.meta = OpenArmLeRobotDatasetMetadata(self.repo_id, self.root)
        self.stats = self.meta.stats
        missing = [str(p) for p in self.get_episodes_file_paths() if not (self.root / p).is_file()]
        if missing:
            raise FileNotFoundError(f"{self.root}: {len(missing)} missing files, e.g. {missing[:3]}")
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        # canonical -> actual video keys; only keys this subset really has get delta indices
        self.delta_indices = {}
        for key, deltas in delta_indices.items():
            if key in CANONICAL_VIEW_KEYS:
                actual = self.meta.view_key_map[key]
                if actual is None:
                    continue
                self.delta_indices[actual] = list(deltas)
            else:
                self.delta_indices[key] = list(deltas)

    def __getitem__(self, idx) -> dict:
        """LeRobotDataset.__getitem__ with FRAME-INDEX video decoding (video frame k == parquet row k)."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
        query_indices, padding = self._get_query_indices(idx, current_ep_idx)
        query_result = self._query_hf_dataset_fast(query_indices)
        item = {**item, **padding}
        for key, val in query_result.items():
            item[key] = val
        if len(self.meta.video_keys) > 0 and self.during_training:
            ep_start = int(self.episode_data_index["from"][current_ep_idx].item())
            for vid_key in self.meta.video_keys:
                if vid_key not in query_indices:
                    continue
                frame_ids = [int(q) - ep_start for q in query_indices[vid_key]]
                video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
                item[vid_key] = decode_frames_by_index(video_path, frame_ids, self.video_backend)
        item["task"] = self.meta.tasks[item["task_index"].item()]
        return item


def decode_frames_by_index(video_path, frame_indices: List[int], backend: Optional[str] = None) -> torch.Tensor:
    """Decode the given within-file frame indices -> float32 [N, C, H, W] in [0, 1] (order preserved)."""
    backend = backend or get_safe_default_codec()
    if backend == "torchcodec":
        try:
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
            batch = decoder.get_frames_at(indices=[int(i) for i in frame_indices])
            return batch.data.to(torch.float32) / 255.0
        except Exception as err:  # e.g. no ffmpeg shared libs (debug pod) -> sequential pyav decode
            warnings.warn(f"torchcodec index decode failed ({type(err).__name__}: {err}); falling back to pyav")
    import av
    wanted = {int(i) for i in frame_indices}
    last = max(wanted)
    got = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        k = 0
        for frame in container.decode(stream):
            if k in wanted:
                got[k] = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            if k >= last:
                break
            k += 1
    missing = [i for i in wanted if i not in got]
    if missing:
        raise ValueError(f"{video_path}: frame indices {missing[:5]} beyond the {k + 1} decodable frames")
    return torch.stack([got[int(i)] for i in frame_indices]).to(torch.float32) / 255.0


class OpenArmMultiLeRobotDataset(MultiLeRobotDataset):
    """MultiLeRobotDataset without the 'drop keys not common to all subsets' rule and without Hub access."""

    def __init__(
        self,
        dataset_dirs: List[str],
        episodes: Optional[dict],
        delta_indices: Dict[str, List[int]],
        tolerances_s: Optional[dict] = None,
        video_backend: Optional[str] = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.dataset_dirs = dataset_dirs
        self.ds_roots = [Path(d) for d in dataset_dirs]
        self.ds_names = list(dataset_dirs)
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(self.ds_names, 0.0001)
        self._datasets = []
        for ds_root, ds_name in zip(self.ds_roots, self.ds_names, strict=True):
            self._datasets.append(
                OpenArmLeRobotDataset(
                    ds_name,
                    root=ds_root,
                    episodes=episodes[ds_name] if episodes else None,
                    delta_indices=delta_indices,
                    tolerance_s=self.tolerances_s[ds_name],
                    video_backend=video_backend,
                )
            )
        self.disabled_features = set()
        self.image_transforms = None
        self.delta_timestamps = None
        self.stats = None  # FastWAM uses its own stats json

    def get_episode_data(self, episode_idx: int) -> dict:
        """Parent version, but string columns (language_instruction) stay numpy instead of failing torch.from_numpy."""
        import pyarrow.parquet as pq
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                table = pq.read_table(str(dataset.root / dataset.meta.get_data_file_path(dataset.episodes[episode_idx])))
                out = {}
                for name in table.column_names:
                    col = table[name]
                    try:
                        arr = col.to_numpy(zero_copy_only=True)
                    except Exception:
                        raw = col.to_numpy(zero_copy_only=False)
                        arr = np.stack(raw) if raw.dtype == object and raw.size and not isinstance(raw[0], str) else raw
                    out[name] = arr if arr.dtype.kind in "OUS" else torch.from_numpy(np.ascontiguousarray(arr))
                return out
            episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")


class OpenArmLerobotDataset(BaseLerobotDataset):
    """BaseLerobotDataset for the OpenArm mixture: frame-index deltas, view aliasing/black fill, loss flags."""

    presample_images = True
    metadata_cls = OpenArmLeRobotDatasetMetadata
    multi_dataset_cls = OpenArmMultiLeRobotDataset

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0,
        obs_size: int = 1,
        past_obs_size: int = 0,
        val_set_proportion: float = 0.0,
        is_training_set: bool = False,
        seed: int = 42,
        global_sample_stride: int = 1,
        image_subsample_stride: int = 1,
        tolerance_s: Optional[float] = None,
        video_backend: Optional[str] = None,
        max_episodes_per_dir: Optional[int] = None,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0 and past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"
        self.dataset_dirs = list(dataset_dirs)
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.image_subsample_stride = image_subsample_stride
        self.global_sample_stride = global_sample_stride
        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set
        self.processor = None

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]
        for meta in self.image_meta:
            meta["lerobot_key"] = f"observation.images.{meta['key']}"
            if meta["lerobot_key"] not in CANONICAL_VIEW_KEYS:
                raise ValueError(f"OpenArm image key must be one of camera_ego_left/right, got {meta['key']}")
        for meta in self.state_meta:
            meta["lerobot_key"] = "observation.state" if meta["key"] == "default" else f"observation.state.{meta['key']}"
        for meta in self.action_meta:
            meta["lerobot_key"] = "action" if meta["key"] == "default" else f"action.{meta['key']}"

        # FRAME-index deltas, identical for every subset whatever its fps (video stride = image_subsample_stride)
        s = int(global_sample_stride)
        delta_indices: Dict[str, List[int]] = {}
        for meta in self.image_meta:
            delta_indices[meta["lerobot_key"]] = [t * s for t in range(0, obs_size, image_subsample_stride)]
        for meta in self.state_meta:
            delta_indices[meta["lerobot_key"]] = [t * s for t in range(0, obs_size)]
        for meta in self.action_meta:
            delta_indices[meta["lerobot_key"]] = [t * s for t in range(0, action_size)]
        self.delta_indices = delta_indices

        metas = [self.metadata_cls(repo_id=d, root=Path(d)) for d in self.dataset_dirs]
        episodes = {}
        for meta in metas:
            # episode ids come from meta/episodes.jsonl, NOT range(total_episodes): rlwrld_human_lerobot dropped 7
            # episodes (wam_human.json dropped_episodes) so its ids run 0..478 with gaps while total_episodes is 472
            all_eps = sorted(int(e) for e in meta.episodes.keys())
            if val_set_proportion < 1e-6:
                eps = list(all_eps)
            else:
                split_idx = int(len(all_eps) * (1 - val_set_proportion))
                eps = list(all_eps)
                np.random.default_rng(seed).shuffle(eps)
                eps = eps[:split_idx] if is_training_set else eps[split_idx:]
            if max_episodes_per_dir is not None:
                eps = eps[: int(max_episodes_per_dir)]
            episodes[meta.repo_id] = eps

        tolerances_s = None if tolerance_s is None else dict.fromkeys(self.dataset_dirs, tolerance_s)
        self.multi_dataset = self.multi_dataset_cls(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_indices=delta_indices,
            tolerances_s=tolerances_s,
            video_backend=video_backend,
        )
        # global episode index (same HACK as the base class)
        episode_data_index, end_index = [], 0
        for dataset in self.multi_dataset._datasets:
            d = {"from": dataset.episode_data_index["from"] + end_index, "to": dataset.episode_data_index["to"] + end_index}
            episode_data_index.append(d)
            end_index = d["to"][-1]
        self.episode_data_index = {
            "from": torch.cat([d["from"] for d in episode_data_index]),
            "to": torch.cat([d["to"] for d in episode_data_index]),
        }
        # per-subset flags + sample-level group ids (0 robot / 1 human) for the balanced batch sampler
        self.subset_is_human = [bool(ds.meta.is_human) for ds in self.multi_dataset._datasets]
        self.subset_action_dim_mask = [ds.meta.action_dim_mask.clone() for ds in self.multi_dataset._datasets]
        self.sample_group_ids = np.concatenate(
            [np.full(ds.num_frames, GROUP_HUMAN if h else GROUP_ROBOT, dtype=np.int8)
             for ds, h in zip(self.multi_dataset._datasets, self.subset_is_human)]
        )
        n_robot = int((self.sample_group_ids == GROUP_ROBOT).sum())
        n_human = int((self.sample_group_ids == GROUP_HUMAN).sum())
        for ds, h in zip(self.multi_dataset._datasets, self.subset_is_human):
            logger.info(
                "[openarm] %s: %s fps=%s episodes=%d frames=%d views=%s action_dims=%d/28",
                ds.root, "HUMAN(video-only)" if h else "robot", ds.meta.fps, ds.num_episodes, ds.num_frames,
                {k.split(".")[-1]: (v.split(".")[-1] if v else "BLACK") for k, v in ds.meta.view_key_map.items()},
                int(ds.meta.action_dim_mask.sum()),
            )
        logger.info("[openarm] samples: robot=%d human=%d total=%d | deltas: %s", n_robot, n_human, len(self), delta_indices)

    def _split_lerobot_sample(self, lerobot_sample):
        """Canonicalize view keys (banana alias), synthesize a BLACK missing view, attach the loss flags."""
        ds_idx = lerobot_sample.get("dataset_index", 0)
        ds_idx = int(ds_idx.item()) if isinstance(ds_idx, torch.Tensor) else int(ds_idx)
        ds = self.multi_dataset._datasets[ds_idx]
        present = None
        for canonical in CANONICAL_VIEW_KEYS:
            actual = ds.meta.view_key_map[canonical]
            if actual is None:
                continue
            if actual != canonical:
                lerobot_sample[canonical] = lerobot_sample.pop(actual)
                lerobot_sample[f"{canonical}_is_pad"] = lerobot_sample.pop(f"{actual}_is_pad")
            present = canonical
        for canonical in CANONICAL_VIEW_KEYS:
            if ds.meta.view_key_map[canonical] is None:
                ref = lerobot_sample[present]
                lerobot_sample[canonical] = torch.zeros_like(ref)  # black view (0.0 in [0,1] -> -1 after Normalize)
                lerobot_sample[f"{canonical}_is_pad"] = lerobot_sample[f"{present}_is_pad"].clone()
        lerobot_sample["is_human"] = torch.tensor(self.subset_is_human[ds_idx])
        return lerobot_sample

    def _get_additional_data(self, sample, lerobot_sample):
        ds_idx = int(lerobot_sample["dataset_index"].item()) if "dataset_index" in lerobot_sample else 0
        is_human = self.subset_is_human[ds_idx]
        sample["is_human"] = torch.tensor(is_human)
        sample["has_action"] = torch.tensor(not is_human)
        mask = self.subset_action_dim_mask[ds_idx].clone()
        if is_human:
            mask[:] = False
        sample["action_dim_mask"] = mask
        sample["dataset_index"] = torch.tensor(ds_idx)
        return sample


class OpenArmRobotVideoDataset(RobotVideoDataset):
    """RobotVideoDataset over the OpenArm mixture (2-view vertical canvas: left on top, right below)."""

    base_dataset_cls = OpenArmLerobotDataset

    def __init__(self, *args, group_fractions=(0.5, 0.5), max_episodes_per_dir=None, require_stats_marker=True, **kwargs):
        self._max_episodes_per_dir = max_episodes_per_dir
        stats_path = kwargs.get("pretrained_norm_stats", None)
        if require_stats_marker:
            if not stats_path:
                raise ValueError(
                    "OpenArm training requires pretrained_norm_stats = dataset_stats.true_global.json "
                    "(scripts/compute_openarm_true_global_stats.py); FastWAM's native stats pass would include the "
                    "human placeholder zeros and collapse q01/q99 to per-episode extremes."
                )
            with open(stats_path) as f:
                if json.load(f).get(STATS_MARKER) is not True:
                    raise ValueError(f"{stats_path} lacks the marker {STATS_MARKER}=true")
        super().__init__(*args, **kwargs)
        if self.concat_multi_camera != "vertical":
            raise ValueError(f"OpenArm canvas must be concat_multi_camera='vertical' (left top / right bottom), got {self.concat_multi_camera}")
        keys = [m["key"] for m in self.lerobot_dataset.image_meta]
        if keys != ["camera_ego_left", "camera_ego_right"]:
            raise ValueError(f"OpenArm shape_meta.images must be [camera_ego_left, camera_ego_right] in this order, got {keys}")
        self.batch_group_ids = self.lerobot_dataset.sample_group_ids
        self.batch_group_fractions = [float(v) for v in group_fractions]
        self._text_cache: Dict[str, Any] = {}

    def _base_dataset_extra_kwargs(self):
        return {"max_episodes_per_dir": self._max_episodes_per_dir}

    def _build_data(self, sample):
        data = super()._build_data(sample)
        data["has_action"] = sample["has_action"]
        data["action_dim_mask"] = sample["action_dim_mask"]
        data["is_human"] = sample["is_human"]
        return data

    def _get_cached_text_context(self, prompt: str):
        hit = self._text_cache.get(prompt)
        if hit is None:
            context, mask = super()._get_cached_text_context(prompt)
            hit = (context.clone(), mask.clone())
            if len(self._text_cache) < 4096:
                self._text_cache[prompt] = hit
        context, mask = hit
        return context.clone(), mask.clone()
