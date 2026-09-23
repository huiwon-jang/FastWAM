"""TRUE global per-dim q01/q99 (+min/max/mean/std) for the OpenArm FastWAM run, written directly in FastWAM's
dataset_stats.json schema with the marker `huiwon_true_global_quantiles: true`.

Why not FastWAM's own stats pass (BaseLerobotDataset.get_dataset_stats): it takes torch.quantile PER EPISODE and
amin/amax-reduces over episodes (-> ~min/max), and it would pool the HUMAN subsets' zero placeholders and banana's
zero-filled absent action groups. Here:
  * only subsets WITHOUT meta/wam_human.json contribute (human zeros never enter any statistic);
  * a subset with meta/wam_action_groups.json contributes only its valid groups' dims (banana: right arm + right
    hand) to both the action and the state pools — the other dims are zero-filled there and loss-masked;
  * training-sample semantics reproduced exactly (OpenArmLerobotDataset + OpenArmRelativeArmTransform): one sample
    per anchor frame i (every anchor), action chunk j = 0..23 uses action[min(i+j, N-1)] with arm dims [2:16]
    RELATIVE to state[i], neck/hands absolute; padded steps (i+j >= N) are EXCLUDED; state = absolute state[i].
  * numpy.percentile over the pooled values, staged per dim as float32 files (one dim in RAM at a time).
Dims whose q99-q01 range is below --collapse-range-below (default 0.02 rad; the OpenArm neck pitch/yaw span 0.011 / 0.001 rad over the whole robot set) are written with
q01 = q99 = mean, so FastWAM's LinearNormalizer treats them as constant (scale 1, centred) instead of blowing
sensor noise up by 1/range.

  python scripts/compute_openarm_true_global_stats.py --root <subset> [--root ...] --out-dir /data/huiwon/data/openarm_fastwam \
      --staging-dir /data/huiwon/tmp/fastwam_openarm_trueq [--horizon 24] [--anchor-stride 1] [--workers 8]
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DIMS = 28
ARM = slice(2, 16)
GROUPS = {"neck_joints": (0, 2), "left_arm_joints": (2, 9), "right_arm_joints": (9, 16),
          "left_hand_joints": (16, 22), "right_hand_joints": (22, 28)}
DIM_NAMES = ([f"neck_{i}" for i in range(2)] + [f"larm_{i}" for i in range(7)] + [f"rarm_{i}" for i in range(7)]
             + [f"lhand_{i}" for i in range(6)] + [f"rhand_{i}" for i in range(6)])
MARKER = "huiwon_true_global_quantiles"


def subset_info(root: Path):
    info = json.load(open(root / "meta" / "info.json"))
    human = False
    p = root / "meta" / "wam_human.json"
    if p.is_file():
        human = bool(json.load(open(p)).get("human", False))
    mask = np.ones(DIMS, dtype=bool)
    p = root / "meta" / "wam_action_groups.json"
    if p.is_file():
        groups = json.load(open(p)).get("valid_groups")
        if groups is not None:
            mask[:] = False
            for g in groups:
                a, b = GROUPS[g]
                mask[a:b] = True
    chunks = int(info["chunks_size"])
    # episode ids from meta/episodes.jsonl (ids may have gaps, e.g. dropped episodes), not range(total_episodes)
    ep_ids = sorted(int(json.loads(l)["episode_index"]) for l in open(root / "meta" / "episodes.jsonl") if l.strip())
    paths = [root / info["data_path"].format(episode_chunk=e // chunks, episode_index=e) for e in ep_ids]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: {len(missing)} parquet files listed in episodes.jsonl are missing, e.g. {missing[:3]}")
    return info, human, mask, paths


def load_episode(path):
    t = pq.read_table(str(path), columns=["frame_index", "observation.state", "action"])
    fi = t.column("frame_index").to_numpy()
    st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    ac = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
    order = np.argsort(fi, kind="stable")
    return st[order], ac[order]


def episode_values(state, action, horizon, stride):
    n = state.shape[0]
    anchors = np.arange(0, n, stride)
    j = np.arange(horizon)
    idx = anchors[:, None] + j[None, :]
    valid = idx < n
    chunk = action[np.minimum(idx, n - 1)].copy()  # [A, H, 28]
    chunk[:, :, ARM] -= state[anchors][:, None, ARM]
    return chunk[valid], int((~valid).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="subset root (repeatable); human subsets are skipped")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--staging-dir", required=True)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--anchor-stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--collapse-range-below", type=float, default=2e-2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_out = out_dir / "dataset_stats.true_global.json"
    trueq_out = out_dir / "true_global_quantiles.json"
    if stats_out.is_file() and not args.force:
        print(f"[openarm-stats] {stats_out} exists (use --force to rebuild)")
        return
    staging = Path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    act_files = [open(staging / f"action_dim{d}.f32", "wb") for d in range(DIMS)]
    st_files = [open(staging / f"state_dim{d}.f32", "wb") for d in range(DIMS)]
    acc = {k: {"n": np.zeros(DIMS, np.int64), "sum": np.zeros(DIMS, np.float64), "sumsq": np.zeros(DIMS, np.float64),
               "min": np.full(DIMS, np.inf), "max": np.full(DIMS, -np.inf)} for k in ("action", "state")}

    def accumulate(kind, arr, mask):
        a = acc[kind]
        for d in np.flatnonzero(mask):
            v = arr[:, d]
            a["n"][d] += v.shape[0]
            a["sum"][d] += float(v.sum(dtype=np.float64))
            a["sumsq"][d] += float((v.astype(np.float64) ** 2).sum())
            a["min"][d] = min(a["min"][d], float(v.min()))
            a["max"][d] = max(a["max"][d], float(v.max()))

    t0 = time.time()
    used, skipped, n_frames, n_padded, n_episodes = [], [], 0, 0, 0
    for root in [Path(r) for r in args.root]:
        info, human, mask, paths = subset_info(root)
        if human:
            skipped.append(str(root))
            print(f"[openarm-stats] SKIP human subset {root} ({info['total_frames']} frames of placeholder zeros)", flush=True)
            continue
        used.append({"root": str(root), "fps": info["fps"], "episodes": len(paths), "frames": info["total_frames"],
                     "valid_dims": int(mask.sum()), "valid_groups": [g for g, (a, b) in GROUPS.items() if mask[a:b].all()]})
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for state, action in ex.map(load_episode, paths, chunksize=8):
                pool, pad = episode_values(state, action, args.horizon, args.anchor_stride)
                n_padded += pad
                n_frames += state.shape[0]
                n_episodes += 1
                accumulate("action", pool, mask)
                accumulate("state", state, mask)
                for d in np.flatnonzero(mask):
                    act_files[d].write(np.ascontiguousarray(pool[:, d]).tobytes())
                    st_files[d].write(np.ascontiguousarray(state[:, d]).tobytes())
        print(f"[openarm-stats] {root}: episodes={len(paths)} frames={info['total_frames']} valid_dims={int(mask.sum())} ({time.time() - t0:.0f}s)", flush=True)
    for f in act_files + st_files:
        f.close()
    if not used:
        sys.exit("[openarm-stats] no robot subset given")

    result = {"meta": {
        "roots": used, "skipped_human_roots": skipped, "episodes": n_episodes, "frames": n_frames,
        "anchor_stride": args.anchor_stride, "horizon": args.horizon, "padded_steps_excluded": n_padded, "dims": DIM_NAMES,
        "semantics": "action = [neck abs(2), arms rel. to anchor state (14), hands abs(12)] over valid chunk steps of every anchor; "
                     "state = absolute 28-dim per frame; human subsets excluded; per-subset absent action groups excluded",
        "method": "numpy.percentile linear over the pooled values (per dim)", "collapse_range_below": args.collapse_range_below,
        "wall_s": round(time.time() - t0, 1),
    }}
    collapsed = {}
    for kind, prefix in (("action", "action_dim"), ("state", "state_dim")):
        a = acc[kind]
        n = a["n"]
        mean = np.where(n > 0, a["sum"] / np.maximum(n, 1), 0.0)
        std = np.sqrt(np.maximum(np.where(n > 0, a["sumsq"] / np.maximum(n, 1) - mean ** 2, 0.0), 0.0))
        q01, q99 = [], []
        for d in range(DIMS):
            v = np.fromfile(staging / f"{prefix}{d}.f32", dtype=np.float32)
            assert v.shape[0] == n[d], (kind, d, v.shape, n[d])
            lo, hi = (np.percentile(v, [1.0, 99.0]) if v.shape[0] else (0.0, 0.0))
            lo, hi = float(lo), float(hi)
            if hi - lo < args.collapse_range_below:
                collapsed.setdefault(kind, []).append({"dim": d, "name": DIM_NAMES[d], "q01": lo, "q99": hi, "mean": float(mean[d])})
                lo = hi = float(mean[d])
            q01.append(lo)
            q99.append(hi)
            del v
            print(f"[openarm-stats] {kind} dim{d:2d} ({DIM_NAMES[d]:8s}): n={n[d]} q01={lo:+.4f} q99={hi:+.4f} min={a['min'][d]:+.4f} max={a['max'][d]:+.4f} mean={mean[d]:+.4f} std={std[d]:.4f}", flush=True)
        result[kind] = {"q01": q01, "q99": q99, "min": [float(x) if np.isfinite(x) else 0.0 for x in a["min"]],
                        "max": [float(x) if np.isfinite(x) else 0.0 for x in a["max"]], "mean": mean.tolist(), "std": std.tolist(), "n": n.tolist()}
    result["meta"]["collapsed_dims"] = collapsed
    with open(str(trueq_out) + ".tmp", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(str(trueq_out) + ".tmp", trueq_out)

    # FastWAM dataset_stats.json schema (LinearNormalizer reads global_* for both action and state)
    def block(kind, steps):
        r = result[kind]
        return {
            "global_min": r["min"], "global_max": r["max"], "global_q01": r["q01"], "global_q99": r["q99"],
            "global_mean": r["mean"], "global_std": r["std"],
            "stepwise_min": [r["min"]] * steps, "stepwise_max": [r["max"]] * steps,
            "stepwise_q01": [r["q01"]] * steps, "stepwise_q99": [r["q99"]] * steps,
            "stepwise_mean": [r["mean"]] * steps, "stepwise_std": [r["std"]] * steps,
        }
    stats = {
        "state": {"default": block("state", args.horizon + 1)},
        "action": {"default": block("action", args.horizon)},
        "num_episodes": n_episodes, "num_transition": n_frames,
        MARKER: True,
        f"{MARKER}_meta": {"source": str(trueq_out), "created": datetime.datetime.utcnow().isoformat() + "Z",
                           "recipe": "openarm_wam_v1 robot-only true global q01/q99", **result["meta"]},
    }
    with open(str(stats_out) + ".tmp", "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(str(stats_out) + ".tmp", stats_out)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[openarm-stats] collapsed dims: {json.dumps(collapsed)}")
    print(f"OPENARM_STATS_OK {stats_out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
