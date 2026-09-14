"""TRUE global per-dim q01/q99 (+min/max/mean/std) of the values that enter FastWAM's normalizer.

Why: FastWAM's own stats tool (BaseLerobotDataset.get_dataset_stats) takes torch.quantile PER EPISODE and
then reduces with amin/amax over episodes and window steps, so `global_q01/q99` ends up at the most extreme
episode's quantile (≈ global min/max over 57k episodes). This script instead pools every value and takes
numpy.percentile over the pool.

Training-sample semantics reproduced exactly (see DroidLerobotDataset / DroidJointGripperTransform):
  * one sample per anchor frame i of an episode of length N (every anchor when --anchor-stride 1);
  * action chunk j = 0..23 uses action[min(i+j, N-1)]; joints (7) are RELATIVE to the anchor state
    state[i], the gripper (1) is absolute; steps with i+j >= N are padded (zeroed and loss-masked in
    training) and are EXCLUDED from the pool here (their count is reported);
  * state = [joint(7), gripper(1)] per frame (the model only consumes state[i], every frame is an anchor).
Values are staged per dim as float32 binary files so the percentile pass holds one dim at a time
(~4 bytes x pool size per dim); CPU only, numpy + pyarrow only.

  python scripts/compute_true_global_quantiles.py --dataset-root /data/shared_dataset/DreamZero-DROID-Data \
      --out /data/huiwon/data/droid_fastwam/true_global_quantiles.json --staging-dir /data/huiwon/tmp/fastwam_trueq \
      [--anchor-stride 1] [--workers 8]
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

HORIZON = 24
ACTION_GRIPPER = slice(12, 13)
ACTION_JOINT = slice(14, 21)
STATE_GRIPPER = slice(6, 7)
STATE_JOINT = slice(7, 14)
DIMS = 8
DIM_NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


def load_episode(path):
    t = pq.read_table(path, columns=["frame_index", "observation.state", "action"])
    fi = t.column("frame_index").to_numpy()
    st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    ac = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
    order = np.argsort(fi, kind="stable")
    st, ac = st[order], ac[order]
    state8 = np.concatenate([st[:, STATE_JOINT], st[:, STATE_GRIPPER]], axis=1)  # [N, 8]
    action8 = np.concatenate([ac[:, ACTION_JOINT], ac[:, ACTION_GRIPPER]], axis=1)  # [N, 8]
    return state8, action8


def episode_values(state8, action8, stride):
    """Return (action_pool [M, 8] valid chunk steps, n_padded) for anchors 0, stride, 2*stride, ..."""
    n = state8.shape[0]
    anchors = np.arange(0, n, stride)
    j = np.arange(HORIZON)
    idx = anchors[:, None] + j[None, :]  # [A, 24]
    valid = idx < n
    idx_c = np.minimum(idx, n - 1)
    chunk = action8[idx_c]  # [A, 24, 8]
    chunk = chunk.copy()
    chunk[:, :, :7] -= state8[anchors, :7][:, None, :]  # relative joints, absolute gripper
    pool = chunk[valid]  # [M, 8]
    return pool, int((~valid).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--staging-dir", required=True)
    ap.add_argument("--anchor-stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-episodes", type=int, default=None, help="debug: only the first K episodes")
    args = ap.parse_args()

    info = json.load(open(os.path.join(args.dataset_root, "meta", "info.json")))
    total = int(info["total_episodes"]) if args.limit_episodes is None else int(args.limit_episodes)
    chunks_size = int(info["chunks_size"])
    data_path = info["data_path"]
    paths = [os.path.join(args.dataset_root, data_path.format(episode_chunk=e // chunks_size, episode_index=e)) for e in range(total)]

    os.makedirs(args.staging_dir, exist_ok=True)
    act_files = [open(os.path.join(args.staging_dir, f"action_dim{d}.f32"), "wb") for d in range(DIMS)]
    st_files = [open(os.path.join(args.staging_dir, f"state_dim{d}.f32"), "wb") for d in range(DIMS)]
    acc = {k: {"n": 0, "sum": np.zeros(DIMS, np.float64), "sumsq": np.zeros(DIMS, np.float64),
               "min": np.full(DIMS, np.inf, np.float64), "max": np.full(DIMS, -np.inf, np.float64)} for k in ("action", "state")}
    n_padded = 0
    n_frames = 0

    def accumulate(kind, arr):
        a = acc[kind]
        a["n"] += arr.shape[0]
        a["sum"] += arr.sum(0, dtype=np.float64)
        a["sumsq"] += (arr.astype(np.float64) ** 2).sum(0)
        a["min"] = np.minimum(a["min"], arr.min(0))
        a["max"] = np.maximum(a["max"], arr.max(0))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, (state8, action8) in enumerate(ex.map(load_episode, paths, chunksize=16)):
            pool, pad = episode_values(state8, action8, args.anchor_stride)
            n_padded += pad
            n_frames += state8.shape[0]
            accumulate("action", pool)
            accumulate("state", state8)
            for d in range(DIMS):
                act_files[d].write(np.ascontiguousarray(pool[:, d]).tobytes())
                st_files[d].write(np.ascontiguousarray(state8[:, d]).tobytes())
            if (k + 1) % 2000 == 0 or k + 1 == total:
                print(f"[trueq] {k + 1}/{total} episodes, frames={n_frames}, action values={acc['action']['n']}, {time.time() - t0:.0f}s", flush=True)
    for f in act_files + st_files:
        f.close()

    result = {"meta": {
        "dataset_root": args.dataset_root, "episodes": total, "frames": n_frames, "anchor_stride": args.anchor_stride,
        "horizon": HORIZON, "padded_steps_excluded": n_padded, "action_values_per_dim": acc["action"]["n"],
        "state_values_per_dim": acc["state"]["n"], "dims": DIM_NAMES,
        "semantics": "action = [joint rel. to anchor state (7), gripper abs (1)] over valid chunk steps of every anchor; state = [joint(7), gripper(1)] per frame",
        "method": "numpy.percentile linear over the pooled values (per dim)", "wall_s": round(time.time() - t0, 1),
    }}
    for kind, prefix in (("action", "action_dim"), ("state", "state_dim")):
        a = acc[kind]
        n = a["n"]
        mean = a["sum"] / n
        std = np.sqrt(np.maximum(a["sumsq"] / n - mean ** 2, 0.0))
        q01, q99 = [], []
        for d in range(DIMS):
            v = np.fromfile(os.path.join(args.staging_dir, f"{prefix}{d}.f32"), dtype=np.float32)
            assert v.shape[0] == n, (v.shape, n)
            lo, hi = np.percentile(v, [1.0, 99.0])
            q01.append(float(lo)); q99.append(float(hi))
            del v
            print(f"[trueq] {kind} dim{d} ({DIM_NAMES[d]}): q01={lo:.4f} q99={hi:.4f} min={a['min'][d]:.4f} max={a['max'][d]:.4f} mean={mean[d]:.4f} std={std[d]:.4f}", flush=True)
        result[kind] = {"q01": q01, "q99": q99, "min": a["min"].tolist(), "max": a["max"].tolist(),
                        "mean": mean.tolist(), "std": std.tolist(), "n": int(n)}
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, args.out)
    shutil.rmtree(args.staging_dir, ignore_errors=True)
    print(f"TRUEQ_OK {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
