"""CPU smoke of the OpenArm FastWAM data pipeline on 2 robot + 2 human episodes (no model, no text cache).
Safe for a 4 GiB pod. Checks shapes, the 2-view vertical canvas (left TOP / right BOTTOM, BLACK where a view is
missing), has_action / action_dim_mask flags, the normalized action range, banana's absent dims, the relative-arm
transform, per-sample latency, and the balanced robot/human batch sampler (incl. resume offsets).

  BASE_DIR=/data/huiwon/fastwam-openarm STATS=/data/huiwon/data/openarm_fastwam/dataset_stats.true_global.json \
    .venv/bin/python scripts/openarm_data_smoke.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import fastwam
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.samplers import BalancedGroupEpochSampler

register_default_resolvers()
BASE = os.environ.get("BASE_DIR", "/data/huiwon/fastwam-openarm")
STATS = os.environ.get("STATS", "/data/huiwon/data/openarm_fastwam/dataset_stats.true_global.json")
OA = os.environ.get("OPENARM_DATA_ROOT", "/data/huiwon/data/openarm_wam_v1")
assert os.path.realpath(fastwam.__file__).startswith(os.path.realpath(BASE)), f"fastwam imported from {fastwam.__file__}, not {BASE}"
work = "/data/huiwon/tmp/fastwam_openarm_smoke"
os.makedirs(work, exist_ok=True)
misc.register_work_dir(work)
with initialize_config_dir(config_dir=os.path.join(BASE, "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=openarm_fastwam_5b_b64_50k", "output_dir=" + work])
d = OmegaConf.to_container(cfg.data.train, resolve=True)
roots = [f"{OA}/robot/openarm_ego_jungwook", f"{OA}/robot/banana_v21_openarm28",
         f"{OA}/human_as_openarm28/rlwrld_human_lerobot", f"{OA}/human_as_openarm28/anyh2r"]
d.update(dataset_dirs=roots, pretrained_norm_stats=STATS, use_text_embed_cache=False, max_episodes_per_dir=1, is_training_set=True)
t0 = time.time()
ds = instantiate(OmegaConf.create(d))
lr = ds.lerobot_dataset
print(f"dataset built in {time.time()-t0:.1f}s len={len(ds)} episodes={lr.multi_dataset.num_episodes} groups={np.bincount(ds.batch_group_ids)} fractions={ds.batch_group_fractions}")
for sub in lr.multi_dataset._datasets:
    print(f"  subset {sub.root.name}: fps={sub.meta.fps} frames={sub.num_frames} human={sub.meta.is_human} views={sub.meta.view_key_map} deltas={ {k: (v[0], v[1], v[-1], len(v)) for k, v in sub.delta_indices.items()} }")
st = json.load(open(STATS))
assert st.get("huiwon_true_global_quantiles") is True
q01, q99 = np.array(st["action"]["default"]["global_q01"]), np.array(st["action"]["default"]["global_q99"])
print("stats action q01:", q01.round(3).tolist()); print("stats action q99:", q99.round(3).tolist())
norm = lr.processor.normalizer.normalizers["action"]["default"]
offset = norm.offset.numpy()
starts = np.cumsum([0] + [sub.num_frames for sub in lr.multi_dataset._datasets])
fails = []
def check(c, m):
    if not c:
        fails.append(m); print("FAIL:", m)
rows = []
for si, sub in enumerate(lr.multi_dataset._datasets):
    for off in (0, sub.num_frames - 1):   # start, END (padded chunk)
        idx = int(starts[si] + off)
        t1 = time.time(); s = ds[idx]; dt = time.time() - t1
        v, a, p = s["video"], s["action"], s["proprio"]
        top, bot = v[:, :, :192, :], v[:, :, 192:, :]
        black_top = bool((top <= -0.999).all()); black_bot = bool((bot <= -0.999).all())
        human = bool(s["is_human"]); has_act = bool(s["has_action"]); dm = s["action_dim_mask"]
        valid_steps = ~s["action_is_pad"]
        a_valid = a[valid_steps][:, dm] if dm.any() else a[valid_steps][:, :0]
        in_range = float((a_valid.abs() <= 1.0).float().mean()) if a_valid.numel() else float("nan")
        rows.append((sub.root.name, off, dt))
        print(f"[{sub.root.name} idx={idx} off={off}] {dt:.2f}s video={tuple(v.shape)} rng=({v.min():.2f},{v.max():.2f}) top_black={black_top} bot_black={black_bot} "
              f"action={tuple(a.shape)} proprio={tuple(p.shape)} pads={int(s['action_is_pad'].sum())} img_pads={int(s['image_is_pad'].sum())} "
              f"human={human} has_action={has_act} dim_mask={int(dm.sum())}/28 |a|<=1 on valid dims/steps: {in_range:.3f} prompt={s['prompt'][:90]!r}")
        check(tuple(v.shape) == (3, 9, 384, 256), f"video shape {tuple(v.shape)}")
        check(tuple(a.shape) == (24, 28) and tuple(p.shape) == (24, 28), "action/proprio shape")
        check(not black_bot, "right view (bottom) must never be black")
        if sub.meta.view_key_map["observation.images.camera_ego_left"] is None:
            check(black_top, "single-view subset: left (top) must be black")
        else:
            check(not black_top, "two-view subset: left (top) must not be black")
            check(not torch.equal(top, bot), "left and right views must differ")
        check(human == sub.meta.is_human and has_act == (not sub.meta.is_human), "has_action/is_human flags")
        if human:
            check(not dm.any(), "human rows must have an all-False action_dim_mask")
        else:
            check(torch.equal(dm, sub.meta.action_dim_mask), "robot action_dim_mask == subset mask")
            check(in_range >= 0.90, f"normalized valid actions within [-1,1] (q01/q99 -> ~2% outside by construction): {in_range:.3f}")
        if "banana" in sub.root.name:
            absent = ~dm
            check(int(absent.sum()) == 15 and dm[9:16].all() and dm[22:28].all(), "banana mask = right arm + right hand")
            # zero-filled raw dims normalize to exactly the offset (x*scale + offset with x=0) -> raw is zero
            check(torch.allclose(a[:, absent], torch.as_tensor(offset[absent.numpy()], dtype=a.dtype).expand(24, -1), atol=1e-5), "banana absent dims are zero (raw)")
            check(torch.allclose(p[:, absent], torch.as_tensor(lr.processor.normalizer.normalizers['state']['default'].offset.numpy()[absent.numpy()], dtype=p.dtype).expand(24, -1), atol=1e-5), "banana absent state dims are zero (raw)")
        if int(s["action_is_pad"].sum()) > 0 and not human:
            # padded steps: relative arm dims were zeroed before normalization -> equal the offset on arm dims
            pad_rows = a[s["action_is_pad"]][:, 2:16]
            check(torch.allclose(pad_rows, torch.as_tensor(offset[2:16], dtype=a.dtype).expand(pad_rows.shape[0], -1), atol=1e-5), "padded steps have zero relative arm dims")
# per-stage latency profile (where does a sample's time go?)
import time as _t
from fastwam.datasets.lerobot.openarm_dataset import decode_frames_by_index
sub0 = lr.multi_dataset._datasets[0]; j = int(starts[0]) + 100
t = _t.time(); row = sub0.hf_dataset[j]; t_row = _t.time() - t
qi, pad = sub0._get_query_indices(j, 0); t = _t.time(); qr = sub0._query_hf_dataset_fast(qi); t_sel = _t.time() - t
vk = sub0.meta.video_keys[0]; ids = [int(q) - int(sub0.episode_data_index["from"][0]) for q in qi[vk]]
t = _t.time(); fr = decode_frames_by_index(sub0.root / sub0.meta.get_video_file_path(sub0.episodes[0], vk), ids, sub0.video_backend); t_dec = _t.time() - t
t = _t.time(); raw_item = lr.multi_dataset[j]; t_item = _t.time() - t
t = _t.time(); full = ds[j]; t_full = _t.time() - t
print(f"profile (backend={sub0.video_backend}, one view {len(ids)} frames idx {ids[0]}..{ids[-1]}): hf_row={t_row:.3f}s hf_select={t_sel:.3f}s decode_one_view={t_dec:.3f}s lerobot_item(2 views)={t_item:.3f}s full_sample(+aug/canvas)={t_full:.3f}s")
# video frame count == parquet rows for the sampled episodes (frame-index decoding assumption)
import av
for sub in lr.multi_dataset._datasets:
    for vid_key in sub.meta.video_keys:
        with av.open(str(sub.root / sub.meta.get_video_file_path(sub.episodes[0], vid_key))) as c:
            n = sum(1 for pk in c.demux(c.streams.video[0]) if pk.pts is not None); rate = float(c.streams.video[0].average_rate)
        check(n == sub.num_frames, f"{sub.root.name}/{vid_key}: video frames {n} != parquet rows {sub.num_frames}")
        print(f"  video check {sub.root.name}/{vid_key.split('.')[-1]}: frames={n} rows={sub.num_frames} tagged_fps={rate:.1f} dataset_fps={sub.meta.fps}")
# relative-arm transform correctness on one raw episode (compare against a manual computation)
sub = lr.multi_dataset._datasets[0]
raw = lr.multi_dataset.get_episode_data(0)
state, action = raw["observation.state"].float(), raw["action"].float()
i = 5
s = ds[int(starts[0] + i)]
man = action[i:i + 24].clone(); man[:, 2:16] -= state[i, 2:16]
man_norm = torch.clamp(man * norm.scale + norm.offset, -5, 5)
check(torch.allclose(s["action"][~s["action_is_pad"]], man_norm[~s["action_is_pad"]], atol=1e-4), "relative-arm transform + q01/q99 normalization matches manual computation")
state_norm = lr.processor.normalizer.normalizers["state"]["default"]
check(torch.allclose(s["proprio"][0], torch.clamp(state[i] * state_norm.scale + state_norm.offset, -5, 5), atol=1e-4), "proprio[0] == normalized state[t0]")
# balanced sampler
gids = ds.batch_group_ids
smp = BalancedGroupEpochSampler(ds, seed=42, batch_size=16, num_processes=4, group_ids=gids, group_fractions=[0.5, 0.5])
idx = np.array(list(iter(smp)))
b = gids[idx].reshape(-1, 16)
check(bool((b[:, :8] == 0).all() and (b[:, 8:] == 1).all()), "every 16-batch is [robot x8 | human x8]")
check(len(idx) == smp.num_batches * 16, "sampler length")
smp.set_resume_sample_offset(16 * 4 * 3)
idx2 = np.array(list(iter(smp)))
check(bool((idx2 == idx[192:]).all()), "resume sample offset slices the same epoch stream")
smp2 = BalancedGroupEpochSampler(ds, seed=42, batch_size=8, num_processes=4, group_ids=gids, group_fractions=[0.5, 0.5])
b8 = gids[np.array(list(iter(smp2)))].reshape(-1, 8)
check(bool((b8[:, :4] == 0).all() and (b8[:, 4:] == 1).all()), "pd8 plate: [robot x4 | human x4]")
print(f"sampler: batches/epoch={smp.num_batches} first batch groups={b[0].tolist()} robot coverage per epoch={smp.num_batches*8/ (gids==0).sum():.2f}x human={smp.num_batches*8/(gids==1).sum():.2f}x")
# collate a mixed batch the way the DataLoader would, and run the loss-mask arithmetic on it
from torch.utils.data import default_collate
batch = default_collate([ds[int(j)] for j in idx[:4].tolist()] + [ds[int(j)] for j in idx[8:12].tolist()])
print("collated:", {k: (tuple(v.shape), str(v.dtype)) for k, v in batch.items() if isinstance(v, torch.Tensor)})
check(tuple(batch["has_action"].shape) == (8,) and batch["has_action"].dtype == torch.bool and int(batch["has_action"].sum()) == 4, "collated has_action [B] bool with 4 robot rows")
check(tuple(batch["action_dim_mask"].shape) == (8, 28), "collated action_dim_mask [B,28]")
mse = torch.rand(8, 24, 28)
dim_valid = batch["action_dim_mask"].float().unsqueeze(1)
tok = (mse * dim_valid).sum(2) / dim_valid.sum(2).clamp(min=1.0)
valid = (~batch["action_is_pad"]).float(); per = (tok * valid).sum(1) / valid.sum(1).clamp(min=1.0)
row = batch["has_action"].float(); loss = (per * row).sum() / row.sum().clamp(min=1.0)
check(bool(torch.allclose(loss, per[batch["has_action"]].mean())), "masked action loss == mean over robot rows only (human rows contribute 0)")
check(bool((tok[~batch["has_action"]] == 0).all()), "human rows: per-token action loss is exactly 0")
# augmentation: train (moderate) vs eval (none) on the same raw frames; black view stays black under aug
from fastwam.datasets.lerobot.transforms.openarm_aug import OpenArmModerateAug
aug = OpenArmModerateAug(crop_area=0.9, rotate_deg=5.0, jitter=0.2)
raw = lr.multi_dataset[int(starts[3])]["observation.images.camera_ego_left"].float()  # [T,3,192,256] in [0,1]
a1, a2 = aug(raw.clone()), aug(raw.clone())
check(a1.shape == raw.shape and float(a1.min()) >= 0 and float(a1.max()) <= 1, "aug keeps shape/range")
check(not torch.equal(a1, raw) and not torch.equal(a1, a2), "aug is random and changes the frames")
check(bool((aug(torch.zeros_like(raw)) == 0).all()), "black view stays black under aug")
check(any("OpenArmModerateAug" in type(t).__name__ for t in lr.processor.train_transforms) and not any("OpenArmModerateAug" in type(t).__name__ for t in lr.processor.val_transforms), "processor: aug in train_transforms only")
print("aug params example:", {k: round(v, 3) if isinstance(v, float) else v for k, v in aug.sample_params(192, 256).items()})
lat = np.array([r[2] for r in rows]); print(f"per-sample latency: mean={lat.mean():.2f}s min={lat.min():.2f}s max={lat.max():.2f}s (first call includes decoder warm-up)")
from PIL import Image
f = ((ds[int(starts[3])]["video"][:, 0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8"); Image.fromarray(f).save(os.path.join(work, "frame0_anyh2r_vconcat.png"))
f = ((ds[int(starts[2])]["video"][:, 0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8"); Image.fromarray(f).save(os.path.join(work, "frame0_rlwrld_human_blacktop.png"))
f = ((ds[int(starts[1])]["video"][:, 0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8"); Image.fromarray(f).save(os.path.join(work, "frame0_banana_vconcat.png"))
print("saved canvases to", work)
if fails:
    print(f"FAILURES: {len(fails)}"); sys.exit(1)
print("OPENARM_DATA_SMOKE_OK")
