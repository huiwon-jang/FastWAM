"""Hydra-compose preflight gate for a FastWAM task (no model build, no GPU, small RAM).

Composes configs/train.yaml + task=<task> + the given overrides, resolves every interpolation, asserts the
values the MLXP launcher relies on, optionally instantiates the (light) processor and the LeRobot metadata,
and prints one banner line. Exit code 0 iff all assertions pass.

  python scripts/compose_check.py --task droid_fastwam_5b_b128_200k --world 4 --expect-effective 128 \
      [--check-paths] [--check-meta] [-- <hydra overrides...>]
"""
import argparse
import os
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--world", type=int, default=int(os.environ.get("NUM_GPUS", "1")))
    ap.add_argument("--expect-effective", type=int, default=None)
    ap.add_argument("--check-paths", action="store_true", help="dataset root / ActionDiT .pt / stats / text cache must exist")
    ap.add_argument("--check-meta", action="store_true", help="load LeRobot metadata (reads meta/*.jsonl)")
    ap.add_argument("--expect-fps", default="15", help="comma list of admissible dataset fps (DROID: 15; OpenArm: 20,30)")
    ap.add_argument("--expect-raw-dims", default="28,14", help="raw action,state dims in meta/info.json (DROID 28,14; OpenArm 28,28)")
    ap.add_argument("--expect-proc-dims", default="8,8", help="processor action_output_dim,proprio_output_dim (DROID 8,8; OpenArm 28,28)")
    ap.add_argument("--expect-layout", default=None, help="required concat_multi_camera (default: no check beyond tmosaic geometry)")
    ap.add_argument("--expect-video-size", default=None, help="required video_size HxW, e.g. 384x256")
    ap.add_argument("--expect-lr", type=float, default=None)
    ap.add_argument("--expect-wd", type=float, default=None)
    ap.add_argument("--expect-warmup", type=int, default=None, help="required warmup_steps (config value)")
    ap.add_argument("--expect-min-lr-ratio", type=float, default=None)
    ap.add_argument("--expect-max-steps", type=int, default=None)
    ap.add_argument("--expect-aug", default=None, help="'moderate' -> OpenArmModerateAug(0.9/5/0.2) must be in train_transforms and absent from val_transforms; 'none' -> no aug")
    ap.add_argument("--expect-frames", default=None, help="num_frames/ratio, e.g. 25/3")
    ap.add_argument("--expect-group-fractions", default=None, help="e.g. 0.5,0.5")
    ap.add_argument("--expect-relative-dims", default=None, help="e.g. 2,16 (OpenArmRelativeArmTransform.relative_dims)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    overrides = [o for o in args.overrides if o != "--"]

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}", "output_dir=/dev/null"] + overrides)
    c = OmegaConf.to_container(cfg, resolve=True)
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    bs, ga, ms = int(c["batch_size"]), int(c["gradient_accumulation_steps"]), int(c["max_steps"])
    eff = bs * args.world * ga
    if args.expect_effective is not None:
        check(eff == args.expect_effective, f"effective batch {eff} != {args.expect_effective}")
    check(int(c["save_every"]) > 0, "save_every must be > 0")
    check(c.get("save_total_limit") is None or int(c["save_total_limit"]) > 0, "save_total_limit must be null or > 0")
    check(int(c["eval_every"]) == 0, "eval must be disabled (eval_every=0) for the unattended MLXP run")
    d = c["data"]["train"]
    nf, ratio = int(d["num_frames"]), int(d["action_video_freq_ratio"])
    check((nf - 1) % ratio == 0 and ((nf - 1) // ratio) % 4 == 0, f"num_frames/ratio invalid: {nf}/{ratio}")
    h, w = d["video_size"]
    check(h % 32 == 0 and w % 32 == 0, f"video_size {d['video_size']} must be multiples of 32")
    if d["concat_multi_camera"] == "tmosaic":
        th, tw = d["tmosaic_top_size"]; bh, bw = d["tmosaic_bottom_size"]
        check(tw == 2 * bw and [th + bh, tw] == [h, w], "tmosaic sizes must sum to video_size")
    p = d["processor"]
    check(p["action_output_dim"] == sum(m["shape"] for m in d["shape_meta"]["action"]), "action_output_dim mismatch")
    check(p["proprio_output_dim"] == sum(m["shape"] for m in d["shape_meta"]["state"]), "proprio_output_dim mismatch")
    m = c["model"]
    check(m["_target_"] == "fastwam.runtime.create_fastwam", f"model target {m['_target_']} is not the uncond FastWAM")
    check(m["redirect_common_files"] is False, "redirect_common_files must be false (local original Wan-AI files)")
    check(m["load_text_encoder"] is False and d["use_text_embed_cache"] is True, "text embed cache mode expected")
    check(m["action_dit_config"]["action_dim"] == p["action_output_dim"], "ActionDiT action_dim mismatch")
    def near(a, b, tol=1e-12):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    if args.expect_lr is not None:
        check(near(c["learning_rate"], args.expect_lr, 1e-9), f"learning_rate {c['learning_rate']} != {args.expect_lr}")
    if args.expect_wd is not None:
        check(near(c["weight_decay"], args.expect_wd, 1e-9), f"weight_decay {c['weight_decay']} != {args.expect_wd}")
    if args.expect_warmup is not None:
        check(c.get("warmup_steps") is not None and int(c["warmup_steps"]) == args.expect_warmup, f"warmup_steps {c.get('warmup_steps')} != {args.expect_warmup}")
    if args.expect_min_lr_ratio is not None:
        check(c.get("min_lr_ratio") is not None and near(c["min_lr_ratio"], args.expect_min_lr_ratio, 1e-9), f"min_lr_ratio {c.get('min_lr_ratio')} != {args.expect_min_lr_ratio}")
    if args.expect_max_steps is not None:
        check(ms == args.expect_max_steps, f"max_steps {ms} != {args.expect_max_steps}")
    if args.expect_frames is not None:
        enf, er = (int(v) for v in args.expect_frames.split("/"))
        check((nf, ratio) == (enf, er), f"num_frames/ratio {nf}/{ratio} != {args.expect_frames}")
    if args.expect_group_fractions is not None:
        ef = [float(v) for v in args.expect_group_fractions.split(",")]
        gf = [float(v) for v in (d.get("group_fractions") or [])]
        check(gf == ef, f"group_fractions {gf} != {ef}")
    if args.expect_relative_dims is not None:
        er = [int(v) for v in args.expect_relative_dims.split(",")]
        tr = [t for t in (p.get("action_state_transforms") or []) if "OpenArmRelativeArmTransform" in str(t.get("_target_"))]
        check(len(tr) == 1 and [int(v) for v in tr[0].get("relative_dims", [])] == er and bool(tr[0].get("zero_padded_relative", False)),
              f"OpenArmRelativeArmTransform(relative_dims={er}, zero_padded_relative=true) missing: {tr}")
        check(p["norm_default_mode"] == "q01/q99" and not p["use_stepwise_action_norm"], "normalization must be global q01/q99")
    if args.expect_aug is not None:
        def aug_of(ts):
            return [t for t in (ts or []) if "OpenArmModerateAug" in str(t.get("_target_"))]
        tr_aug, va_aug = aug_of(p.get("train_transforms")), aug_of(p.get("val_transforms"))
        if args.expect_aug == "moderate":
            ok = len(tr_aug) == 1 and near(tr_aug[0].get("crop_area", -1), 0.9, 1e-9) and near(tr_aug[0].get("rotate_deg", -1), 5.0, 1e-9) \
                 and near(tr_aug[0].get("jitter", -1), 0.2, 1e-9) and bool(tr_aug[0].get("enabled", True))
            check(ok, f"train_transforms must carry OpenArmModerateAug(crop_area=0.9, rotate_deg=5, jitter=0.2): {tr_aug}")
            check(not va_aug, "val_transforms must not augment")
        elif args.expect_aug == "none":
            check(not tr_aug and not va_aug, "no augmentation expected")
        else:
            check(False, f"unknown --expect-aug {args.expect_aug}")
    if args.check_paths:
        check(os.path.isfile(os.path.join(d["dataset_dirs"][0], "meta", "info.json")), f"dataset root missing: {d['dataset_dirs'][0]}")
        check(os.path.isfile(m["action_dit_pretrained_path"]), f"ActionDiT backbone missing: {m['action_dit_pretrained_path']}")
        check(d["pretrained_norm_stats"] and os.path.isfile(d["pretrained_norm_stats"]), f"stats json missing: {d['pretrained_norm_stats']}")
        check(os.path.isdir(d["text_embedding_cache_dir"]), f"text cache dir missing: {d['text_embedding_cache_dir']}")
    fps_ok = {int(v) for v in str(args.expect_fps).split(",") if v}
    raw_a, raw_s = (int(v) for v in str(args.expect_raw_dims).split(","))
    proc_a, proc_s = (int(v) for v in str(args.expect_proc_dims).split(","))
    if args.expect_layout is not None:
        check(d["concat_multi_camera"] == args.expect_layout, f"layout {d['concat_multi_camera']} != {args.expect_layout}")
    if args.expect_video_size is not None:
        eh, ew = (int(v) for v in args.expect_video_size.lower().split("x"))
        check([h, w] == [eh, ew], f"video_size {d['video_size']} != {args.expect_video_size}")
    if args.check_meta:
        is_openarm = "openarm" in str(d["_target_"])
        if is_openarm:
            from fastwam.datasets.lerobot.openarm_dataset import OpenArmLeRobotDatasetMetadata as MetaCls, CANONICAL_VIEW_KEYS
        else:
            from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata as MetaCls
        n_human = 0
        for root in d["dataset_dirs"]:
            meta = MetaCls(repo_id=root, root=root)
            check(int(meta.fps) in fps_ok, f"{root}: fps {meta.fps} not in {sorted(fps_ok)}")
            if is_openarm:
                check(meta.view_key_map["observation.images.camera_ego_right"] is not None, f"{root}: no right view")
                n_human += int(meta.is_human)
            else:
                check(set(meta.video_keys) >= {f"observation.images.{im['key']}" for im in d["shape_meta"]["images"]}, f"{root}: video keys {meta.video_keys}")
            check(tuple(meta.features["action"]["shape"]) == (raw_a,) and tuple(meta.features["observation.state"]["shape"]) == (raw_s,), f"{root}: raw action/state dims != ({raw_a},{raw_s})")
            print(f"[compose] meta: {root} codebase={meta.info['codebase_version']} fps={meta.fps} episodes={meta.total_episodes} frames={meta.total_frames} tasks={len(meta.tasks)} video_keys={meta.video_keys}" + (f" human={meta.is_human} action_dims={int(meta.action_dim_mask.sum())}" if is_openarm else ""))
        if is_openarm:
            print(f"[compose] openarm subsets: {len(d['dataset_dirs'])} (human {n_human}, robot {len(d['dataset_dirs']) - n_human})")
    proc = instantiate(cfg.data.train.processor)  # light: no model, no dataset
    check(proc.action_output_dim == proc_a and proc.proprio_output_dim == proc_s, f"processor dims != ({proc_a},{proc_s})")
    print(
        f"[compose] task={args.task} model={m['_target_'].split('.')[-1]} eff_batch={eff} = {args.world} gpu x pd{bs} x GA{ga} "
        f"| max_steps={ms} save_every={c['save_every']} keep={c.get('save_total_limit')} eval_every={c['eval_every']} "
        f"| lr={c['learning_rate']} wd={c['weight_decay']} sched={c['lr_scheduler_type']} bf16={c['mixed_precision']=='bf16'} "
        f"| frames={nf} ratio={ratio} video={h}x{w} layout={d['concat_multi_camera']} norm={p['norm_default_mode']} "
        f"| warmup={c.get('warmup_steps')} min_lr_ratio={c.get('min_lr_ratio')} aug={[t['_target_'].split('.')[-1] for t in (p.get('train_transforms') or [])]} "
        f"| grad_ckpt={m['mot_checkpoint_mixed_attn']} resume={c.get('resume')} wandb={c['wandb']['project']}/{c['wandb']['name']}"
    )
    if fails:
        for f in fails:
            print("[compose] FAIL:", f)
        print(f"FAILURES: {len(fails)}")
        sys.exit(1)
    print("FAILURES: 0")


if __name__ == "__main__":
    main()
