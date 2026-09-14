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
    if args.check_paths:
        check(os.path.isfile(os.path.join(d["dataset_dirs"][0], "meta", "info.json")), f"dataset root missing: {d['dataset_dirs'][0]}")
        check(os.path.isfile(m["action_dit_pretrained_path"]), f"ActionDiT backbone missing: {m['action_dit_pretrained_path']}")
        check(d["pretrained_norm_stats"] and os.path.isfile(d["pretrained_norm_stats"]), f"stats json missing: {d['pretrained_norm_stats']}")
        check(os.path.isdir(d["text_embedding_cache_dir"]), f"text cache dir missing: {d['text_embedding_cache_dir']}")
    if args.check_meta:
        from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata
        root = d["dataset_dirs"][0]
        meta = LeRobotDatasetMetadata(repo_id=root, root=root)
        check(meta.fps == 15, f"fps {meta.fps} != 15")
        check(set(meta.video_keys) >= {f"observation.images.{im['key']}" for im in d["shape_meta"]["images"]}, f"video keys {meta.video_keys}")
        check(tuple(meta.features["action"]["shape"]) == (28,) and tuple(meta.features["observation.state"]["shape"]) == (14,), "raw action/state dims changed")
        print(f"[compose] meta: codebase={meta.info['codebase_version']} episodes={meta.total_episodes} frames={meta.total_frames} tasks={len(meta.tasks)} video_keys={meta.video_keys}")
    proc = instantiate(cfg.data.train.processor)  # light: no model, no dataset
    check(proc.action_output_dim == 8 and proc.proprio_output_dim == 8, "processor dims")
    print(
        f"[compose] task={args.task} model={m['_target_'].split('.')[-1]} eff_batch={eff} = {args.world} gpu x pd{bs} x GA{ga} "
        f"| max_steps={ms} save_every={c['save_every']} keep={c.get('save_total_limit')} eval_every={c['eval_every']} "
        f"| lr={c['learning_rate']} wd={c['weight_decay']} sched={c['lr_scheduler_type']} bf16={c['mixed_precision']=='bf16'} "
        f"| frames={nf} ratio={ratio} video={h}x{w} layout={d['concat_multi_camera']} norm={p['norm_default_mode']} "
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
