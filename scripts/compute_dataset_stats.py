"""Compute FastWAM's `dataset_stats.json` for a task config on CPU (no model, no GPU).

Equivalent to what `RobotVideoDataset.__init__` does on the first run when `pretrained_norm_stats` is null
(BaseLerobotDataset.get_dataset_stats: per-episode parquet pass, stats taken AFTER action_state_transforms),
but as a standalone, idempotent step so every (re)start of the training job — including resumes — loads
the same stats file instantly.

Usage (inside the training Job, from the repo root):
  python scripts/compute_dataset_stats.py task=droid_fastwam_5b_b128_200k +stats_output_dir=/data/huiwon/data/droid_fastwam
Writes <stats_output_dir>/dataset_stats.json atomically (tmp + rename). Exit 0 on success.
"""
import json
import os
import shutil
import tempfile
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging()
    out_dir = cfg.get("stats_output_dir")
    if not out_dir:
        raise ValueError("pass +stats_output_dir=<dir>")
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, "dataset_stats.json")
    if os.path.isfile(final_path) and not bool(cfg.get("overwrite", False)):
        logger.info("dataset_stats.json already exists at %s (pass +overwrite=true to recompute)", final_path)
        print("STATS_EXISTS", final_path)
        return

    work_dir = tempfile.mkdtemp(prefix="fastwam_stats_", dir=out_dir)
    misc.register_work_dir(work_dir)  # RobotVideoDataset writes <work_dir>/dataset_stats.json
    ds_cfg = OmegaConf.to_container(cfg.data.train, resolve=True)
    ds_cfg["pretrained_norm_stats"] = None  # force the stats pass
    t0 = time.time()
    ds = instantiate(OmegaConf.create(ds_cfg))
    tmp_path = os.path.join(work_dir, "dataset_stats.json")
    if not os.path.isfile(tmp_path):
        raise RuntimeError(f"dataset did not write {tmp_path}")
    os.replace(tmp_path, final_path)
    shutil.rmtree(work_dir, ignore_errors=True)
    with open(final_path) as f:
        stats = json.load(f)
    logger.info(
        "wrote %s in %.1fs: num_episodes=%s num_transition=%s dataset_len=%d",
        final_path, time.time() - t0, stats.get("num_episodes"), stats.get("num_transition"), len(ds),
    )
    for kind in ("action", "state"):
        for key, st in stats[kind].items():
            logger.info("  %s[%s] global_q01=%s", kind, key, [round(v, 4) for v in st["global_q01"]])
            logger.info("  %s[%s] global_q99=%s", kind, key, [round(v, 4) for v in st["global_q99"]])
    print("STATS_OK", final_path)


if __name__ == "__main__":
    main()
