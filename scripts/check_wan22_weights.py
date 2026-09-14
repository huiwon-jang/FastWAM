"""Verify the local Wan2.2-TI2V-5B weight files against FastWAM's hash registry.

FastWAM (DiffSynth-style) identifies each weight file by an md5 over its *key names + shapes*
(`fastwam.models.wan22.helpers.io.hash_model_file`) and refuses to load anything unknown.
This script recomputes those hashes WITHOUT materialising tensors (safetensors are opened
lazily; .pth files are torch.load'ed with mmap=True), so it is safe on a small-RAM pod.

Usage:
  python scripts/check_wan22_weights.py --base /data/huiwon/checkpoints/fastwam_base [--skip-t5]
Exit code 0 iff every checked file matches the registry hash.
"""
import argparse
import glob
import hashlib
import os
import sys

import torch
from safetensors import safe_open

from fastwam.models.wan22.helpers.loader import WAN22_MODEL_REGISTRY


def _keys_from_safetensors(path):
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            out[key] = list(f.get_slice(key).get_shape())
    return out


def _keys_from_bin(path):
    sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if len(sd) == 1:
        for wrapper in ("state_dict", "module", "model_state"):
            if wrapper in sd:
                sd = sd[wrapper]
                break

    def conv(d):
        o = {}
        for k, v in d.items():
            o[k] = list(v.shape) if isinstance(v, torch.Tensor) else conv(v)
        return o

    return conv(sd)


def _to_str(keys_dict, with_shape=True):
    # verbatim port of io._convert_keys_dict_to_single_str
    keys = []
    for key, value in keys_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                keys.append(key + "|" + _to_str(value, with_shape=with_shape))
            else:
                if with_shape:
                    keys.append(key + ":" + "_".join(map(str, list(value))))
                keys.append(key)
    keys.sort()
    return ",".join(keys)


def hash_files(paths):
    merged = {}
    for p in paths:
        merged.update(_keys_from_safetensors(p) if p.endswith(".safetensors") else _keys_from_bin(p))
    return hashlib.md5(_to_str(merged).encode("UTF-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="DIFFSYNTH_MODEL_BASE_PATH")
    ap.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    ap.add_argument("--skip-t5", action="store_true", help="skip the 11 GB umT5 encoder (.pth)")
    ap.add_argument("--skip-vae", action="store_true")
    args = ap.parse_args()

    registry = {c["model_name"]: c["model_hash"] for c in WAN22_MODEL_REGISTRY}
    root = os.path.join(args.base, args.model_id)
    checks = [("wan_video_dit", sorted(glob.glob(os.path.join(root, "diffusion_pytorch_model*.safetensors"))))]
    if not args.skip_vae:
        checks.append(("wan_video_vae", [os.path.join(root, "Wan2.2_VAE.pth")]))
    if not args.skip_t5:
        checks.append(("wan_video_text_encoder", [os.path.join(root, "models_t5_umt5-xxl-enc-bf16.pth")]))

    ok = True
    for name, paths in checks:
        if not paths or not all(os.path.isfile(p) for p in paths):
            print(f"[FAIL] {name}: missing file(s) {paths}")
            ok = False
            continue
        h = hash_files(paths)
        match = h == registry[name]
        ok &= match
        print(f"[{'OK' if match else 'FAIL'}] {name}: hash={h} expected={registry[name]} files={[os.path.basename(p) for p in paths]}")
    tok = os.path.join(args.base, "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl")
    tok_ok = all(os.path.isfile(os.path.join(tok, f)) for f in ("tokenizer.json", "spiece.model", "tokenizer_config.json"))
    print(f"[{'OK' if tok_ok else 'FAIL'}] tokenizer dir {tok}")
    ok &= tok_ok
    print("WEIGHTS_OK" if ok else "WEIGHTS_BAD")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
