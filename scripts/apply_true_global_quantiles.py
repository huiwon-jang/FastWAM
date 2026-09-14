"""Build a FastWAM dataset_stats.json whose q01/q99 fields are TRUE global quantiles.

Reads the FastWAM-native stats file (kept untouched), replaces action/state `global_q01`/`global_q99` with
the values from compute_true_global_quantiles.py, broadcasts them into `stepwise_q01`/`stepwise_q99` (only
consumed with use_stepwise_action_norm=true, kept consistent anyway), leaves every other field as-is (schema
unchanged) and adds the marker `"huiwon_true_global_quantiles": true` that the launcher gate requires.

  python scripts/apply_true_global_quantiles.py --native dataset_stats.json --true true_global_quantiles.json \
      --out dataset_stats.true_global.json [--force]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--true", dest="true_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.force:
        print(f"[apply] {args.out} exists (use --force to rebuild)")
        return
    native = json.load(open(args.native))
    trueq = json.load(open(args.true_path))
    if native.get("huiwon_true_global_quantiles"):
        sys.exit(f"[apply] refusing: --native {args.native} is already an override file")
    out = json.loads(json.dumps(native))  # deep copy
    report = []
    for kind in ("action", "state"):
        keys = list(out[kind].keys())
        if len(keys) != 1:
            sys.exit(f"[apply] expected exactly one {kind} key, got {keys}")
        key = keys[0]
        st = out[kind][key]
        q01, q99 = trueq[kind]["q01"], trueq[kind]["q99"]
        if len(q01) != len(st["global_q01"]):
            sys.exit(f"[apply] dim mismatch for {kind}: {len(q01)} vs {len(st['global_q01'])}")
        for d in range(len(q01)):
            rng_native = st["global_q99"][d] - st["global_q01"][d]
            rng_true = q99[d] - q01[d]
            report.append((kind, d, st["global_q01"][d], q01[d], st["global_q99"][d], q99[d],
                           rng_true / rng_native if rng_native > 1e-8 else float("nan")))
        st["global_q01"], st["global_q99"] = [float(v) for v in q01], [float(v) for v in q99]
        steps = np.array(st["stepwise_q01"]).shape[0]
        st["stepwise_q01"] = [list(map(float, q01))] * steps
        st["stepwise_q99"] = [list(map(float, q99))] * steps
    out["huiwon_true_global_quantiles"] = True
    out["huiwon_true_global_quantiles_meta"] = {
        "source": os.path.abspath(args.true_path), "native": os.path.abspath(args.native),
        "created": datetime.datetime.utcnow().isoformat() + "Z", **trueq.get("meta", {}),
    }
    with open(args.out + ".tmp", "w") as f:
        json.dump(out, f, indent=2)
    os.replace(args.out + ".tmp", args.out)
    print(f"{'kind':6s} {'dim':>3s} {'q01_native':>11s} {'q01_true':>10s} {'q99_native':>11s} {'q99_true':>10s} {'range_true/native':>17s}")
    for kind, d, a, b, c, e, r in report:
        print(f"{kind:6s} {d:3d} {a:11.4f} {b:10.4f} {c:11.4f} {e:10.4f} {r:17.3f}")
    print(f"APPLY_OK {args.out}")


if __name__ == "__main__":
    main()
