"""
aggregate_multiseed.py - Aggregate multi-seed stability results (revision K1).

For every cell (model, ablation, head, tune) that was re-seeded, collects the
5-fold accuracy from the original seed-42 run (reports/ or Drive reports/) and
from each additional seed (reports_multiseed/seedNN/), then reports per cell:
  - per-seed 5-fold mean,
  - across-seed mean +/- std (the stability number the editor asked for),
  - pooled fold-level mean +/- std.
Also prints a global summary (mean and worst-case across-seed std over all cells).
Writes reports_multiseed/multiseed_summary.json (consumed by make_multiseed_table.py).

Env knobs: MULTISEED_SEEDS, MULTISEED_MODELS, MULTISEED_ABLATIONS (shared with trainer).
"""
import os
import re
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

SEEDS_NEW = [int(s) for s in os.environ.get("MULTISEED_SEEDS", "123,2024").split(",") if s.strip()]
MODELS = [m.strip() for m in os.environ.get(
    "MULTISEED_MODELS",
    "DenseNet121,EfficientNetV2B0,ResNet50,MobileNetV3Small,VGG16").split(",") if m.strip()]
ABLATIONS = [a.strip().upper() for a in os.environ.get("MULTISEED_ABLATIONS", "FULL").split(",") if a.strip()]
ORIG_SEED = 42
CELL_RE = re.compile(r"kfold_summary_(standard|attention|spatial_pooling)_(standard|progressive)\.json$")


def fold_scores(roots, model, ablation, head, tune):
    rel = os.path.join(model, ablation.upper(), f"kfold_summary_{head}_{tune}.json")
    for root in roots:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return json.load(open(p)).get("fold_scores")
    return None


def discover_cells(model, ablation):
    found = set()
    roots = [os.path.join(config.BASE_DIR, "reports"), os.path.join(config.DRIVE_DIR, "reports")]
    roots += [os.path.join(config.BASE_DIR, "reports_multiseed", f"seed{s}") for s in SEEDS_NEW]
    roots += [os.path.join(config.DRIVE_DIR, "revize", "multiseed", f"seed{s}", "reports") for s in SEEDS_NEW]
    for root in roots:
        d = os.path.join(root, model, ablation.upper())
        if os.path.isdir(d):
            for f in os.listdir(d):
                m = CELL_RE.match(f)
                if m:
                    found.add((m.group(1), m.group(2)))
    return sorted(found)


def run():
    seed42_roots = [os.path.join(config.BASE_DIR, "reports"), os.path.join(config.DRIVE_DIR, "reports")]
    MIN_FOLD = float(os.environ.get("MULTISEED_MIN_FOLD", "0.5"))  # folds below this threshold are treated as diverged
    cells = {}
    all_std = []
    corrupted = []
    for ab in ABLATIONS:
        for model in MODELS:
            for head, tune in discover_cells(model, ab):
                sf = {}
                f42 = fold_scores(seed42_roots, model, ab, head, tune)
                if f42:
                    sf[ORIG_SEED] = f42
                for s in SEEDS_NEW:
                    roots = [os.path.join(config.BASE_DIR, "reports_multiseed", f"seed{s}"),
                             os.path.join(config.DRIVE_DIR, "revize", "multiseed", f"seed{s}", "reports")]
                    fr = fold_scores(roots, model, ab, head, tune)
                    if fr:
                        sf[s] = fr
                # Exclude seed-runs that contain any fold below MIN_FOLD (diverged or OOM).
                for s in list(sf.keys()):
                    if sf[s] and min(sf[s]) < MIN_FOLD:
                        corrupted.append({"cell": f"{model}|{ab}|{head}|{tune}", "seed": int(s),
                                          "fold_scores": [round(x, 4) for x in sf[s]]})
                        del sf[s]
                if not sf:
                    continue
                per_seed = {int(k): float(np.mean(v)) for k, v in sf.items()}
                pooled = [x for v in sf.values() for x in v]
                means = list(per_seed.values())
                std = float(np.std(means))
                key = f"{model}|{ab}|{head}|{tune}"
                cells[key] = {
                    "model": model, "ablation": ab, "head": head, "tune": tune,
                    "seeds_used": sorted(per_seed.keys()),
                    "per_seed_mean": per_seed,
                    "across_seed_mean": float(np.mean(means)),
                    "across_seed_std": std,
                    "pooled_fold_mean": float(np.mean(pooled)),
                    "pooled_fold_std": float(np.std(pooled)),
                    "n_seeds": len(per_seed),
                }
                if len(per_seed) > 1:
                    all_std.append(std)
                print(f"{key:55s} seeds={cells[key]['seeds_used']} "
                      f"mean={cells[key]['across_seed_mean']*100:6.2f} "
                      f"std={std*100:.2f}pp")

    summary = {
        "n_cells": len(cells),
        "mean_across_seed_std_pp": float(np.mean(all_std) * 100) if all_std else None,
        "max_across_seed_std_pp": float(np.max(all_std) * 100) if all_std else None,
        "cells_with_multiple_seeds": len(all_std),
        "corrupted_runs": len(corrupted),
    }
    out = os.path.join(config.BASE_DIR, "reports_multiseed", "multiseed_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"original_seed": ORIG_SEED, "new_seeds": SEEDS_NEW,
                   "summary": summary, "corrupted": corrupted, "cells": cells}, f, indent=2)
    print(f"\n[AGG] {summary['n_cells']} clean cells | mean std "
          f"{summary['mean_across_seed_std_pp']}pp | max std {summary['max_across_seed_std_pp']}pp")
    if corrupted:
        print(f"[AGG] {len(corrupted)} seed-run(s) excluded (diverged fold < {MIN_FOLD}); re-run:")
        for c in corrupted:
            print(f"        {c['cell']}  seed {c['seed']}  fold scores {c['fold_scores']}")
    print(f"[AGG] -> {out}")


if __name__ == "__main__":
    run()
