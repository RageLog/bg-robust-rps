"""
verify_multiseed_splits.py - Pre-flight leakage check for the multi-seed run.

For every (ablation, seed) pair, reconstructs the StratifiedGroupKFold partition
that cv_data_loader.get_cv_folds() would produce and asserts zero group overlap
between train and val in every fold. No model training or GPU required.
The fold split is determined solely by ablation data and seed; head/tune are irrelevant.

Environment variables (shared with train_multiseed.py):
  MULTISEED_SEEDS     comma-separated seed integers (default: "123,2024")
  MULTISEED_ABLATIONS comma-separated ablation tags in uppercase (default: "FULL")
  MULTISEED_CV        number of folds (default: "5")
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from sklearn.model_selection import StratifiedGroupKFold

SEEDS = [int(s) for s in os.environ.get("MULTISEED_SEEDS", "123,2024").split(",") if s.strip()]
ABLATIONS = [a.strip().upper() for a in os.environ.get("MULTISEED_ABLATIONS", "FULL").split(",") if a.strip()]
CV = int(os.environ.get("MULTISEED_CV", "5"))


def gather(ablation):
    label_to_index = {n: i for i, n in enumerate(config.CLASS_NAMES)}
    target = os.path.join(config.BASE_DIR, "datasets", f"synthetic_{ablation.lower()}")
    paths, labels = [], []
    for pool in ("train", "val"):
        base = os.path.join(target, pool)
        if not os.path.isdir(base):
            continue
        for cls in config.CLASS_NAMES:
            d = os.path.join(base, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".png")):
                    paths.append(os.path.join(d, f))
                    labels.append(label_to_index[cls])
    paths = np.array(paths)
    labels = np.array(labels)
    groups = []
    for p in paths:
        fn = os.path.basename(p)
        try:
            base = fn.split("_", 1)[1].rsplit("_syn", 1)[0]
        except IndexError:
            base = fn
        groups.append(base)
    return paths, labels, np.array(groups)


def run():
    out = {"cv": CV, "ablations": {}}
    all_ok = True
    missing = []

    for ablation in ABLATIONS:
        paths, labels, groups = gather(ablation)
        if len(paths) == 0:
            print(f"[VERIFY][WARN] no data found under datasets/synthetic_{ablation.lower()} -- skipping {ablation}")
            missing.append(ablation)
            continue
        print(f"[VERIFY] {ablation}: {len(paths)} images | {len(np.unique(groups))} groups | "
              f"class counts {np.bincount(labels, minlength=len(config.CLASS_NAMES)).tolist()}")
        ab_rec = {"n_images": int(len(paths)), "n_groups": int(len(np.unique(groups))), "seeds": {}}
        for seed in SEEDS:
            sgkf = StratifiedGroupKFold(n_splits=CV, shuffle=True, random_state=seed)
            rec = {"leakage_free": True, "folds": []}
            for fold, (tr, va) in enumerate(sgkf.split(paths, labels, groups), 1):
                overlap = len(set(groups[tr]) & set(groups[va]))
                if overlap > 0:
                    rec["leakage_free"] = False
                    all_ok = False
                rec["folds"].append({"fold": fold, "train": int(len(tr)), "val": int(len(va)),
                                     "group_overlap": overlap})
                tag = "OK" if overlap == 0 else f"LEAK={overlap}"
                print(f"   {ablation} seed {seed} fold {fold}: "
                      f"train {len(tr):5d} / val {len(va):5d} | overlap {overlap} [{tag}]")
            ab_rec["seeds"][str(seed)] = rec
        out["ablations"][ablation] = ab_rec

    rd = os.path.join(config.BASE_DIR, "reports_multiseed")
    os.makedirs(rd, exist_ok=True)
    path = os.path.join(rd, "split_verification.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[VERIFY] {'all checked splits are leakage-free' if all_ok else 'LEAKAGE DETECTED'} -> {path}")
    if missing:
        print(f"[VERIFY][WARN] no data for ablation(s): {missing} -- stage the corresponding synthetic sets first")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    run()
