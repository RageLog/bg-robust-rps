"""
verify_grouping_logic.py — Local proof that cv_data_loader.py's grouping
prevents AUGMENTATION leakage (a synthetic kid never lands in a different
fold from its parent raw image), without requiring the synthetic dataset.

Produces:
  paper/data_provenance/grouping_proof.md
  paper/data_provenance/grouping_proof.json

Method:
  1. Build a synthetic-style filename pool from raw images (5 augmentations
     per parent), mirroring exactly how generate_data.py names files:
       <class>_<basename>_syn<ii>.jpg
  2. Apply the same group attribute extraction as cv_data_loader.py:62-69:
       basename = filename.split('_', 1)[1].rsplit('_syn', 1)[0]
  3. Run sklearn.model_selection.StratifiedGroupKFold(n_splits=5,
     shuffle=True, random_state=42) (identical config).
  4. For each fold, intersect train/val group sets. If the intersection
     is empty for ALL folds, the loader provably prevents augmentation
     leakage on this filename schema.

NOTE: This proves AUGMENTATION-leakage prevention, NOT subject-leakage.
The Kaggle subset used does not contain per-subject metadata, so
true subject-level grouping is not possible from filenames alone.
This limitation must be disclosed in the manuscript.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "reports" / "aggregated"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["paper", "rock", "scissors"]  # 'none' generated downstream
AUG_FACTOR = 5
SEED = 42
K = 5


def derive_group(fname: str) -> str:
    """Replicates cv_data_loader.py:62-69 verbatim."""
    try:
        base = fname.split("_", 1)[1].rsplit("_syn", 1)[0]
    except IndexError:
        base = fname
    return base


def main():
    paths = []
    labels = []
    for label_idx, c in enumerate(CLASSES):
        cdir = RAW / c
        if not cdir.exists():
            continue
        # mirror generate_data.py:88 naming exactly
        for raw_img in sorted(cdir.iterdir()):
            if raw_img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            base = raw_img.stem
            for i in range(AUG_FACTOR):
                synthetic_name = f"{c}_{base}_syn{i:02d}.jpg"
                paths.append(synthetic_name)
                labels.append(label_idx)

    paths = np.array(paths)
    labels = np.array(labels)
    groups = np.array([derive_group(p) for p in paths])

    n_total = len(paths)
    n_unique_groups = len(np.unique(groups))
    print(f"[INFO] Built simulated synthetic pool: {n_total} files, "
          f"{n_unique_groups} unique groups (parent images)")
    assert n_unique_groups * AUG_FACTOR == n_total, (
        "Group derivation lost or duplicated files; investigate naming.")

    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)

    fold_reports = []
    all_clean = True
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(paths, labels, groups), 1):
        g_train = set(groups[train_idx])
        g_val = set(groups[val_idx])
        intersect = g_train & g_val
        clean = len(intersect) == 0
        all_clean = all_clean and clean
        # class balance
        cb_train = {CLASSES[c]: int((labels[train_idx] == c).sum()) for c in range(len(CLASSES))}
        cb_val = {CLASSES[c]: int((labels[val_idx] == c).sum()) for c in range(len(CLASSES))}
        fold_reports.append({
            "fold": fold,
            "n_train_files": int(len(train_idx)),
            "n_val_files": int(len(val_idx)),
            "n_train_groups": int(len(g_train)),
            "n_val_groups": int(len(g_val)),
            "group_overlap": int(len(intersect)),
            "leakage_clean": clean,
            "class_balance_train": cb_train,
            "class_balance_val": cb_val,
        })
        status = "CLEAN" if clean else "LEAK"
        print(f"  Fold {fold}: {len(train_idx):6d} train / {len(val_idx):5d} val | "
              f"group overlap = {len(intersect):3d} [{status}]")

    out_json = {
        "schema": "<class>_<basename>_syn<ii>.jpg (mirrors generate_data.py:88)",
        "group_extraction_rule": "filename.split('_', 1)[1].rsplit('_syn', 1)[0]",
        "splitter": f"StratifiedGroupKFold(n_splits={K}, shuffle=True, random_state={SEED})",
        "raw_image_count": n_unique_groups,
        "augmentation_factor": AUG_FACTOR,
        "simulated_synthetic_count": n_total,
        "all_folds_clean": all_clean,
        "folds": fold_reports,
        "interpretation": (
            "PROVES that the cv_data_loader.py grouping prevents "
            "augmentation-leakage (synthetic siblings never appear in different "
            "folds from their parent raw image) on the production filename "
            "schema. Does NOT prove subject-leakage prevention, which would "
            "require per-subject metadata that is absent from the public Kaggle "
            "subset used in this study. This caveat must appear in Methods and "
            "Limitations."
        ),
    }
    (OUT / "grouping_proof.json").write_text(json.dumps(out_json, indent=2))

    md_lines = [
        "# Augmentation-Leakage Grouping Proof (Local)\n",
        f"- **Splitter**: `{out_json['splitter']}`\n",
        f"- **Filename schema**: `{out_json['schema']}`\n",
        f"- **Group extraction rule**: `{out_json['group_extraction_rule']}`\n",
        f"- **Raw parent images**: {n_unique_groups:,}\n",
        f"- **Simulated synthetic files (factor {AUG_FACTOR})**: {n_total:,}\n",
        f"- **All folds clean (zero group overlap)**: **{all_clean}**\n\n",
        "| Fold | Train files | Val files | Train groups | Val groups | Group overlap | Status |",
        "|-----:|------------:|----------:|-------------:|-----------:|--------------:|:------:|",
    ]
    for f in fold_reports:
        md_lines.append(
            f"| {f['fold']} | {f['n_train_files']:,} | {f['n_val_files']:,} | "
            f"{f['n_train_groups']:,} | {f['n_val_groups']:,} | "
            f"{f['group_overlap']} | {'CLEAN' if f['leakage_clean'] else 'LEAK'} |"
        )
    md_lines += [
        "",
        "## Interpretation",
        out_json["interpretation"],
    ]
    (OUT / "grouping_proof.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n[SAVED] {OUT / 'grouping_proof.json'}")
    print(f"[SAVED] {OUT / 'grouping_proof.md'}")
    print(f"\n[VERDICT] {'AUGMENTATION-LEAKAGE PROVABLY PREVENTED' if all_clean else 'LEAK DETECTED'}")


if __name__ == "__main__":
    main()
