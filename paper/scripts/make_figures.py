"""
make_figures.py — Generate paper-ready vector figures (PDF) from Q1 CSVs and per-fold JSON.

Outputs (paper/figures/):
  F01_synth_vs_no_synth.pdf       Bar chart, mean acc per backbone, FULL vs NO_SYNTH
  F02_ablation_heatmap.pdf        Heatmap, accuracy(model x ablation), standard/standard
  F03_generalization_gap.pdf      Scatter, in-dist vs zero-shot, per backbone
  F04_tune_paired_dotplot.pdf     Per-config standard-vs-progressive dot plot
  F05_topcfg_confusion.pdf        4x4 confusion matrix for the top FULL config

All vector PDFs, ~3.5in single-column IEEE width, color-blind safe palette.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
AGG = REPO / "reports" / "aggregated"
Q1 = AGG
STATS = AGG
PROV = AGG
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # editable text
})

# Color-blind friendly palette (Okabe-Ito subset)
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"[SAVED] {p}  ({p.stat().st_size//1024} KB)")


# ---------------------------------------------------------------- F01
def f01_synth_vs_no_synth():
    df = pd.read_csv(STATS / "per_config_summary.csv")
    pivot_full = df[(df.ablation == "FULL") & (df["head"] == "standard") & (df.tune == "standard")]
    pivot_no = df[(df.ablation == "NO_SYNTH") & (df["head"] == "standard") & (df.tune == "standard")]
    models = sorted(set(pivot_full["model"]) & set(pivot_no["model"]))
    full_means = [pivot_full[pivot_full.model == m]["acc_mean"].iloc[0] * 100 for m in models]
    full_stds = [pivot_full[pivot_full.model == m]["acc_std"].iloc[0] * 100 for m in models]
    no_means = [pivot_no[pivot_no.model == m]["acc_mean"].iloc[0] * 100 for m in models]
    no_stds = [pivot_no[pivot_no.model == m]["acc_std"].iloc[0] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(models))
    w = 0.38
    ax.bar(x - w/2, no_means, w, yerr=no_stds, color=PAL[1],
           label="NO_SYNTH", capsize=2, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, full_means, w, yerr=full_stds, color=PAL[0],
           label="FULL", capsize=2, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("EfficientNetV2B0", "EffV2B0").replace("MobileNetV3Small", "MobNetV3S")
                         for m in models], rotation=20, ha="right")
    ax.set_ylabel("5-fold validation accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    save(fig, "F01_synth_vs_no_synth.pdf")


# ---------------------------------------------------------------- F02
def f02_ablation_heatmap():
    df = pd.read_csv(STATS / "per_config_summary.csv")
    sub = df[(df["head"] == "standard") & (df.tune == "standard")]
    pv = sub.pivot_table(index="model", columns="ablation", values="acc_mean") * 100
    # Ordering
    model_order = ["EfficientNetV2B0", "MobileNetV3Small", "ResNet50", "DenseNet121", "VGG16"]
    ab_order = ["NO_SYNTH", "REMBG_ONLY", "RANDBG", "INDOOR", "GAN", "STYLE_TRANSFER",
                "RATIO_1X", "NO_SHIFT", "NO_ALPHA", "FULL"]
    model_order = [m for m in model_order if m in pv.index]
    ab_order = [a for a in ab_order if a in pv.columns]
    pv = pv.reindex(index=model_order, columns=ab_order)

    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    im = ax.imshow(pv.values, aspect="auto", cmap="viridis", vmin=40, vmax=100)
    ax.set_xticks(range(len(ab_order)))
    ax.set_xticklabels(ab_order, rotation=30, ha="right")
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels([m.replace("EfficientNetV2B0", "EffV2B0")
                          .replace("MobileNetV3Small", "MobNetV3S") for m in model_order])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy (%)")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                         color="white" if v < 75 else "black", fontsize=6.5)
    save(fig, "F02_ablation_heatmap.pdf")


# ---------------------------------------------------------------- F03
def f03_generalization_gap():
    df = pd.read_csv(Q1 / "Q1_Table4_Generalization_Gap.csv", sep=";")
    for col in ["İç Veri: Mean Acc", "Dış Veri: Zero-Shot Acc", "Generalization Gap"]:
        df[col] = df[col].str.rstrip("%").astype(float)
    df = df[(df["Head"] == "Standard")]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    models = sorted(df["Model"].unique())
    for i, m in enumerate(models):
        sub = df[df["Model"] == m]
        ax.scatter(sub["İç Veri: Mean Acc"], sub["Dış Veri: Zero-Shot Acc"],
                    color=PAL[i % len(PAL)], label=m.replace("EfficientNetV2B0", "EffV2B0")
                                                      .replace("MobileNetV3Small", "MobNetV3S"),
                    s=18, alpha=0.85, edgecolor="black", linewidth=0.3)
    lims = [40, 100]
    ax.plot(lims, lims, "k--", lw=0.6, alpha=0.5, label="$y=x$ (no gap)")
    # MediaPipe+SVM external accuracy from T14 as a horizontal reference
    ax.axhline(59.17, color=PAL[5], linestyle=":", lw=1.0, alpha=0.85,
               label="MediaPipe+SVM (ext. 59.2%)")
    ax.set_xlim(*lims); ax.set_ylim(50, 100)
    ax.set_xlabel("In-distribution accuracy (%)")
    ax.set_ylabel("HaGRID zero-shot accuracy (%)")
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=7)
    ax.grid(linestyle=":", alpha=0.4)
    save(fig, "F03_generalization_gap.pdf")


# ---------------------------------------------------------------- F04
def f04_tune_paired_dotplot():
    df = pd.read_csv(STATS / "pairwise_results.csv")
    sub = df[df["family"] == "C_tune_standard_vs_progressive"].copy()
    sub["label"] = sub["model"].str.replace("EfficientNetV2B0", "EffV2B0")\
                               .str.replace("MobileNetV3Small", "MobNetV3S") + "/" + sub["ablation"]
    sub = sub.sort_values("delta")
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(3.5, max(2.5, 0.18 * len(sub))))
    ax.scatter(sub["mean_progressive"]*100, y, color=PAL[1], label="Progressive",
               s=18, edgecolor="black", linewidth=0.3)
    ax.scatter(sub["mean_standard"]*100, y, color=PAL[0], label="Standard",
               s=18, edgecolor="black", linewidth=0.3)
    for i, row in enumerate(sub.itertuples()):
        ax.plot([row.mean_progressive*100, row.mean_standard*100], [i, i],
                color="gray", lw=0.5, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(sub["label"])
    ax.set_xlabel("5-fold mean accuracy (%)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    save(fig, "F04_tune_paired_dotplot.pdf")


# ---------------------------------------------------------------- F05
def f05_topcfg_confusion():
    """3x2 confusion-matrix grid: per-backbone GLOBAL-BEST (n=5 complete)
    configuration. Complementary to F06, which fixes the protocol at
    FULL/std/std for all backbones; F05 lets every backbone pick its own
    optimum (different ablation may win for different backbones).
    """
    df = pd.read_csv(STATS / "per_config_summary.csv")
    df = df[df["n_folds_complete"] == True]
    base = REPO / "reports"
    classes = ["rock", "paper", "scissors", "none"]
    backbones = ["DenseNet121", "EfficientNetV2B0", "ResNet50",
                 "MobileNetV3Small", "VGG16"]

    # Per-backbone global-best (acc_mean) cell, then aggregate its 5 folds
    cm_data = {}
    for b in backbones:
        sub = df[df["model"] == b].sort_values("acc_mean", ascending=False)
        if not len(sub):
            continue
        top = sub.iloc[0]
        cm_total = np.zeros((4, 4), dtype=int)
        for fold in range(1, 6):
            p = base / top["model"] / top["ablation"] / (
                f"report_{top['model']}_{top['ablation'].lower()}"
                f"_{top['head']}_{top['tune']}_fold_{fold}.json"
            )
            if p.exists():
                with open(p) as fh:
                    d = json.load(fh)
                cm = np.array(d.get("confusion_matrix", []))
                if cm.shape == (4, 4):
                    cm_total += cm
        if cm_total.sum() == 0:
            print(f"[WARN] F05: no CM data for {b} (best {top['ablation']})")
            continue
        cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True) * 100
        cm_data[b] = (top, cm_norm)

    # Global empirical vmin for shared colorbar
    if cm_data:
        all_nz = np.concatenate(
            [m[m > 0].ravel() for _, m in cm_data.values() if (m > 0).any()])
        vmin_global = float(all_nz.min()) * 0.9 if len(all_nz) else 0
    else:
        print("[WARN] F05: no data, skipping")
        return

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.5),
                             constrained_layout=True)
    flat = axes.flatten()
    last_im = None
    for idx, b in enumerate(backbones):
        ax = flat[idx]
        if b not in cm_data:
            ax.set_title(f"{b} (no data)"); ax.axis("off"); continue
        top, cm_norm = cm_data[b]
        last_im = ax.imshow(cm_norm, cmap="Blues",
                            vmin=vmin_global, vmax=100)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(classes, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(
            f"{b}\nbest: {top['ablation']}/{top['head']}/{top['tune']}  "
            f"acc={top['acc_mean']*100:.2f}%", fontsize=7.5)
        for i in range(4):
            for j_ in range(4):
                v = cm_norm[i, j_]
                ax.text(j_, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 50 else "black", fontsize=6.5)
    for spare_idx in range(len(backbones), len(flat)):
        flat[spare_idx].axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     fraction=0.025, pad=0.02, label="Row-norm (%)")
    save(fig, "F05_topcfg_confusion.pdf")


def f06_best_cm_grid():
    """3x2 row-normalised confusion-matrix grid for the five
    backbone/FULL/std/std cells, aggregated across all 5 folds.
    The 6th panel hosts the colorbar.
    """
    base = REPO / "reports"
    targets = [
        ("DenseNet121",      "FULL"),
        ("EfficientNetV2B0", "FULL"),
        ("ResNet50",         "FULL"),
        ("MobileNetV3Small", "FULL"),
        ("VGG16",            "FULL"),
    ]
    classes = ["rock", "paper", "scissors", "none"]

    # Pre-load all 4 cm_norm matrices to compute a global empirical vmin
    cm_norms = {}
    for model, ablation in targets:
        cm_total_pre = np.zeros((4, 4), dtype=int)
        folds_found = 0
        for fold in range(1, 6):
            p = base / model / ablation / (
                f"report_{model}_{ablation.lower()}"
                f"_standard_standard_fold_{fold}.json"
            )
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            cm = np.array(d.get("confusion_matrix", []))
            if cm.shape == (4, 4):
                cm_total_pre += cm
                folds_found += 1
        if folds_found > 0:
            cm_norms[(model, ablation)] = (
                cm_total_pre / cm_total_pre.sum(axis=1, keepdims=True) * 100,
                folds_found,
            )
    all_nz = np.concatenate(
        [m[m > 0].ravel() for m, _ in cm_norms.values() if (m > 0).any()]
    ) if cm_norms else np.array([])
    vmin_global = float(all_nz.min()) * 0.9 if len(all_nz) else 0

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.5), constrained_layout=True)
    flat_axes = axes.flatten()
    last_im = None
    for idx, (model, ablation) in enumerate(targets):
        ax = flat_axes[idx]
        if (model, ablation) not in cm_norms:
            ax.set_title(f"{model} {ablation} (no data)")
            ax.axis("off")
            continue
        cm_norm, n_folds = cm_norms[(model, ablation)]
        last_im = ax.imshow(cm_norm, cmap="Blues", vmin=vmin_global, vmax=100)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(classes, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(f"{model} (n={n_folds} folds)", fontsize=8)
        for i in range(4):
            for j in range(4):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 50 else "black", fontsize=6.5)
    # Hide spare 6th panel (only 5 backbones)
    for spare_idx in range(len(targets), len(flat_axes)):
        flat_axes[spare_idx].axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     fraction=0.025, pad=0.02, label="Row-norm (%)")
    save(fig, "F06_best_cm_grid.pdf")


def f07_gradcam_grid():
    """5x2 grid of Grad-CAM activations: 5 backbones x (correct, error).

    Top row shows a correctly classified sample for each backbone (FULL
    pipeline preferred); bottom row shows a misclassified sample
    sourced from the most informative ablation cell that produced one.
    For DenseNet121, no FULL/standard error sample exists; we show a
    RATIO_1X error instead (rock misread as scissors).
    """
    rep = REPO / "reports"
    backbones = ["DenseNet121", "EfficientNetV2B0", "ResNet50",
                 "MobileNetV3Small", "VGG16"]
    correct = {
        "DenseNet121":      rep / "DenseNet121/FULL_standard_progressive/gradcam/paper/correct/CORRECT_True[paper]_Pred[paper]_100.0pct_paper_Nm8PHXHrGlZy0d6Q_syn04.jpg",
        "EfficientNetV2B0": rep / "EfficientNetV2B0/FULL_standard_progressive/gradcam/paper/correct/CORRECT_True[paper]_Pred[paper]_77.3pct_paper_paper04-108_syn02.jpg",
        "ResNet50":         rep / "ResNet50/FULL_standard_progressive/gradcam/rock/correct/CORRECT_True[rock]_Pred[rock]_100.0pct_rock_rock_513_syn01.jpg",
        "MobileNetV3Small": rep / "MobileNetV3Small/FULL_standard_progressive/gradcam/paper/correct/CORRECT_True[paper]_Pred[paper]_68.8pct_paper_zXY93m62vUNIH4a0_syn04.jpg",
        "VGG16":            rep / "VGG16/FULL_standard_progressive/gradcam/paper/correct/CORRECT_True[paper]_Pred[paper]_100.0pct_paper_HQSUE6P23pvLctuy_syn03.jpg",
    }
    correct_caption = {
        "DenseNet121":      ("paper / FULL", "100%"),
        "EfficientNetV2B0": ("paper / FULL", "77%"),
        "ResNet50":         ("rock / FULL",  "100%"),
        "MobileNetV3Small": ("paper / FULL", "69%"),
        "VGG16":            ("paper / FULL", "100%"),
    }
    error = {
        "DenseNet121":      rep / "DenseNet121/RATIO_1X_standard_standard/gradcam/rock/error/ERROR_True[rock]_Pred[scissors]_57.6pct_rock_gutk3kRhu9AfjYWQ_syn04.jpg",
        "EfficientNetV2B0": rep / "EfficientNetV2B0/FULL_standard_progressive/gradcam/paper/error/ERROR_True[paper]_Pred[scissors]_81.4pct_paper_paper_47_syn03.jpg",
        "ResNet50":         rep / "ResNet50/FULL_standard_standard/gradcam/paper/error/ERROR_True[paper]_Pred[scissors]_99.2pct_paper_paper_254_syn01.jpg",
        "MobileNetV3Small": rep / "MobileNetV3Small/FULL_standard_progressive/gradcam/paper/error/ERROR_True[paper]_Pred[rock]_53.0pct_paper_paper_334_syn04.jpg",
        "VGG16":            rep / "VGG16/GAN_standard_progressive/gradcam/rock/error/ERROR_True[rock]_Pred[none]_84.4pct_rock_rock_128_syn03.jpg",
    }
    error_caption = {
        "DenseNet121":      ("rock→scissors / RATIO_1X",      "58%"),
        "EfficientNetV2B0": ("paper→scissors / FULL",         "81%"),
        "ResNet50":         ("paper→scissors / FULL/std",     "99%"),
        "MobileNetV3Small": ("paper→rock / FULL",             "53%"),
        "VGG16":            ("rock→none / GAN",                "84%"),
    }

    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.4),
                             constrained_layout=True)
    for col, b in enumerate(backbones):
        for row, src in enumerate([correct, error]):
            ax = axes[row, col]
            p = src[b]
            cap_label, cap_conf = (correct_caption if row == 0 else error_caption)[b]
            if p.exists():
                ax.imshow(plt.imread(str(p)))
                ax.set_title(f"{cap_label}\nconf={cap_conf}", fontsize=6.5)
            else:
                ax.text(0.5, 0.5, "no sample", ha="center", va="center",
                        fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        # Backbone label as the column "super-title"
        axes[0, col].set_xlabel(
            b.replace("EfficientNetV2B0", "EffV2B0")
             .replace("MobileNetV3Small", "MobNetV3S"),
            fontsize=7.5, labelpad=2)
        axes[0, col].xaxis.set_label_position("top")
    # Row labels
    axes[0, 0].set_ylabel("Correct", fontsize=8)
    axes[1, 0].set_ylabel("Misclassified", fontsize=8)
    save(fig, "F07_gradcam_grid.pdf")


def f09_zeroshot_classwise_bars():
    """Grouped bar chart of per-class zero-shot F1 for the BEST cell of
    each of the five CNN backbones, plus the auxiliary MediaPipe+SVM
    landmark baseline (overall F1 only, shown as a horizontal reference
    line because per-class breakdown is not retained).

    Source: 005_dev_code/reports/cross_dataset/global_cross_dataset_evaluation.json
    + paper_overleaf/tables/T14_mediapipe_svm.tex (overall macro-F1 58.04 %)
    """
    j = json.loads((AGG / "global_cross_dataset_evaluation.json").read_text())
    mr = j["model_results"]
    backbones = ["DenseNet121", "EfficientNetV2B0", "ResNet50",
                 "MobileNetV3Small", "VGG16"]
    classes = ["rock", "paper", "scissors", "none"]
    best = {}
    for k, v in mr.items():
        for b in backbones:
            if k.startswith(f"RPS_Synthetic_V1_{b}_"):
                acc = v.get("accuracy", 0)
                if b not in best or acc > best[b][1]:
                    best[b] = (k, acc, v)
                break
    # Build matrix [backbone, class] of F1 (%)
    f1 = np.zeros((len(backbones), len(classes)))
    for bi, b in enumerate(backbones):
        if b in best:
            cr = best[b][2]["classification_report"]
            for ci, c in enumerate(classes):
                f1[bi, ci] = cr.get(c, {}).get("f1-score", 0) * 100

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(backbones))
    w = 0.20
    cls_colors = [PAL[0], PAL[1], PAL[2], PAL[3]]
    for ci, c in enumerate(classes):
        ax.bar(x + (ci - 1.5) * w, f1[:, ci], w, color=cls_colors[ci],
               label=c, edgecolor="black", linewidth=0.3)
    # MediaPipe+SVM macro-F1 (58.04 %) as a horizontal reference
    ax.axhline(58.04, color=PAL[5], linestyle="--", lw=1.0,
               label="MediaPipe+SVM macro-F1 (ext. 58.0%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [b.replace("EfficientNetV2B0", "EffV2B0")
          .replace("MobileNetV3Small", "MobNetV3S") for b in backbones],
        rotation=15, ha="right")
    ax.set_ylabel("Per-class zero-shot $F_1$ (%)")
    ax.set_ylim(40, 105)
    ax.legend(loc="lower center", frameon=False, ncol=5, fontsize=7,
              bbox_to_anchor=(0.5, -0.45))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    save(fig, "F09_zeroshot_classwise_bars.pdf")


def f08_cnn_vs_landmark():
    """Bar chart comparing the 5 CNN backbones (FULL/std/std)
    against the auxiliary MediaPipe Hands + RBF-SVM landmark baseline.

    For CNNs we report:
      - In-distribution mean accuracy (5-fold validation, FULL/std/std)
      - Zero-shot HaGRID accuracy (best head/tune for that backbone, from
        Q1_Table4_Generalization_Gap.csv, FULL ablation only)

    For MediaPipe+SVM we report the single auxiliary external accuracy
    from T14_mediapipe_svm.tex (59.17 %).
    """
    # In-distribution accuracies from per_config_summary
    summary = pd.read_csv(STATS / "per_config_summary.csv")
    sub = summary[(summary["ablation"] == "FULL")
                  & (summary["head"] == "standard")
                  & (summary["tune"] == "standard")]
    backbones = ["EfficientNetV2B0", "MobileNetV3Small", "ResNet50",
                 "DenseNet121", "VGG16"]
    in_dist = []
    for m in backbones:
        row = sub[sub["model"] == m]
        in_dist.append(float(row["acc_mean"].iloc[0]) * 100 if len(row) else 0.0)

    # Zero-shot HaGRID accuracies from Q1_Table4 (best head/tune per backbone, FULL only)
    gap = pd.read_csv(Q1 / "Q1_Table4_Generalization_Gap.csv", sep=";")
    for col in ["İç Veri: Mean Acc", "Dış Veri: Zero-Shot Acc",
                "Generalization Gap"]:
        gap[col] = gap[col].str.rstrip("%").astype(float)
    full_gap = gap[gap["Ablasyon"] == "FULL"]
    zero_shot = []
    for m in backbones:
        rows = full_gap[full_gap["Model"] == m]
        zero_shot.append(float(rows["Dış Veri: Zero-Shot Acc"].max())
                         if len(rows) else 0.0)

    mediapipe_acc = 59.17  # from T14_mediapipe_svm.tex (auxiliary)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.arange(len(backbones))
    w = 0.36
    ax.bar(x - w/2, in_dist, w, color=PAL[0], label="In-distribution (5-fold)",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, zero_shot, w, color=PAL[1], label="HaGRID zero-shot",
           edgecolor="black", linewidth=0.4)
    # MediaPipe+SVM as a horizontal reference line
    ax.axhline(mediapipe_acc, color=PAL[2], linestyle="--", linewidth=1.2,
               label=f"MediaPipe+SVM (ext. {mediapipe_acc:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("EfficientNetV2B0", "EffV2B0")
          .replace("MobileNetV3Small", "MobNetV3S") for m in backbones],
        rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, -0.45), ncol=1)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    save(fig, "F08_cnn_vs_landmark.pdf")


def main():
    f01_synth_vs_no_synth()
    f02_ablation_heatmap()
    f03_generalization_gap()
    f04_tune_paired_dotplot()
    f05_topcfg_confusion()
    f06_best_cm_grid()
    f07_gradcam_grid()
    f08_cnn_vs_landmark()
    f09_zeroshot_classwise_bars()
    print(f"\n[INFO] All figures written to: {OUT}")


if __name__ == "__main__":
    main()
