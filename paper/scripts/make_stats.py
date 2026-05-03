"""
make_stats.py — Q1-grade statistical analysis from per-fold JSON reports.

Inputs : paper/data_provenance/all_per_fold_results.csv (long-form, fold-level)
Outputs: paper/stats/main_comparisons.md
         paper/stats/pairwise_results.csv
         paper/stats/per_config_summary.csv

Tests:
  - Within-config descriptive: mean, std, median, IQR, 95% CI (bootstrap, n=10000)
  - Pairwise paired Wilcoxon signed-rank (per-fold paired across configs)
  - Multiple-testing correction: Benjamini-Hochberg FDR (within each comparison family)
  - Effect size: Cohen's d (paired, uncorrected), Hedges' g (paired, bias-corrected), paired Cliff's delta
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data_provenance" / "all_per_fold_results.csv"
OUT_DIR = ROOT / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
N_BOOT = 10_000


def bootstrap_ci(values, ci=0.95, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def cohens_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    if diff.std(ddof=1) == 0:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def hedges_g_paired(a, b):
    """Bias-corrected paired effect size (Hedges' g)."""
    a = np.asarray(a); b = np.asarray(b)
    diff = a - b
    n = len(diff)
    if n < 2 or diff.std(ddof=1) == 0:
        return 0.0
    d = float(diff.mean() / diff.std(ddof=1))
    J = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)  # Hedges correction
    return d * J


def cliffs_delta_paired(a, b):
    """Paired Cliff's delta (rank-biserial on signed differences)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    if n == 0:
        return float("nan")
    gt = int(np.sum(diff > 0)); lt = int(np.sum(diff < 0))
    return float((gt - lt) / n)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = q
    return np.clip(out, 0, 1)


def paired_wilcoxon(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b) or len(a) < 3:
        return float("nan"), float("nan")
    if np.allclose(a, b):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def paired_t(a, b):
    """Paired t-test on a-b. Returns (t, p, shapiro_p_of_diff).

    With n=5 folds, paired t can yield p<0.05 when |Cohen's d| > ~1.5,
    while Wilcoxon's two-sided minimum is ~0.0625. Both are reported.
    Caller should consult the Shapiro p-value of the differences before
    trusting the t-test.
    """
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b) or len(a) < 3:
        return float("nan"), float("nan"), float("nan")
    diff = a - b
    if np.allclose(diff, 0):
        return 0.0, 1.0, float("nan")
    try:
        t, p = stats.ttest_rel(a, b)
        # Shapiro requires variance > 0
        if diff.std(ddof=1) > 0 and len(diff) >= 3:
            sh_p = float(stats.shapiro(diff).pvalue)
        else:
            sh_p = float("nan")
        return float(t), float(p), sh_p
    except Exception:
        return float("nan"), float("nan"), float("nan")


def main():
    df = pd.read_csv(RAW_CSV)
    df["config"] = (df["model"] + "|" + df["ablation"] + "|" + df["head"] + "|" + df["tune"])
    print(f"[INFO] Loaded {len(df)} per-fold records, {df['config'].nunique()} configs.")

    # 1. Per-config descriptive summary
    rows = []
    for cfg, g in df.groupby("config"):
        g = g.sort_values("fold")
        accs = g["accuracy"].to_numpy()
        f1s = g["macro_f1"].to_numpy()
        if len(accs) < 3:
            continue
        m = accs.mean(); s = accs.std(ddof=1); med = np.median(accs)
        q25, q75 = np.percentile(accs, [25, 75])
        ci_lo, ci_hi = bootstrap_ci(accs)
        rows.append({
            "config": cfg, "n_folds": len(accs),
            "n_folds_complete": len(accs) == 5,
            "acc_mean": m, "acc_std": s, "acc_median": med,
            "acc_iqr_lo": q25, "acc_iqr_hi": q75,
            "acc_ci95_lo": ci_lo, "acc_ci95_hi": ci_hi,
            "macro_f1_mean": f1s.mean(), "macro_f1_std": f1s.std(ddof=1),
        })
    summ = pd.DataFrame(rows)
    summ[["model","ablation","head","tune"]] = summ["config"].str.split("|", expand=True)
    summ.to_csv(OUT_DIR / "per_config_summary.csv", index=False)
    print(f"[SAVED] {OUT_DIR / 'per_config_summary.csv'}  ({len(summ)} configs)")

    # Helper: pivot per-fold accuracy for paired tests
    # NOTE: use df["head"] (column access) not df.head (method conflict).
    def fold_vec(model, ablation, head, tune):
        g = df[(df["model"] == model) & (df["ablation"] == ablation)
               & (df["head"] == head) & (df["tune"] == tune)].sort_values("fold")
        if len(g) != 5:
            return None
        return g["accuracy"].to_numpy()

    # 2. Family A: Synthetic vs NO_SYNTH (same model, head=standard, tune=standard)
    family_A = []
    for model in sorted(df["model"].unique()):
        for ab in ("FULL", "INDOOR", "RANDBG", "GAN", "STYLE_TRANSFER",
                   "REMBG_ONLY", "NO_ALPHA", "NO_SHIFT", "RATIO_1X"):
            a = fold_vec(model, ab, "standard", "standard")
            b = fold_vec(model, "NO_SYNTH", "standard", "standard")
            if a is None or b is None:
                continue
            w_stat, w_p = paired_wilcoxon(a, b)
            t_stat, t_p, shapiro_p = paired_t(a, b)
            cohen_d_uncorrected = cohens_d_paired(a, b)
            hedges_g = hedges_g_paired(a, b)
            cd = cliffs_delta_paired(a, b)
            family_A.append({
                "family": "A_synth_vs_no_synth",
                "model": model,
                "ablation_a": ab, "ablation_b": "NO_SYNTH",
                "head": "standard", "tune": "standard",
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "delta": float(a.mean() - b.mean()),
                "wilcoxon_stat": w_stat, "wilcoxon_p_raw": w_p,
                "t_stat": t_stat, "t_p_raw": t_p,
                "shapiro_p_diff": shapiro_p,
                "cohen_d_uncorrected": cohen_d_uncorrected, "hedges_g": hedges_g,
                "cliffs_delta": cd,
                "n_pairs": 5,
                "p_raw": t_p,  # primary p-value used for FDR (t-test, since n=5 under-powers Wilcoxon)
            })

    # 3. Family B: Backbone comparison on FULL/standard/standard (pairwise)
    family_B = []
    full_models = [m for m in sorted(df["model"].unique())
                   if fold_vec(m, "FULL", "standard", "standard") is not None]
    for i, m1 in enumerate(full_models):
        for m2 in full_models[i+1:]:
            a = fold_vec(m1, "FULL", "standard", "standard")
            b = fold_vec(m2, "FULL", "standard", "standard")
            w_stat, w_p = paired_wilcoxon(a, b)
            t_stat, t_p, shapiro_p = paired_t(a, b)
            cohen_d_uncorrected = cohens_d_paired(a, b)
            hedges_g = hedges_g_paired(a, b)
            cd = cliffs_delta_paired(a, b)
            family_B.append({
                "family": "B_backbone_pairwise_FULL",
                "model_a": m1, "model_b": m2,
                "ablation": "FULL", "head": "standard", "tune": "standard",
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "delta": float(a.mean() - b.mean()),
                "wilcoxon_stat": w_stat, "wilcoxon_p_raw": w_p,
                "t_stat": t_stat, "t_p_raw": t_p,
                "shapiro_p_diff": shapiro_p,
                "cohen_d_uncorrected": cohen_d_uncorrected, "hedges_g": hedges_g,
                "cliffs_delta": cd,
                "n_pairs": 5,
                "p_raw": t_p,
            })

    # 4. Family C: Standard vs Progressive tune (same model, ablation, head=standard)
    family_C = []
    for model in sorted(df["model"].unique()):
        for ab in ("FULL", "INDOOR", "RANDBG", "GAN"):
            a = fold_vec(model, ab, "standard", "standard")
            b = fold_vec(model, ab, "standard", "progressive")
            if a is None or b is None:
                continue
            w_stat, w_p = paired_wilcoxon(a, b)
            t_stat, t_p, shapiro_p = paired_t(a, b)
            cohen_d_uncorrected = cohens_d_paired(a, b)
            hedges_g = hedges_g_paired(a, b)
            cd = cliffs_delta_paired(a, b)
            family_C.append({
                "family": "C_tune_standard_vs_progressive",
                "model": model, "ablation": ab,
                "head": "standard",
                "mean_standard": float(a.mean()), "mean_progressive": float(b.mean()),
                "delta": float(a.mean() - b.mean()),
                "wilcoxon_stat": w_stat, "wilcoxon_p_raw": w_p,
                "t_stat": t_stat, "t_p_raw": t_p,
                "shapiro_p_diff": shapiro_p,
                "cohen_d_uncorrected": cohen_d_uncorrected, "hedges_g": hedges_g,
                "cliffs_delta": cd,
                "n_pairs": 5,
                "p_raw": t_p,
            })

    # 5. Family D: Head ablation on EfficientNetV2B0 FULL standard (vs standard head)
    family_D = []
    if "EfficientNetV2B0" in df["model"].unique():
        base = fold_vec("EfficientNetV2B0", "FULL", "standard", "standard")
        if base is not None:
            for head in ("attention", "spatial_pooling"):
                b = fold_vec("EfficientNetV2B0", "FULL", head, "standard")
                if b is None: continue
                w_stat, w_p = paired_wilcoxon(base, b)
                t_stat, t_p, shapiro_p = paired_t(base, b)
                family_D.append({
                    "family": "D_head_vs_standard",
                    "model": "EfficientNetV2B0", "ablation": "FULL",
                    "head_a": "standard", "head_b": head,
                    "mean_standard_head": float(base.mean()),
                    "mean_other_head": float(b.mean()),
                    "delta": float(base.mean() - b.mean()),
                    "wilcoxon_stat": w_stat, "wilcoxon_p_raw": w_p,
                    "t_stat": t_stat, "t_p_raw": t_p,
                    "shapiro_p_diff": shapiro_p,
                    "cohen_d_uncorrected": cohens_d_paired(base, b),
                    "hedges_g": hedges_g_paired(base, b),
                    "cliffs_delta": cliffs_delta_paired(base, b),
                    "n_pairs": 5,
                    "p_raw": t_p,
                })

    # Apply BH-FDR within each family
    all_results = []
    for fam in (family_A, family_B, family_C, family_D):
        if not fam: continue
        fdf = pd.DataFrame(fam)
        valid = fdf["p_raw"].notna()
        if valid.sum() > 0:
            fdf.loc[valid, "p_bh_fdr"] = bh_fdr(fdf.loc[valid, "p_raw"].values)
            fdf.loc[~valid, "p_bh_fdr"] = float("nan")
        else:
            fdf["p_bh_fdr"] = float("nan")
        fdf["sig_q05"] = fdf["p_bh_fdr"] < 0.05
        all_results.append(fdf)

    pairwise = pd.concat(all_results, ignore_index=True, sort=False)
    pairwise.to_csv(OUT_DIR / "pairwise_results.csv", index=False)
    print(f"[SAVED] {OUT_DIR / 'pairwise_results.csv'}  ({len(pairwise)} comparisons)")

    # Markdown report
    md = ["# Q1 Statistical Analysis — RPS Hand Gesture Recognition\n"]
    md.append(f"Source: `{RAW_CSV}`  ({len(df)} per-fold rows, {df['config'].nunique()} configs)\n")
    md.append("Tests reported per pair: paired t-test (primary, used for FDR) AND paired Wilcoxon "
              "signed-rank (non-parametric reference).\n")
    md.append("Effect size: paired Cohen's d (uncorrected), Hedges' g (bias-corrected paired), "
              "and paired Cliff's delta (rank-biserial on signed differences). 95% CI: bootstrap (n=10000).\n")
    md.append("Configurations with `n_folds_complete=False` are excluded from all paired tests; "
              "they are listed in the per-config table for transparency.\n")
    md.append("Multiple comparisons: Benjamini-Hochberg FDR within each family · α=0.05.\n")
    md.append("Normality of paired differences: Shapiro-Wilk p-value reported per pair "
              "(if shapiro_p < 0.05, prefer Wilcoxon).\n\n")
    md.append("> **Statistical caveat (n=5 limitation):** With only k=5 folds per cell, the "
              "two-sided minimum achievable p-value of the paired Wilcoxon signed-rank test is "
              "$2/2^5 \\approx 0.0625$. Hence Wilcoxon **cannot** reach p<0.05 at this fold count, "
              "even for the largest observed effects. The paired t-test does not share this floor "
              "and is therefore used as the primary FDR input, with Shapiro–Wilk reported for "
              "every pair to flag normality concerns. We recommend reporting **both** tests in "
              "the manuscript and disclosing the n=5 limitation explicitly in Methods.\n\n")

    def fmt_family(name, fam_df, key_cols):
        if fam_df.empty: return ""
        out = [f"## Family {name}\n", f"N comparisons: {len(fam_df)}\n"]
        out.append(fam_df.to_markdown(index=False, floatfmt=".4f"))
        out.append("\n\n")
        return "\n".join(out)

    md.append("## Per-config descriptive (top 15 by mean accuracy)\n")
    top = summ.sort_values("acc_mean", ascending=False).head(15)
    md.append(top[["model","ablation","head","tune","n_folds",
                   "acc_mean","acc_std","acc_ci95_lo","acc_ci95_hi",
                   "macro_f1_mean","macro_f1_std"]]
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n")

    md.append("## Per-config descriptive (bottom 10 by mean accuracy)\n")
    bot = summ.sort_values("acc_mean").head(10)
    md.append(bot[["model","ablation","head","tune","n_folds",
                   "acc_mean","acc_std","acc_ci95_lo","acc_ci95_hi"]]
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("\n\n")

    for fam_name, fam_data in [("A_synth_vs_no_synth", family_A),
                                 ("B_backbone_pairwise_FULL", family_B),
                                 ("C_tune_standard_vs_progressive", family_C),
                                 ("D_head_vs_standard", family_D)]:
        if not fam_data: continue
        sub = pairwise[pairwise["family"] == fam_name].copy()
        if sub.empty: continue
        md.append(f"## Family {fam_name}\n")
        md.append(f"N comparisons: {len(sub)}\n\n")
        md.append(sub.to_markdown(index=False, floatfmt=".4f"))
        md.append("\n\n")

    (OUT_DIR / "main_comparisons.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[SAVED] {OUT_DIR / 'main_comparisons.md'}")
    print()
    print("=== HIGHLIGHTS (primary p = paired t-test, BH-FDR within family) ===")
    A = pairwise[pairwise.family == "A_synth_vs_no_synth"]
    if not A.empty:
        print(f"A: synth vs NO_SYNTH — t-test BH-significant: {(A.sig_q05).sum()}/{len(A)}")
        print(f"   Mean delta range: [{A.delta.min():+.4f}, {A.delta.max():+.4f}]")
        print(f"   Wilcoxon BH-significant: {(bh_fdr(A.wilcoxon_p_raw.fillna(1).values) < 0.05).sum()}/{len(A)}")
        print(f"   Hedges' g range: [{A.hedges_g.min():+.4f}, {A.hedges_g.max():+.4f}]")
    B = pairwise[pairwise.family == "B_backbone_pairwise_FULL"]
    if not B.empty:
        print(f"B: backbone pairwise (FULL/std/std) — t-test BH-significant: "
              f"{(B.sig_q05).sum()}/{len(B)}")
        print(f"   Hedges' g range: [{B.hedges_g.min():+.4f}, {B.hedges_g.max():+.4f}]")
    C = pairwise[pairwise.family == "C_tune_standard_vs_progressive"]
    if not C.empty:
        print(f"C: standard vs progressive — t-test BH-significant: {(C.sig_q05).sum()}/{len(C)}; "
              f"std-better: {(C.delta > 0).sum()}, prog-better: {(C.delta < 0).sum()}")
        print(f"   Hedges' g range: [{C.hedges_g.min():+.4f}, {C.hedges_g.max():+.4f}]")
    D = pairwise[pairwise.family == "D_head_vs_standard"]
    if not D.empty:
        print(f"D: head vs standard (EffNetV2B0/FULL/std) — t-test BH-significant: "
              f"{(D.sig_q05).sum()}/{len(D)}")
        print(f"   Hedges' g range: [{D.hedges_g.min():+.4f}, {D.hedges_g.max():+.4f}]")


if __name__ == "__main__":
    main()
