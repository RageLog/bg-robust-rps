# Reports

This folder hosts experimental outputs at two levels of granularity.

## `aggregated/`

Pre-shipped, paper-driving artefacts. Every figure and table in
`paper/` reads from one of these files. Re-running the experiments is
not required to inspect or rebuild the figures and tables.

| File | Purpose |
|---|---|
| `Q1_Table1_Model_Complexity.csv` | Trainable-parameter footprint per backbone (paper Tab. II) |
| `Q1_Table2_KFold_Statistical_Consistency.csv` | 5-fold mean ± std for every configuration (paper Tab. XII) |
| `Q1_Table3_Classwise_ZeroShot_F1_Scores.csv` | Per-class HaGRID zero-shot F1 (paper Tab. IX) |
| `Q1_Table4_Generalization_Gap.csv` | In-distribution vs.\ zero-shot accuracy and gap (paper Tab. V, XIII; Fig. F03, F08, F09) |
| `Q1_Table5_Ablation_Impact.csv` | Pooled per-mode mean ± std accuracy (paper Tab. IV; Fig. F02) |
| `Q1_Table6_Precision_vs_Recall_Tradeoff.csv` | Macro precision/recall trade-off (paper Tab. X) |
| `Q1_Table7_Internal_Classwise_Stability.csv` | Per-class F1 stability across folds (paper Tab. XI) |
| `all_per_fold_results.csv` | 363 fold runs in long form |
| `per_config_summary.csv` | 73 configurations with descriptive stats and `n_folds_complete` flag |
| `pairwise_results.csv` | All paired comparisons (Family A/B/C/D) with Hedges' g and BH-FDR |
| `main_comparisons.md` | Human-readable statistical report |
| `grouping_proof.md` and `grouping_proof.json` | Image-grouped leakage proof (5 of 5 folds with zero overlap) |
| `temporal_smoothing_optimization.json` | Deployment-side N×θ grid (paper Tab. XV; deployment study) |
| `rembg_evaluation_summary.json` | Foreground-extraction quality (pseudo-IoU, edge-contrast) |
| `global_cross_dataset_evaluation.json` | Zero-shot HaGRID per trained checkpoint (basis of Fig. F09) |
| `raw_dataset_inventory.json` | Raw image inventory (paper Tab. I) |

## At runtime: per-fold artefacts

When `run_all_experiments.py` finishes, the per-fold JSON reports
land directly under this `reports/` folder, organised as:

```
reports/
└── {Backbone}/
    └── {ABLATION}/
        ├── report_{Backbone}_{ablation}_{head}_{tune}_fold_{N}.json
        └── ... (per-fold confusion matrices, training history,
                 classification_report)
```

These per-fold files are not checked in to the public repository
because they total several thousand JSONs. The aggregated CSVs in
`aggregated/` already carry every number used in the paper.

## Regeneration

To regenerate the aggregated CSVs from a fresh per-fold run:

```bash
python regenerate_reports.py
```

To regenerate the long-form, statistics, and grouping-proof artefacts:

```bash
python paper/scripts/make_stats.py
python paper/scripts/verify_grouping_logic.py
```
