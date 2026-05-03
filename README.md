# Background-Robust Static Hand-Gesture Recognition

> **An Image-Grouped Multi-Backbone Ablation**
> Companion repository for the paper of the same title (2026).
> Repository version: **v3** (paper-ready release).

This repository contains everything needed to reproduce the paper's
results: source code for the synthetic-augmentation pipeline, training
scripts for the five CNN backbones, the auxiliary MediaPipe + RBF-SVM
baseline, the aggregated experimental outputs, and the LaTeX sources
of the paper itself (figures, tables, references).

## What is in this repository

```
.
├── src/                          Core pipeline (data gen, train, eval, gradcam)
├── run_all_experiments.py        Auto-pilot for the full experimental grid
├── train_on_colab.ipynb          Hosted-runtime entry point (Google Colab)
├── verify_leakage.py             Image-grouped leakage check (run on disk)
├── count_subjects.py             Raw image inventory and ID extraction
├── regenerate_reports.py         Refresh aggregated CSV reports
├── zip_datasets_for_colab.py     Pack synthetic data for cloud upload
├── reports/aggregated/           Statistical artefacts driving the paper
│   ├── Q1_Table[1-7]_*.csv       Source tables for paper §IV
│   ├── all_per_fold_results.csv  363 per-fold runs in long form
│   ├── per_config_summary.csv    73 configurations, descriptive stats
│   ├── pairwise_results.csv      All paired comparisons (Hedges' g + BH-FDR)
│   ├── main_comparisons.md       Human-readable statistical report
│   ├── grouping_proof.{md,json}  Leakage proof (5/5 folds zero overlap)
│   ├── temporal_smoothing_*.json Deployment-side N×θ grid output
│   ├── rembg_*.json              Foreground-extraction quality probe
│   └── global_cross_dataset_*.json   Zero-shot HaGRID per-model scores
├── paper/
│   ├── main.tex                  Full paper source (IEEEtran)
│   ├── references.bib            BibTeX (30 verified entries)
│   ├── IEEEtran.cls              IEEE conference template
│   ├── figures/F01..F09.pdf      Vector figures used in the paper
│   ├── tables/T01..T15.tex       booktabs LaTeX tables
│   └── scripts/                  Scripts that regenerate every figure / table
├── docs/                         Reproduction guide + changelog
└── models/                       Trained checkpoints land here at runtime
```

## Highlights from the paper

- 5 ImageNet-pretrained CNN backbones (EfficientNetV2B0, ResNet50,
  MobileNetV3Small, DenseNet121, VGG16) trained under an
  **image-grouped** 5-fold protocol so that augmented siblings of one
  parent image always stay in the same fold. The grouping rule is
  locally verifiable on the production filename schema; the proof is
  stored as `reports/aggregated/grouping_proof.md` (5 of 5 folds with
  zero group overlap).
- 73 configurations (10 augmentation modes × 5 backbones × 3 heads × 2
  fine-tuning protocols, with the head and tune ablations restricted
  to the representative \texttt{FULL/EfficientNetV2B0} cell), 363
  per-fold runs, single random seed (42).
- All 43 paired synthetic-vs.-baseline contrasts pass
  Benjamini--Hochberg correction (paired t-test, q < 0.05); paired
  Wilcoxon is reported as a backup, with the n=5 two-sided floor
  (~0.0625) acknowledged.
- Honest synthetic-to-real measurement on a held-out HaGRID slice:
  zero-shot accuracy spans 66.2 % to 87.7 % across the 67 augmented
  configurations, mean gap −17.6 pp.
- Auxiliary MediaPipe Hands + RBF-SVM landmark baseline reaches 59.2 %
  external accuracy; bounded by a 37.3 % landmark-detection rate on
  the external slice.
- Negative finding for progressive layer unfreezing on lightweight
  backbones: 18 of 18 paired contrasts favour standard fine-tuning
  (13 of 18 significant after BH-FDR).
- Grad-CAM evidence per backbone (correct + misclassified samples).

## Quick start

### Local (single-GPU)

```bash
git clone https://github.com/RageLog/RPC_Article.git
cd RPC_Article
pip install -r requirements.txt
pip install rembg[gpu]    # if not pulled by requirements.txt

# 1. Place the raw RPS dataset inside datasets/raw/
#    The expected hierarchy:  datasets/raw/{rock,paper,scissors,none}/*.jpg
# 2. Place background scenes inside datasets/backgrounds/

# 3. Generate synthetic data + train the entire grid
python run_all_experiments.py --run-mode missing --tune_strategy both
```

The orchestrator runs in stages: dataset preparation, the N × M
training grid (cv = 5), classical baseline, head ablation, temporal
smoothing optimisation, segmentation evaluation, cross-dataset
evaluation, Grad-CAM extraction, and vector-graphics export. Trained
.keras artefacts land in `models/`; per-fold JSON reports land in
`reports/`.

### Hosted (Google Colab)

1. Pack the raw dataset as `datasets.zip` and upload it to
   `MyDrive/RPC_Colab/datasets.zip`.
2. Open `train_on_colab.ipynb` in Colab.
3. Run the **Environmental Setup** block (mounts Drive, clones the
   repo, installs dependencies, stages the dataset).
4. Either step through the modular experiment cells or run the
   **Automated Pipeline** cell to launch the full grid.

A more detailed reproduction guide lives in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Reproducing the paper figures and tables

```bash
# After a successful run that has populated reports/, regenerate the
# aggregated CSVs into the same shape as paper/ expects:
python regenerate_reports.py

# Then rebuild the per-paper artefacts:
python paper/scripts/make_stats.py        # statistical analysis (Hedges' g + BH-FDR)
python paper/scripts/make_tables.py       # T01..T08
python paper/scripts/make_extra_tables.py # T09..T13
python paper/scripts/make_table15.py      # T15 (temporal smoothing)
python paper/scripts/make_figures.py      # F01..F09 (vector PDF)
```

If you only want to compile the paper itself, change directory into
`paper/` and run your usual `pdflatex` chain on `main.tex`. The bib
file is `references.bib`. All figures are vector PDFs already; tables
are stand-alone `.tex` files included via `\input{tables/T..}`.

## How to cite

If you use this software, please cite both the paper and the
repository (the citation file `CITATION.cff` is also picked up
automatically by GitHub):

```bibtex
@inproceedings{rpc_article_2026,
  author    = {{RPC_Article authors}},
  title     = {Background-Robust Static Hand-Gesture Recognition:
               An Image-Grouped Multi-Backbone Ablation},
  year      = {2026},
  url       = {https://github.com/RageLog/RPC_Article}
}
```

## Licence

Source code, scripts, and documentation in this repository are
released under the **Apache 2.0** licence (see [`LICENSE`](LICENSE)).
The figures and the LaTeX sources in `paper/` are released under
**CC-BY 4.0** for the manuscript and Apache 2.0 for the
generation scripts. Datasets are not redistributed; please obtain
them from their original sources (links in the paper's
`references.bib`).

## Acknowledgements

This study would not have been possible without the publicly released
RPS Kaggle datasets (`drgfreeman2018rps`, `frtgnn2020rps`,
`sanikamal2019rps`, `alexandredj2021rps`), the HaGRID 30k slice
(`innominate817_hagrid30k`) and the HaGRIDv2 release
(`nuzhdin2024hagridv21mimagesstatic`).
