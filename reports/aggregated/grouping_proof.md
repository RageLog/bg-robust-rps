# Augmentation-Leakage Grouping Proof (Local)

- **Splitter**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`

- **Filename schema**: `<class>_<basename>_syn<ii>.jpg (mirrors generate_data.py:88)`

- **Group extraction rule**: `filename.split('_', 1)[1].rsplit('_syn', 1)[0]`

- **Raw parent images**: 7,830

- **Simulated synthetic files (factor 5)**: 39,150

- **All folds clean (zero group overlap)**: **True**


| Fold | Train files | Val files | Train groups | Val groups | Group overlap | Status |
|-----:|------------:|----------:|-------------:|-----------:|--------------:|:------:|
| 1 | 31,320 | 7,830 | 6,264 | 1,566 | 0 | CLEAN |
| 2 | 31,320 | 7,830 | 6,264 | 1,566 | 0 | CLEAN |
| 3 | 31,320 | 7,830 | 6,264 | 1,566 | 0 | CLEAN |
| 4 | 31,320 | 7,830 | 6,264 | 1,566 | 0 | CLEAN |
| 5 | 31,320 | 7,830 | 6,264 | 1,566 | 0 | CLEAN |

## Interpretation
PROVES that the cv_data_loader.py grouping prevents augmentation-leakage (synthetic siblings never appear in different folds from their parent raw image) on the production filename schema. Does NOT prove subject-leakage prevention, which would require per-subject metadata that is absent from the public Kaggle subset used in this study. This caveat must appear in Methods and Limitations.