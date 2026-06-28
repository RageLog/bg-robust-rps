# Trained model artefacts

Trained Keras checkpoints (`.keras`) and the auxiliary
MediaPipe + RBF-SVM joblib (`.joblib`) land in this folder at runtime.
They are deliberately excluded from version control because each
checkpoint is between 5 and 120 MB and a full grid run produces a few
hundred of them.

## Naming convention

```
RPS_Synthetic_V1_{backbone}_{ablation}_{head}_{tune}_fold_{N}_best.keras
```

Examples:

```
RPS_Synthetic_V1_DenseNet121_full_standard_standard_fold_3_best.keras
RPS_Synthetic_V1_EfficientNetV2B0_indoor_attention_progressive_fold_5_best.keras
baseline_svm_full.joblib
```

The classical baseline checkpoint is `baseline_svm_full.joblib` and
contains the trained `sklearn.svm.SVC` instance fitted on
21×3 MediaPipe Hands landmark vectors.

## Restoring archived checkpoints

The orchestrator backs up every checkpoint to
`MyDrive/RPC_Colab/models/` during a Colab run; on subsequent runs
`run_all_experiments.py` automatically restores any missing local
checkpoint from the Drive cache before re-training. See
`run_all_experiments.py:resolve_model_path` for the exact lookup
order.
