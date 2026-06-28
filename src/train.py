"""
train.py - Stable Training Loop
================================
XLA and Autotune disabled for stability.
"""
import os
import sys
import argparse
import numpy as np
import json

# --- GPU & PERFORMANCE SETTINGS ---
# 1. Disable unnecessary TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable Mixed Precision (Half-Precision) for modern GPUs (e.g., A100)!
# Reduces memory footprint by half and accelerates training significantly.
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"[INFO] Mixed Precision ENABLED: {policy.compute_dtype}")
except Exception as e:
    print(f"[WARN] Failed to enable Mixed Precision. Error: {e}")

# GPU Memory Management (prevent OOM)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# -----------------------------------------------

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_datasets
from cv_data_loader import get_cv_folds
from models import build_model, unfreeze_model, WARMUP_EPOCHS, unfreeze_model_progressive
import plot_vector_graphics

def get_callbacks(checkpoint_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN() 
    ]

def save_training_report(model, dataset, model_name, ablation_mode, history_obj, fold=None, head_strategy="standard", tune_strategy="standard"):
    """Save JSON reports for vector graphics generation (plot_vector_graphics.py)."""
    from sklearn.metrics import confusion_matrix, classification_report
    
    report_dir = os.path.join(config.REPORTS_DIR, model_name, ablation_mode.upper())
    os.makedirs(report_dir, exist_ok=True)

    # Predict on the dataset
    y_true, y_pred_list = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_list.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred_list)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Merge histories from Phase 1 + Phase 2 if both exist
    history_dict = {}
    if history_obj and hasattr(history_obj, 'history'):
        history_dict = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}

    # Determine empirically which classes are actually present in the true/pred labels
    # to avoid "Number of classes does not match size of target_names" errors in ablation subsets
    present_integer_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    dynamic_target_names = [config.CLASS_NAMES[int(idx)] for idx in present_integer_classes]

    suffix = f"_fold_{fold}" if fold else ""
    report_data = {
        "model_name": model_name,
        "ablation_mode": ablation_mode,
        "head_strategy": head_strategy,
        "class_names": config.CLASS_NAMES,
        "confusion_matrix": cm,
        "history": history_dict,
        "classification_report": classification_report(
            y_true, y_pred, target_names=dynamic_target_names, output_dict=True, zero_division=0
        )
    }

    file_name = f"report_{model_name}_{ablation_mode.lower()}_{head_strategy}_{tune_strategy}_fold_{fold}.json" if fold else f"report_{model_name}_{ablation_mode.lower()}_{head_strategy}_{tune_strategy}_full.json"
    report_path = os.path.join(report_dir, file_name)
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
        
    print(f"[INFO] JSON Report saved: {report_path}")
    
    # Generate high-resolution vector charts immediately
    try:
        plot_vector_graphics.plot_confusion_matrix_from_json(report_path, report_dir)
        plot_vector_graphics.plot_training_history_from_json(report_path, report_dir)
    except Exception as e:
        print(f"[WARN] Failed to plot vector graphics: {e}")

def train(ablation_mode="FULL", model_name="EfficientNetV2B0", cv=None, tune_strategy="standard", head_strategy="standard"):
    if cv and cv > 1:
        print(f"\n[INFO] K-FOLD TRAINING STARTING: {config.RUN_NAME} ({cv}-Fold) | Mode: {ablation_mode} | Model: {model_name} | Strategy: {tune_strategy} | Head: {head_strategy}")
    else:
        print(f"\n[INFO] TRAINING STARTING: {config.RUN_NAME} (Standard Split) | Mode: {ablation_mode} | Model: {model_name} | Strategy: {tune_strategy} | Head: {head_strategy}")
    
    # Ensure GPU Memory Growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Active GPUs: {len(gpus)}")
        except RuntimeError as e:
            print(e)
            

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[INFO] Mixed Precision (float16) currently active")
    # Global determinism (python/numpy/tf) for the fixed-seed pipeline.
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    try:
        if cv and cv > 1:
            # --- K-FOLD CROSS VALIDATION ---
            print(f"[INFO] Preparing K-Fold data ({cv} folds)...")
            cv_results = []
            
            for fold, train_ds, val_ds in get_cv_folds(k_folds=cv, ablation_mode=ablation_mode):
                print(f"\n{'='*40}")
                print(f" STARTING FOLD {fold}/{cv}")
                print(f"{'='*40}\n")
                
                # Check if this fold has already been trained and evaluated
                report_dir = os.path.join(config.REPORTS_DIR, model_name, ablation_mode.upper())
                report_filename = f"report_{model_name}_{ablation_mode.lower()}_{head_strategy}_{tune_strategy}_fold_{fold}.json"
                local_report = os.path.join(report_dir, report_filename)
                # Attempt to find it in Drive if not local
                if not os.path.exists(local_report) and os.path.exists(config.DRIVE_DIR):
                    drive_report = os.path.join(config.DRIVE_DIR, "reports", model_name, ablation_mode.upper(), report_filename)
                    if os.path.exists(drive_report):
                        os.makedirs(report_dir, exist_ok=True)
                        import shutil
                        shutil.copy2(drive_report, local_report)
                        
                if os.path.exists(local_report):
                    print(f"[INFO] Fold {fold} already completed! Skipping training.")
                    # Load previous accuracy to ensure summary calculation works
                    with open(local_report, "r") as f:
                        data = json.load(f)
                        cv_results.append(data.get("val_accuracy", 0.0))
                    continue
                
                model = build_model(model_name=model_name, compile_model=True, head_strategy=head_strategy)
                # Differentiate model names based on ablation mode, fold, and head strategy and tune strategy
                checkpoint_path = os.path.join(config.MODELS_DIR, f"{config.RUN_NAME}_{model_name}_{ablation_mode.lower()}_{head_strategy}_{tune_strategy}_fold_{fold}_best.keras")
                callbacks = get_callbacks(checkpoint_path)
                
                # Phase 1: Warm-up (frozen backbone, train head only)
                remaining_epochs = config.EPOCHS
                if WARMUP_EPOCHS > 0:
                    print(f"[INFO] Phase 1: Warm-up ({WARMUP_EPOCHS} epochs, backbone frozen)")
                    model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=WARMUP_EPOCHS,
                        callbacks=callbacks,
                        verbose=1
                    )
                    remaining_epochs = config.EPOCHS - WARMUP_EPOCHS
                
                # Phase 2: Fine-tuning (unfreeze backbone)
                if remaining_epochs > 0:
                    if tune_strategy == "progressive":
                        print(f"[INFO] Phase 2: Progressive Unfreezing Stage 1 (Top 50% backbone)")
                        stage1_epochs = remaining_epochs // 2
                        model = unfreeze_model_progressive(model, model_name=model_name, stage=1)
                        model.fit(
                            train_ds, validation_data=val_ds,
                            initial_epoch=WARMUP_EPOCHS, epochs=WARMUP_EPOCHS + stage1_epochs,
                            callbacks=callbacks, verbose=1
                        )
                        remaining_epochs -= stage1_epochs
                        
                        print(f"[INFO] Phase 3: Progressive Unfreezing Stage 2 (100% backbone)")
                        model = unfreeze_model_progressive(model, model_name=model_name, stage=2)
                        model.fit(
                            train_ds, validation_data=val_ds,
                            initial_epoch=WARMUP_EPOCHS + stage1_epochs, epochs=config.EPOCHS,
                            callbacks=callbacks, verbose=1
                        )
                    else:
                        print(f"[INFO] Phase 2: Fine-tuning ({remaining_epochs} epochs, backbone unfrozen)")
                        model = unfreeze_model(model, model_name=model_name)
                        model.fit(
                            train_ds, validation_data=val_ds,
                            initial_epoch=WARMUP_EPOCHS, epochs=config.EPOCHS,
                            callbacks=callbacks, verbose=1
                        )
                
                print(f"\n[INFO] Fold {fold} Validation Result:")
                val_loss, val_acc = model.evaluate(val_ds)
                cv_results.append(val_acc)
                print(f"Fold {fold} Accuracy: {val_acc:.4f}\n")

                # Save JSON report for this fold
                save_training_report(model, val_ds, model_name, ablation_mode, history_obj=None, fold=fold, head_strategy=head_strategy, tune_strategy=tune_strategy)
                
            
            print(f"\n{'='*40}")
            print("[INFO] K-FOLD CV COMPLETED!")
            
            mean_acc = np.mean(cv_results)
            std_acc = np.std(cv_results)
            
            print(f"  All Fold Scores: {[f'{s:.4f}' for s in cv_results]}")
            print(f"  Average Accuracy: {mean_acc:.4f} (±{std_acc:.4f})")
            print(f"{'='*40}\n")
            
            # Save K-Fold Summary Report
            summary_data = {
                "model_name": model_name,
                "ablation_mode": ablation_mode,
                "head_strategy": head_strategy,
                "k_folds": cv,
                "fold_scores": [float(s) for s in cv_results],
                "mean_accuracy": float(mean_acc),
                "std_accuracy": float(std_acc),
                "formatted_result": f"{mean_acc:.4f} ± {std_acc:.4f}"
            }
            report_dir = os.path.join(config.REPORTS_DIR, model_name, ablation_mode.upper())
            os.makedirs(report_dir, exist_ok=True)
            summary_path = os.path.join(report_dir, f"kfold_summary_{head_strategy}_{tune_strategy}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"[SUCCESS] K-Fold Summary Output: {summary_path}")
            
            
        else:
            # --- STANDARD (SINGLE SPLIT) TRAINING ---
            print("[INFO] Checking if Standard Training is already completed...")
            report_dir = os.path.join(config.REPORTS_DIR, model_name, ablation_mode.upper())
            report_filename = f"report_{model_name}_{ablation_mode.lower()}_{head_strategy}_{tune_strategy}_full.json"
            local_report = os.path.join(report_dir, report_filename)
            
            if not os.path.exists(local_report) and os.path.exists(config.DRIVE_DIR):
                drive_report = os.path.join(config.DRIVE_DIR, "reports", model_name, ablation_mode.upper(), report_filename)
                if os.path.exists(drive_report):
                    os.makedirs(report_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(drive_report, local_report)
                    
            if os.path.exists(local_report):
                print(f"[INFO] Standard training for {model_name} ({ablation_mode}) already completed! Skipping.")
                return

            print("[INFO] Loading data...")
            train_ds, val_ds, test_ds = load_datasets(ablation_mode=ablation_mode)
            
            model = build_model(model_name=model_name, compile_model=True, head_strategy=head_strategy)
            checkpoint_path = os.path.join(config.MODELS_DIR, f"{config.RUN_NAME}_{model_name}_{ablation_mode.lower()}_{head_strategy}_{tune_strategy}_best.keras")
            callbacks = get_callbacks(checkpoint_path)
            
            # Phase 1: Warm-up (frozen backbone, train head only)
            remaining_epochs = config.EPOCHS
            if WARMUP_EPOCHS > 0:
                print(f"\n[INFO] Phase 1: Warm-up ({WARMUP_EPOCHS} epochs, backbone frozen)")
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=WARMUP_EPOCHS,
                    callbacks=callbacks
                )
                remaining_epochs = config.EPOCHS - WARMUP_EPOCHS
            
            # Phase 2: Fine-tuning (unfreeze backbone)
            if remaining_epochs > 0:
                if tune_strategy == "progressive":
                    print(f"\n[INFO] Phase 2: Progressive Unfreezing Stage 1 (Top 50% backbone)")
                    stage1_epochs = remaining_epochs // 2
                    model = unfreeze_model_progressive(model, model_name=model_name, stage=1)
                    history_2 = model.fit(
                        train_ds, validation_data=val_ds,
                        initial_epoch=WARMUP_EPOCHS, epochs=WARMUP_EPOCHS + stage1_epochs,
                        callbacks=callbacks
                    )
                    
                    print(f"\n[INFO] Phase 3: Progressive Unfreezing Stage 2 (100% backbone)")
                    model = unfreeze_model_progressive(model, model_name=model_name, stage=2)
                    history_3 = model.fit(
                        train_ds, validation_data=val_ds,
                        initial_epoch=WARMUP_EPOCHS + stage1_epochs, epochs=config.EPOCHS,
                        callbacks=callbacks
                    )
                    history = history_3 # Keep latest for report saving
                else:
                    print(f"\n[INFO] Phase 2: Fine-tuning ({remaining_epochs} epochs, backbone unfrozen)")
                    model = unfreeze_model(model, model_name=model_name)
                    history = model.fit(
                        train_ds, validation_data=val_ds,
                        initial_epoch=WARMUP_EPOCHS, epochs=config.EPOCHS,
                        callbacks=callbacks
                    )
            
            print("\n[INFO] Test Results:")
            model.evaluate(test_ds)

            # Save JSON report with training history and confusion matrix
            save_training_report(model, test_ds, model_name, ablation_mode, history_obj=history, head_strategy=head_strategy, tune_strategy=tune_strategy)

            print(f"\n[SUCCESS] Model Saved: {checkpoint_path}")
            
    except KeyboardInterrupt:
        print("\n[WARN] Training stopped interactively by user.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPS Arena Training Script")
    parser.add_argument("--model", type=str, default="EfficientNetV2B0", 
                        choices=["EfficientNetV2B0", "ResNet50", "MobileNetV3Small", "DenseNet121", "VGG16"],
                        help="CNN Architecture Backbone")
    parser.add_argument("--ablation", type=str, default="FULL", choices=["FULL", "INDOOR", "RANDBG", "REMBG_ONLY", "BASELINE", "NO_SYNTH", "STYLE_TRANSFER", "GAN", "RATIO_1X", "NO_SHIFT", "NO_ALPHA"],
                        help="Ablation study mode")
    parser.add_argument("--cv", type=int, default=None, help="Number of folds for Cross Validation (e.g., 5)")
    parser.add_argument("--tune_strategy", type=str, choices=["standard", "progressive"], default="standard",
                        help="Fine-tuning strategy: standard (1-step unfreeze) or progressive (multi-stage unfreeze with descending LR)")
    parser.add_argument("--head_strategy", type=str, choices=["standard", "attention", "spatial_pooling"], default="standard",
                        help="Classification Head architecture ablation.")
    args = parser.parse_args()
    
    train(ablation_mode=args.ablation, model_name=args.model, cv=args.cv, tune_strategy=args.tune_strategy, head_strategy=args.head_strategy)