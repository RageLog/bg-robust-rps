"""
benchmark.py - ULTIMATE MODEL ANALYSIS AND REPORTING
==================================================
This script analyzes not only the accuracy but also the character of the model.
Outputs are saved in the 'reports/BENCHMARK_TIMESTAMP' folder.
"""

import os
import sys
import time
import json
import platform
import psutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_datasets

# Matplotlib backend setting (Error prevention)
import matplotlib
matplotlib.use('Agg')

def get_system_info():
    """Gathers system hardware information."""
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "os": platform.system(),
        "release": platform.release(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    
    gpus = tf.config.list_physical_devices('GPU')
    info["gpu_available"] = len(gpus) > 0
    info["gpu_names"] = [gpu.name for gpu in gpus] if gpus else ["CPU Only"]
    
    return info

def plot_confusion_matrix_advanced(y_true, y_pred, classes, save_dir):
    """Draws both numerical and normalized Confusion Matrix."""
    
    # 1. Numerical Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Count)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cm_counts.png'), dpi=300)
    plt.close()

    # 2. Normalized Matrix (%)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Percentage)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cm_normalized.png'), dpi=300)
    plt.close()

def plot_multiclass_roc(y_true, y_probs, classes, save_dir):
    """Draws ROC Curves for each class."""
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10')
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(i), lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300)
    plt.close()

def plot_confidence_histograms(y_true, y_pred, y_probs, save_dir):
    """Confidence distribution of Correct and Incorrect predictions."""
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]
    
    correct_conf = np.max(y_probs[correct_indices], axis=1) if len(correct_indices) > 0 else []
    incorrect_conf = np.max(y_probs[incorrect_indices], axis=1) if len(incorrect_indices) > 0 else []
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(correct_conf, bins=20, alpha=0.7, color='green', label=f'Correct Predictions ({len(correct_conf)})')
    plt.hist(incorrect_conf, bins=20, alpha=0.7, color='red', label=f'Incorrect Predictions ({len(incorrect_indices)})')
    
    plt.xlabel('Confidence (Model Confidence)')
    plt.ylabel('Number of Samples')
    plt.title('Prediction Confidence Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), dpi=300)
    plt.close()

def benchmark_latency(model, sample_input, num_samples=1000):
    """Latency and Throughput test."""
    print(f"\n⏱  Latency Test Starting ({num_samples} loops)...")
    
    # Warmup
    print("    GPU/CPU ...")
    for _ in range(50):
        model.predict(sample_input, verbose=0)
        
    latencies = []
    
    print("    Measuring...")
    start_total = time.time()
    for _ in range(num_samples):
        t0 = time.time()
        model.predict(sample_input, verbose=0)
        latencies.append((time.time() - t0) * 1000) # in ms
    end_total = time.time()
    
    latencies = np.array(latencies)
    total_time = end_total - start_total
    
    stats = {
        "avg_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput_fps": num_samples / total_time
    }
    
    return stats, latencies

def main():
    # 1. Folder Preparation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config.REPORTS_DIR, f"BENCHMARK_{config.RUN_NAME}_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\n STARTING ULTIMATE BENCHMARK: {config.RUN_NAME}")
    print(f" Report Folder: {report_dir}")
    print("=" * 60)

    # 2. System Information
    sys_info = get_system_info()
    print(f"  System: {sys_info['os']} | CPU: {sys_info['cpu_count']} core | GPU: {sys_info['gpu_names']}")
    
    # 3. Load Model
    model_path = os.path.join(config.MODELS_DIR, f"{config.RUN_NAME}_best.keras")
    if not os.path.exists(model_path):
        print(f" Model not found: {model_path}")
        return
        
    print("\n Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    # 4. Dataset
    print(" Loading test data (via data_loader)...")
    try:
        # get only test set from data_loader
        if hasattr(load_datasets, '__code__') and load_datasets.__code__.co_argcount == 0:
             # If called without arguments
             _, _, test_ds = load_datasets()
        else:
             # If it requires arguments (old versions)
             _, _, test_ds = load_datasets() 
    except Exception as e:
        print(f" data_loader error: {e}. Trying manual load...")
        # Fallback: Manual load
        test_dir = os.path.join(config.SYNTHETIC_DIR, 'test')
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, shuffle=False
        ).map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

    # 5. Predictions (Batch Inference)
    print("\n Predicting the entire test set (This process may take time depending on data size)...")
    y_true = []
    y_pred_probs = []
    
    # Prediction with progress bar
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("    Predictions completed.")

    # 6. Analysis and Reporting
    print("\n Calculating Graphics and Metrics...")
    
    # A. General Metrics
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics_summary = {
        "accuracy": acc,
        "mcc": mcc,
        "total_samples": len(y_true)
    }
    
    print(f"    Accuracy: {acc:.4f}")
    print(f"    MCC Score: {mcc:.4f}")

    # B. Graphics
    plot_confusion_matrix_advanced(y_true, y_pred, config.CLASS_NAMES, report_dir)
    plot_multiclass_roc(y_true, y_pred_probs, config.CLASS_NAMES, report_dir)
    plot_confidence_histograms(y_true, y_pred, y_pred_probs, report_dir)
    
    # C. Classification Report
    clf_report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    with open(os.path.join(report_dir, 'classification_report.txt'), 'w') as f:
        f.write(clf_report)
    
    # 7. Latency Test
    # Create a single sample
    sample_img = tf.random.normal((1, *config.IMG_SIZE, 3))
    latency_stats, latencies_raw = benchmark_latency(model, sample_img)
    
    print(f"    Average Latency: {latency_stats['avg_latency_ms']:.2f} ms")
    print(f"    P99 Latency: {latency_stats['p99_latency_ms']:.2f} ms (Worst 1%)")
    print(f"    FPS: {latency_stats['throughput_fps']:.2f}")

    # Latency Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(latencies_raw, bins=30, color='purple', alpha=0.7)
    plt.axvline(latency_stats['avg_latency_ms'], color='k', linestyle='dashed', linewidth=1, label='Avg')
    plt.axvline(latency_stats['p99_latency_ms'], color='r', linestyle='dashed', linewidth=1, label='P99')
    plt.title('Inference Latency Distribution')
    plt.xlabel('Time (ms)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'latency_dist.png'))
    plt.close()

    # 8. Save all results as JSON
    final_report = {
        "system_info": sys_info,
        "model_info": {
            "name": config.RUN_NAME,
            "backbone": config.MODEL_BACKBONE,
            "input_size": config.IMG_SIZE
        },
        "performance_metrics": metrics_summary,
        "latency_metrics": latency_stats,
        "classification_report": classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)
    }
    
    with open(os.path.join(report_dir, 'FULL_REPORT.json'), 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print("\n" + "="*60)
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f" All Reports: {report_dir}")
    print("="*60)

if __name__ == "__main__":
    main()