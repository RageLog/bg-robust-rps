"""
test_cross_dataset.py
=====================
This script evaluates how well the trained models generalize on completely
unseen external datasets (e.g., HaGRID, ASL).

Usage Example:
python src/test_cross_dataset.py --model_path models/RPS_EfficientNetV2B0_full_best.keras --model_name EfficientNetV2B0 --dataset_dir /content/drive/MyDrive/RPC_Colab/external_test
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_preprocessing_function(model_name):
    """Returns the Keras preprocess_input function appropriate for the model architecture."""
    model_name = model_name.lower()
    if 'resnet' in model_name:
        return tf.keras.applications.resnet50.preprocess_input
    elif 'vgg' in model_name:
        return tf.keras.applications.vgg16.preprocess_input
    elif 'mobilenet' in model_name:
        return tf.keras.applications.mobilenet_v3.preprocess_input
    elif 'dense' in model_name:
        return tf.keras.applications.densenet.preprocess_input
    else:
        # Dummy pass-through for models like EfficientNet with built-in preprocessing
        return lambda x: x

def evaluate_on_external(model_path, model_name, dataset_dir, ablation="FULL", head_strategy="standard", tune_strategy="standard"):
    print(f"\n==============================================")
    print(f"[INFO] CROSS-DATASET EVALUATION STARTING ")
    print(f" Model: {model_name}")
    print(f" Weights: {os.path.basename(model_path)}")
    print(f" Dataset: {dataset_dir}")
    print(f" Ablation: {ablation} | Head: {head_strategy} | Tune: {tune_strategy}")
    print(f"==============================================\n")
    
    # Try to extract fold from weight name to avoid overwriting reports
    # E.g., RPC_ResNet50_full_standard_standard_fold_1_best.keras
    import re
    fold_match = re.search(r'_fold_(\d+)', os.path.basename(model_path))
    fold_suffix = f"_fold_{fold_match.group(1)}" if fold_match else "_full"
    
    report_filename = f"cross_dataset_{model_name}_{ablation.lower()}_{head_strategy}_{tune_strategy}{fold_suffix}.json"
    report_dir = os.path.join(config.REPORTS_DIR, model_name, ablation.upper())
    local_report = os.path.join(report_dir, report_filename)
    
    # Early out if report already exists
    if not os.path.exists(local_report) and os.path.exists(config.DRIVE_DIR):
        drive_report = os.path.join(config.DRIVE_DIR, "reports", model_name, ablation.upper(), report_filename)
        if os.path.exists(drive_report):
            os.makedirs(report_dir, exist_ok=True)
            import shutil
            shutil.copy2(drive_report, local_report)
            
    if os.path.exists(local_report):
        print(f"[INFO] Cross-Dataset Evaluation for {model_name} {ablation} {fold_suffix} already completed! Skipping.")
        return
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
        
    import download_external_dataset
    download_external_dataset.download_and_prepare(dataset_dir)
    
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        raise FileNotFoundError(f"[ERROR] External dataset folder is desperately missing: {dataset_dir}")
    # 1. Load Model
    print("[INFO] Loading model into memory...")
    
    # Keras Application Specific Preprocess Function
    preprocessor = get_preprocessing_function(model_name)
    import keras
    # Keras 3 Serialization Fix for Lambda layers using built-in functions
    @keras.saving.register_keras_serializable(package="builtins", name="preprocess_input")
    def _dummy_preprocess(*args, **kwargs):
        return preprocessor(*args, **kwargs)
        
    model = keras.models.load_model(model_path, custom_objects={'preprocess_input': _dummy_preprocess}, safe_mode=False)
    
    def process_image(img_path, label):
        """Reads and pre-processes the image for the model."""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, config.IMG_SHAPE[:2])
        # The image is currently in the 0-255 range.
        # Since the models were trained with images mapped to [0,1] in data_loader,
        # we apply the same scaling here. The built-in Rescaling layer in models.py 
        # (lambda x: x * 255.0) will handle the conversion back to the format 
        # required by the specific Keras application's preprocess_input.
        img = tf.cast(img, tf.float32) / 255.0 
        
        return img, label

    # 2. Find Dataset Paths
    label_map = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
    img_paths = []
    labels = []
    
    for cls_name in config.CLASS_NAMES:
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"    [WARN] Class folder not found, skipping: {cls_dir}")
            continue
            
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_paths.append(os.path.join(cls_dir, fname))
                labels.append(label_map[cls_name])
                
    if not img_paths:
        print(f"\n[ERROR] No test images found in '{dataset_dir}'.")
        print(f"        For Cross-Dataset Evaluation, the external test images MUST be placed inside subfolders named exactly as the target classes.")
        print(f"        Expected structure:")
        print(f"          {dataset_dir}/")
        for c in config.CLASS_NAMES:
            print(f"            ├── {c}/")
            print(f"            │   ├── image1.jpg")
            print(f"            │   └── ...")
        print(f"        If you just have a bunch of images in the root folder, this script cannot measure accuracy because it needs the ground-truth labels (indicated by the folder names).")
        print(f"        Please organize the dataset and try again.\n")
        raise ValueError(f"Missing class subfolders or images in '{dataset_dir}'.")
        
    print(f"[INFO] Total {len(img_paths)} external test images found.")
    
    # tf.data Pipeline
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 3. Model Inference
    print("\n[INFO] Running predictions...")
    predictions = model.predict(ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.array(labels)
    
    # 4. Metrics and Reporting
    print("\n[RESULT] CLASSIFICATION REPORT")
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print(report)
    
    report_dir = os.path.join(config.REPORTS_DIR, model_name, f"{ablation.upper()}_{head_strategy}_{tune_strategy}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save Confusion Matrix (SVG/PDF High Res)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title(f'Cross-Dataset Confusion Matrix ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"cross_dataset_cm.pdf"), format='pdf', dpi=300)
    plt.savefig(os.path.join(report_dir, f"cross_dataset_cm.svg"), format='svg', dpi=300)
    plt.close()
    
    # 5. Advanced Metrics: ROC-AUC & Precision-Recall
    # Convert y_true to one-hot encoding for multi-class ROC
    from tensorflow.keras.utils import to_categorical
    y_true_onehot = to_categorical(y_true, num_classes=len(config.CLASS_NAMES))
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(config.CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC Curve ({model_name})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"cross_dataset_roc.pdf"), format='pdf', dpi=300)
    plt.savefig(os.path.join(report_dir, f"cross_dataset_roc.svg"), format='svg', dpi=300)
    plt.close()
    
    # Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(config.CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], predictions[:, i])
        ap = average_precision_score(y_true_onehot[:, i], predictions[:, i])
        plt.plot(recall, precision, lw=2, label=f'{cls} (AP = {ap:.3f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"cross_dataset_pr.pdf"), format='pdf', dpi=300)
    plt.savefig(os.path.join(report_dir, f"cross_dataset_pr.svg"), format='svg', dpi=300)
    plt.close()
    
    print(f"[SUCCESS] High-resolution charts (SVG/PDF) saved to {report_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval external dataset (e.g., HaGRID)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .keras file")
    parser.add_argument("--model_name", type=str, default="EfficientNetV2B0", help="Architecture name")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root directory of external test data")
    parser.add_argument("--ablation", type=str, default="FULL", help="Ablation logic to categorize output")
    parser.add_argument("--head_strategy", type=str, default="standard", help="Head strategy logic")
    parser.add_argument("--tune_strategy", type=str, default="standard", help="Tuning strategy logic")
    
    args = parser.parse_args()
    evaluate_on_external(args.model_path, args.model_name, args.dataset_dir, args.ablation, args.head_strategy, args.tune_strategy)
