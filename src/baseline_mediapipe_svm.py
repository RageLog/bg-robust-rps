"""
baseline_mediapipe_svm.py - Baseline Model Comparison
======================================================
This script evaluates a classical Machine Learning (SVM) + MediaPipe Hand Skeleton Extraction 
approach to benchmark the performance against our Deep Learning (CNN) models.
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Version-agnostic mediapipe import
# MediaPipe >=0.10.21 removed mp.solutions namespace; use Tasks API as fallback.
_USE_TASKS_API = False
_hand_landmarker = None

try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    print("[INFO] MediaPipe legacy Solutions API loaded.")
except (AttributeError, TypeError):
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )
        import urllib.request

        # Download hand_landmarker model if not present
        _model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(_model_path):
            print("[INFO] Downloading hand_landmarker.task model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                _model_path
            )

        try:
            _options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_model_path, delegate=BaseOptions.Delegate.GPU),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.3
            )
            _hand_landmarker = HandLandmarker.create_from_options(_options)
            _USE_TASKS_API = True
            hands = None  # Not used with Tasks API
            print("[INFO] MediaPipe Tasks API loaded (>=0.10.21) with GPU Delegate.")
        except Exception as gpu_err:
            print(f"[WARN] GPU Delegate initialization failed: {gpu_err}. Falling back to CPU...")
            _options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.3
            )
            _hand_landmarker = HandLandmarker.create_from_options(_options)
            _USE_TASKS_API = True
            hands = None
            print("[INFO] MediaPipe Tasks API loaded (>=0.10.21) with CPU.")
    except Exception as e:
        print(f"[ERROR] MediaPipe initialization failed: {e}")
        print("[TIP] Run: pip install mediapipe==0.10.14  (or any version <0.10.21)")
        sys.exit(1)

def extract_features(img_path):
    """Extracts MediaPipe 21x3=63 dimensional landmarks from the image. Returns zeros if not found."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(63)
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features = np.zeros(63) # x, y, z coordinates of 21 points

    if _USE_TASKS_API:
        # MediaPipe Tasks API (>=0.10.21)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = _hand_landmarker.detect(mp_image)
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            landmarks = results.hand_landmarks[0]
            for i, lm in enumerate(landmarks):
                features[i*3] = lm.x
                features[i*3 + 1] = lm.y
                features[i*3 + 2] = lm.z
    else:
        # Legacy Solutions API
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            for i, lm in enumerate(landmarks):
                features[i*3] = lm.x
                features[i*3 + 1] = lm.y
                features[i*3 + 2] = lm.z
            
    return features

def load_data(split_path, class_names):
    """Reads images from the folder and returns feature/label arrays."""
    X = []
    y = []
    
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.exists(class_dir):
            continue
            
        img_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.png"))
        
        for img_path in tqdm(img_paths, desc=f"{split_path.split(os.sep)[-1]} - {class_name}"):
            feats = extract_features(img_path)
            X.append(feats)
            y.append(label_idx)
            
    return np.array(X), np.array(y)

def run_baseline(ablation_mode):
    print(f"\n==============================================")
    print(f"[INFO] BASELINE TRAINING: MediaPipe + SVM ({ablation_mode})")
    print(f"==============================================\n")
    
    config.SYNTHETIC_DIR = os.path.join(config.BASE_DIR, "datasets", f"synthetic_{ablation_mode.lower()}")
    
    if not os.path.exists(config.SYNTHETIC_DIR):
        print(f"[ERROR] Data folder not found: {config.SYNTHETIC_DIR}")
        print(f"[INFO] First, generate data (generate_data.py) and split it (split_data.py)")
        return
        
    train_dir = os.path.join(config.SYNTHETIC_DIR, "train")
    val_dir = os.path.join(config.SYNTHETIC_DIR, "val")
    test_dir = os.path.join(config.SYNTHETIC_DIR, "test")
    
    print("[INFO] 1. Feature Extraction Starting (MediaPipe)...")
    start_time = time.time()
    
    # Merge Train and Val data for SVM
    X_train_raw, y_train_raw = load_data(train_dir, config.CLASS_NAMES)
    X_val, y_val = load_data(val_dir, config.CLASS_NAMES)
    
    X_train = np.vstack((X_train_raw, X_val)) if len(X_val) > 0 else X_train_raw
    y_train = np.hstack((y_train_raw, y_val)) if len(y_val) > 0 else y_train_raw
    
    X_test, y_test = load_data(test_dir, config.CLASS_NAMES)
    
    ext_time = time.time() - start_time
    print(f"\n[INFO] Feature extraction completed ({ext_time:.2f} seconds).")
    print(f"[INFO] Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")
    
    print("\n[INFO] 2. SVM Model Training (RBF Kernel)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=config.RANDOM_SEED)
    
    train_start = time.time()
    svm_model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"[INFO] Training completed ({train_time:.2f} seconds).")
    
    print("\n[INFO] 3. Evaluating on Test Data...")
    y_pred = svm_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Baseline (MediaPipe+SVM) Accuracy: {acc:.4f}\n")
    
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES)
    print(report)
    
    # Save Reports
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Save SVM Model
    model_path = os.path.join(config.MODELS_DIR, f"baseline_svm_{ablation_mode.lower()}.joblib")
    joblib.dump(svm_model, model_path)
    print(f"\n[INFO] Model Saved: {model_path}")
    
    # Save Report
    report_path = os.path.join(config.REPORTS_DIR, f"baseline_report_{ablation_mode.lower()}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"BASELINE: MediaPipe + SVM ({ablation_mode})\n")
        f.write("=========================================\n")
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")
        f.write(f"Training Time: {train_time:.2f} sec\n")
        f.write(f"Feature Extraction Time: {ext_time:.2f} sec\n")
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES, 
                yticklabels=config.CLASS_NAMES)
    plt.title(f'Baseline Confusion Matrix ({ablation_mode})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    fig_path = os.path.join(config.REPORTS_DIR, f"baseline_cm_{ablation_mode.lower()}.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Reports saved to {config.REPORTS_DIR} directory.")
    
    # Explicitly close the Tasks API object to prevent __del__ errors during interpreter shutdown
    global _hand_landmarker
    if _USE_TASKS_API and _hand_landmarker is not None:
        try:
            _hand_landmarker.close()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default='FULL')
    args = parser.parse_args()
    
    run_baseline(args.ablation)
