"""
evaluate_cross_dataset.py - Zero-Shot / Cross-Dataset Evaluation
================================================================
Evaluates the best trained model on an entirely unseen, external
dataset (e.g., HaGRID, ASL, or web-scraped images) to prove
the model's true generalization capability, not just over-fitting
to the primary dataset's lighting/camera settings.
"""
import os
import sys
import json
import glob
import shutil
import zipfile
import urllib.request
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Mappings from external dataset class names to our intrinsic class names
# Example: If testing on HaGRID, "peace" means "scissors".
CLASS_MAPPING = {
    "rock": "rock",
    "fist": "rock",
    "stone": "rock",
    
    "paper": "paper",
    "palm": "paper",
    "flat": "paper",
    "five": "paper",
    
    "scissors": "scissors",
    "peace": "scissors",
    "v": "scissors",
    "two": "scissors",
    
    "none": "none",
    "background": "none",
    "empty": "none"
}

def load_external_dataset(external_dir):
    """Loads images from external_dir, mapping their subfolder names to our classes."""
    images = []
    labels = []
    found_classes = set()
    
    for ext_class_name in os.listdir(external_dir):
        class_dir = os.path.join(external_dir, ext_class_name)
        if not os.path.isdir(class_dir):
            continue
            
        ext_class_lower = ext_class_name.lower()
        if ext_class_lower not in CLASS_MAPPING:
            print(f"[WARN] Unknown external class '{ext_class_name}'. Skipping.")
            continue
            
        target_class = CLASS_MAPPING[ext_class_lower]
        if target_class not in config.CLASS_NAMES:
            print(f"[WARN] Target class '{target_class}' not in our config.CLASS_NAMES. Skipping.")
            continue
            
        target_idx = config.CLASS_NAMES.index(target_class)
        found_classes.add(target_class)
        
        img_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                    glob.glob(os.path.join(class_dir, "*.png")) + \
                    glob.glob(os.path.join(class_dir, "*.jpeg"))
                    
        for p in img_paths:
            try:
                img = tf.keras.preprocessing.image.load_img(p, target_size=config.IMG_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(target_idx)
            except Exception as e:
                pass # Skip corrupt images
                
    return np.array(images), np.array(labels), list(found_classes)

def download_and_prepare_dataset(external_dir):
    """
    Downloads the external dataset automatically if it doesn't exist.
    Caches it to Google Drive for faster future runs.
    """
    # Try recovering from Drive first for speed
    drive_cache_zip = "/content/drive/MyDrive/RPC_Colab/external_test.zip"
    local_zip = os.path.join(config.BASE_DIR, "datasets", "external_test.zip")
    
    if not os.listdir(external_dir):
        print("[INFO] Local external_test dir is empty. Searching for dataset...")
        
        # 1. Check if cached in Drive
        if os.path.exists(drive_cache_zip):
            print(f"[INFO] Found cached dataset in Drive: {drive_cache_zip}")
            shutil.copy2(drive_cache_zip, local_zip)
        else:
            print("[INFO] No Drive cache found. Attempting to download external dataset...")
            # Placeholder URL: Replace with actual HaGRID/ASL slice URL (e.g., Dropbox/Gdrive link)
            # For demonstration, we'll gracefully handle a missing URL by prompting the user
            ext_url = getattr(config, 'EXTERNAL_DATASET_URL', None)
            
            if ext_url:
                try:
                    print(f"[INFO] Downloading from {ext_url} ...")
                    urllib.request.urlretrieve(ext_url, local_zip)
                    
                    # Cache to Drive
                    drive_dir = os.path.dirname(drive_cache_zip)
                    if os.path.exists(drive_dir):
                        print("[INFO] Caching downloaded dataset to Google Drive...")
                        shutil.copy2(local_zip, drive_cache_zip)
                except Exception as e:
                    print(f"[ERROR] Auto-download failed: {e}")
                    return False
            else:
                 print("[WARN] No config.EXTERNAL_DATASET_URL defined. Auto-download skipped.")
                 print("-> ACTION REQUIRED: Place external dataset folders into datasets/external_test/ manually, OR define EXTERNAL_DATASET_URL in config.py")
                 return False

        # 2. Extract
        if os.path.exists(local_zip):
            print(f"[INFO] Extracting {local_zip}...")
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(external_dir)
            os.remove(local_zip)
            return True
            
    return len(os.listdir(external_dir)) > 0


def get_preprocessing_function(model_name):
    """Returns the Keras preprocess_input function appropriate for the model architecture."""
    if "ResNet50" in model_name:
        return tf.keras.applications.resnet50.preprocess_input
    elif "VGG16" in model_name:
        return tf.keras.applications.vgg16.preprocess_input
    elif "MobileNetV3Small" in model_name:
        return tf.keras.applications.mobilenet_v3.preprocess_input
    elif "DenseNet121" in model_name:
        return tf.keras.applications.densenet.preprocess_input
    else: # EfficientNetV2B0 or generic fallbacks
        return lambda x: x

import keras

# Keras 3 Serialization Fix for Lambda layers using built-in functions
# Must be registered globally before loading any models
CURRENT_PREPROCESSOR = None

@keras.saving.register_keras_serializable(package="builtins", name="preprocess_input")
def _dummy_preprocess(*args, **kwargs):
    global CURRENT_PREPROCESSOR
    if CURRENT_PREPROCESSOR is not None:
        return CURRENT_PREPROCESSOR(*args, **kwargs)
    return args[0]

def main():
    print("[INFO] Starting Cross-Dataset (Zero-Shot) Evaluation...")
    
    external_dir = os.path.join(config.BASE_DIR, "datasets", "external_test")
    os.makedirs(external_dir, exist_ok=True)
    
    import prepare_hagrid_annotations
    dataset_ready = prepare_hagrid_annotations.crop_hagrid_dataset()
    if not dataset_ready:
        print("[SKIP] External dataset is missing and could not be auto-downloaded.")
        return
        
    # Get ALL trained models
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_files = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('.keras')]
    if not model_files:
        print("[ERROR] No trained models found in the models directory to evaluate.")
        return
        
    print(f"[INFO] Found {len(model_files)} models. Evaluating all models on cross-dataset...")
    
    X_test, y_test, found_classes = load_external_dataset(external_dir)
    if len(X_test) == 0:
        print("[ERROR] No valid mapped images found in the external dataset directory.")
        return
        
    print(f"[INFO] External dataset loaded: {len(X_test)} images mapped to classes: {found_classes}")
    
    out_dir = os.path.join(config.REPORTS_DIR, "cross_dataset")
    os.makedirs(out_dir, exist_ok=True)
    
    global_report = {}
    
    for model_file in model_files:
        model_path = os.path.join(config.MODELS_DIR, model_file)
        model_name = model_file.replace('.keras', '')
        print(f"\n[EVAL] Testing model: {model_name}")
        
        try:
            # Set global preprocessor for this specific architecture
            global CURRENT_PREPROCESSOR
            CURRENT_PREPROCESSOR = get_preprocessing_function(model_name)
                
            model = keras.models.load_model(model_path, custom_objects={'preprocess_input': _dummy_preprocess}, safe_mode=False)
            preds = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            
            cr = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, 
                                       labels=range(len(config.CLASS_NAMES)), output_dict=True, zero_division=0)
            
            acc = cr['accuracy']
            print(f" -> Accuracy: {acc:.4f}")
            
            global_report[model_name] = {
                "accuracy": acc,
                "classification_report": cr
            }
            
            # Plot specific Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=range(len(config.CLASS_NAMES)))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
            plt.title(f"Cross-Dataset CM: {model_name}")
            plt.ylabel('True Class (External GT)')
            plt.xlabel('Predicted Class (Our Model)')
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"cm_{model_name}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
        except Exception as e:
            print(f" -> [ERROR] Failed to evaluate {model_name}: {e}")
            
    # Save global aggregated JSON Report
    report_data = {
        "external_database_path": external_dir,
        "sample_size": len(X_test),
        "classes_found": found_classes,
        "model_results": global_report
    }
    
    report_path = os.path.join(out_dir, "global_cross_dataset_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print(f"\n[SUCCESS] Cross-Dataset Evaluation Complete! Aggregated report saved to: {report_path}")

if __name__ == "__main__":
    main()
