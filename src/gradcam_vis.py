"""
gradcam_vis.py - Grad-CAM Visualization
========================================
This script visualizes "which pixels" (regions) the trained models focus on
when predicting a specific image using a heatmap.
Provides explainability for the thesis.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_last_conv_layer_name(model):
    """
    Recursively finds the name of the last convolutional layer in the model.
    Handles nested Functional models/Sequentials (common in Keras Applications).
    Returns (conv_layer_name, inner_model) where inner_model is the sub-model if applicable.
    """
    for layer in reversed(model.layers):
        # Check if the layer contains other layers (Functional or Sequential)
        if hasattr(layer, 'layers'):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, tf.keras.layers.Conv2D):
                    return inner_layer.name, layer
                    
        # Standard model layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, None
            
    raise ValueError("[ERROR] No Conv2D layer found in the model architecture.")

def build_gradcam_model(model, last_conv_layer_name, inner_model=None):
    """
    Safely constructs the split Grad-CAM models depending on architecture nesting.
    Returns (grad_model, classifier_model)
    """
    if inner_model is not None:
        # 1. The backbone feature extractor (Image -> Conv Features)
        grad_model = tf.keras.models.Model(
            inner_model.inputs, 
            [inner_model.get_layer(last_conv_layer_name).output, inner_model.output]
        )
        
        # 2. The classification head (Conv Features/Pools -> Predictions)
        # Recreate the path from inner_model.output to final prediction
        classifier_input = tf.keras.Input(shape=inner_model.output.shape[1:])
        x = classifier_input
        
        # Flag to start capturing layers AFTER the inner_model
        capture = False
        for layer in model.layers:
            if layer.name == inner_model.name:
                capture = True
                continue
            if not capture:
                continue
            # Skip preprocessing lambda layers that alter input shape unnecessarily
            if isinstance(layer, tf.keras.layers.Lambda) and 'preprocess' in layer.name.lower():
                continue
            x = layer(x)
            
        classifier_model = tf.keras.models.Model(classifier_input, x)
        return grad_model, classifier_model
        
    else:
        # Flat model (e.g. basic Sequential)
        grad_model = tf.keras.models.Model(
            model.inputs, 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        # There is no standalone classifier model in a flat architecture for this simple setup
        return grad_model, None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, inner_model=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap robustly with mathematical safety nets.
    """
    grad_model, classifier_model = build_gradcam_model(model, last_conv_layer_name, inner_model)

    with tf.GradientTape() as tape:
        if classifier_model:
            # Inputs -> [Backbone] -> Conv Output & Global Features
            # Global Features -> [Classifier Head] -> Preds
            # Note: We simulate any input preprocessing required by the backbone
            preprocessed_input = img_array * 255.0
            last_conv_layer_output, backbone_out = grad_model(preprocessed_input)
            preds = classifier_model(backbone_out)
        else:
            last_conv_layer_output, preds = grad_model(img_array)
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Mathematical Safety: Check if gradients died (all zeros or NaNs)
    if grads is None:
        print("[WARN] Gradient tape returned None. Returning blank heatmap.")
        return np.zeros((last_conv_layer_output.shape[1], last_conv_layer_output.shape[2]))
        
    # Average gradients per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply gradients with feature map
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (only consider positive effects)
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize securely
    max_val = tf.math.reduce_max(heatmap)
    
    if max_val == 0.0 or tf.math.is_nan(max_val):
        # Gradients collapsed or feature maps are dead
        print(f"[WARN] Heatmap max is zero or NaN. Returning blank/normalized map.")
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / max_val
    
    heatmap_numpy = heatmap.numpy()
    
    # Final NaN scrubber
    if np.isnan(heatmap_numpy).any():
        heatmap_numpy = np.nan_to_num(heatmap_numpy)
        
    return heatmap_numpy

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """Overlays the heatmap onto the original image and saves it."""
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image at {img_path} for Grad-CAM overlay.")
        return
        
    # Guard against completely empty/invalid heatmaps returned by safety nets
    if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
        print(f"[WARN] Heatmap for {img_path} is completely empty. Skipping overlay.")
        return
        
    # Mixed precision models return float16, which OpenCV resize does NOT support. Cast to float32.
    heatmap = np.float32(heatmap)
        
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to 0-255 RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose with the original image
    superimposed_img = heatmap * alpha + img
    
    cv2.imwrite(cam_path, superimposed_img)

def run_gradcam_on_folder(model_path, images_dir, output_dir):
    print(f"\n======================================")
    print(f"[INFO] GRAD-CAM VISUALIZATION STARTING")
    print(f" Model: {model_path}")
    print(f"======================================\n")
    
    # Load Model
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found! {model_path}")
        return
        
    # Apply standard project-wide Lambda deserialization fix for custom preprocessors
    import test_cross_dataset
    model_name = "EfficientNetV2B0" # default fallback
    for name in config.CLASS_NAMES + ["ResNet50", "VGG16", "MobileNet", "DenseNet121", "EfficientNetV2B0"]:
        if name.lower() in model_path.lower():
            model_name = name
            break
            
    preprocessor = test_cross_dataset.get_preprocessing_function(model_name)
    import keras
    @keras.saving.register_keras_serializable(package="builtins", name="preprocess_input")
    def _dummy_preprocess(*args, **kwargs):
        return preprocessor(*args, **kwargs)
        
    try:
        model = keras.models.load_model(model_path, custom_objects={'preprocess_input': _dummy_preprocess}, safe_mode=False)
    except Exception as e:
        print(f"[ERROR] Failed to load model for Grad-CAM: {e}")
        return
    
    try:
        conv_layer, inner_model = get_last_conv_layer_name(model)
        inner_name = inner_model.name if inner_model else ''
        print(f"[INFO] Detected Last Conv Layer: {inner_name + '/' if inner_name else ''}{conv_layer}")
    except Exception as e:
        print(f"[ERROR] Could not find Conv layer in Keras architecture. {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Find images
    valid_exts = ('.jpg', '.jpeg', '.png')
    img_paths = []
    
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                img_paths.append(os.path.join(root, f))
                
    if not img_paths:
        print(f"[WARN] No test images found in {images_dir}.")
        return
        
    # To find misclassifications efficiently, we shuffle and scan
    np.random.shuffle(img_paths)
    scan_limit = min(200, len(img_paths))
    
    correct_cases = []
    misclassified_cases = []
    
    print(f"[INFO] Scanning up to {scan_limit} images for Misclassification Analysis...\n")
    
    # Avoid verbose output during scan
    for img_path in img_paths[:scan_limit]:
        true_class = os.path.basename(os.path.dirname(img_path)).lower()
        if true_class not in config.CLASS_NAMES:
            true_class = "unknown" # Fallback if folder isn't named strictly as class
            
        cv_img = cv2.imread(img_path)
        if cv_img is None: continue
        
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(cv_img_rgb, config.IMG_SIZE)
        img_array = np.expand_dims(resized_img, axis=0)
        img_array = img_array.astype('float32') / 255.0

        preds = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_class = config.CLASS_NAMES[pred_idx].lower()
        confidence = preds[0][pred_idx]
        
        info = {
            'path': img_path,
            'true': true_class,
            'pred': pred_class,
            'conf': confidence,
            'img_array': img_array,
            'pred_idx': pred_idx
        }
        
        if true_class == pred_class:
            correct_cases.append(info)
        else:
            misclassified_cases.append(info)
            
    # Take up to 10 correct and 10 misclassified
    selected_cases = correct_cases[:10] + misclassified_cases[:10]
    
    print(f"[INFO] Found {len(misclassified_cases)} misclassifications out of {scan_limit} images scanned.")
    print(f"[INFO] Generating heatmaps for {min(10, len(correct_cases))} Correct and {min(10, len(misclassified_cases))} Misclassified images...\n")
    
    for i, info in enumerate(selected_cases):
        img_path = info['path']
        true_c = info['true']
        pred_c = info['pred']
        conf = info['conf']
        img_array = info['img_array']
        pred_idx = info['pred_idx']
        
        heatmap = make_gradcam_heatmap(img_array, model, conv_layer, inner_model=inner_model, pred_index=pred_idx)
        
        status = "CORRECT" if true_c == pred_c else "ERROR"
        base_name = os.path.basename(img_path)
        save_name = f"{status}_True[{true_c}]_Pred[{pred_c}]_{conf*100:.1f}pct_{base_name}"
        
        # Save nested by the true class first, then status for better organization
        sub_dir = os.path.join(output_dir, true_c, status.lower())
        os.makedirs(sub_dir, exist_ok=True)
        save_path = os.path.join(sub_dir, save_name)
        
        save_and_display_gradcam(img_path, heatmap, cam_path=save_path)
        
        print(f"  [{i+1}/{len(selected_cases)}] -> {status} | True: {true_c}, Pred: {pred_c} ({conf*100:.1f}%) -> Saved")

    print(f"\n[SUCCESS] All operations completed. Outputs: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Full path to the trained .keras model")
    parser.add_argument('--test_dir', type=str, required=True, help="Directory containing test images (e.g. datasets/synthetic_full/test/rock)")
    parser.add_argument('--out_dir', type=str, default="reports/gradcam", help="Output directory")
    args = parser.parse_args()
    
    run_gradcam_on_folder(args.model_path, args.test_dir, args.out_dir)
