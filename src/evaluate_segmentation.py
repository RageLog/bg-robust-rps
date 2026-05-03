"""
evaluate_segmentation.py - Quantitative Evaluation of Rembg Segmentation
========================================================================
Because we lack pixel-perfect manual Ground Truth (GT) masks, this script
uses an unsupervised/pseudo-GT approach to evaluate the precision of the
U-2-Net based rembg component.

Metrics Evaluated:
1. Pseudo-IoU (Intersection Over Union vs MediaPipe Hands Bounding Box)
2. Edge Contrast Quality (Sharpness of the boundary F1-proxy)
"""
import os
import sys
import cv2
import glob
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import importlib.util

# Conditional rembg import (handles some environments without it)
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

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

        _options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3
        )
        _hand_landmarker = HandLandmarker.create_from_options(_options)
        _USE_TASKS_API = True
        hands = None  # Not used with Tasks API
        print("[INFO] MediaPipe Tasks API loaded (>=0.10.21).")
    except Exception as e:
        print(f"[ERROR] MediaPipe initialization failed: {e}")
        print("[TIP] Run: pip install mediapipe==0.10.14  (or any version <0.10.21)")
        sys.exit(1)

def get_mediapipe_bounding_box(image_rgb):
    """Returns the bounding box [x_min, y_min, x_max, y_max] of a hand using MediaPipe."""
    h, w, _ = image_rgb.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    found = False
    if _USE_TASKS_API:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = _hand_landmarker.detect(mp_image)
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            found = True
            for lm in results.hand_landmarks[0]:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
    else:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            found = True
            for landmark in results.multi_hand_landmarks[0].landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
                
    if not found:
        return None
        
    # Add padding to bounding box
    pad = int(w * 0.05)
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)
    
    return [x_min, y_min, x_max, y_max]

def calculate_iou(boxA, boxB):
    """Calculates Intersection Over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_edge_contrast(grayscale_mask):
    """
    Evaluates how sharp the segmentation edges are.
    Higher means crisp, confident boundaries. Low means blurry/uncertain.
    """
    # Calculate gradients
    sobelx = cv2.Sobel(grayscale_mask, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(grayscale_mask, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Average gradient magnitude along non-zero edge pixels
    edge_pixels = magnitude[magnitude > 0]
    if len(edge_pixels) == 0:
        return 0.0
    return np.mean(edge_pixels)

def main():
    if not REMBG_AVAILABLE:
        print("[ERROR] rembg is not installed. Cannot evaluate segmentation.")
        return
        
    print("[INFO] Starting Rembg Segmentation Quality Evaluation...")
    # Select a small subset of absolute raw original data
    raw_dir = os.path.join(config.BASE_DIR, "datasets", "raw")
    if not os.path.exists(raw_dir):
         print(f"[ERROR] Raw data not found at: {raw_dir}")
         return
         
    classes = [c for c in config.CLASS_NAMES if c != "none"]
    all_images = []
    for c in classes:
        class_dir = os.path.join(raw_dir, c)
        if os.path.exists(class_dir):
            images = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.png"))
            # Take up to 20 images per class for evaluation speed
            all_images.extend(images[:20])
            
    if not all_images:
        print("[ERROR] No images found for segmentation evaluation.")
        return
        
    print(f"[INFO] Evaluating {len(all_images)} images across {len(classes)} classes...")
    
    session = new_session()
    
    iou_scores = []
    edge_quality_scores = []
    
    pbar = tqdm(total=len(all_images), desc="Evaluating")
    for img_path in all_images:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            pbar.update(1)
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Get Pseudo-GT Bounding Box from MediaPipe
        mp_bbox = get_mediapipe_bounding_box(img_rgb)
        
        # 2. Extract Rembg Mask
        # rembg returns an RGBA image. The Alpha channel is our mask.
        rembg_rgba = remove(img_bgr, session=session)
        mask = rembg_rgba[:, :, 3] 
        
        # Calculate Bounding Box of Rembg Mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour (the hand)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            rembg_bbox = [x, y, x + w, y + h]
            
            # Compare bounding boxes (Pseudo-IoU)
            if mp_bbox:
                iou = calculate_iou(mp_bbox, rembg_bbox)
                iou_scores.append(iou)
                
            # Evaluate Edge Sharpness Quality
            edge_quality = calculate_edge_contrast(mask)
            edge_quality_scores.append(edge_quality)
        
        pbar.update(1)
    pbar.close()
    
    # Calculate Statistical Summary
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    std_iou = float(np.std(iou_scores)) if iou_scores else 0.0
    mean_edge = float(np.mean(edge_quality_scores)) if edge_quality_scores else 0.0
    
    print("\n--- SEGMENTATION EVALUATION RESULTS ---")
    print(f"Mean Pseudo-IoU (vs MediaPipe) : {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Mean Edge Contrast Quality     : {mean_edge:.4f}")
    print("---------------------------------------")
    
    report_data = {
        "evaluation_count": len(iou_scores),
        "metrics": {
            "pseudo_iou_mean": mean_iou,
            "pseudo_iou_std": std_iou,
            "edge_contrast_quality_mean": mean_edge
        },
        "description": "IoU calculated against MediaPipe hand detection bounding boxes as pseudo-ground truth. Edge contrast represents the Sharpness/Gradient Magnitude of the alpha mask."
    }
    
    report_dir = os.path.join(config.REPORTS_DIR, "segmentation")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "rembg_evaluation_summary.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print(f"[SUCCESS] Saved quantitative segmentation report to {report_path}")
    
    # Explicitly close the Tasks API object to prevent __del__ errors during interpreter shutdown
    global _hand_landmarker
    if _USE_TASKS_API and _hand_landmarker is not None:
        try:
            _hand_landmarker.close()
        except:
            pass
        finally:
            # Suppress stderr right before exit to hide MediaPipe's unavoidable __del__ NoneType C++ bindings Bug on script termination
            import sys as _sys
            import os as _os
            _sys.stderr = open(_os.devnull, 'w')

if __name__ == "__main__":
    main()
