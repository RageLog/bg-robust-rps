"""
optimize_temporal_smoothing.py - Grid Search for Temporal Smoothing
===================================================================
Simulates temporal streams from the test dataset to optimize
N (history size) and Confidence Threshold for realtime_demo.py.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque
import itertools
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def load_simulated_stream():
    print("[INFO] Loading test dataset for simulation...")
    test_dir = os.path.join(config.BASE_DIR, "datasets", "synthetic_full", "test")
    if not os.path.exists(test_dir):
        print(f"[ERROR] Test directory not found: {test_dir}")
        return None, None
        
    stream_x = []
    stream_y = []
    
    none_idx = config.CLASS_NAMES.index("none") if "none" in config.CLASS_NAMES else -1
    none_files = []
    if none_idx != -1:
        none_dir = os.path.join(test_dir, "none")
        if os.path.exists(none_dir):
            none_files = [os.path.join(none_dir, f) for f in os.listdir(none_dir) if f.lower().endswith(('.jpg', '.png'))]
            # Use up to 50 none images
            none_files = none_files[:50]
    
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        if class_name == "none": continue
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir): continue
        
        # 30 examples per gesture to simulate bursts
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png'))][:30] 
        for f in files:
            img = tf.keras.preprocessing.image.load_img(f, target_size=config.IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            
            # Gesture frame burst (User holds gesture)
            for _ in range(15):
                stream_x.append(img_array)
                stream_y.append(class_idx)
                
            # Transition frame burst (User lowers hand -> none class)
            if none_files:
                nf = np.random.choice(none_files)
                n_img = tf.keras.preprocessing.image.load_img(nf, target_size=config.IMG_SIZE)
                n_array = tf.keras.preprocessing.image.img_to_array(n_img) / 255.0
                for _ in range(5):
                    stream_x.append(n_array)
                    stream_y.append(none_idx)
                    
    return np.array(stream_x), np.array(stream_y)

def main():
    import glob
    model_pattern = os.path.join(config.MODELS_DIR, f"{config.RUN_NAME}_EfficientNetV2B0_full_*best.keras")
    model_paths = glob.glob(model_pattern)
    
    if not model_paths:
        print(f"[ERROR] Model not found to run optimization matching: {model_pattern}")
        return

    # Pick the first matching model (preferably fold_1 and 'both' strategy if multiple exist, but any is fine)
    model_path = sorted(model_paths)[0]

    print(f"[INFO] Evaluating model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    X, y = load_simulated_stream()
    if X is None or len(X) == 0:
        print("[ERROR] Failed to load simulation data.")
        return
    
    print(f"[INFO] Running predictions on {len(X)} simulated frames (This may take a moment)...")
    # Pre-calculate base predictions to speed up grid search
    base_preds = model.predict(X, batch_size=config.BATCH_SIZE, verbose=1)
    
    N_values = [1, 3, 5, 7, 10, 15]
    Thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_n = 5
    best_t = 0.6
    best_score = 0
    
    print("\n--- GRID SEARCH RESULTS ---")
    print(f"{'N':<5} | {'Threshold':<10} | {'Accuracy':<10} | {'False Positive Ratio':<20}")
    print("-" * 55)
    
    none_idx = config.CLASS_NAMES.index("none") if "none" in config.CLASS_NAMES else -1
    
    results_list = []
    
    for n, t in itertools.product(N_values, Thresholds):
        history = deque(maxlen=n)
        correct = 0
        false_positives = 0
        
        for i in range(len(base_preds)):
            history.append(base_preds[i])
            avg_preds = np.mean(history, axis=0)
            class_idx = np.argmax(avg_preds)
            confidence = avg_preds[class_idx]
            
            final_pred = class_idx
            if confidence < t:
                final_pred = none_idx if none_idx != -1 else final_pred
            
            if final_pred == y[i]:
                correct += 1
            elif y[i] == none_idx and final_pred != none_idx:
                # Actual is None, but we predicted a gesture!
                false_positives += 1
                
        acc = correct / len(base_preds)
        # FPR relative to total negatives (none occurrences)
        total_nones = np.sum(y == none_idx)
        fpr = false_positives / total_nones if total_nones > 0 else 0
        
        print(f"{n:<5} | {t:<10.2f} | {acc:.4f}     | {fpr:.4f}")
        
        # Simple heuristic score: High Accuracy, Heavy Penalty for False Positives
        score = acc - (fpr * 1.5) 
        if score > best_score:
            best_score = score
            best_n = n
            best_t = t
            
        results_list.append({
            "N": n,
            "threshold": float(t),
            "accuracy": float(acc),
            "fpr": float(fpr),
            "score": float(score)
        })
            
    print("-" * 55)
    print(f"[SUCCESS] Optimal values found: N = {best_n}, Threshold = {best_t} (Score: {best_score:.4f})")
    print(" -> Recommendation: Update realtime_demo.py with these values for best smoothness.")
    
    # Save the report for auto-run inclusion
    report_data = {
         "optimal_N": best_n,
         "optimal_threshold": float(best_t),
         "best_score": float(best_score),
         "raw_results": results_list
    }
    
    out_dir = os.path.join(config.REPORTS_DIR, "EfficientNetV2B0", "FULL")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "temporal_smoothing_optimization.json")
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[SUCCESS] Saved temporal smoothing report to {out_path}")
    
if __name__ == "__main__":
    main()
