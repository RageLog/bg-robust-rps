"""
generate_style_transfer.py - Fast OpenCV Stylization
====================================================
Creates the 'STYLE_TRANSFER' ablation dataset by applying
OpenCV stylization filters to the raw images to simulate
artistic/texture-shifted synthetic data.
"""
import os
import sys
import cv2
import glob
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def apply_style_transfer():
    print("[INFO] Starting OpenCV Fast Stylization...")
    input_dir = config.RAW_DATA_DIR
    output_dir = os.path.join(config.BASE_DIR, "datasets", "synthetic_style_transfer")
    
    if not os.path.exists(input_dir):
        print(f"[ERROR] Raw data directory not found: {input_dir}")
        return
        
    for cls in config.CLASS_NAMES:
        cls_in = os.path.join(input_dir, cls)
        if not os.path.exists(cls_in): continue
        
        cls_out = os.path.join(output_dir, cls)
        os.makedirs(cls_out, exist_ok=True)
        
        files = glob.glob(os.path.join(cls_in, "*.jpg")) + glob.glob(os.path.join(cls_in, "*.png"))
        
        for f in tqdm(files, desc=f"Stylizing {cls}"):
            img = cv2.imread(f)
            if img is None: continue
            
            # Apply OpenCV Stylization
            stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
            
            # Save preserving base name for leakage checks
            out_path = os.path.join(cls_out, os.path.basename(f))
            cv2.imwrite(out_path, stylized)
            
    print(f"[SUCCESS] Style Transfer data generated at: {output_dir}")

if __name__ == "__main__":
    apply_style_transfer()
