"""
generate_gan_mock.py - GAN Data Placeholder
===========================================
Creates the 'synthetic_gan' folder structure and populates it
with heavily augmented raw images to act as a placeholder for GAN data.
This ensures the NxM pipeline doesn't crash if real GAN data isn't provided.
"""
import os
import sys
import cv2
import glob
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def apply_gan_mock():
    print("[INFO] Creating mock GAN data (Heavily augmented placeholder)...")
    input_dir = config.RAW_DATA_DIR
    output_dir = os.path.join(config.BASE_DIR, "datasets", "synthetic_gan")
    
    if not os.path.exists(input_dir):
        print(f"[ERROR] Raw data directory not found: {input_dir}")
        return
        
    for cls in config.CLASS_NAMES:
        cls_in = os.path.join(input_dir, cls)
        if not os.path.exists(cls_in): continue
        
        cls_out = os.path.join(output_dir, cls)
        os.makedirs(cls_out, exist_ok=True)
        
        files = glob.glob(os.path.join(cls_in, "*.jpg")) + glob.glob(os.path.join(cls_in, "*.png"))
        
        for f in tqdm(files, desc=f"Mocking GAN data {cls}"):
            img = cv2.imread(f)
            if img is None: continue
            
            # Simulate GAN generation artifacts: add noise and slight blur
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            img_noisy = cv2.add(img, noise)
            img_gan_mock = cv2.GaussianBlur(img_noisy, (5, 5), 0)
            
            # Keep base names to prevent data leakage issues in splitting
            out_path = os.path.join(cls_out, os.path.basename(f))
            cv2.imwrite(out_path, img_gan_mock)
            
    print(f"[SUCCESS] GAN Mock data generated at: {output_dir}")
    print("[NOTE] Replace this directory with actual GAN Models' Output in the future.")

if __name__ == "__main__":
    apply_gan_mock()
