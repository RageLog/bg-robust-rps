import os
import json
import kagglehub
import shutil
import random
import numpy as np
import zipfile
from PIL import Image
import urllib.request
import sys

# Append parent dir so config can be loaded
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import config
except ImportError:
    class DummyConfig:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DRIVE_DIR = "/content/drive/MyDrive/RPC_Colab"
    config = DummyConfig()

MAX_IMAGES_PER_CLASS = 150

CLASS_MAP = {
    "fist": "rock",
    "palm": "paper",
    "peace": "scissors",
    "two": "scissors"
    # no_gesture is hard to crop since it has no bounding box, we will generate synthetic none images or skip
}

def crop_hagrid_dataset():
    print("[INFO] Starting HaGRID Annotation-Based Bounding Box Cropping...")
    target_dir = os.path.join(config.BASE_DIR, "datasets", "external_test")
    drive_zip = os.path.join(config.DRIVE_DIR, "external_test_cropped.zip")
    
    # Check if already cached in Drive
    if os.path.exists(drive_zip):
        print(f"[INFO] Found previously cropped cache in Drive: {drive_zip}")
        print("Restoring...")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(drive_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("[SUCCESS] Restored from Drive cache.")
        return True
        
    print("[INFO] Downloading/Locating Kaggle Dataset: innominate817/hagrid-sample-30k-384p ...")
    try:
        p = kagglehub.dataset_download('innominate817/hagrid-sample-30k-384p')
    except Exception as e:
        print(f"[ERROR] Failed to download from Kaggle: {e}")
        return False
        
    hagrid_root = os.path.join(p, 'hagrid-sample-30k-384p')
    if not os.path.exists(hagrid_root):
        hagrid_root = p # Fallback if structure is flat
        
    ann_dir = os.path.join(hagrid_root, 'ann_train_val')
    img_dir_root = os.path.join(hagrid_root, 'hagrid_30k')
    
    if not os.path.exists(ann_dir) or not os.path.exists(img_dir_root):
        print(f"[ERROR] Expected folder structure not found in {hagrid_root}")
        return False
        
    # Prepare target dirs
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    for cls in ["rock", "paper", "scissors", "none"]:
        os.makedirs(os.path.join(target_dir, cls), exist_ok=True)
        
    # Process positive classes
    for hagrid_cls in ["fist", "palm", "peace", "two"]:
        json_path = os.path.join(ann_dir, f"{hagrid_cls}.json")
        if not os.path.exists(json_path):
            continue
            
        target_cls = CLASS_MAP.get(hagrid_cls)
        if not target_cls: continue
        
        print(f"[INFO] Processing annotations for '{hagrid_cls}' -> '{target_cls}'...")
        
        with open(json_path, 'r') as f:
            ann_data = json.load(f)
            
        # Keys are image IDs (without .jpg). Values have 'bboxes', 'labels'
        img_ids = list(ann_data.keys())
        random.shuffle(img_ids) # Randomize to get a good spread
        
        extracted_count = 0
        for img_id in img_ids:
            if extracted_count >= MAX_IMAGES_PER_CLASS:
                break
                
            entry = ann_data[img_id]
            bboxes = entry.get('bboxes', [])
            if not bboxes: continue
            
            # Find the actual image file globally in img_dir_root
            # because the image might be stored in 'train_val_fist' or similar subfolders
            found_img_path = None
            for root, dirs, files in os.walk(img_dir_root):
                if f"{img_id}.jpg" in files:
                    found_img_path = os.path.join(root, f"{img_id}.jpg")
                    break
                if f"{img_id}.jpeg" in files:
                    found_img_path = os.path.join(root, f"{img_id}.jpeg")
                    break
                    
            if not found_img_path:
                continue
                
            try:
                # Open image
                with Image.open(found_img_path) as img:
                    img_w, img_h = img.size
                    # Process the first bounding box
                    bbox = bboxes[0] # [x_min, y_min, width, height] relative
                    rel_x, rel_y, rel_w, rel_h = bbox
                    
                    # Absolute coordinates
                    left = int(rel_x * img_w)
                    top = int(rel_y * img_h)
                    right = int((rel_x + rel_w) * img_w)
                    bottom = int((rel_y + rel_h) * img_h)
                    
                    # Add padding (10% around the hand for context)
                    pad_w = int(0.10 * (right - left))
                    pad_h = int(0.10 * (bottom - top))
                    
                    left = max(0, left - pad_w)
                    top = max(0, top - pad_h)
                    right = min(img_w, right + pad_w)
                    bottom = min(img_h, bottom + pad_h)
                    
                    # Crop
                    crop_img = img.crop((left, top, right, bottom))
                    
                    # --- STANDARDIZATION & PREPROCESSING ---
                    # 1. Make square (Pad shorter side with black)
                    target_w, target_h = config.IMG_SIZE # usually (224, 224)
                    cw, ch = crop_img.size
                    max_dim = max(cw, ch)
                    square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0)) # Black padding
                    paste_x = (max_dim - cw) // 2
                    paste_y = (max_dim - ch) // 2
                    square_img.paste(crop_img, (paste_x, paste_y))
                    
                    # 2. Resize to exact config dimensions
                    final_img = square_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    
                    # Save
                    save_path = os.path.join(target_dir, target_cls, f"{hagrid_cls}_{img_id}.jpg")
                    final_img.save(save_path)
                    extracted_count += 1
            except Exception as e:
                # Corrupted image or read error
                continue
                
        print(f" -> Extracted {extracted_count} cropped images for {target_cls}.")

    # Finally, generate a "none" class
    print(f"[INFO] Generating 'none' class images (Solid Colors/Noise fallback)...")
    none_dir = os.path.join(target_dir, "none")
    for i in range(MAX_IMAGES_PER_CLASS):
        if i % 2 == 0:
            c = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        else:
            color = np.random.randint(0, 256, 3).tolist()
            c = np.full((224, 224, 3), color, dtype=np.uint8)
        Image.fromarray(c).save(os.path.join(none_dir, f"noise_{i}.jpg"))
        
    print("[SUCCESS] HaGRID Cropping Completed.")
    
    # Backup to Drive
    if os.path.exists(os.path.dirname(drive_zip)):
        print(f"[INFO] Backing up Cropped Dataset to Drive: {drive_zip}...")
        # Zip target_dir into drive_zip
        # Because we want the contents of target_dir to be at the root of the zip
        shutil.make_archive(drive_zip.replace('.zip', ''), 'zip', target_dir)
        print("[SUCCESS] Backup completed!")
        
    return True

if __name__ == "__main__":
    crop_hagrid_dataset()
