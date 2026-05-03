import os
import sys
import shutil
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import config
except ImportError:
    config = None

def get_kaggle_hagrid_subset(target_dir, max_images_per_class=150):
    """
    Downloads the kaggle 'innominate817/hagrid-sample-30k-384p' dataset.
    This contains a perfect subset of HaGRID (384p).
    """
    try:
        import kagglehub
    except ImportError:
        print("[INFO] 'kagglehub' not found. Installing it automatically...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    print(f"\n[INFO] Downloading/Locating Kaggle Dataset: innominate817/hagrid-sample-30k-384p ...")
    try:
        # Download latest version using KaggleHub
        dataset_path = kagglehub.dataset_download("innominate817/hagrid-sample-30k-384p")
        print(f"[SUCCESS] Dataset cached at: {dataset_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download Kaggle dataset. Please ensure internet access. Error: {e}")
        return False

    # The kaggle dataset contains images structured in a folder (often 'hagrid_30k' or similar)
    # inside the downloaded path. It contains classes like 'fist', 'palm', 'peace', 'no_gesture', etc.
    
    print(f"  -> Organizing and Mapping HaGRID classes from Kaggle to target directory...")
    
    for tgt in ["rock", "paper", "scissors", "none"]:
        os.makedirs(os.path.join(target_dir, tgt), exist_ok=True)
        
    found_any = False
    
    # Recursively search the downloaded Kaggle path for the target folders
    for root, dirs, files in os.walk(dataset_path):
        for d in dirs:
            dl = d.lower()
            
            target_class = None
            if "fist" in dl:
                target_class = "rock"
            elif "palm" in dl:
                target_class = "paper"
            elif "peace" in dl or "two" in dl:
                target_class = "scissors"
            elif "no_gesture" in dl or "none" in dl:
                target_class = "none"
                
            if target_class:
                src_cls = os.path.join(root, d)
                dst_cls = os.path.join(target_dir, target_class)
                
                # Gather images
                images = [f for f in os.listdir(src_cls) if os.path.isfile(os.path.join(src_cls, f))]
                
                # We want real-world robust testing, let's pick random images
                if len(images) > max_images_per_class:
                    images = random.sample(images, max_images_per_class)
                    
                for item in images:
                    s = os.path.join(src_cls, item)
                    d_item = os.path.join(dst_cls, item)
                    shutil.copy2(s, d_item)
                    found_any = True
                    
    # Validation
    if not found_any:
        print(f"[ERROR] Could not find the expected gesture folders (fist, palm, peace, no_gesture) inside {dataset_path}")
        return False
        
    # Check if 'none' class is empty (unlikely with this specific dataset, but just in case)
    none_dir = os.path.join(target_dir, "none")
    if len(os.listdir(none_dir)) == 0:
        print("[WARN] Dataset did not contain 'no_gesture'. Generating fallback empty images...")
        import numpy as np
        from PIL import Image
        for i in range(20):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            img = Image.new('RGB', (224, 224), color=color)
            img.save(os.path.join(none_dir, f"fallback_none_{i:03d}.jpg"))

    print(f"[SUCCESS] Curated authentic HaGRID mapping completed into {target_dir}!")
    return True

def download_and_prepare(target_dir):
    """
    Main entry point for dataset preparation.
    """
    # 1. Check local preparation
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) >= 4:
        is_ready = True
        for c in ["rock", "paper", "scissors", "none"]:
            d = os.path.join(target_dir, c)
            if not os.path.exists(d) or len(os.listdir(d)) == 0:
                is_ready = False
                break
        if is_ready:
            return True

    drive_dir = getattr(config, 'DRIVE_DIR', '/content/drive/MyDrive/RPC_Colab') if config else '/content/drive/MyDrive/RPC_Colab'
    drive_external_dir = os.path.join(drive_dir, "external_test")

    # 2. Check Google Drive backup
    if os.path.exists(drive_external_dir) and len(os.listdir(drive_external_dir)) >= 4:
        is_ready = True
        for c in ["rock", "paper", "scissors", "none"]:
            d = os.path.join(drive_external_dir, c)
            if not os.path.exists(d) or len(os.listdir(d)) == 0:
                is_ready = False
                break
        if is_ready:
            print(f"[INFO] Found prepared HaGRID subset in Google Drive. Copying to local {target_dir}...")
            shutil.copytree(drive_external_dir, target_dir, dirs_exist_ok=True)
            return True

    # 3. If not ready, construct the dataset from Kaggle
    # Using 150 images per class which is large enough for statistical significance (600 total images)
    # yet small enough to evaluate in < 30 seconds during pipeline.
    success = get_kaggle_hagrid_subset(target_dir, max_images_per_class=150) 
    
    # 4. Backup
    if success and os.path.exists(drive_dir):
        print(f"[INFO] Backing up curated HaGRID subset to Google Drive: {drive_external_dir}...")
        if os.path.exists(drive_external_dir):
            shutil.rmtree(drive_external_dir)
        shutil.copytree(target_dir, drive_external_dir, dirs_exist_ok=True)
        print(f"[SUCCESS] Backup completed!")
        
    return success

if __name__ == "__main__":
    download_and_prepare(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "external_test"))
