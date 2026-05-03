"""
generate_data.py - Synthetic Data Generator
==========================================
This script:
1. Reads hand images from 'datasets/raw'.
2. Removes the background using 'rembg' (makes it transparent).
3. Selects a random background from 'datasets/backgrounds'.
4. Composites the hand onto the new background.
5. Saves the results to the 'datasets/synthetic' directory.
"""

import os
import sys
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw
from rembg import remove, new_session
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_image_files(directory):
    """Lists all image files in a directory."""
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(valid_exts):
                files.append(os.path.join(root, filename))
    return files

def process_single_image(args):
    """
    Generates synthetic data for a single image.
    Args: (img_path, bg_files, output_dir, class_name, session)
    """
    img_path, bg_files, output_dir, class_name, session = args
    
    try:
        # 1. Load original image
        original_img = Image.open(img_path).convert("RGBA")
        
        # 2. Remove background (rembg) - Keep only the hand
        # Alpha matting softens edges, but is disabled in NO_ALPHA ablation
        use_alpha_matting = config.ALPHA_MATTING and ("NO_ALPHA" not in config.SYNTHETIC_DIR)
        no_bg_img = remove(original_img, session=session, alpha_matting=use_alpha_matting)
        
        # Resize image to config size (fit while maintaining aspect ratio)
        no_bg_img.thumbnail(config.IMG_SIZE, Image.Resampling.LANCZOS)
        
        # 3. Augmentation Loop
        # Try AUGMENTATION_FACTOR different backgrounds for each hand image
        for i in range(config.AUGMENTATION_FACTOR):
            # Select a random background
            bg_path = random.choice(bg_files)
            bg_img = Image.open(bg_path).convert("RGBA")
            
            # Resize background to target size (Crop and fill)
            bg_img = bg_img.resize(config.IMG_SIZE, Image.Resampling.LANCZOS) # Simple resize
            
            # 4. Compositing
            # Place the hand in the center of the background (or slightly random position)
            
            # Slight random positioning (Shift)
            if "NO_SHIFT" in config.SYNTHETIC_DIR:
                shift_x, shift_y = 0, 0
            else:
                max_shift_x = int(config.IMG_SIZE[0] * 0.1)
                max_shift_y = int(config.IMG_SIZE[1] * 0.1)
                shift_x = random.randint(-max_shift_x, max_shift_x)
                shift_y = random.randint(-max_shift_y, max_shift_y)
            
            # Calculate coordinates to center the hand image
            bg_w, bg_h = bg_img.size
            fg_w, fg_h = no_bg_img.size
            x = (bg_w - fg_w) // 2 + shift_x
            y = (bg_h - fg_h) // 2 + shift_y
            
            # Paste
            comp_img = bg_img.copy()
            comp_img.paste(no_bg_img, (x, y), no_bg_img) # 3rd parameter used as mask
            
            # 5. Save
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_name = f"{class_name}_{base_name}_syn{i:02d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            
            # Convert to RGB and save as JPG
            comp_img.convert("RGB").save(save_path, quality=95)
            
    except Exception as e:
        # Skip corrupted images
        # print(f"Error: {img_path} - {e}")
        pass

def main():
    print(f"[INFO] STARTING SYNTHETIC DATA GENERATION: {config.RUN_NAME}")
    print(f"[INFO] Source: {config.RAW_DATA_DIR}")
    print(f"[INFO] Backgrounds: {config.BACKGROUND_DIR}")
    print(f"[INFO] Target: {config.SYNTHETIC_DIR}")
    
    # Evaluate RATIO_1X ablation
    if "RATIO_1X" in config.SYNTHETIC_DIR:
        config.AUGMENTATION_FACTOR = 1
        print("[INFO] ABLATION TRIGGERED: AUGMENTATION_FACTOR forced to 1x")
        
    print(f"[INFO] Effective Augmentation Factor: {config.AUGMENTATION_FACTOR}x")
    
    # 1. Preparation
    if not os.path.exists(config.BACKGROUND_DIR):
        print("[ERROR] Backgrounds folder not found!")
        return

    # Build background pool
    all_bg_files = get_image_files(config.BACKGROUND_DIR)
    if len(all_bg_files) == 0:
        print("[ERROR] No background images found!")
        return
    print(f"[INFO] Loaded {len(all_bg_files)} backgrounds.")
    
    # rembg session
    # Force GPU usage
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = new_session("u2net", providers=providers) 
    except Exception as e:
        print(f"[WARN] Failed to initialize GPU provider, falling back to default. {e}")
        session = new_session("u2net")
    
    # Clear/create folders
    if os.path.exists(config.SYNTHETIC_DIR):
        import shutil
        shutil.rmtree(config.SYNTHETIC_DIR)
    
    # We will do Train/Val/Test split later, for now put everything in class folders
    # However, we will do special processing for the "None" class.
    
    # 2. Process Gesture Classes (Rock, Paper, Scissors)
    for class_name in ['rock', 'paper', 'scissors']:
        print(f"\n[INFO] Processing: {class_name.upper()}")
        
        # Target folder
        output_dir = os.path.join(config.SYNTHETIC_DIR, class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Source images
        raw_class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if not os.path.exists(raw_class_dir):
            print(f"[WARN] {class_name} folder not found, skipping.")
            continue
            
        img_files = get_image_files(raw_class_dir)
        print(f"[INFO] Found {len(img_files)} source images.")
        print(f"[INFO] Estimated output: {len(img_files) * config.AUGMENTATION_FACTOR} images.")
        
        # Prepare task list
        tasks = []
        for img_path in img_files:
            tasks.append((img_path, all_bg_files, output_dir, class_name, session))
        
        # Parallel Processing
        print(f"[INFO] Multi-threading active. Maximizing GPU utilization...")
        max_workers = min(32, os.cpu_count() + 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc=f"Generating {class_name}"))

    # 3. Process None (Background) Class
    print(f"\n[INFO] Processing: NONE (Pure Backgrounds)")
    none_output_dir = os.path.join(config.SYNTHETIC_DIR, 'none')
    os.makedirs(none_output_dir, exist_ok=True)
    
    # How much data did we generate in other classes? (Average)
    # We will take about that many backgrounds to prevent imbalance.
    
    total_gesture_imgs = 0
    gesture_classes = 0
    for cls in ['rock', 'paper', 'scissors']:
        d = os.path.join(config.SYNTHETIC_DIR, cls)
        if os.path.exists(d):
            count = len(os.listdir(d))
            if count > 0:
                total_gesture_imgs += count
                gesture_classes += 1
    
    if gesture_classes > 0:
        target_none_count = int(total_gesture_imgs / gesture_classes)
    else:
        target_none_count = 1000 # Fallback
        
    print(f"[INFO] Target NONE count: {target_none_count} (For balancing)")
    
    # Process raw/none if it exists FIRST
    raw_none_dir = os.path.join(config.RAW_DATA_DIR, 'none')
    none_generated = 0
    if os.path.exists(raw_none_dir):
        none_img_files = get_image_files(raw_none_dir)
        if none_img_files:
            print(f"[INFO] Found {len(none_img_files)} source images for NONE class. Augmenting negative examples...")
            none_tasks = []
            for img_path in none_img_files:
                none_tasks.append((img_path, all_bg_files, none_output_dir, 'none', session))
                
            max_workers = min(32, os.cpu_count() + 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(process_single_image, none_tasks), total=len(none_tasks), desc="Generating None (from raw negative examples)"))
            none_generated = len(none_img_files) * config.AUGMENTATION_FACTOR

    remaining_none = max(0, target_none_count - none_generated)
    if remaining_none > 0:
        print(f"[INFO] Adding target {remaining_none} backgrounds to NONE class (50% Pure, 50% Hard Negatives)...")
        selected_bgs = random.sample(all_bg_files, min(len(all_bg_files), remaining_none))
        
        for idx, bg_path in enumerate(tqdm(selected_bgs, desc="Generating None (Background & Hard Negatives)")):
            try:
                bg_img = Image.open(bg_path).convert("RGB")
                bg_img = bg_img.resize(config.IMG_SIZE, Image.Resampling.LANCZOS)
                save_path = os.path.join(none_output_dir, f"none_bg_{idx:05d}.jpg")
                
                # Add a "Hard Negative" (Glove, Skin Colored Spot, etc.) to 1 out of every 2 backgrounds.
                # If "raw/hard_negatives" exists, fetch from there, otherwise draw synthetic skin/distractor.
                is_hard_negative = (idx % 2 == 0)
                if is_hard_negative:
                    hard_neg_dir = os.path.join(config.RAW_DATA_DIR, 'hard_negatives')
                    composited = False
                    
                    if os.path.exists(hard_neg_dir):
                        neg_files = get_image_files(hard_neg_dir)
                        if neg_files:
                            neg_img_path = random.choice(neg_files)
                            neg_img = cv2.imread(neg_img_path)
                            if neg_img is not None:
                                use_alpha_matting = config.ALPHA_MATTING and ("NO_ALPHA" not in config.SYNTHETIC_DIR)
                                no_bg_neg = remove(neg_img, session=session, alpha_matting=use_alpha_matting)
                                
                                fg_pil_neg = Image.fromarray(cv2.cvtColor(no_bg_neg, cv2.COLOR_BGRA2RGBA))
                                scale = random.uniform(0.6, 1.2)
                                new_w, new_h = int(config.IMG_SIZE[0] * scale), int(config.IMG_SIZE[1] * scale)
                                fg_pil_neg = fg_pil_neg.resize((new_w, new_h), Image.Resampling.LANCZOS)
                                
                                shift_x, shift_y = 0, 0
                                if "NO_SHIFT" not in config.SYNTHETIC_DIR:
                                    max_shift_x = int(config.IMG_SIZE[0] * 0.1)
                                    max_shift_y = int(config.IMG_SIZE[1] * 0.1)
                                    shift_x = random.randint(-max_shift_x, max_shift_x)
                                    shift_y = random.randint(-max_shift_y, max_shift_y)
                                    
                                paste_x = (config.IMG_SIZE[0] - new_w) // 2 + shift_x
                                paste_y = (config.IMG_SIZE[1] - new_h) // 2 + shift_y
                                bg_img.paste(fg_pil_neg, (paste_x, paste_y), fg_pil_neg)
                                composited = True

                    if not composited:
                        # Fallback: Draw a synthetic skin-colored blob/ellipse as a distractor
                        draw = ImageDraw.Draw(bg_img)
                        # Typical skin tones (light to dark)
                        skin_colors = [(255, 224, 189), (241, 194, 125), (224, 172, 105), (141, 85, 36), (198, 134, 66)]
                        color = random.choice(skin_colors)
                        
                        bw = config.IMG_SIZE[0]
                        bh = config.IMG_SIZE[1]
                        ex1 = random.randint(int(bw*0.1), int(bw*0.4))
                        ey1 = random.randint(int(bh*0.1), int(bh*0.4))
                        ex2 = ex1 + random.randint(int(bw*0.3), int(bw*0.5))
                        ey2 = ey1 + random.randint(int(bh*0.3), int(bh*0.5))
                        
                        # Add a fake bounding/distractor
                        draw.ellipse([ex1, ey1, ex2, ey2], fill=color)
                        
                bg_img.save(save_path, quality=95)
            except Exception as e:
                pass
    else:
        print(f"[INFO] Negative examples fulfilled the target count. Minimum pure backgrounds added.")

    print("\n[SUCCESS] ALL OPERATIONS COMPLETED!")
    print(f"[INFO] Data generated at: {config.SYNTHETIC_DIR}")

if __name__ == "__main__":
    import argparse
    import config
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default='FULL')
    parser.add_argument('--model', type=str, default='EfficientNetV2B0')
    args = parser.parse_args()
    
    # Update SYNTHETIC_DIR based on ablation parameter
    config.SYNTHETIC_DIR = os.path.join(config.BASE_DIR, 'datasets', f'synthetic_{args.ablation.lower()}')
    
    main()
