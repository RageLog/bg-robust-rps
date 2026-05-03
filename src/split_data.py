"""
split_data.py - Synthetic Data Splitter
========================================
Distributes synthetic data into train/val/test folders.
"""
import os
import shutil
import random
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    print(f"[INFO] STARTING DATA SPLIT...")
    print(f"[INFO] Source: {config.SYNTHETIC_DIR}")
    
    # Check if already split
    if all(os.path.exists(os.path.join(config.SYNTHETIC_DIR, s)) for s in ['train', 'val', 'test']):
        print("[INFO] Data appears to be already split into train/val/test folders. Skipping split.")
        return
        
    # Check classes
    classes = [d for d in os.listdir(config.SYNTHETIC_DIR) 
               if os.path.isdir(os.path.join(config.SYNTHETIC_DIR, d)) 
               and d in config.CLASS_NAMES]
    
    if not classes:
        print("[ERROR] Synthetic data folders not found!")
        print("[INFO] You must run 'python src/generate_data.py' first.")
        return

    files_map = {}
    for cls in classes:
        cls_dir = os.path.join(config.SYNTHETIC_DIR, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) 
                 if f.lower().endswith(('.jpg', '.png'))]
        files_map[cls] = files
        print(f"[INFO] {cls.upper()}: {len(files)} images found.")

    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(config.SYNTHETIC_DIR, split, cls), exist_ok=True)

    print("\n[INFO] Moving files (Subject-Level Split)...")
    for cls, files in files_map.items():
        # Group files by base_name to prevent data leakage
        groups = {}
        for f in files:
            file_name = os.path.basename(f)
            try:
                # Format is usually: {class}_{base_name}_syn{XX}.jpg
                base_name = file_name.split('_', 1)[1].rsplit('_syn', 1)[0]
            except IndexError:
                base_name = file_name # Fallback
                
            if base_name not in groups:
                 groups[base_name] = []
            groups[base_name].append(f)
            
        group_keys = list(groups.keys())
        random.shuffle(group_keys)
        total_groups = len(group_keys)
        
        # Ratios based on groups, not individual files
        n_train_g = int(total_groups * 0.70)
        n_val_g = int(total_groups * 0.15)
        
        train_groups = group_keys[:n_train_g]
        val_groups = group_keys[n_train_g:n_train_g+n_val_g]
        test_groups = group_keys[n_train_g+n_val_g:]
        
        splits = {
            'train': [f for g in train_groups for f in groups[g]],
            'val': [f for g in val_groups for f in groups[g]],
            'test': [f for g in test_groups for f in groups[g]]
        }
        
        for split_name, split_files in splits.items():
            for src_path in tqdm(split_files, desc=f"{cls} -> {split_name}", leave=False):
                file_name = os.path.basename(src_path)
                dst_path = os.path.join(config.SYNTHETIC_DIR, split_name, cls, file_name)
                
                shutil.move(src_path, dst_path)
        
        try:
            os.rmdir(os.path.join(config.SYNTHETIC_DIR, cls))
        except:
            pass 

    print("\n[SUCCESS] SPLIT COMPLETED!")
    print(f"[INFO] Data has been moved to train/val/test folders under {config.SYNTHETIC_DIR}.")

if __name__ == "__main__":
    import argparse
    import config
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default='FULL')
    args = parser.parse_args()
    
    # Update SYNTHETIC_DIR based on ablation parameter
    config.SYNTHETIC_DIR = os.path.join(config.BASE_DIR, 'datasets', f'synthetic_{args.ablation.lower()}')
    
    random.seed(config.RANDOM_SEED)
    main()
