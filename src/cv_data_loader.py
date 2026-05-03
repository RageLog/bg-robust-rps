"""
cv_data_loader.py - K-Fold Cross Validation Loader
=====================================================
This module skips split train/val folders,
reads folders under 'datasets/synthetic' and
directly creates Subject-Level/Stratified K-Fold split in memory.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_cv_folds(k_folds=5, ablation_mode="FULL"):
    """
    Returns a generator producing (train_ds, val_ds) pairs for K-Fold.
    """
    print(f"\n Preparing K-Fold CV ({k_folds}-Fold)... Ablation Mode: {ablation_mode}")
    
    all_img_paths = []
    all_labels = []
    
    label_to_index = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
    
    # 1. Find dataset path based on Ablation Mode
    # Ex: FULL -> datasets/synthetic_full
    target_dir = os.path.join(config.BASE_DIR, 'datasets', f"synthetic_{ablation_mode.lower()}")
    
    # Target folders (if split_data has run, inside train/val, else directly in class folders)
    pool_dirs = ['train', 'val']
    
    for pd in pool_dirs:
        base_path = os.path.join(target_dir, pd)
        if not os.path.exists(base_path):
            continue
            
        for cls_name in config.CLASS_NAMES:
            cls_dir = os.path.join(base_path, cls_name)
            if not os.path.exists(cls_dir):
                continue
                
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(('.jpg', '.png')):
                    continue
                    
                full_path = os.path.join(cls_dir, fname)
                all_img_paths.append(full_path)
                all_labels.append(label_to_index[cls_name])
                
    if len(all_img_paths) == 0:
        raise ValueError(f"No data found! Please check '{target_dir}' folder.")
        
    print(f"   Total {len(all_img_paths)} images found. (Classes: {len(np.unique(all_labels))})")

    all_img_paths = np.array(all_img_paths)
    all_labels = np.array(all_labels)
    
    # 1.5. Create group list (base_name)
    groups = []
    for path in all_img_paths:
        file_name = os.path.basename(path)
        try:
            base_name = file_name.split('_', 1)[1].rsplit('_syn', 1)[0]
        except IndexError:
            base_name = file_name
        groups.append(base_name)
    groups = np.array(groups)

    # 2. StratifiedGroupKFold (Preserves class balance and group isolation)
    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=config.RANDOM_SEED)
    
    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, config.IMG_SIZE)
        # One-hot encode label
        label = tf.one_hot(label, config.NUM_CLASSES)
        return image, label

    def create_dataset(paths, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=config.RANDOM_SEED)
            
        ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(config.BATCH_SIZE)
        
        # Augmentation (Only in FULL or specific modes)
        # Keep the hand static in Random background and no background (Rembg) modes to measure ablation effect.
        if training and ablation_mode not in ["REMBG_ONLY", "BASELINE"]:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomBrightness(0.1),
            ])
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
            
        # Normalize
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
                    
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    # Fold generator
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(all_img_paths, all_labels, groups)):
        print(f"\n   [Fold {fold+1}/{k_folds}] Preparing -> Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        train_ds = create_dataset(all_img_paths[train_idx], all_labels[train_idx], training=True)
        val_ds = create_dataset(all_img_paths[val_idx], all_labels[val_idx], training=False)
        
        yield fold + 1, train_ds, val_ds
