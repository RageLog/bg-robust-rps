"""
data_loader.py - Simplified Data Loader
==========================================
Loads synthetic data. Minimal augmentation required.
"""
import os
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def load_datasets(ablation_mode="FULL"):
    print(f"\n[INFO] Preparing data pipeline... Ablation Mode: {ablation_mode}")
    # Set target directory
    target_dir = os.path.join(config.BASE_DIR, 'datasets', f"synthetic_{ablation_mode.lower()}")
    
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError("Data folders not found. Please run 'split_data.py' first.")

    # --- TRAIN ---
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=config.CLASS_NAMES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    # Ablation Logic
    if ablation_mode in ["REMBG_ONLY", "BASELINE"]:
        print(f"    [INFO] Augmentation DISABLED (Ablation: {ablation_mode})")
        # Normalize only
        train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                                num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Standard Augmentation
        print("    [INFO] Augmentation ENABLED")
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.1),
        ])
        
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        # Normalize (0-255 -> 0-1)
        train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                                num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

    # --- VAL & TEST ---
    def process_val_test(directory):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            class_names=config.CLASS_NAMES,
            image_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    val_ds = process_val_test(val_dir)
    test_ds = process_val_test(test_dir)
    
    print("    [SUCCESS] Datasets are ready.")
    return train_ds, val_ds, test_ds