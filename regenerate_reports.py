import os
import sys
import argparse
import glob
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

import config
from data_loader import load_datasets
from train import save_training_report
from run_all_experiments import backup_results, prepare_data
from test_cross_dataset import get_preprocessing_function

def extract_metadata_from_filename(filename):
    """Parses standard model filenames to extract required args."""
    # Example format: RPC_EfficientNetV2B0_full_standard_standard_fold_1_best.keras
    # Removing extension and prefix
    name = filename.replace(".keras", "").replace(f"{config.RUN_NAME}_", "")
    parts = name.split("_")
    
    # We expect something like: ['EfficientNetV2B0', 'full', 'standard', 'standard', 'fold', '1', 'best']
    model_name = parts[0]
    ablation = parts[1].upper()
    head_strategy = parts[2]
    tune_strategy = parts[3]
    
    fold = None
    if "fold" in parts:
        idx = parts.index("fold")
        fold = int(parts[idx+1])
        
    return model_name, ablation, head_strategy, tune_strategy, fold

def regenerate_all_reports():
    print("[INFO] Searching for trained models in Local & Drive...")
    local_models = glob.glob(os.path.join(BASE_DIR, "models", "*.keras"))
    drive_models = glob.glob(os.path.join(config.DRIVE_DIR, "models", "*.keras")) if os.path.exists(config.DRIVE_DIR) else []
    
    model_files = list(set(local_models + drive_models))
    
    if not model_files:
        print("[WARN] No models found in Local or Drive. Cannot regenerate reports.")
        return
        
    print(f"[INFO] Found {len(model_files)} models. Regenerating missing JSON reports...")
    
    # Cache loaded datasets to save time across folds
    loaded_datasets = {}
    
    for mf in model_files:
        filename = os.path.basename(mf)
        try:
            model_name, ablation, head_strategy, tune_strategy, fold = extract_metadata_from_filename(filename)
        except Exception as e:
            print(f"[WARN] Failed to parse {filename}: {e}")
            continue
            
        print(f"\n[EXEC] Processing {filename}...")
        
        # Check if report already exists so we don't duplicate effort
        report_dir = os.path.join(config.REPORTS_DIR, model_name, ablation)
        if fold:
             expected_json = f"report_{model_name}_{ablation.lower()}_{head_strategy}_{tune_strategy}_fold_{fold}.json"
        else:
             expected_json = f"report_{model_name}_{ablation.lower()}_{head_strategy}_{tune_strategy}_full.json"
             
        expected_json_path = os.path.join(report_dir, expected_json)
             
        if not os.path.exists(expected_json_path) and os.path.exists(config.DRIVE_DIR):
            drive_report_path = os.path.join(config.DRIVE_DIR, "reports", model_name, ablation, expected_json)
            if os.path.exists(drive_report_path):
                os.makedirs(report_dir, exist_ok=True)
                import shutil
                shutil.copy2(drive_report_path, expected_json_path)
                
        if os.path.exists(expected_json_path):
            print(f"  -> Report already exists locally or restored from Drive. Skipping regeneration.")
            continue
            
        # 1. Load Data (use cache if possible)
        if ablation not in loaded_datasets:
            print(f"  -> Ensuring dataset for {ablation} is extracted from Drive...")
            try:
                prepare_data(ablation)
            except Exception as e:
                print(f"  -> [ERROR] Failed to prepare data for {ablation}: {e}")
                continue
                
            print(f"  -> Loading dataset for {ablation}...")
            _, val_ds, _ = load_datasets(ablation_mode=ablation)
            loaded_datasets[ablation] = val_ds
            
        val_ds = loaded_datasets[ablation]
        
        # 2. Load Model 
        print(f"  -> Loading weights...")
        
        preprocessor = get_preprocessing_function(model_name)
        import keras
        @keras.saving.register_keras_serializable(package="builtins", name="preprocess_input")
        def _dummy_preprocess(*args, **kwargs):
            return preprocessor(*args, **kwargs)
            
        try:
            # compile=False since we only need inference evaluating
            model = keras.models.load_model(mf, compile=False, custom_objects={'preprocess_input': _dummy_preprocess}, safe_mode=False) 
        except Exception as e:
            print(f"  -> [ERROR] Failed to load model weights: {e}")
            continue
            
        # 3. Generate Evaluation
        print(f"  -> Evaluating and Saving Report...")
        try:
            save_training_report(
                model=model, 
                dataset=val_ds, 
                model_name=model_name, 
                ablation_mode=ablation, 
                history_obj=None, 
                fold=fold, 
                head_strategy=head_strategy, 
                tune_strategy=tune_strategy
            )
        except Exception as e:
            print(f"  -> [ERROR] Evaluation failed: {e}")

if __name__ == "__main__":
    import tensorflow as tf
    # Limit GPU usage for quick inference
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    regenerate_all_reports()
    
    # Also attempt to run cross-dataset eval since reports were missed
    print("\n[INFO] Starting Cross-Dataset evaluations...")
    import subprocess
    subprocess.run("python src/evaluate_cross_dataset.py", shell=True)
    
    # After recovering any missing json reports, ALSO recover the missing SVG/PDF Vector Graphics
    print("\n[INFO] Forcing generation of SVG/PDF Vector Graphics for all existing JSON reports...")
    try:
        sys.path.append(os.path.join(BASE_DIR, "src"))
        import plot_vector_graphics
        plot_vector_graphics.main()
    except Exception as e:
        print(f"[ERROR] Failed to regenerate SVG/PDF Vector Graphics: {e}")
    
    print("\n[INFO] Saving all models and reports to Google Drive...")
    backup_results()
    
    print("\n[SUCCESS] Reporting and Backup Recovery Complete.")
