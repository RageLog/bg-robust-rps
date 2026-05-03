import os
import shutil
import subprocess
import time
import argparse
import concurrent.futures
import threading

backup_lock = threading.Lock()
_uploaded_cache = {} # Cache to prevent redundant Drive I/O checks

# --- CONFIGURATION ---
DRIVE_DIR = "/content/drive/MyDrive/RPC_Colab"
BASE_DIR = "/content/rpc_project/005_dev_code"

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))
import config

# All Ablations and Models for N x M Grid Search
ABLATIONS = ["NO_SYNTH", "REMBG_ONLY", "RANDBG", "INDOOR", "FULL", "STYLE_TRANSFER", "GAN", "RATIO_1X", "NO_SHIFT", "NO_ALPHA"]
MODELS = ["EfficientNetV2B0", "ResNet50", "MobileNetV3Small", "DenseNet121", "VGG16"]

CV_FOLDS = 5

def run_cmd(cmd, log_file=None):
    """Runs a shell command. If log_file is given, captures output to file instead of stdout."""
    if log_file:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"[EXEC] Running: {cmd}\n")
            lf.write("=" * 60 + "\n")
            result = subprocess.run(cmd, shell=True, stdout=lf, stderr=lf)
            lf.write("\n" + "=" * 60 + "\n")
            if result.returncode != 0:
                lf.write(f"[ERROR] Command failed (exit code {result.returncode}): {cmd}\n")
            else:
                lf.write(f"[OK] Command completed successfully.\n")
        return result.returncode == 0
    else:
        print(f"\n[EXEC] Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"[ERROR] Command failed: {cmd}")
        return result.returncode == 0

def _run_cmd_with_log(args):
    """Wrapper: runs a command with its designated log file path."""
    cmd, log_path = args
    return run_cmd(cmd, log_file=log_path), cmd, log_path

def run_tasks_in_parallel(commands, max_workers=2):
    """Runs a list of shell commands in parallel. Each task writes to its own log file.
       After all tasks complete, logs are printed sequentially — no interleaving."""
    if not commands:
        return
    
    log_dir = os.path.join(BASE_DIR, "logs", "parallel_runs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Assign a unique log file to each command
    tasks = []
    for i, cmd in enumerate(commands):
        # Create a meaningful name from the command instead of just task_000
        safe_name = cmd.replace("python ", "").replace("src/", "").replace(".py", "").replace(" --", "_").replace(" ", "_").replace("/", "_")
        if len(safe_name) > 60:
            safe_name = safe_name[:60]
        log_path = os.path.join(log_dir, f"log_{i:03d}_{safe_name}.txt")
        tasks.append((cmd, log_path))
    
    total = len(tasks)
    print(f"\n[INFO] Starting {total} tasks in parallel (max_workers={max_workers})...")
    print(f"[INFO] Individual logs will be saved to: {log_dir}")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_cmd_with_log, t): t for t in tasks}
        done_count = 0
        for future in concurrent.futures.as_completed(future_map):
            done_count += 1
            try:
                success, cmd, log_path = future.result()
                status = "✅" if success else "❌"
                results.append((success, cmd, log_path))
                print(f"  [{done_count}/{total}] {status} {cmd}")
                
                # Backup immediately after each model finishes
                backup_results()
            except Exception as e:
                _, (cmd, log_path) = future_map[future], future_map[future]
                results.append((False, cmd, log_path))
                print(f"  [{done_count}/{total}] ❌ EXCEPTION: {e}")
    
    # Print logs sequentially, in original submission order
    print(f"\n{'=' * 70}")
    print(f"[INFO] ALL {total} PARALLEL TASKS COMPLETED. Printing logs in order:")
    print(f"{'=' * 70}")
    for success, cmd, log_path in sorted(results, key=lambda x: x[2]):
        status_tag = "SUCCESS" if success else "FAILED"
        print(f"\n{'─' * 70}")
        print(f"[{status_tag}] {cmd}")
        print(f"{'─' * 70}")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                print(lf.read())
    print(f"{'=' * 70}\n")

def extract_zip(zip_path, extract_to):
    import zipfile
    print(f"[INFO] Extracting: {os.path.basename(zip_path)} -> {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def prepare_data(ablation_mode):
    """Fetches or generates data for the specified ablation mode."""
    synthetic_dir = os.path.join(BASE_DIR, "datasets", f"synthetic_{ablation_mode.lower()}")
    zip_name = f"synthetic_{ablation_mode.lower()}.zip"
    drive_zip_path = os.path.join(DRIVE_DIR, zip_name)

    if not os.path.exists(synthetic_dir):
        if os.path.exists(drive_zip_path):
            print(f"[INFO] Backup {zip_name} found in Drive. Restoring...")
            extract_zip(drive_zip_path, synthetic_dir)
        else:
            if ablation_mode == "NO_SYNTH":
                print(f"[INFO] Preparing non-synthetic raw data...")
                raw_dir = os.path.join(BASE_DIR, "datasets", "raw")
                if os.path.exists(raw_dir):
                    shutil.copytree(raw_dir, synthetic_dir)
                    os.makedirs(os.path.join(synthetic_dir, "none"), exist_ok=True) # Keras expects 4 classes
                    success = True
                else:
                    print(f"[ERROR] {raw_dir} not found!")
                    success = False
            else:
                print(f"[INFO] Generating synthetic data... Mode: {ablation_mode}")
                success = run_cmd(f"python src/generate_data.py --ablation {ablation_mode} --model {MODELS[0]}")
                
            if success:
                print(f"[INFO] Backing up to Drive to prevent data loss...")
                shutil.make_archive(synthetic_dir, 'zip', synthetic_dir)
                shutil.copy2(f"{synthetic_dir}.zip", drive_zip_path)
                print(f"[SUCCESS] Backup completed: {drive_zip_path}")
    else:
        print(f"[INFO] Synthetic data '{synthetic_dir}' already exists.")

    # Split data into train/val/test
    print(f"[INFO] Splitting dataset... Mode: {ablation_mode}")
    run_cmd(f"python src/split_data.py --ablation {ablation_mode}")

def pre_flight_checks():
    """Runs fast environment and dependency checks before starting the multi-hour pipeline."""
    print("\n[INFO] Running Pre-Flight Checks...")
    
    # 1. Check Google Drive Mount (Only if on Linux/Colab)
    if not sys.platform.startswith('win'):
        if not os.path.exists(DRIVE_DIR):
            print(f"[ERROR] Google Drive is not mounted at {DRIVE_DIR}!")
            print(" -> Please ensure Google Drive is mounted before running the pipeline.")
            return False
            
        # 2. Check Local Working Directory
        if not os.path.exists(BASE_DIR):
            print(f"[ERROR] Local project directory not found at {BASE_DIR}!")
            print(" -> Please clone the repository properly into the local Colab storage.")
            return False
        
    # 3. Check Important Dependencies (Skip heavy imports on local Windows sanity check)
    if not sys.platform.startswith('win'):
        try:
            import mediapipe
            import tensorflow as tf
            import cv2
            import rembg
        except ImportError as e:
            print(f"[ERROR] Missing dependency detected: {e}")
            print(" -> Please ensure all requirements are installed (pip install mediapipe rembg opencv-python tensorflow)")
            return False
            
        # 4. Verify Keras Benchmark Architectures
        try:
            for model in MODELS:
                if not hasattr(tf.keras.applications, model):
                    print(f"[ERROR] Model {model} is missing in tf.keras.applications!")
                    return False
        except Exception as e:
            print(f"[ERROR] TensorFlow verification failed: {e}")
            return False
        
    # 5. Warn about missing external dataset
    external_test_dir = os.path.join(DRIVE_DIR, "external_test")
    if not os.path.exists(external_test_dir):
        external_test_dir = os.path.join(BASE_DIR, "datasets", "external_test")
        if not os.path.exists(external_test_dir):
            print(f"[WARN] External test directory not found: {external_test_dir}")
            print(" -> Stage 7 (Cross-Dataset Evaluation) will fail if this is not provided.")
        
    print("[SUCCESS] All pre-flight checks passed.\n")
    return True

def _do_backup():
    if not os.path.exists(DRIVE_DIR):
        return # Skip if Drive is not mounted
        
    backup_models_dir = os.path.join(DRIVE_DIR, "models")
    backup_reports_dir = os.path.join(DRIVE_DIR, "reports")
    
    os.makedirs(backup_models_dir, exist_ok=True)
    os.makedirs(backup_reports_dir, exist_ok=True)
    
    local_models = os.path.join(BASE_DIR, "models")
    local_reports = os.path.join(BASE_DIR, "reports")
    
    if os.path.exists(local_models):
        for file in os.listdir(local_models):
            if file.endswith(".keras") or file.endswith(".joblib"):
                src = os.path.join(local_models, file)
                dst = os.path.join(backup_models_dir, file)
                
                # Check cache first to avoid slow Drive I/O
                local_mtime = os.path.getmtime(src)
                if _uploaded_cache.get(src) == local_mtime:
                    continue 

                try:
                    if not os.path.exists(dst) or local_mtime > os.path.getmtime(dst):
                        shutil.copy2(src, dst)
                        _uploaded_cache[src] = local_mtime # Update cache
                        print(f"[SUCCESS] Model backed up to Drive: {file}")
                        # os.remove(src)  <-- Disabled to avoid race conditions with parallel evaluation scripts
                        # print(f"[INFO] Local model deleted to save disk space: {file}")
                    else:
                        _uploaded_cache[src] = local_mtime # It's already there and up to date
                        # os.remove(src)  <-- Disabled to avoid race conditions with parallel evaluation scripts
                        # print(f"[INFO] Local model deleted to save disk space (already in Drive): {file}")
                except Exception as e:
                    print(f"[ERROR] Failed to backup or delete model {file}: {e}")

    if os.path.exists(local_reports):
        for root, _, files in os.walk(local_reports):
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(src, local_reports)
                dst = os.path.join(backup_reports_dir, rel_path)
                
                # Check cache for INDIVIDUAL files
                local_mtime = os.path.getmtime(src)
                if _uploaded_cache.get(src) == local_mtime:
                    continue

                os.makedirs(os.path.dirname(dst), exist_ok=True)

                try:
                    if not os.path.exists(dst) or local_mtime > os.path.getmtime(dst):
                        shutil.copy2(src, dst)
                        _uploaded_cache[src] = local_mtime
                        print(f"[SUCCESS] Report backed up to Drive: {rel_path}")
                    else:
                        _uploaded_cache[src] = local_mtime
                except Exception as e:
                    print(f"[ERROR] Failed to backup report {rel_path}: {e}")

def backup_results_sync():
    """Forces a blocking backup to Drive. Used at the very end of the script."""
    with backup_lock:
        _do_backup()

def _backup_worker():
    # Only run if no other backup is currently processing
    if backup_lock.acquire(blocking=False):
        try:
            _do_backup()
        finally:
            backup_lock.release()

def backup_results():
    """Triggers an asynchronous, non-blocking backup to Drive."""
    threading.Thread(target=_backup_worker, daemon=True).start()

def resolve_model_path(model_filename):
    """Finds a model file from local or Drive backup, restoring from Drive if needed."""
    local_path = os.path.join(BASE_DIR, "models", model_filename)
    if os.path.exists(local_path):
        return local_path

    drive_path = os.path.join(DRIVE_DIR, "models", model_filename)
    if os.path.exists(drive_path):
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        shutil.copy2(drive_path, local_path)
        print(f"[INFO] Restored model from Drive: {model_filename}")
        return local_path

    return None

def is_model_trained(model_name, ablation, head_strategy, tune_strategy, cv_folds):
    """Checks if the Keras model(s) for the given ablation, model, head, and tune strategy combine already exist."""
    drive_models = os.path.join(DRIVE_DIR, "models")
    local_models = os.path.join(BASE_DIR, "models")
    
    def exists(name):
        return os.path.exists(os.path.join(local_models, name)) or os.path.exists(os.path.join(drive_models, name))
        
    if cv_folds and cv_folds > 1:
        # Check if ALL folds exist
        for f in range(1, cv_folds + 1):
            if not exists(f"{config.RUN_NAME}_{model_name}_{ablation.lower()}_{head_strategy}_{tune_strategy}_fold_{f}_best.keras"):
                return False
        return True
    else:
        return exists(f"{config.RUN_NAME}_{model_name}_{ablation.lower()}_{head_strategy}_{tune_strategy}_best.keras")

def is_baseline_trained(ablation):
    """Checks if the baseline SVM is trained."""
    name = f"baseline_svm_{ablation.lower()}.joblib"
    return os.path.exists(os.path.join(BASE_DIR, "models", name)) or \
           os.path.exists(os.path.join(DRIVE_DIR, "models", name))

def main():
    parser = argparse.ArgumentParser(description="Automated Experiment Pipeline")
    parser.add_argument("--run-mode", type=str, choices=["missing", "full"], default="missing",
                        help="missing: skip already trained models to save time. full: train everything from scratch.")
    parser.add_argument("--tune_strategy", type=str, choices=["standard", "progressive", "both"], default="both",
                        help="Fine-tuning strategy: standard, progressive, or both (runs both in parallel pipeline)")
    args = parser.parse_args()

    print("=========================================================")
    print(f"[STARTED] AUTOMATED EXPERIMENT PIPELINE (Mode: {args.run_mode.upper()})")
    print("=========================================================\n")
    
    if not pre_flight_checks():
        print("[CRITICAL] Pre-flight checks failed. Aborting pipeline execution to save time.")
        return

    os.chdir(BASE_DIR)

    # --- STAGE 1: DATA PREPARATION (ALL ABLATION MODES) ---
    print("\n--- STAGE 1: DATA PREPARATION (ALL ABLATION MODES) ---")
    for ablation in ABLATIONS:
        print(f"\n[INIT] Data prep: Ablation={ablation} ")
        try:
            prepare_data(ablation)
        except Exception as e:
            print(f"[ERROR] Failed preparing {ablation} data: {e}")
            continue

    if args.tune_strategy == "both":
        tune_strategies_to_run = ["standard", "progressive"]
    else:
        tune_strategies_to_run = [args.tune_strategy]

    # --- STAGE 2: N x M GRID TRAINING ---
    print("\n--- STAGE 2: N x M GRID TRAINING (ALL ABLATIONS x ALL MODELS x TUNING STRATS) ---")
    training_cmds = []
    for ablation in ABLATIONS:
        for model in MODELS:
            for strategy in tune_strategies_to_run:
                if args.run_mode == "missing" and is_model_trained(model, ablation, "standard", strategy, CV_FOLDS):
                    print(f"[SKIP] Models for Ablation={ablation}, Model={model}, Tune={strategy} already exist. Skipping.")
                    continue
                    
                cmd = f"python src/train.py --ablation {ablation} --model {model} --cv {CV_FOLDS} --tune_strategy {strategy}"
                training_cmds.append(cmd)
                
    # Run training commands in parallel
    if training_cmds:
        run_tasks_in_parallel(training_cmds, max_workers=2) # Adjust max_workers based on GPU VRAM

    # --- STAGE 4: CLASSICAL ML BASELINE (MEDIA PIPE + SVM) ---
    print("\n--- STAGE 4: CLASSICAL ML BASELINE (MEDIA PIPE + SVM) ---")
    if args.run_mode == "missing" and is_baseline_trained("FULL"):
        print(f"\n[SKIP] Baseline SVM already exists. Skipping.")
    else:
        try:
            cmd = f"python src/baseline_mediapipe_svm.py --ablation FULL"
            run_cmd(cmd)
            backup_results()
        except Exception as e:
            print(f"[ERROR] Baseline SVM failed: {e}")

    # --- STAGE 8: CLASSIFICATION HEAD ABLATION ---
    print("\n--- STAGE 8: CLASSIFICATION HEAD ARCHITECTURE ABLATION ---")
    head_strategies = ["spatial_pooling", "attention"]
    best_backbone = "EfficientNetV2B0"
    ablation = "FULL"
    head_cmds = []
    
    for head_strat in head_strategies:
        for strategy in tune_strategies_to_run:
            if args.run_mode == "missing" and is_model_trained(best_backbone, ablation, head_strat, strategy, CV_FOLDS):
                print(f"[SKIP] Head Ablation Model={best_backbone}, Head={head_strat}, Tune={strategy} already exists. Skipping.")
                continue
                
            cmd = f"python src/train.py --ablation {ablation} --model {best_backbone} --cv {CV_FOLDS} --tune_strategy {strategy} --head_strategy {head_strat}"
            head_cmds.append(cmd)
            
    if head_cmds:
        run_tasks_in_parallel(head_cmds, max_workers=2)

    # --- STAGE 8.5: BULK MODEL RESTORATION FROM DRIVE ---
    print("\n--- STAGE 8.5: BULK MODEL RESTORATION FROM DRIVE ---")
    drive_models = os.path.join(DRIVE_DIR, "models")
    local_models = os.path.join(BASE_DIR, "models")
    if os.path.exists(drive_models):
        print("[INFO] Pre-fetching previously trained models from Drive (if they don't exist locally)...")
        os.makedirs(local_models, exist_ok=True)
        
        models_to_copy = []
        for model_file in os.listdir(drive_models):
            if model_file.endswith(".keras") or model_file.endswith(".joblib"):
                # ONLY FETCH MODELS FOR CURRENT RUN or Baseline SVMs
                is_current_run = model_file.startswith(config.RUN_NAME)
                is_baseline = model_file.startswith("baseline_svm")
                
                if not (is_current_run or is_baseline):
                    continue
                    
                local_path = os.path.join(local_models, model_file)
                drive_path = os.path.join(drive_models, model_file)
                if not os.path.exists(local_path):
                    models_to_copy.append((drive_path, local_path, model_file))
                    
        if models_to_copy:
            print(f" -> Found {len(models_to_copy)} models missing locally. Starting parallel restore...")
            def _copy_model(args):
                src, dst, name = args
                shutil.copy2(src, dst)
                return name
                
            # Use threading to speed up Google Drive I/O
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(_copy_model, item) for item in models_to_copy]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        name = future.result()
                        print(f" -> Restored: {name}")
                    except Exception as e:
                        print(f" -> [ERROR] Failed to restore model: {e}")
            print("[SUCCESS] Bulk restoration complete.")
        else:
            print(" -> All models are already present locally.")
    
    # --- STAGE 9: TEMPORAL SMOOTHING OPTIMIZATION ---
    print("\n--- STAGE 9: TEMPORAL SMOOTHING OPTIMIZATION (INFERENCE) ---")
    try:
        print("[INIT] Running Temporal Smoothing grid search on test dataset...")
        # Ensure the model is available locally before running the script
        for strategy in tune_strategies_to_run:
            model_name_base = f"{config.RUN_NAME}_EfficientNetV2B0_full_standard_{strategy}"
            model_path = resolve_model_path(f"{model_name_base}_fold_1_best.keras")
            if not model_path:
                model_path = resolve_model_path(f"{model_name_base}_best.keras")
            if model_path:
                break # We found a model to use

        cmd = f"python src/optimize_temporal_smoothing.py"
        run_cmd(cmd)
        backup_results()
    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")

    # --- STAGE 10: SEGMENTATION QUALITY EVALUATION ---
    print("\n--- STAGE 10: SEGMENTATION QUALITY EVALUATION (REMBG) ---")
    try:
        print("[INIT] Running Unsupervised Segmentation Evaluation...")
        cmd = f"python src/evaluate_segmentation.py"
        run_cmd(cmd)
        backup_results()
    except Exception as e:
        print(f"[ERROR] Segmentation evaluation failed: {e}")

    # --- STAGE 11: CROSS-DATASET EVALUATION ---
    print("\n--- STAGE 11: CROSS-DATASET EVALUATION (ZERO-SHOT) ---")
    try:
        print("[INIT] Evaluating flagship model on external dataset (if exists)...")
        cmd = f"python src/evaluate_cross_dataset.py"
        run_cmd(cmd)
        backup_results()
    except Exception as e:
        print(f"[ERROR] Cross-Dataset Evaluation failed: {e}")

    # --- STAGE 5: GRAD-CAM VISUALIZATION ---
    print("\n--- STAGE 5: EXPLAINABILITY (GRAD-CAM HEATMAPS) ---")
    try:
        gradcam_cmds = []
        for ablation in ABLATIONS:
            for model in MODELS:
                for head_strat in ["standard", "spatial_pooling", "attention"]:
                    # We only ran spatial_pooling/attention on the best backbone (EfficientNetV2B0), skip if it's not that model
                    if head_strat != "standard" and model != "EfficientNetV2B0":
                        continue
                        
                    for strategy in tune_strategies_to_run:
                        # Try fold_1 first, then non-fold fallback
                        model_name_base = f"{config.RUN_NAME}_{model}_{ablation.lower()}_{head_strat}_{strategy}"
                        model_path = resolve_model_path(f"{model_name_base}_fold_1_best.keras")
                        if not model_path:
                            model_path = resolve_model_path(f"{model_name_base}_best.keras")
                            
                        if not model_path:
                            continue
                            
                        # Directory containing test images
                        test_dir = os.path.join(BASE_DIR, "datasets", f"synthetic_{ablation.lower()}", "test")
                        if not os.path.exists(test_dir):
                            test_dir = os.path.join(BASE_DIR, "datasets", f"synthetic_{ablation.lower()}", "val")
                        if not os.path.exists(test_dir):
                            test_dir = os.path.join(BASE_DIR, "datasets", f"synthetic_{ablation.lower()}")
                            
                        out_dir = os.path.join(BASE_DIR, "reports", model, f"{ablation.upper()}_{head_strat}_{strategy}", "gradcam")
                        
                        if os.path.exists(test_dir):
                            cmd = f"python src/gradcam_vis.py --model_path {model_path} --test_dir {test_dir} --out_dir {out_dir}"
                            gradcam_cmds.append(cmd)
                            
        if gradcam_cmds:
            run_tasks_in_parallel(gradcam_cmds, max_workers=2)
    except Exception as e:
        print(f"[ERROR] Grad-CAM generation failed: {e}")

    # --- STAGE 6: VECTOR GRAPHICS GENERATION ---
    print("\n--- STAGE 6: PUBLICATION QUALITY GRAPHICS (PDF/SVG) ---")
    try:
        cmd = f"python src/plot_vector_graphics.py"
        run_cmd(cmd)
        backup_results() # Backup reports mainly for inserted PDFs
    except Exception as e:
        print(f"[ERROR] Vector graphics generation failed: {e}")

    # --- STAGE 7: CROSS-DATASET EVALUATION ---
    print("\n--- STAGE 7: EXTERNAL DATASET EVALUATION (CROSS-DATASET) ---")
    try:
        external_test_dir = os.path.join(DRIVE_DIR, "external_test")
        if not os.path.exists(external_test_dir):
            external_test_dir = os.path.join(BASE_DIR, "datasets", "external_test")
            
        if os.path.exists(external_test_dir):
            eval_cmds = []
            for ablation in ABLATIONS:
                for model in MODELS:
                    for head_strat in ["standard", "spatial_pooling", "attention"]:
                        if head_strat != "standard" and model != "EfficientNetV2B0":
                            continue
                            
                        for strategy in tune_strategies_to_run:
                            model_name_base = f"{config.RUN_NAME}_{model}_{ablation.lower()}_{head_strat}_{strategy}"
                            model_path = resolve_model_path(f"{model_name_base}_fold_1_best.keras")
                            if not model_path:
                                model_path = resolve_model_path(f"{model_name_base}_best.keras")
                                
                            if model_path:
                                cmd = f"python src/test_cross_dataset.py --model_path {model_path} --model_name {model} --dataset_dir {external_test_dir} --ablation {ablation} --head_strategy {head_strat} --tune_strategy {strategy}"
                                eval_cmds.append(cmd)
                                
            if eval_cmds:
                run_tasks_in_parallel(eval_cmds, max_workers=2)
        else:
            print(f"[WARN] External test data not found: {external_test_dir}")
            print(" -> Please create 'external_test' in Drive or local datasets folder and insert images inside.")
    except Exception as e:
        print(f"[ERROR] Cross-Dataset evaluation failed: {e}")

    print("\n[INFO] Performing final synchronous backup to Google Drive...")
    backup_results_sync()

    print("\n=========================================================")
    print("[SUCCESS] ALL EXPERIMENTS COMPLETED AND BACKED UP!")
    print("=========================================================")

if __name__ == "__main__":
    main()
