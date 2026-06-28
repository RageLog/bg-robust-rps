"""
build_external_test_v2.py - V2 external-test builder and zero-shot evaluator.

Constructs external_test_v2 by reusing the RPS gesture split from the primary
external_test and replacing the synthetic ``none`` class with a three-component
mix: random-noise tiles, solid-colour tiles, and real HaGRID frames of non-target
gestures (auto-discovered from ann_train_val/*.json). HaGRID is never used for
training or model selection, so the real-hand subset is leakage-free.
Evaluation overlaps Drive-to-local checkpoint staging (thread pool) with GPU
inference. Reports are written to DRIVE_DIR/revize/ when Drive is mounted.
"""
import os
# Disable XLA auto-clustering before TensorFlow import; per-graph compilation
# adds overhead without benefit for zero-shot inference over many checkpoints.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import sys
import json
import queue
import shutil
import random
import threading
import concurrent.futures

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import prepare_hagrid_annotations as prep
from evaluate_cross_dataset import (
    load_external_dataset,
    get_preprocessing_function,
    _dummy_preprocess,
)

SEED = 42
N_NOISE = 50
N_SOLID = 50
N_REAL = 50
TARGET = config.IMG_SIZE  # (224, 224)

# Restrict evaluation to checkpoints whose filename contains this substring.
# Empty evaluates every checkpoint found. Example: "full_standard_standard".
MODEL_FILTER = os.environ.get("MODEL_FILTER", "")
# Parallel Drive->local copy workers and how far staging may run ahead of inference.
STAGE_WORKERS = int(os.environ.get("STAGE_WORKERS", "12"))
STAGE_AHEAD = 24

RPS_TOKENS = ("fist", "palm", "peace", "two")
SKIP_TOKENS = ("no_gesture", "none", "background", "empty")


def _generate_synthetic_none(none_dir):
    """Write N_NOISE random-noise and N_SOLID solid-colour tiles (deterministic)."""
    rng = np.random.default_rng(SEED)
    for i in range(N_NOISE):
        arr = rng.integers(0, 256, (TARGET[1], TARGET[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(none_dir, f"noise_{i:03d}.jpg"))
    for i in range(N_SOLID):
        color = rng.integers(0, 256, 3).tolist()
        arr = np.full((TARGET[1], TARGET[0], 3), color, dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(none_dir, f"solid_{i:03d}.jpg"))


def _resolve_hagrid_paths():
    """Resolve the HaGRID annotation and image roots (mirrors prepare_hagrid_annotations)."""
    import kagglehub
    p = kagglehub.dataset_download("innominate817/hagrid-sample-30k-384p")
    root = os.path.join(p, "hagrid-sample-30k-384p")
    if not os.path.exists(root):
        root = p
    return os.path.join(root, "ann_train_val"), os.path.join(root, "hagrid_30k")


def _index_images(img_root):
    """Map image basename (without extension) -> full path, over the whole image tree."""
    idx = {}
    for r, _dirs, files in os.walk(img_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[os.path.splitext(f)[0]] = os.path.join(r, f)
    return idx


def _add_real_non_rps(none_dir, n_real):
    """Sample n_real real HaGRID non-RPS gesture frames into none_dir (leakage-free)."""
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])

    ann_dir, img_root = _resolve_hagrid_paths()
    if not os.path.isdir(ann_dir):
        raise SystemExit(f"[V2 ERROR] HaGRID annotation dir not found: {ann_dir}")

    all_classes = [os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".json")]
    non_rps = [c for c in all_classes
               if not any(t in c.lower() for t in RPS_TOKENS)
               and not any(t in c.lower() for t in SKIP_TOKENS)]
    if not non_rps:
        raise SystemExit(f"[V2 ERROR] No non-RPS classes among {sorted(all_classes)}.")

    idx = _index_images(img_root)
    rng = random.Random(SEED)
    per = max(1, n_real // len(non_rps))
    picks = []
    for c in sorted(non_rps):
        try:
            ids = list(json.load(open(os.path.join(ann_dir, c + ".json"))).keys())
        except Exception:
            continue
        rng.shuffle(ids)
        got = 0
        for iid in ids:
            if got >= per:
                break
            src = idx.get(iid)
            if src:
                picks.append((c, src))
                got += 1
    rng.shuffle(picks)
    picks = picks[:n_real]

    used = {}
    for i, (c, src) in enumerate(picks):
        try:
            img = Image.open(src).convert("RGB").resize(TARGET, Image.Resampling.LANCZOS)
        except Exception:
            continue
        img.save(os.path.join(none_dir, f"real_nonrps_{c}_{i:03d}.jpg"))
        used[c] = used.get(c, 0) + 1

    print(f"[V2] non-RPS classes available: {sorted(non_rps)}")
    print(f"[V2] real non-RPS samples used per class: {used}")
    if sum(used.values()) == 0:
        raise SystemExit("[V2 ERROR] Located non-RPS annotations but no matching images.")
    return sum(used.values()), used


def build_v2(v1_dir, v2_dir):
    """Build external_test_v2: gestures copied from V1, none rebuilt as the V2 mix."""
    gestures = ["rock", "paper", "scissors"]
    have_v1 = os.path.isdir(v1_dir) and all(
        os.path.isdir(os.path.join(v1_dir, c)) and os.listdir(os.path.join(v1_dir, c))
        for c in gestures
    )
    if not have_v1:
        print("[V2] Primary external_test missing; building it first ...")
        if not prep.crop_hagrid_dataset():
            raise SystemExit("[V2 ERROR] Could not build the primary external_test.")

    if os.path.exists(v2_dir):
        shutil.rmtree(v2_dir)
    for c in gestures + ["none"]:
        os.makedirs(os.path.join(v2_dir, c), exist_ok=True)

    for c in gestures:
        src_c = os.path.join(v1_dir, c)
        for f in os.listdir(src_c):
            shutil.copy2(os.path.join(src_c, f), os.path.join(v2_dir, c, f))

    none_dir = os.path.join(v2_dir, "none")
    _generate_synthetic_none(none_dir)
    n_real, used = _add_real_non_rps(none_dir, N_REAL)
    print(f"[V2] none = {N_NOISE} noise + {N_SOLID} solid + {n_real} real non-RPS hands")
    return {"noise": N_NOISE, "solid": N_SOLID,
            "real_non_rps": n_real, "real_non_rps_classes": used}


def _find_checkpoints():
    """Locate .keras checkpoints in local models/ and the Drive models cache."""
    search_dirs = [config.MODELS_DIR, os.path.join(config.DRIVE_DIR, "models")]
    found = {}
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if not f.endswith(".keras") or f in found:
                    continue
                if MODEL_FILTER and MODEL_FILTER.lower() not in f.lower():
                    continue
                found[f] = os.path.join(root, f)
    return found


def evaluate_v2(v2_dir, out_dir, none_info):
    """Run zero-shot evaluation on external_test_v2.

    Stages checkpoints from Drive to local disk in parallel (thread pool) while
    the main thread runs GPU inference. Staged files are removed after each
    inference pass to bound local disk usage.
    """
    import gc
    import keras
    import evaluate_cross_dataset as ev
    from sklearn.metrics import classification_report

    os.makedirs(out_dir, exist_ok=True)
    X, y, found = load_external_dataset(v2_dir)
    if len(X) == 0:
        raise SystemExit("[V2 ERROR] No mapped images in external_test_v2.")

    checkpoints = _find_checkpoints()
    if not checkpoints:
        raise SystemExit(
            "[V2 ERROR] No '.keras' checkpoints in local models/ or "
            f"{os.path.join(config.DRIVE_DIR, 'models')}. Restore checkpoints from Drive first."
        )
    total = len(checkpoints)
    stage_dir = os.path.join(config.BASE_DIR, "models_stage")
    os.makedirs(stage_dir, exist_ok=True)
    print(f"[V2] staging and evaluating {total} checkpoint(s) on {len(X)} images "
          f"({STAGE_WORKERS} I/O workers)", flush=True)

    work_q = queue.Queue(maxsize=STAGE_AHEAD)

    def _stage_one(fname, path):
        if not path.startswith(config.DRIVE_DIR):
            return fname, path, False
        local = os.path.join(stage_dir, fname)
        try:
            if not os.path.exists(local) or os.path.getsize(local) == 0:
                shutil.copy2(path, local)
            return fname, local, True
        except Exception:
            return fname, path, False

    def producer():
        with concurrent.futures.ThreadPoolExecutor(max_workers=STAGE_WORKERS) as ex:
            futs = [ex.submit(_stage_one, fn, p) for fn, p in checkpoints.items()]
            for fut in concurrent.futures.as_completed(futs):
                work_q.put(fut.result())
        work_q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    results, skipped = {}, []
    done = 0
    while True:
        item = work_q.get()
        if item is None:
            break
        fname, path, was_staged = item
        name = fname.replace(".keras", "")
        done += 1
        model = None
        try:
            ev.CURRENT_PREPROCESSOR = get_preprocessing_function(name)
            model = keras.models.load_model(
                path, custom_objects={"preprocess_input": _dummy_preprocess}, safe_mode=False,
            )
            try:
                model.jit_compile = False  # disable XLA compilation at inference time
            except Exception:
                pass
            preds = model.predict(X, batch_size=config.BATCH_SIZE, verbose=0)
            y_pred = np.argmax(preds, axis=1)
            cr = classification_report(
                y, y_pred, target_names=config.CLASS_NAMES,
                labels=range(len(config.CLASS_NAMES)), output_dict=True, zero_division=0,
            )
            results[name] = {"accuracy": cr["accuracy"], "classification_report": cr}
            print(f"  [{done}/{total}] {name}: acc={cr['accuracy']:.4f} "
                  f"none_f1={cr['none']['f1-score']:.4f}", flush=True)
        except Exception as e:
            skipped.append({"model": name, "error": str(e)[:200]})
            print(f"  [{done}/{total}] [SKIP] {name}: {str(e)[:100]}", flush=True)
        finally:
            del model
            gc.collect()
            if was_staged and path.startswith(stage_dir):
                try:
                    os.remove(path)
                except OSError:
                    pass

    print(f"[V2] evaluated {len(results)} | skipped {len(skipped)}", flush=True)

    data = {
        "variant": "V2_noise+solid+real_non_rps_hands",
        "none_composition": none_info,
        "seed": SEED,
        "external_dir": v2_dir,
        "sample_size": int(len(X)),
        "classes_found": found,
        "n_evaluated": len(results),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "model_results": results,
    }
    out_path = os.path.join(out_dir, "global_cross_dataset_evaluation_v2.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[V2] report saved: {out_path}")
    return out_path


def _backup_v2_dataset(v2_dir, out_dir):
    """Archive the V2 dataset under a new name in out_dir (does not touch V1 caches)."""
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, "external_test_v2")
    shutil.make_archive(archive, "zip", v2_dir)
    print(f"[V2] dataset archived (new name): {archive}.zip")


def main():
    v1_dir = os.path.join(config.BASE_DIR, "datasets", "external_test")
    v2_dir = os.path.join(config.BASE_DIR, "datasets", "external_test_v2")
    if os.path.isdir(config.DRIVE_DIR):
        out_dir = os.path.join(config.DRIVE_DIR, "revize")
    else:
        out_dir = os.path.join(config.REPORTS_DIR, "revize")

    none_info = build_v2(v1_dir, v2_dir)
    _backup_v2_dataset(v2_dir, out_dir)
    evaluate_v2(v2_dir, out_dir, none_info)


if __name__ == "__main__":
    main()
