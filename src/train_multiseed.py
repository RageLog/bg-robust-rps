"""
train_multiseed.py - Parallel multi-seed stability replication.

Launches each (ablation, model, head, tune, seed) cell as an independent subprocess
(_multiseed_worker.py) that patches config in-process and calls train.train()
unmodified. Cells are discovered from the seed-42 kfold summaries in reports/ or
Drive reports/; if no summaries are found, the fallback grid (standard/standard) is
used. Up to MULTISEED_PARALLEL workers share the GPU concurrently via memory growth.
Per-cell stdout is captured to reports_multiseed/logs/<cell>.log.

Environment variables:
  MULTISEED_SEEDS     comma-separated seed integers (default: "123,2024")
  MULTISEED_MODELS    comma-separated backbone names (default: five standard backbones)
  MULTISEED_ABLATIONS comma-separated ablation tags in uppercase (default: "FULL")
  MULTISEED_CV        number of folds (default: "5")
  MULTISEED_PARALLEL  concurrent subprocess workers (default: "2")
  MULTISEED_HEADS     fallback head strategies when no seed-42 cells are found
  MULTISEED_TUNES     fallback tune strategies when no seed-42 cells are found
"""
import os
import re
import sys
import time
import shutil
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

BASE = config.BASE_DIR
ORIG_DRIVE = config.DRIVE_DIR
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_multiseed_worker.py")

SEEDS = [int(s) for s in os.environ.get("MULTISEED_SEEDS", "123,2024").split(",") if s.strip()]
MODELS = [m.strip() for m in os.environ.get(
    "MULTISEED_MODELS",
    "DenseNet121,EfficientNetV2B0,ResNet50,MobileNetV3Small,VGG16").split(",") if m.strip()]
ABLATIONS = [a.strip().upper() for a in os.environ.get("MULTISEED_ABLATIONS", "FULL").split(",") if a.strip()]
CV = int(os.environ.get("MULTISEED_CV", "5"))
PARALLEL = max(1, int(os.environ.get("MULTISEED_PARALLEL", "2")))
FB_HEADS = [h.strip() for h in os.environ.get("MULTISEED_HEADS", "standard").split(",") if h.strip()]
FB_TUNES = [t.strip() for t in os.environ.get("MULTISEED_TUNES", "standard").split(",") if t.strip()]

CELL_RE = re.compile(r"kfold_summary_(standard|attention|spatial_pooling)_(standard|progressive)\.json$")


def discover_cells(model, ablation):
    found = set()
    for root in (os.path.join(BASE, "reports"), os.path.join(ORIG_DRIVE, "reports")):
        d = os.path.join(root, model, ablation.upper())
        if os.path.isdir(d):
            for f in os.listdir(d):
                m = CELL_RE.match(f)
                if m:
                    found.add((m.group(1), m.group(2)))
    if not found:
        found = {(h, t) for h in FB_HEADS for t in FB_TUNES}
    return sorted(found)


def seed_dirs(seed):
    return (os.path.join(BASE, "models_multiseed", f"seed{seed}"),
            os.path.join(BASE, "reports_multiseed", f"seed{seed}"),
            os.path.join(ORIG_DRIVE, "revize", "multiseed", f"seed{seed}"))


def sync_new(src, dst):
    """Recursively copy files from src to dst, skipping files that already exist."""
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        d = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(d, exist_ok=True)
        for f in files:
            t = os.path.join(d, f)
            if not os.path.exists(t):
                shutil.copy2(os.path.join(root, f), t)


def none_count(ablation):
    """Return the number of 'none'-class images staged for this ablation.

    A count of zero indicates a 3-class dataset (e.g. NO_SYNTH), which is not
    comparable to the 4-class k-fold reported in the paper and must be excluded.
    """
    base = os.path.join(BASE, "datasets", f"synthetic_{ablation.lower()}")
    for pool in ("train", "val"):
        nd = os.path.join(base, pool, "none")
        if os.path.isdir(nd):
            n = sum(1 for f in os.listdir(nd) if f.lower().endswith((".jpg", ".png")))
            if n > 0:
                return n
    return 0


def run():
    logdir = os.path.join(BASE, "reports_multiseed", "logs")
    os.makedirs(logdir, exist_ok=True)

    for seed in SEEDS:
        md, rd, _ = seed_dirs(seed)
        os.makedirs(md, exist_ok=True)
        os.makedirs(rd, exist_ok=True)

    # Exclude ablations whose staged dataset contains no 'none' class (3-class, not comparable).
    usable = []
    for ab in ABLATIONS:
        if none_count(ab) == 0:
            print(f"[MULTISEED][SKIP] {ab}: none-class count is 0 (3-class dataset, excluded from multi-seed run)")
        else:
            usable.append(ab)

    # Job ordering: ablation -> model -> (head, tune) -> seed, so all seeds for a given
    # cell complete together. A partial run still yields complete cells with valid std.
    jobs = []
    for ab in usable:
        for model in MODELS:
            for head, tune in discover_cells(model, ab):
                for seed in SEEDS:
                    md, rd, dd = seed_dirs(seed)
                    env = dict(os.environ,
                               MS_SEED=str(seed), MS_RUN_NAME=f"RPS_Synthetic_V1_seed{seed}",
                               MS_MODELS_DIR=md, MS_REPORTS_DIR=rd, MS_DRIVE_DIR=dd,
                               MS_ABLATION=ab, MS_MODEL=model, MS_HEAD=head, MS_TUNE=tune, MS_CV=str(CV))
                    jobs.append((f"{ab}|{model}|{head}|{tune}|seed{seed}", env))

    print(f"[MULTISEED] seeds={SEEDS} | ablations={usable} | models={MODELS}")
    print(f"[MULTISEED] {len(jobs)} jobs | parallel={PARALLEL} | ~{len(jobs) * CV} fold-trainings")
    print(f"[MULTISEED] job order: ablation -> model -> seed (complete cells prioritised)")
    print(f"[MULTISEED] per-cell logs: {logdir}/")

    running = []   # (label, proc, logfile)
    i = done = 0
    while i < len(jobs) or running:
        while len(running) < PARALLEL and i < len(jobs):
            label, env = jobs[i]
            i += 1
            lf = open(os.path.join(logdir, label.replace("|", "__") + ".log"), "w")
            p = subprocess.Popen([sys.executable, WORKER], env=env, stdout=lf, stderr=subprocess.STDOUT)
            running.append((label, p, lf))
            print(f"[LAUNCH {i}/{len(jobs)}] {label}", flush=True)
        time.sleep(5)
        for label, p, lf in running[:]:
            if p.poll() is not None:
                lf.close()
                done += 1
                print(f"[DONE {done}/{len(jobs)} rc={p.returncode}] {label}", flush=True)
                running.remove((label, p, lf))
                # Incremental Drive sync: copy only new files to avoid overwriting existing results.
                try:
                    seed_n = int(label.rsplit("seed", 1)[1])
                    _, rd_s, dd_s = seed_dirs(seed_n)
                    rep_dst = os.path.join(dd_s, "reports")
                    os.makedirs(rep_dst, exist_ok=True)
                    os.system(f'cp -rn "{rd_s}/." "{rep_dst}/" 2>/dev/null')
                except Exception:
                    pass

    for seed in SEEDS:
        _, rd, dd = seed_dirs(seed)
        try:
            os.makedirs(dd, exist_ok=True)
            shutil.copytree(rd, os.path.join(dd, "reports"), dirs_exist_ok=True)
            print(f"[MULTISEED] synced reports -> {dd}/reports")
        except Exception as e:
            print(f"[MULTISEED][WARN] Drive sync skipped for seed{seed} ({e})")

    print("\n[MULTISEED] complete. Run src/aggregate_multiseed.py to aggregate results.")


if __name__ == "__main__":
    run()
