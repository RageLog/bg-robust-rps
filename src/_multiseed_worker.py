"""
_multiseed_worker.py - Single-cell subprocess for the multi-seed stability run.

Reads MS_* environment variables set by train_multiseed.py, patches the config
module with seed-isolated paths and the target random seed, then calls
train.train() for one (ablation, model, head, tune) cell. Each worker process
is fully independent and deterministic; parallelism is achieved by running
multiple workers concurrently with TensorFlow memory growth enabled.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import tensorflow as tf
import train as train_mod

s = int(os.environ["MS_SEED"])
config.RANDOM_SEED = s
config.RUN_NAME = os.environ["MS_RUN_NAME"]
config.MODELS_DIR = os.environ["MS_MODELS_DIR"]
config.REPORTS_DIR = os.environ["MS_REPORTS_DIR"]
config.DRIVE_DIR = os.environ["MS_DRIVE_DIR"]
os.environ["PYTHONHASHSEED"] = str(s)
tf.keras.utils.set_random_seed(s)

train_mod.train(
    ablation_mode=os.environ["MS_ABLATION"],
    model_name=os.environ["MS_MODEL"],
    cv=int(os.environ["MS_CV"]),
    tune_strategy=os.environ["MS_TUNE"],
    head_strategy=os.environ["MS_HEAD"],
)
