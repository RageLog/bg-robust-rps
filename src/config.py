"""
config.py - Project Configuration
=================================================
Centralized configuration parameters and file paths.
"""
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVE_DIR = "/content/drive/MyDrive/RPC_Colab"
RAW_DATA_DIR = os.path.join(BASE_DIR, 'datasets', 'raw')            
BACKGROUND_DIR = os.path.join(BASE_DIR, 'datasets', 'backgrounds')
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'datasets', 'synthetic') 
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Provide a direct link to a ZIP file containing 'rock', 'paper', 'scissors' folders.
# If None, the system will automatically fallback to a public dataset (DrGFreeman rps-cv).
EXTERNAL_DATASET_URL = None

CLASS_NAMES = ['rock', 'paper', 'scissors', 'none']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
MODEL_BACKBONE = 'EfficientNetV2B0'  
TRAINABLE_BACKBONE = True            
DROPOUT_RATE = 0.3                   
AUGMENTATION_FACTOR = 5 
ALPHA_MATTING = True 
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
RUN_NAME = "RPS_Synthetic_V1"
# Replace these with your camera/video stream URLs (or 0/1 for the
# default local webcam) before running the realtime / web demos.
CAM_ID_MAIN = "http://<camera-host>:<port>/video"
CAM_ID_SEC = "http://<camera-host>:<port>/video"
PORT = 5000
HOST = '0.0.0.0'
