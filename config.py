# config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "dataset"
INTERIM_DIR = BASE_DIR / "interim"
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"
PREDICTIONS_DIR = BASE_DIR / "predictions"

# Create directories
for directory in [INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FEATURES_DIR, PREDICTIONS_DIR]:
    directory.mkdir(exist_ok=True)

# File paths
RAW_DATA_PATH = DATA_DIR / "data.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation_data.csv"
CLEANED_DATA_PATH = INTERIM_DIR / "cleaned_data.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DIR / "train_processed.csv"
PROCESSED_VAL_PATH = PROCESSED_DIR / "val_processed.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
MAX_FEATURES = 5000
N_GRAM_RANGE = (1, 2)
MAX_ITER = 1000

# Preprocessing
CUSTOM_STOPWORDS = {'said', 'would', 'could', 'also', 'us', 'one', 'two', 'like', 'get'}