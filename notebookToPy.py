from pathlib import Path
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import matplotlib.pyplot as plt

SEED = 42
seed_everything(SEED, workers=True)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

DATA_PATH = Path("combined_scaled_battery_data.csv")
MODEL_TARGET_COL = "target_scaled"
MODEL_TIME_COL = "cycle_scaled"

RENAMES = {
    "Cycle_Index": "time_idx",
    "Discharge Time (s)": "discharge_time",
    "Decrement 3.6-3.4V (s)": "decrement_36_34",
    "Max. Voltage Dischar. (V)": "max_voltage_discharge",
    "Min. Voltage Charg. (V)": "min_voltage_charge",
    "Time at 4.15V (s)": "time_at_415",
    "Time constant current (s)": "time_constant_current",
    "Charging time (s)": "charging_time",
    "RUL": "target",
}

RAW_FEATURE_COLS = [
    "discharge_time",
    "decrement_36_34",
    "max_voltage_discharge",
    "min_voltage_charge",
    "time_at_415",
    "time_constant_current",
    "charging_time",
]

# Keep VAE_WINDOW_LENGTH aligned with TFT history length so each latent describes
# the same recent degradation context the TFT sees.
VAE_WINDOW_LENGTH = 20
MAX_ENCODER_LENGTH = 20
MAX_PREDICTION_LENGTH = 1

LATENT_DIM = 8
VAE_HIDDEN_SIZE = 64
VAE_BATCH_SIZE = 256
VAE_EPOCHS = 20
VAE_LR = 1e-3
VAE_BETA = 1e-3

TFT_BATCH_SIZE = 64
TFT_SOURCE_EPOCHS = 15
TFT_TRANSFER_EPOCHS = 10
# Target-only starts from random weights, so give it a larger early-stopped budget while keeping the same target batteries.
TFT_TARGET_ONLY_EPOCHS = 20
TFT_SOURCE_LR = 3e-2
TFT_TRANSFER_LR = 5e-3

# Fraction of target-domain training batteries used for fine-tuning.
# Increase to 1.0 if you want full-target fine-tuning instead of the small-data setup.
TARGET_TRAIN_FRACTION = 0.35

# False matches the VAE-transfer idea: raw features -> frozen VAE encoder -> latent -> TFT head -> normalized RUL.
# Set True for a hybrid TFT that sees both raw cycle features and frozen VAE latents.
USE_RAW_FEATURES_IN_TFT = False

# RUL cannot be negative; clipping is applied only after converting predictions back to cycles.
CLIP_NEGATIVE_RUL_PREDICTIONS = True
