"""
Configuration file for Drone Radar Classification System.
Dataset: 77 GHz FMCW Radar (SAAB SIRS 1600 - Zenodo 5896641)
"""

import os
from pathlib import Path

# Configure matplotlib to avoid slow font scan in sandboxed/headless environments
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", ".")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    fm.findSystemFonts = lambda *args, **kwargs: []
except Exception:
    pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "DATA" / "raw" / "data_SAAB_SIRS_77GHz_FMCW.npy"
PROCESSED_DIR = BASE_DIR / "DATA" / "processed"
MODEL_SAVE_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Radar Physical Parameters
RADAR_FC = 77e9           # 77 GHz center frequency
C = 3e8                   # Speed of light (m/s)
WAVELENGTH = C / RADAR_FC # ~3.9 mm
DWELL_TIME = 0.009        # 9 ms dwell time per scan
PRF = 17000               # Pulse Repetition Frequency (Hz)
RANGE_BINS = 5            # Range samples centered on target
AZIMUTH_PULSES = 256      # Pulses per dwell segment

# Class Mappings
# 3-Class (Airspace Surveillance: Bird vs Drone vs Human)
COARSE_CLASSES = {
    0: "Bird",
    1: "Drone",
    2: "Human"
}
COARSE_LABEL_TO_IDX = {"bird": 0, "drone": 1, "human": 2}

# Fine-grained 8-Class (Model Identification)
FINE_CLASSES = {
    0: "Bird",
    1: "Human",
    2: "D1",
    3: "D2",
    4: "D3",
    5: "D4",
    6: "D5",
    7: "D6"
}

# Leave-One-Drone-Out (LODO) setting for Out-of-Distribution Moat Evaluation
LODO_HOLDOUT_DRONE = "D6"  # Model never seen during training

# Feature Configuration
FEATURE_SIZE = (64, 64)   # Standardized (Height, Width) for CNN input
INPUT_CHANNELS = 2        # Channel 0: Range-Doppler Map, Channel 1: Micro-Doppler STFT

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0           # Set to 0 for Windows compatibility
SEED = 42
