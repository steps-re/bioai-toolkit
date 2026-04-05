"""Path constants for the ExoPred module."""

from pathlib import Path

EXOPRED_DIR = Path(__file__).parent
TOOLKIT_DIR = EXOPRED_DIR.parent
DATA_DIR = TOOLKIT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
CHECKPOINT_DIR = EXOPRED_DIR / "checkpoints"
