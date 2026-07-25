"""Small, project-specific training and analysis utilities."""

from .config import TRAIN_CONFIG
from .model import TrainableBFTransformer

__all__ = ["TRAIN_CONFIG", "TrainableBFTransformer"]
