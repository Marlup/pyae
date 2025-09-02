# training_config.py

import torch
from dataclasses import dataclass
from typing import Dict


# Public package API
__all__ = (
    "TrainingConfig"
)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_decay: float = 1e-5

    # Adaptive LR scheduler: step_size and gamma
    adaptive_lr_config: Dict[str, float] = None  # e.g. {"step_size": 10, "gamma": 0.5}

    # Early stopping: tolerance and max no. improvements
    early_stopping_config: Dict[str, float] = None  # e.g. {"tol": 1e-4, "max_no_improvements": 10}

    # Checkpointing: frequency and flag
    checkpoint_config: Dict[str, float] = None  # e.g. {"frequency": 5, "enabled": True}

    # Model save path
    models_path: str = "./checkpoints"
