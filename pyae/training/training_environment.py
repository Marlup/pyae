# training_environment.py

from dataclasses import dataclass
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader
from typing import Callable, Optional

from pyae.early_stopper import EarlyStopper
from pyae.training.training_config import TrainingConfig


# Public package API
__all__ = (
    "TrainingEnvironment",
)


@dataclass
class TrainingEnvironment:
    optimizer: Optimizer
    lr_scheduler: Optional[_LRScheduler]
    loss_fn: Callable
    early_stopper: EarlyStopper
    train_loader: DataLoader
    eval_loader: Optional[DataLoader] = None

    DEFAULT_OPTIMIZER: Optimizer = Adam

    @classmethod
    def from_model_and_config(cls, model, loss_fn, train_dataset, eval_dataset, config: TrainingConfig):
        optimizer = cls.DEFAULT_OPTIMIZER(model.parameters(), lr=config.adaptive_lr_config["initial_lr"], weight_decay=config.weight_decay)

        scheduler = StepLR(
            optimizer,
            step_size=config.adaptive_lr_config["step_size"],
            gamma=config.adaptive_lr_config["gamma"]
        ) if config.adaptive_lr_config else None

        early_stopper = EarlyStopper(**config.early_stopping_config)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False) if eval_dataset else None

        return cls(
            optimizer=optimizer,
            lr_scheduler=scheduler,
            loss_fn=loss_fn,
            early_stopper=early_stopper,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )
