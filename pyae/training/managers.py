# Standard library
import os
from copy import deepcopy

# Third-party
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from IPython.display import clear_output

from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader
from typing import Callable, Optional

# Local modules
from pyae.utils.decorators import results_training_epoch, results_evaluation_epoch
from pyae.utils.miscellaneous import get_timestamp
from pyae.training.training_config import TrainingConfig
from pyae.training.training_environment import TrainingEnvironment

################## 
#### Training ####
##################


# Public package API
__all__ = (
    "TrainingManager", 
    "KFoldManager",
)


class KFoldManager:
    """
    Coordina el entrenamiento k-fold utilizando configuraciones modulares y entornos desacoplados.
    """

    def __init__(
        self,
        train_data,
        train_target,
        eval_data,
        eval_target,
        groups,
        base_config,  # TrainingConfig
        build_model_fn,  # Callable[[], nn.Module]
        build_environment_fn,  # Callable[[model, config, train_data, val_data], TrainingEnvironment]
        data_module,  # instancia de DataModule
        cv: int = 5
    ):
        self.train_data = train_data
        self.train_target = train_target
        self.eval_data = eval_data
        self.eval_target = eval_target
        self.groups = train_target if groups is None else groups

        self.config = base_config
        self.build_model_fn = build_model_fn
        self.build_environment_fn = build_environment_fn
        self.data_module = data_module
        self.cv = cv

        self.splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=base_config.early_stopping_config.get("seed", 42))
        self.training_log = {}
        self._init_state_dicts_saved = False

    def k_fold_train_model(self):
        for fold_idx, (train_idx, val_idx) in enumerate(self.splitter.split(self.train_data, self.train_target, self.groups)):
            print(f"\n🔁 Fold {fold_idx + 1}/{self.cv}")

            x_train, y_train = self.train_data[train_idx], self.train_target[train_idx]
            x_val, y_val = self.train_data[val_idx], self.train_target[val_idx]

            model = self.build_model_fn()
            train_loader, val_loader = self.data_module.make_dataloaders(
                x_train, y_train, x_val, y_val,
                batch_size=self.config.batch_size,
                device=self.config.device
            )
            env = self.build_environment_fn(model, self.config, train_loader, val_loader)

            if not self._init_state_dicts_saved:
                self._save_training_state_dicts(model, env)
                self._init_state_dicts_saved = True
            else:
                self._reset_training_state(model, env)

            manager = TrainingManager(model, self.config, env)
            manager.train_model()

            self.training_log[fold_idx] = {
                "model": model,
                "train_losses": manager.train_losses,
                "validation_losses": manager.eval_losses,
                "validation_score": manager.eval_losses[-1] if manager.eval_losses else None
            }

    def _save_training_state_dicts(self, model, env):
        self.init_model_state_dict = deepcopy(model.state_dict())
        self.init_optimizer_state_dict = deepcopy(env.optimizer.state_dict())
        if env.lr_scheduler:
            self.init_scheduler_state_dict = deepcopy(env.lr_scheduler.state_dict())

    def _reset_training_state(self, model, env):
        model.load_state_dict(self.init_model_state_dict)
        env.optimizer.load_state_dict(self.init_optimizer_state_dict)
        if env.lr_scheduler:
            env.lr_scheduler.load_state_dict(self.init_scheduler_state_dict)

    def predict(self, x):
        return [entry["model"](x) for entry in self.training_log.values()]

    def get_summary(self):
        scores = torch.tensor([entry["validation_score"] for entry in self.training_log.values()])
        return {
            "mean": scores.mean().item(),
            "std": scores.std().item(),
            "median": scores.median().item(),
            "iqr": (scores.quantile(0.75) - scores.quantile(0.25)).item()
        }

# ---- TrainingManager ----

class TrainingManager:
    def __init__(self, model: nn.Module, config: TrainingConfig, env: TrainingEnvironment):
        """
        TrainingManager coordina el entrenamiento de un modelo
        basado en objetos de configuración y entorno.

        Args:
            model (nn.Module): modelo a entrenar.
            config (TrainingConfig): parámetros de entrenamiento.
            env (TrainingEnvironment): entorno con componentes funcionales (optimizador, dataloaders...).
        """
        self.model = model
        self.config = config
        self.env = env
        self.prev_loss = float("inf")

        if config.checkpoint_config.get("enabled", False):
            os.makedirs(config.models_path, exist_ok=True)

        self.train_losses = []
        self.eval_losses = []

    def _get_optimizer_from_env(self) -> Optimizer:
        return self.env.optimizer

    def train_model(self):
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 40)

            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)

            if self.env.lr_scheduler:
                self.env.lr_scheduler.step()

            if self.env.eval_loader:
                stop = self._evaluate_and_check_early_stop()
                if stop:
                    break

            if self._should_checkpoint(epoch):
                self._save_state(epoch, train_loss)

            if epoch > 0 and epoch % self.config.checkpoint_config.get("n_display_reset", 10) == 0:
                clear_output()

        if self.config.checkpoint_config.get("enabled", False):
            self._save_state(epoch=self.config.epochs, loss=self.prev_loss, on_last_model_checkpoint=True)

        self.model.eval()

    def _should_checkpoint(self, epoch):
        freq = self.config.checkpoint_config.get("frequency", self.config.epochs)
        return self.config.checkpoint_config.get("enabled", False) and (epoch + 1) % freq == 0

    def _evaluate_and_check_early_stop(self):
        eval_loss = self._eval_epoch()
        self.eval_losses.append(eval_loss)
        stop = self.env.early_stopper.check(eval_loss)
        if stop:
            print(f"Early stopping triggered.")
        return stop

    @results_training_epoch
    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        dataset_size = len(self.env.train_loader.dataset)

        for batch in self.env.train_loader:
            self.env.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.env.loss_fn(outputs, batch["y"])
            loss.backward()
            self.env.optimizer.step()
            total_loss += loss.item()

        return total_loss / dataset_size

    @results_evaluation_epoch
    def _eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        dataset_size = len(self.env.eval_loader.dataset)

        with torch.no_grad():
            for batch in self.env.eval_loader:
                outputs = self.model(batch)
                loss = self.env.loss_fn(outputs, batch["y"])
                total_loss += loss.item()

        return total_loss / dataset_size

    def _save_state(self, epoch, loss, on_last_model_checkpoint=False):
        suffix = "_last_" if on_last_model_checkpoint else "_"
        filename = f"model{suffix}checkpoint_epoch_{epoch}_at_{get_timestamp()}.pt"
        path = os.path.join(self.config.models_path, filename)

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.env.optimizer.state_dict(),
                'loss': loss,
            }, 
            path)
