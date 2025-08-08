# Standard library
import os
from copy import deepcopy
from typing import Dict

# Third-party
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from IPython.display import clear_output

# Local modules
from constant import RANDOM_STATE
from dataset.classifier import EMIDatasetClassifier
from training.early_stopper import EarlyStopper
from utils.decorators import results_training_epoch, results_evaluation_epoch
from utils.loss_handler import get_loss_handler


################## 
#### Training ####
##################

class KFoldManager:
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_DEVICE = "cpu"
    DEFAULT_GENERATOR = torch.Generator("cpu")

    def __init__(
        self, 
        train_data,
        train_target,
        eval_data,
        eval_target,
        groups,
        dataloader_config,
        cv,
        training_manager_template
    ):
        self.train_data = train_data
        self.train_target = train_target
        self.eval_data = eval_data
        self.eval_target = eval_target
        self.groups = train_target if groups is None else groups
        self.dataloader_config = dataloader_config
        self.cv = cv

        self.splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        self.training_manager_template = training_manager_template
        self.training_log = {}

        # Placeholder for reference state dicts
        self._init_state_dicts_saved = False

    def k_fold_train_model(self):
        for fold_idx, (train_idx, val_idx) in enumerate(self.splitter.split(self.train_data, self.train_target, self.groups)):
            print(f"\nTraining fold {fold_idx + 1}/{self.cv}")

            train_loader, val_loader = self._make_dataloaders(
                self.train_data[train_idx], self.train_target[train_idx],
                self.train_data[val_idx], self.train_target[val_idx]
            )

            training_manager = self.training_manager_template()
            training_manager.train_loader = train_loader
            training_manager.eval_loader = val_loader

            if not self._init_state_dicts_saved:
                self._save_training_state_dicts(training_manager)
                self._init_state_dicts_saved = True
            else:
                self._reset_training_state(training_manager)

            training_manager.train_model()
            self.training_log[fold_idx] = {
                "model": training_manager.model,
                "train_losses": training_manager.train_losses,
                "validation_losses": training_manager.eval_losses,
                "validation_score": training_manager.evaluate_model()
            }

    def _make_dataloaders(self, x_train, y_train, x_val, y_val):
        batch_size = self.dataloader_config.get("batch_size", self.DEFAULT_BATCH_SIZE)
        device = self.dataloader_config.get("device", self.DEFAULT_DEVICE)
        generator = self.dataloader_config.get("generator", self.DEFAULT_GENERATOR)

        train_loader = DataLoader(
            EMIDatasetClassifier(
                x=torch.tensor(x_train, device=device, dtype=torch.float32),
                y=torch.tensor(y_train, device=device, dtype=torch.float32)
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=generator
        )

        val_loader = DataLoader(
            EMIDatasetClassifier(
                x=torch.tensor(x_val, device=device, dtype=torch.float32),
                y=torch.tensor(y_val, device=device, dtype=torch.float32)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader

    def _save_training_state_dicts(self, training_manager):
        self.init_model_state_dict = deepcopy(training_manager.model.state_dict())
        self.init_optimizer_state_dict = deepcopy(training_manager.optimizer.state_dict())
        self.init_scheduler_state_dict = deepcopy(training_manager.lr_scheduler.state_dict())

    def _reset_training_state(self, training_manager):
        training_manager.model.load_state_dict(self.init_model_state_dict)
        training_manager.optimizer.load_state_dict(self.init_optimizer_state_dict)
        training_manager.lr_scheduler.load_state_dict(self.init_scheduler_state_dict)
        training_manager.train_losses.clear()
        training_manager.eval_losses.clear()

    def predict(self, x):
        return [log["model"](x) for _, log in self.training_log.items()]

    def get_summary(self):
        scores = torch.tensor([log["validation_score"] for log in self.training_log.values()])
        return {
            "mean": scores.mean().item(),
            "std": scores.std().item(),
            "median": scores.median().item(),
            "iqr": (scores.quantile(0.75) - scores.quantile(0.25)).item()
        }

class TrainingConfig():
    epochs: int
    batch_size: int
    device: torch.DeviceObjType
    weigth_decay: float
    adaptive_lr_config: Dict[str, float]
    early_stopping_config: Dict[str, float]
    checkpoint_config: Dict[str, float]
    models_path: str

class TrainingEnvironment():
    """
    Defines
        - Optimizer
        - Loss function
        - Regularizers
        - Callbacks (early stopping, adaptive LR, etc)
        - Dataloaders
    """

class TrainingManager():
    def __init__(
        self, 
        model, 
        train_loader,
        eval_loader=None,
        group_test_loaders=None,
        discriminator=None,
        optimizer=None,
        optimizer_discriminator=None,
        criterion=None, 
        criterion_discriminator=None, 
        metrics=None, 
        device=None,
        lr_scheduler=None, 
        epochs=10, 
        tol=4e-5,
        max_no_improvements=5,
        checkpoint_frequency=-1,
        model_checkpoint_directory="",
        n_display_reset=6,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.group_test_loaders = group_test_loaders
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.optimizer_discriminator = optimizer_discriminator
        self.criterion = criterion
        self.criterion_discriminator = criterion_discriminator
        self.metrics = metrics
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.tol = tol
        self.max_no_improvements = max_no_improvements
        self.checkpoint_frequency = epochs if checkpoint_frequency < 1 else checkpoint_frequency
        self.on_model_checkpoint = bool(model_checkpoint_directory)
        self.model_checkpoint_directory = model_checkpoint_directory
        os.makedirs(model_checkpoint_directory, exist_ok=True) if self.on_model_checkpoint else None
        self.n_display_reset = n_display_reset

        param_groups = self.optimizer.defaults
        self.initial_lr = param_groups["lr"]
        self.weight_decay = param_groups["weight_decay"]
        self.step_size = self.lr_scheduler.step_size
        self.gamma = self.lr_scheduler.gamma

        self.train_losses = []
        self.eval_losses = []
        self.early_stopper = EarlyStopper(tol=self.tol, max_no_improvements=self.max_no_improvements)

        self.loss_handler = get_loss_handler(self.mode, self)

    def train_model(self):
        self._train_main_loop()

    def _train_main_loop(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}\n" + "-" * 40)

            epoch_loss = self._train_epoch()
            self._log_losses(epoch_loss)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.eval_loader:
                if self._evaluate_and_check_early_stop():
                    break

            if self._is_checkpoint(epoch):
                self._save_state(epoch, epoch_loss)

            if epoch > 0 and epoch % self.n_display_reset == 0:
                clear_output()

        if self.on_model_checkpoint:
            self._save_state(epoch=self.epochs, loss=self.prev_loss, on_last_model_checkpoint=True)

        self.model.eval()

    def _log_losses(self, epoch_loss):
        self.train_losses.append(epoch_loss)

    def _is_checkpoint(self, epoch):
        return self.on_model_checkpoint and self.checkpoint_frequency > 0 and (epoch + 1) % self.checkpoint_frequency == 0

    def _evaluate_and_check_early_stop(self):
        eval_loss = self._eval_epoch()
        self.eval_losses.append(eval_loss)
        return self._early_stopping(eval_loss)

    @results_training_epoch
    def _train_epoch(self):
        self.model.train()
        if self._should_update_p_target():
            self._update_p_target()

        total_loss = 0.0
        n_samples = len(self.train_loader.dataset)

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self.model(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_samples

    @results_evaluation_epoch
    def _eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        n_samples = len(self.eval_loader.dataset)

        with torch.no_grad():
            for batch in self.eval_loader:
                loss = self.model(batch)
                total_loss += loss.item()

        return total_loss / n_samples