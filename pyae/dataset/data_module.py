# data_module.py

import torch
from torch.utils.data import DataLoader
from pyae.dataset import EMIDatasetClassifier


# Public package API
__all__ = (
    "DataModule",
)



class DataModule:
    """
    Encapsula la lógica para construir DataLoaders a partir de datos crudos (x, y).
    Compatible con estructuras de entrenamiento modulares y configurables.
    """

    def __init__(self, dataset_cls=EMIDatasetClassifier):
        self.dataset_cls = dataset_cls

    def make_dataloaders(
        self,
        x_train,
        y_train,
        x_val = None,
        y_val = None,
        batch_size: int = 64,
        device: torch.device  ="cpu",
        generator: torch.Generator = None
    ) -> tuple[DataLoader, DataLoader]:
        """
        Devuelve los DataLoaders de entrenamiento y validación.

        Args:
            x_train, y_train: datos de entrenamiento
            x_val, y_val: datos de validación
            batch_size: tamaño del batch
            device: dispositivo para los tensores
            generator: generador aleatorio de PyTorch

        Returns:
            Tuple[train_loader, val_loader]
        """

        # Train data
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

        train_dataset = self.dataset_cls(x=x_train_tensor, y=y_train_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator
        )
        
        # Validation data
        if x_val_tensor and x_val_tensor:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)

            val_dataset = self.dataset_cls(x=x_val_tensor, y=y_val_tensor)

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            return train_loader, val_loader
        return train_loader
