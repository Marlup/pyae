import os
import numpy as np
from h5py import File as File_hdf
import torch


__all__ = (
    "ModelLoader",
    "HDF5Handler"
)


class ModelLoader():
    """
    A helper class to load PyTorch models and state dictionaries.
    """

    @staticmethod
    def load_model(model: torch.nn.Module, path: str, eval_mode: bool = True):
        model.load_state_dict(torch.load(path))
        if eval_mode:
            model.eval()
        return model

    @staticmethod
    def load_state_dict(path: str, eval_mode: bool = True):
        model = torch.load(path)
        if eval_mode:
            model.eval()
        return model

class HDF5Handler():
    """
    A utility class for reading and writing HDF5 files with datasets.
    """

    @staticmethod
    def write(path: str, dataset_name: str, data: np.ndarray, strict: bool = True) -> bool:
        if not path or not dataset_name or data is None:
            raise ValueError("'path', 'dataset_name', and 'data' must be provided.")
        if os.path.exists(path):
            print(f"File {path} already exists.")
            return False
        mode = "w-" if strict else "w"
        with File_hdf(path, mode) as f:
            f.create_dataset(dataset_name, data=data)
        return True

    @staticmethod
    def read(path: str, dataset_name: str) -> np.ndarray:
        if not path or not dataset_name:
            raise ValueError("'path' and 'dataset_name' must be provided.")
        with File_hdf(path, "r") as f:
            return np.array(f[dataset_name])

