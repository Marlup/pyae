from typing import Optional, Callable
import torch
from torch import Tensor
from torch.utils.data import Dataset


# Public package API
__all__ = (
    "EMIDatasetBase",
)


class EMIDatasetBase(Dataset):
    def __init__(
        self,
        x: Tensor,
        ids: Optional[Tensor] = None,
        noise: float = 0.0,
        transform: Optional[Callable] = None,
    ) -> None:
        self.x = x
        self.ids = ids
        self.noise = noise
        self.transform = transform

        if ids is not None and len(ids) != len(x):
            raise ValueError("IDs and input data x must have the same length.")

    def __len__(self) -> int:
        return len(self.x)

    def apply_noise(self, sample: Tensor) -> Tensor:
        return sample + self.noise * torch.rand_like(sample) if self.noise > 0.0 else sample

    def get_id(self, index: int) -> Tensor:
        return self.ids[index] if self.ids is not None else torch.tensor([])

    def apply_transform(self, x: Tensor) -> Tensor:
        return self.transform(x) if self.transform else x
