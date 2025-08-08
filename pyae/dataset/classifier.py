from typing import Optional, Dict, Union
from torch import Tensor
from .base import EMIDatasetBase
from .types import EMISample

class EMIDatasetClassifier(EMIDatasetBase):
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        x_categories: Optional[Dict[str, Tensor]] = None,
        ids: Optional[Tensor] = None,
        noise: float = 0.0,
        transform=None,
    ) -> None:
        super().__init__(x, ids, noise, transform)
        self.y = y
        self.x_categories = x_categories or {}

        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")

        for name, tensor_ in self.x_categories.items():
            if len(tensor_) != len(x):
                raise ValueError(f"Category '{name}' has length {len(tensor_)}, expected {len(x)}")

    def __getitem__(self, index: int) -> EMISample:
        x = self.apply_noise(self.x[index])
        x = self.apply_transform(x)

        return {
            "x": x,
            "y": self.y[index],
            "ids": self.get_id(index),
            "x_categories": {k: v[index] for k, v in self.x_categories.items()},
        }

    def as_numpy(self, index: int) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        item = self[index]
        return {
            k: v.numpy() if isinstance(v, Tensor) else {ik: iv.numpy() for ik, iv in v.items()}
            for k, v in item.items()
        }
