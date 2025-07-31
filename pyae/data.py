from typing import Optional, Dict, Union, TypedDict
import torch
from torch import Tensor
from torch.utils.data import Dataset


class EMISample(TypedDict, total=False):
    x: Tensor
    y: Tensor
    ids: Tensor
    x_categories: Dict[str, Tensor]


class EMIDatasetBase(Dataset):
    """Base dataset to share common utilities for EMI datasets.

    Parameters
    ----------
    x : Tensor
        Input features.
    ids : Tensor, optional
        Tensor of identifiers with the same length as ``x``.
    noise : float, default 0.0
        Amount of random noise added to each sample.
    transform : callable, optional
        Optional transform applied on each sample.
    """

    def __init__(
        self,
        x: Tensor,
        ids: Optional[Tensor] = None,
        noise: float = 0.0,
        transform=None,
    ) -> None:
        self.x = x
        self.ids = ids
        self.noise = noise
        self.transform = transform

        if ids is not None and len(ids) != len(x):
            raise ValueError("IDs and input data x must have the same length.")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.x)

    def apply_noise(self, sample: Tensor) -> Tensor:
        """Add random noise to ``sample`` if noise level is greater than zero."""
        return sample + self.noise * torch.rand_like(sample) if self.noise > 0.0 else sample

    def get_id(self, index: int) -> Tensor:
        """Return the ID for ``index`` or an empty tensor if IDs are not provided."""
        return self.ids[index] if self.ids is not None else torch.tensor([])

    def apply_transform(self, x: Tensor) -> Tensor:
        """Apply the optional transform to ``x``."""
        return self.transform(x) if self.transform else x


class EMIDataset(EMIDatasetBase):
    """Dataset for auto-encoding tasks using EMI data."""

    def __init__(
        self,
        x: Tensor,
        x_categories: Optional[Dict[str, Tensor]] = None,
        ids: Optional[Tensor] = None,
        target_feature_index: int = 0,
        noise: float = 0.0,
        transform=None,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        x : Tensor
            Input data of shape ``(n, *)``.
        x_categories : dict[str, Tensor], optional
            Additional categorical information aligned with ``x``.
        ids : Tensor, optional
            Sample identifiers.
        target_feature_index : int, default 0
            Index of the feature to be used as target when ``x`` is 3-D.
        noise : float, default 0.0
            Noise level applied to ``x``.
        transform : callable, optional
            Optional transform applied to each sample.
        """

        super().__init__(x, ids, noise, transform)
        self.x_categories = x_categories or {}
        self.target_feature_index = target_feature_index

        for name, tensor_ in self.x_categories.items():
            if len(tensor_) != len(x):
                raise ValueError(
                    f"Category '{name}' has length {len(tensor_)}, expected {len(x)}"
                )

    def __getitem__(self, index: int) -> EMISample:
        """Return a single sample.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        EMISample
            Dictionary with ``x``, ``y``, ``ids`` and ``x_categories``.
        """

        x = self.apply_noise(self.x[index])
        x = self.apply_transform(x)

        if self.x.ndim == 3:
            y = self.x[index, [self.target_feature_index]]
        else:
            y = self.x[index]

        return {
            "x": x,
            "y": y,
            "ids": self.get_id(index),
            "x_categories": {k: v[index] for k, v in self.x_categories.items()},
        }

    def as_numpy(self, index: int) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Return a sample as NumPy arrays."""
        item = self[index]
        return {
            k: v.numpy() if isinstance(v, Tensor) else {ik: iv.numpy() for ik, iv in v.items()}
            for k, v in item.items()
        }


class EMIDatasetClassifier(EMIDatasetBase):
    """Dataset for supervised classification using EMI data."""

    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        x_categories: Optional[Dict[str, Tensor]] = None,
        ids: Optional[Tensor] = None,
        noise: float = 0.0,
        transform=None,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        x : Tensor
            Input features.
        y : Tensor
            Target labels.
        x_categories : dict[str, Tensor], optional
            Additional categorical information.
        ids : Tensor, optional
            Sample identifiers.
        noise : float, default 0.0
            Noise level applied to ``x``.
        transform : callable, optional
            Optional transform applied to each sample.
        """

        super().__init__(x, ids, noise, transform)
        self.y = y
        self.x_categories = x_categories or {}

        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        for name, tensor_ in self.x_categories.items():
            if len(tensor_) != len(x):
                raise ValueError(
                    f"Category '{name}' has length {len(tensor_)}, expected {len(x)}"
                )

    def __getitem__(self, index: int) -> EMISample:
        """Return a single sample with label."""

        x = self.apply_noise(self.x[index])
        x = self.apply_transform(x)

        return {
            "x": x,
            "y": self.y[index],
            "ids": self.get_id(index),
            "x_categories": {k: v[index] for k, v in self.x_categories.items()},
        }

    def as_numpy(self, index: int) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Return a sample as NumPy arrays."""
        item = self[index]
        return {
            k: v.numpy() if isinstance(v, Tensor) else {ik: iv.numpy() for ik, iv in v.items()}
            for k, v in item.items()
        }
