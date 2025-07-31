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
    def __init__(
        self,
        x: Tensor,
        ids: Optional[Tensor] = None,
        noise: float = 0.0,
        transform=None
    ):
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


class EMIDataset(EMIDatasetBase):
    def __init__(
        self,
        x: Tensor,
        x_categories: Optional[Dict[str, Tensor]] = None,
        ids: Optional[Tensor] = None,
        target_feature_index: int = 0,
        noise: float = 0.0,
        transform=None
    ):
        super().__init__(x, ids, noise, transform)
        self.x_categories = x_categories or {}
        self.target_feature_index = target_feature_index

        for name, tensor_ in self.x_categories.items():
            if len(tensor_) != len(x):
                raise ValueError(f"Category '{name}' has length {len(tensor_)}, expected {len(x)}")

    def __getitem__(self, index: int) -> EMISample:
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
            "x_categories": {k: v[index] for k, v in self.x_categories.items()}
        }

    def as_numpy(self, index: int) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        item = self[index]
        return {k: v.numpy() if isinstance(v, Tensor) else {ik: iv.numpy() for ik, iv in v.items()}
                for k, v in item.items()}


class EMIDatasetClassifier(EMIDatasetBase):
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        x_categories: Optional[Dict[str, Tensor]] = None,
        ids: Optional[Tensor] = None,
        noise: float = 0.0,
        transform=None
    ):
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
            "x_categories": {k: v[index] for k, v in self.x_categories.items()}
        }

    def as_numpy(self, index: int) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        item = self[index]
        return {k: v.numpy() if isinstance(v, Tensor) else {ik: iv.numpy() for ik, iv in v.items()}
                for k, v in item.items()}

class EMIDataset(Dataset):
    def __init__(
        self, 
        x, 
        x_categories=None, 
        ids=None, 
        target_feature_index=0, 
        noise=0.0
    ):
        self.x = x
        self.x_categories = x_categories if x_categories is not None else []
        self.ids = ids
        self.target_feature_index = target_feature_index
        self.noise = noise
    
    def __getitem__(self, index):
        data_output = {}
        
        if self.noise > 0.0:
            x = self.x[index] + self.noise * rand_like(self.x[index])
        else:
            x = self.x[index]
        
        data_output.update({"x": x})
        
        if self.x_categories:
            # Suponiendo que x_categories es una lista de tensores one-hot
            categories = [self.x_categories[i][index] for i in range(len(self.x_categories))]
            data_output.update({"x_categories": categories})
        
        if len(self.x.shape) == 3:
            y = self.x[index, [self.target_feature_index]]  # to keep dims
        elif len(self.x.shape) == 2:
            y = self.x[index]
        else:
            y = self.x[index]
        
        data_output.update({"y": y})
        
        if self.ids is not None:
            data_output.update({"ids": self.ids[index]})
        else:
            data_output.update({"ids": tensor([])})
        
        return data_output
    
    def __len__(self):
        return len(self.x)

class EMIDatasetClassifier(Dataset):
    def __init__(self, x, y, x_category=None, ids=None, noise=0.0):
        self.x = x
        self.y = y
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length.")
        
        self.x_category = x_category
        self.ids = ids
        self.noise = noise
    
    def __getitem__(self, index):
        
        # Add x tensor
        if self.noise > 0.0:
            x = self.x[index] + self.noise * rand_like(self.x[index])
        else:
            x = self.x[index]
        
        data_output = {
            "x": x,
            "y": self.y[index]
            }
        
        # Add x_category tensor
        data_output["x_category"] = self.x_category[index] if self.x_category is not None else tensor([])
        
        # Add IDs tensor
        data_output["ids"] = self.ids[index] if self.ids is not None else tensor([])
        
        return data_output
    
    def __len__(self):
        return len(self.x)