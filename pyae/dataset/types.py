from typing import TypedDict, Dict
from torch import Tensor

class EMISample(TypedDict, total=False):
    x: Tensor
    y: Tensor
    ids: Tensor = None
    x_categories: Dict[str, Tensor] = None
