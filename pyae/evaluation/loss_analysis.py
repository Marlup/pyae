import torch
from pyae.architecture.forward import compute_forward_loss

class LossAnalyzer():
    def __init__(self, model, mode="standard"):
        self.model = model
        self.mode = mode

    def compute(self, dataloader, loss_func, on_vae=False):
        self.model.eval()
        losses = [
            compute_forward_loss(self.model,
                                 batch,
                                 loss_func,
                                 mode=self.mode)[0].item()
            for batch in dataloader
        ]
        return torch.tensor(losses)

    def statistics(self, dataloader, loss_func, on_vae=False):
        losses = self.compute(dataloader, loss_func, on_vae)
        return {
            "mean": losses.mean(),
            "std": losses.std(),
            "median": losses.median(),
            "iqr": losses.quantile(0.75) - losses.quantile(0.25)
        }

    def all_statistics(self, dataloader, losses_dict, on_vae=False):
        result = {}
        for name, loss_func in losses_dict.items():
            stats = self.statistics(dataloader, loss_func, on_vae)
            for k, v in stats.items():
                result[f"{name}_{k}"] = v.item()
        return result
