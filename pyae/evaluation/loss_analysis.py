import torch

class LossAnalyzer():
    def __init__(self, model):
        self.model = model

    def compute(self, dataloader, loss_func, on_vae=False):
        self.model.eval()
        losses = [
            self.model._compute_forward_loss(batch, loss_func=loss_func).item()
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
