import matplotlib.pyplot as plt
from pyae.utils.miscellaneous import get_exp_adaptive_learning


class ReconstructionPlotter:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def plot_basic_reconstructions(self, n=20, cols=4, suptitle="Reconstruction of signals"):
        self.model.eval()
        rows = n // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.ravel()

        for i, batch in enumerate(self.dataloader):
            if i >= n:
                break

            deviation, recon = self.model._compute_forward_loss(batch, return_outputs=True)
            axes[i].plot(batch["y"].squeeze().cpu(), label="Original")
            axes[i].plot(recon.squeeze().cpu(), label="Predicted")
            axes[i].set_title(f"Loss: {deviation.item():.4f}")
            axes[i].legend()

        plt.suptitle(suptitle)
        plt.tight_layout()

    def plot_with_confidence_intervals(self, model_lower, model_upper, n=20):
        self.model.eval()
        rows = n // 4
        fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
        axes = axes.ravel()

        for i, batch in enumerate(self.dataloader):
            if i >= n:
                break

            loss, recon = self.model._compute_forward_loss(batch, return_outputs=True)
            low = model_lower(batch["x"]).squeeze().cpu()
            up = model_upper(batch["x"]).squeeze().cpu()
            y = batch["y"].squeeze().cpu()

            ax = axes[i]
            ax.plot(y, label="Original")
            ax.plot(recon.squeeze().cpu(), label="Predicted")
            ax.plot(low, label="Lower band")
            ax.plot(up, label="Upper band")
            ax.set_title(f"Loss: {loss.item():.4f}")
            ax.legend()

        plt.tight_layout()

def plot_losses(train_losses, eval_losses=None, **kwargs):
    fig = plt.figure()
    y_lim = kwargs.get("y_lim", None)
    epochs = kwargs.get("epochs", None)
    learning_rate = kwargs.get("learning_rate", None)
    gamma = kwargs.get("gamma", None)
    step_size = kwargs.get("step_size", None)
    color = kwargs.get("color", "red")
    
    plt.plot(train_losses, label='Training Loss')
    if eval_losses:
        plt.plot(eval_losses, label='Validation Loss')
    
    if learning_rate and gamma and step_size and epochs:
        for position, _ in enumerate(get_exp_adaptive_learning(epochs, learning_rate, gamma, step_size)):
            x = step_size * (position + 1)
            plt.axvline(x, 0.0, 1.0, color=color, alpha=0.2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.ylim(y_lim)
    plt.legend()
    plt.show()
    return fig