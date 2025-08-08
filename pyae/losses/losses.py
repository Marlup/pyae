# custom_losses.py

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


def quantile_loss(outputs, target, quantile):
    """
    Compute quantile regression loss.
    """
    assert 0.0 < quantile < 1.0
    errors = target - outputs
    loss = torch.max((quantile - 1.0) * errors, quantile * errors)
    return torch.abs(loss).mean()


def kl_divergence(q, p_target, eps=1e-3):
    """
    Compute KL divergence with numerical stability.
    """
    q = q + eps
    return (p_target * torch.log(p_target / q)).sum()


def mse_hybrid_reduction(x, y):
    """
    Compute MSE: sum across features, mean across batch.
    """
    return torch.mean(torch.sum(torch.pow(x - y, 2), axis=1), axis=0)


def vae_hybrid_loss(outputs, x, mean, log_var, reduction="sum"):
    """
    Compute reconstruction + KL loss for VAE.
    """
    recon_loss = mse_hybrid_reduction(outputs, x)
    if reduction == "mean":
        kl = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    else:
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / len(mean)
    return recon_loss, kl


class AreaBetweenCurvesLoss(nn.Module):
    """
    Compute area between two curves using trapezoidal integration.
    """
    def __init__(self, dx=1, dim=-1, on_reduce=True):
        super().__init__()
        self.dx = dx
        self.dim = dim
        self.on_reduce = on_reduce

    def forward(self, batch):
        y_true, y_pred = batch["target"], batch["preds"]
        diff = torch.abs(y_pred - y_true)
        area = torch.trapezoid(diff, dx=self.dx, dim=self.dim)
        return area.mean() if self.on_reduce else area


class VariationalLoss(nn.Module):
    """
    VAE loss: reconstruction + KL divergence (beta scaled).
    """
    def __init__(self, beta=0.5, reduction="sum"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, outputs, batch):
        x = batch["x"]
        mean, logvar = batch["mean"], batch["logvar"]
        recon_loss = F.mse_loss(outputs, x, reduction=self.reduction)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl


class DCECLoss(nn.Module):
    """
    DCEC loss: MSE reconstruction + KL clustering.
    """
    def __init__(self, gamma=0.1, denominator_eps=1e-3):
        super().__init__()
        self.gamma = gamma
        self.eps = denominator_eps

    def forward(self, outputs, batch):
        target = batch["x"]
        q = batch.get("q")
        p_target = batch.get("p_target")

        recon_loss = F.mse_loss(outputs, target)
        kl = kl_divergence(q, p_target, eps=self.eps) if (q is not None and p_target is not None) else 0.0
        return recon_loss + self.gamma * kl


class QuantileLoss(_Loss):
    """
    Quantile regression loss.
    """
    def __init__(self, quantile=0.5, reduction="mean"):
        super().__init__(reduction=reduction)
        self.quantile = quantile

    def forward(self, outputs, batch):
        target = batch["target"]
        return quantile_loss(outputs, target, self.quantile)


class FactorVAELoss(nn.Module):
    """
    FactorVAE loss: reconstruction + KL + TC (discriminator-based).
    """
    def __init__(self, beta: float=0.5, gamma: float=5.0, reduction: str="sum"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

def forward(self, outputs, batch):
    x = batch["x"]
    mu = batch["mean"]
    logvar = batch["logvar"]
    tc_logits = batch["tc_logits"]

    recon, kl = vae_hybrid_loss(outputs, x, mu, logvar, self.reduction)
    tc = F.binary_cross_entropy_with_logits(tc_logits, torch.ones_like(tc_logits), reduction=self.reduction)
    if self.reduction == "mean":
        tc /= x.size(0)

    return recon + self.beta * kl + self.gamma * tc


class FactorVAEDiscriminatorLoss(nn.Module):
    """
    Discriminator loss for FactorVAE (real vs permuted z).
    """
    def forward(self, outputs, batch):
        logits_real = batch["logits_real"]
        logits_fake = batch["logits_fake"]

        loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real), reduction="sum")
        loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake), reduction="sum")
        return loss_real + loss_fake

