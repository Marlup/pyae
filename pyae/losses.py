import torch
from torch import nn
from torch.nn.modules.loss import _Loss

# Custom loss implementation (function vs class): https://discuss.pytorch.org/t/custom-loss-implementation-function-vs-class/143880/2
################# 
#### losses ####
#################

class AreaBetweenCurvesLoss(nn.Module):
    """
    Calculates the area between two curves.
    Parameters:
        - dx (float): Spacing along the specified dimension.
        - dim (int): Dimension along which to compute the integral.


    Forward call:
        Calculates the area between the true and predicted curves.

    Methods:
        - _is_tensor: Checks if input is a tensor.
        - _cast_to_tensor: Casts input to a tensor.
    """

    def __init__(self, dx=1, dim=-1, on_reduce=True):
        super(AreaBetweenCurvesLoss, self).__init__()
    
        self.dx = dx
        self.dim = dim
        self.on_reduce = on_reduce
    
    def forward(self, y_true, y_pred):
        """
        Calculates the area between two curves.

        Parameters:
            - y_true (tensor): True curve.
            - y_pred (tensor): Predicted curve.

        Returns:
            Area between the curves.
        """
        if not self._is_tensor(y_true):
            y_true = self._cast_to_tensor(y_true)
        if not self._is_tensor(y_pred):
            y_pred = self._cast_to_tensor(y_pred)
        
        absolute_difference = torch.abs(y_pred - y_true)
        area_between_curves = torch.trapezoid(absolute_difference, dx=self.dx, dim=self.dim)
        
        if self.on_reduce:
            return area_between_curves.mean()
        
        return area_between_curves

    def _is_tensor(self, x):
        """
        Checks if input is a tensor.

        Parameters:
            - x: Input object.

        Returns:
            True if input is a tensor, False otherwise.
        """
        return isinstance(x, torch.Tensor)
        
    def _cast_to_tensor(self, x):
        """
        Casts input to a tensor.

        Parameters:
            - x: Input object.

        Returns:
            Tensor version of the input.
        """
        return torch.tensor(x)
    
    @property
    def __name__(self):
        return "abc"

# Full credits to Matthew N. Bernstein from https://mbernste.github.io/posts/vae/
class VariationalLoss(nn.Module):
    """
    Variational loss function for Variational Autoencoders (VAEs).
    Parameters:
        - var_eps (float): Epsilon value for numerical stability.

    Forward call:
        Calculates the total loss including reconstruction loss and KL divergence loss.

    Methods:
        - _is_tensor: Checks if input is a tensor.
        - _cast_to_tensor: Casts input to a tensor.
    """

    def __init__(self, var_eps=0.001):
        super(VariationalLoss, self).__init__()

        self.var_eps = var_eps
    
    def forward(self, output, x, mean, log_var):
        """
        Calculates the total loss including reconstruction loss and KL divergence loss.

        Parameters:
            - output (tensor): Output of the VAE decoder.
            - x (tensor): Input to the VAE encoder.
            - mean (tensor): Mean of the latent space.
            - log_var (tensor): Log variance of the latent space.

        Returns:
            Total loss.
        """
        if not self._is_tensor(output):
            output = self._cast_to_tensor(output)
        if not self._is_tensor(x):
            x = self._cast_to_tensor(x)

        reconstruction_loss = nn.functional.mse_loss(output, x, reduction="sum") / len(output)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reconstruction_loss + kl_loss + self.var_eps
    
    def _is_tensor(self, x):
        """
        Checks if input is a tensor.

        Parameters:
            - x: Input object.

        Returns:
            True if input is a tensor, False otherwise.
        """
        return isinstance(x, torch.Tensor)
        
    def _cast_to_tensor(self, x):
        """
        Casts input to a tensor.

        Parameters:
            - x: Input object.

        Returns:
            Tensor version of the input.
        """
        return torch.tensor(x)
    
    @property
    def __name__(self):
        return "VariationalLoss"

class DCECLoss(nn.Module):
    """
    Deep Convolutional Embedded Clustering (DCEC) loss composed of reconstruction loss
    and clustering loss (Kullback-Leibler).
            Parameters:
            - gamma (float): Weight for the clustering loss.
            - denominator_eps (float): Epsilon value added to the denominator for numerical stability.
    """
    def __init__(self, gamma=0.1, on_only_kl=False, denominator_eps=0.001):
        """
        Initializes the DCEC loss.
        """
        super(DCECLoss, self).__init__()
        
        self.gamma = gamma
        self.denominator_eps = denominator_eps
    
    def forward(self, outputs, target, q, p_target):
        """
        Calculates the DCEC loss.

        Parameters:
            - outputs (tensor): Model output.
            - target (tensor): Target data.
            - q (tensor): Predicted distribution.
            - p_target (tensor): Target distribution.

        Returns:
            DCEC loss if both q and p_target are not nan/None, otherwise MSE loss
            is returned.
        """
        mse_loss = self._compute_mse_loss(outputs, target)
        
        on_clustering_forward = (q is not None) and (p_target is not None)
    
        if on_clustering_forward:
            kl_loss = self._compute_kl_loss(q, p_target)
        else: 
            kl_loss = 0.0
        
        return mse_loss + self.gamma * kl_loss
    
    def _compute_mse_loss(self, outputs, target):
        loss = nn.functional.mse_loss(outputs, target, reduction="mean")
        
        return loss
    
    def _compute_kl_loss(self, q, p_target):
        kl_div = kl_divergence(q, p_target)
        
        return self.gamma * kl_div 
    
    @property
    def __name__(self):
        """
        Returns the name of the loss.
        """
        return "DCECLoss"

class QuantileLoss(_Loss):
    """
    Quantile loss function.

    Args:
        quantile (float): Quantile value.
        size_average (bool, optional): Deprecated (see reduction). By default, None.
        reduce (bool, optional): Deprecated (see reduction). By default, None.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                    'none': no reduction will be applied,
                                    'mean': the sum of the output will be divided by the number of elements in the output,
                                    'sum': the output will be summed. By default, 'mean'.
    """
    __constants__ = ['reduction']

    def __init__(self, quantile=0.5, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.quantile = quantile
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return quantile_loss(input, target, self.quantile)

    @property
    def __name__(self):
        return "QuantileLoss"

def quantile_loss(preds, target, quantile):
    """
    Quantile loss function.

    Args:
        preds (torch.Tensor): Predicted values.
        target (torch.Tensor): Target values.
        quantile (float): Quantile value.

    Returns:
        torch.Tensor: Quantile loss.
    """
    assert 0.0 < quantile < 1.0, "Quantile should be in (0, 1) range"
    errors = target - preds
    loss = torch.max((quantile - 1.0) * errors, quantile * errors)
    return torch.abs(loss).mean()

def kl_divergence(q, p_target, eps=1e-3):
    """
    Kullback-Leibler divergence loss function.

    Args:
        q_pred (torch.Tensor): Predicted distribution.
        p_target (torch.Tensor): Target distribution.
        eps (float, optional): Smoothing factor to avoid division by zero. Defaults to 1e-3.

    Returns:
        torch.Tensor: Kullback-Leibler divergence loss.
    """
    q += eps
    return (p_target * torch.log(p_target / q)).sum()