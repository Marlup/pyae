import torch
import numpy as np
import xarray as xr

############################
#### Data preprocessing #### 
############################

def generate_noisy_signals(x, amplitudes=None):
    """
    Return a set of noisy signals by adding white noise, Norm(mu=0, std=1) to 
    the original signals.

    Args:
        x (ndarray): Array of signals with shape (examples, steps).
        amplitudes (float, ndarray, list, tuple, optional): Amplitude(s) of the white noise.
            If None, default amplitudes [0.01, 0.02] are used.
    
    Returns:
        ndarray: Array of noisy signals.
    """
    if not isinstance(amplitudes, (float, np.ndarray, list, tuple, type(None))):
        raise Exception("'amplitudes' must be either an ndarray, list, tuple, float, or None.")

    if isinstance(amplitudes, (float, )):
        amplitudes = [amplitudes]
        
    # If amplitudes are not defined, set default values
    if amplitudes is None:
        amplitudes = [0.01, 0.02]
    
    noisy_signals = []
    for amplitude in amplitudes:
        noisy_signals.append(x + amplitude * np.random.randn(*x.shape))

    return np.vstack(noisy_signals)

def generate_synthetic_signal(x, probabilities_to_positive=None, axis=-3, return_array=True):
    """
    Generate synthetic signals by skewing positive values based on given probabilities.

    Args:
        - x (ndarray): Input signal data with either 
        
                4-dimensional array -> (*samples, sweeps, categories (sensors), steps)
                    or
                3-dimensional array -> (sweeps, categories (sensors), steps)
          
          Note that dimension '*samples' is optional, 
        
        - probabilities_to_positive (float, ndarray, list, tuple, optional): Probabilities of being positive.
          If None, default probabilities [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99] are used, 10 values. Therefore, 
          10 synthetic signals are generated per category (sensor).
          
        - axis (int, optional): Axis along which to compute the minimum and maximum vectors.
        - return_array (bool, optional): Whether to return the synthetic signals as a NumPy array.

    Returns:
        ndarray or list: 3-dimensional array (stacked_signals, categories (sensors), steps) of synthetic signals.
    """
    if not isinstance(probabilities_to_positive, (float, np.ndarray, list, tuple, type(None))):
        raise Exception("Parameter error. 'probabilities_to_positive' must be either an ndarray, list, tuple, float, or None.")
    
    if isinstance(probabilities_to_positive, (float,)):
        probabilities_to_positive = [probabilities_to_positive]
    
    # If probabilities are not defined, set default values
    if probabilities_to_positive is None:
        probabilities_to_positive = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]

    n_categories, n_steps = x.shape[-2:]
    
    # Compute min-max vectors along the specified axis
    min_max = _generate_min_max(x, axis)
    # Compute the difference between max and min vectors
    difference_signal = np.squeeze(np.diff(min_max, axis=0))
    # Compute the middle vector between max and min vectors
    middle_signal = min_max[0] + difference_signal / 2

    # Generate a matrix (categories, points) of positive values between 0 and 1
    positive_variation_ratio = _generate_positive_variation(n_steps, n_categories)
    # Compute a matrix of variation from the difference vector
    common_variation = positive_variation_ratio * difference_signal

    # Iterate over probabilities to generate signals with different densities of positive values
    synthetic_signals = []
    for probability_to_positive in probabilities_to_positive:
        # Generate a vector of probabilities of being positive
        positive_skewed_signal = _skew_to_positive_signal(n_steps, probability_to_positive)
        # Multiply common variation and the vector of probabilities to yield a matrix
        # of variations with certain skewness towards positive values
        variation = common_variation * positive_skewed_signal
        # Generate the synthetic signal by adding the variation to the middle vector
        synthetic_signal = middle_signal + variation
        # Store the new signal
        synthetic_signals.append(synthetic_signal)

    synthetic_signals = np.array(synthetic_signals)
    
    return synthetic_signals

def _generate_positive_variation(n_points, n_categories):
    """
    Generate a matrix of positive variation ratios between 0 and 1.

    Args:
        n_points (int): Number of points.
        n_categories (int): Number of categories.

    Returns:
        ndarray: Matrix of positive variation ratios.
    """
    rng = np.random.default_rng()
    positive_variation_ratio = rng.uniform(
        np.zeros((n_points,), dtype=np.float16),
        np.ones((n_points,), dtype=np.float16),
        size=(n_categories, n_points)
    )
    return positive_variation_ratio

def _skew_to_positive_signal(size, probability_to_positive=0.5):
    """
    Generate a vector with positive skewness based on given probability.

    Args:
        size (int): Size of the vector.
        probability_to_positive (float, optional): Probability of being positive. Default is 0.5.

    Returns:
        ndarray: Vector with positive skewness.
    """
    rng = np.random.default_rng()
    
    if not isinstance(probability_to_positive, (float, )) \
    or \
       not (0.0 <= probability_to_positive <= 1.0):
        raise Exception("Parameter error. 'probability_to_positive' must be float type and within the range [0.0, 1.0], both included.")
    
    p = [1 - probability_to_positive, probability_to_positive]
    return rng.choice([-1, 1], 
                      size=size, 
                      p=p, 
                      replace=True) \
              .astype(float)

def _generate_min_max(x, axis):
    """
    Compute the minimum and maximum vectors along the specified axis.

    Args:
        x (ndarray): Input array.
        axis (int): Axis along which to compute the minimum and maximum vectors.

    Returns:
        ndarray: Array containing the minimum and maximum vectors.
    """
    return np.array(
        [
            x.min(axis),
            x.max(axis),
        ]
    )

def min_max_scale(x, axis=-1, keepdims=True, eps=1e-9):
    """
    Apply min-max normalization on data 'x', along the dimension 'axis' parameter.
    
    Args:
        x (numpy.ndarray or torch.Tensor or xarray.DataArray): A data array or tensor.
        axis (int): The axis along which to apply the min and max aggregations.
        keepdims (bool): If True, the aggregated axis is not squeezed, i.e., it is not dropped 
                        from the data.
        eps (float): A small value to avoid division by zero.

    Returns:
        numpy.ndarray or torch.Tensor or xarray.DataArray: A normalized array or tensor.
    """
    if not isinstance(x, (np.ndarray, torch.Tensor, xr.DataArray)):
        raise TypeError("'x' should be either a numpy.ndarray, a torch.Tensor, or a xarray.DataArray.")
    
    # Compute min and max values along the specified axis
    min_val = x.min(axis=axis, keepdims=keepdims)
    max_val = x.max(axis=axis, keepdims=keepdims)
    
    # Apply min-max normalization
    scaled_x = (x - min_val) / (max_val - min_val + eps)
    return scaled_x