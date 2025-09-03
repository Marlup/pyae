from typing import Callable
from time import perf_counter


"""
A collection of reusable decorators for timing and status reporting.
"""

@staticmethod
def report_time(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = round(perf_counter() - start, 3)
        print(f"({func.__name__}) Execution time: {elapsed}s")
        return result
    return wrapper

@staticmethod
def report_end(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"({func.__name__}) done with param 1: {str(args[0])}")
        return result
    return wrapper

@staticmethod
def results_training_epoch(func: Callable) -> Callable:
    """
    Wrapper function to print the loss and learning rate during a training epoch.
    
    Args:
        func (function): The function being wrapped.

    Returns:
        function: The wrapper function.
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        results = func(*args, **kwargs)
        
        # Print Loss and learning rate
        param = self._get_optimizer_from_env().param_groups[0]
        loss = results[0] if isinstance(results, (list, tuple)) else results
        print(f"\tLearning Rate: {param['lr']}")
        print(f"\tTraining loss: {round(loss, 6)}")
        return results
    
    return wrapper

@staticmethod
def results_evaluation_epoch(func: Callable) -> Callable:
    """
    Wrapper function to print the loss during an evaluation epoch.
    
    Args:
        func (function): The function being wrapped.

    Returns:
        function: The wrapper function.
    """
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        
        # Print Loss
        eval_loss = results[0]
        print(f"\tEval loss: {round(eval_loss, 6)}")
        print(40 * "-")
        return results
    
    return wrapper