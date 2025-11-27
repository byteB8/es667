"""Optimizer factory for creating different optimizers."""

from typing import Dict, Any, Optional
import torch
import torch.optim as optim
from torch.nn import Module


def create_optimizer(
    model: Module,
    optimizer_name: str,
    learning_rate: float = 0.001,
    **kwargs: Any,
) -> optim.Optimizer:
    """
    Create optimizer for model.

    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('sgd', 'adam', 'rmsprop', 'adagrad', 'momentum', 'gd')
        learning_rate: Learning rate
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Optimizer instance
    """
    params = model.parameters()

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        momentum = kwargs.get("momentum", 0.0)
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    elif optimizer_name == "momentum":
        momentum = kwargs.get("momentum", 0.9)
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    elif optimizer_name == "adam":
        betas = kwargs.get("betas", (0.9, 0.999))
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.Adam(params, lr=learning_rate, betas=betas, weight_decay=weight_decay)

    elif optimizer_name == "rmsprop":
        alpha = kwargs.get("alpha", 0.99)
        momentum = kwargs.get("momentum", 0.0)
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.RMSprop(
            params, lr=learning_rate, alpha=alpha, momentum=momentum, weight_decay=weight_decay
        )

    elif optimizer_name == "adagrad":
        lr_decay = kwargs.get("lr_decay", 0.0)
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.Adagrad(
            params, lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay
        )

    elif optimizer_name == "gd":
        # Gradient Descent (SGD with momentum=0)
        weight_decay = kwargs.get("weight_decay", 0.0)
        return optim.SGD(params, lr=learning_rate, momentum=0.0, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_optimizer_config(optimizer_name: str) -> Dict[str, Any]:
    """
    Get default configuration for optimizer.

    Args:
        optimizer_name: Name of optimizer

    Returns:
        Dictionary with default parameters
    """
    configs = {
        "sgd": {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0},
        "momentum": {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0},
        "adam": {"learning_rate": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.0},
        "rmsprop": {"learning_rate": 0.001, "alpha": 0.99, "momentum": 0.0, "weight_decay": 0.0},
        "adagrad": {"learning_rate": 0.01, "lr_decay": 0.0, "weight_decay": 0.0},
        "gd": {"learning_rate": 0.01, "weight_decay": 0.0},
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name not in configs:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return configs[optimizer_name].copy()

