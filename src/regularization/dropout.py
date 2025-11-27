"""Dropout configuration utilities."""

from typing import Dict, Any
import torch.nn as nn


def configure_dropout(model: nn.Module, dropout_rate: float) -> None:
    """
    Configure dropout rate for model.

    Note: Dropout should be configured during model creation.
    This function is for reference/documentation.

    Args:
        model: PyTorch model
        dropout_rate: Dropout rate (0.0 to 1.0)
    """
    # Dropout is typically configured in model architecture
    # This function serves as a utility/documentation
    pass


def get_dropout_config(dropout_rate: float) -> Dict[str, Any]:
    """
    Get dropout configuration.

    Args:
        dropout_rate: Dropout rate

    Returns:
        Configuration dictionary
    """
    return {"dropout_rate": dropout_rate, "enabled": dropout_rate > 0.0}

