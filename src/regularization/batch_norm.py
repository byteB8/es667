"""Batch normalization utilities."""

from typing import Dict, Any
import torch.nn as nn


def configure_batch_norm(model: nn.Module, use_batch_norm: bool) -> None:
    """
    Configure batch normalization for model.

    Note: Batch normalization should be configured during model creation.
    This function is for reference/documentation.

    Args:
        model: PyTorch model
        use_batch_norm: Whether to use batch normalization
    """
    # Batch normalization is typically configured in model architecture
    # This function serves as a utility/documentation
    pass


def get_batch_norm_config(use_batch_norm: bool) -> Dict[str, Any]:
    """
    Get batch normalization configuration.

    Args:
        use_batch_norm: Whether to use batch normalization

    Returns:
        Configuration dictionary
    """
    return {"use_batch_norm": use_batch_norm, "enabled": use_batch_norm}

