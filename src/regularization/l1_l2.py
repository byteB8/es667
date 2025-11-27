"""L1 and L2 regularization utilities."""

from typing import Dict
import torch
import torch.nn as nn


def apply_l1_regularization(model: nn.Module, lambda_l1: float) -> torch.Tensor:
    """
    Calculate L1 regularization term.

    Args:
        model: PyTorch model
        lambda_l1: L1 regularization coefficient

    Returns:
        L1 regularization loss term
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss


def apply_l2_regularization(model: nn.Module, lambda_l2: float) -> torch.Tensor:
    """
    Calculate L2 regularization term.

    Note: L2 regularization is typically handled via weight_decay in optimizers,
    but this function provides manual L2 calculation if needed.

    Args:
        model: PyTorch model
        lambda_l2: L2 regularization coefficient

    Returns:
        L2 regularization loss term
    """
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)
    return lambda_l2 * l2_loss


def get_regularization_loss(
    model: nn.Module,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
) -> torch.Tensor:
    """
    Get combined L1 and L2 regularization loss.

    Args:
        model: PyTorch model
        lambda_l1: L1 regularization coefficient
        lambda_l2: L2 regularization coefficient

    Returns:
        Combined regularization loss
    """
    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    if lambda_l1 > 0:
        reg_loss += apply_l1_regularization(model, lambda_l1)

    if lambda_l2 > 0:
        reg_loss += apply_l2_regularization(model, lambda_l2)

    return reg_loss

