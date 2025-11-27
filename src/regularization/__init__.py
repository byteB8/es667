"""Regularization techniques."""

from .l1_l2 import apply_l1_regularization, apply_l2_regularization
from .dropout import configure_dropout
from .batch_norm import configure_batch_norm

__all__ = [
    "apply_l1_regularization",
    "apply_l2_regularization",
    "configure_dropout",
    "configure_batch_norm",
]

