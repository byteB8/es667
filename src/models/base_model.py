"""Base model class."""

from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """Base class for all models."""

    def __init__(self, num_classes: int):
        """
        Initialize base model.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {"num_classes": self.num_classes}

