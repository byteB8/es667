"""Device management utilities."""

import torch
import random
import numpy as np
import os
from typing import Optional


def is_lightning_ai() -> bool:
    """
    Detect if running on Lightning AI platform.

    Returns:
        True if running on Lightning AI
    """
    return (
        os.getenv("LIGHTNING_CLOUD_PROJECT_ID") is not None
        or os.getenv("LIGHTNING_STUDIO_ID") is not None
        or os.getenv("TEAMSPACE_ID") is not None
        or "/teamspace/" in os.getcwd()
        or "/lightning/" in os.getcwd()
    )


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device (CPU or GPU).

    Args:
        device: Optional device string ('cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        torch.device: The device to use
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Log environment info
    if is_lightning_ai():
        print("🌩️  Detected Lightning AI environment")
        if torch.cuda.is_available():
            print(f"   GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("   Using CPU")
    
    return torch.device(device)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

