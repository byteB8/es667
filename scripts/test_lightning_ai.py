"""Test script to detect Lightning AI environment."""

import os
import torch
from src.utils.logger import is_lightning_ai as logger_is_lightning_ai
from src.utils.device import is_lightning_ai, get_device
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Test Lightning AI detection."""
    print("=" * 60)
    print("Lightning AI Environment Detection Test")
    print("=" * 60)

    # Check environment variables
    print("\n📋 Environment Variables:")
    lightning_vars = [
        "LIGHTNING_CLOUD_PROJECT_ID",
        "LIGHTNING_STUDIO_ID",
        "TEAMSPACE_ID",
        "LIGHTNING_DIR",
    ]
    found_vars = []
    for var in lightning_vars:
        value = os.getenv(var)
        if value:
            print(f"   {var} = {value}")
            found_vars.append(var)

    if not found_vars:
        print("   No Lightning AI environment variables found")

    # Check working directory
    print(f"\n📁 Current Directory: {os.getcwd()}")
    if "/teamspace/" in os.getcwd() or "/lightning/" in os.getcwd():
        print("   ✓ Lightning AI path detected in working directory")
    else:
        print("   ✗ No Lightning AI path in working directory")

    # Test detection functions
    print("\n🔍 Detection Results:")
    device_detected = is_lightning_ai()
    logger_detected = logger_is_lightning_ai()

    print(f"   device.is_lightning_ai() = {device_detected}")
    print(f"   logger.is_lightning_ai() = {logger_detected}")

    if device_detected or logger_detected:
        print("\n   ✅ Lightning AI environment detected!")
    else:
        print("\n   ❌ Not running on Lightning AI")

    # Test device detection
    print("\n🖥️  Device Information:")
    device = get_device()
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   No GPU available")

    # Test W&B detection
    print("\n📊 W&B Status:")
    try:
        import wandb
        try:
            api = wandb.Api()
            print("   ✅ W&B is logged in")
        except Exception as e:
            print(f"   ⚠️  W&B not logged in: {e}")
            print("   💡 Tip: Run 'wandb login' or use --no-wandb flag")
    except ImportError:
        print("   ⚠️  W&B not installed")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
