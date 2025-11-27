"""Weights & Biases logging utilities."""

from typing import Dict, Optional, Any
import os
import wandb


def is_lightning_ai() -> bool:
    """
    Detect if running on Lightning AI platform.

    Returns:
        True if running on Lightning AI
    """
    # Check for Lightning AI environment variables
    return (
        os.getenv("LIGHTNING_CLOUD_PROJECT_ID") is not None
        or os.getenv("LIGHTNING_STUDIO_ID") is not None
        or os.getenv("TEAMSPACE_ID") is not None
        or "/teamspace/" in os.getcwd()
        or "/lightning/" in os.getcwd()
    )


def setup_wandb(
    project_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    mode: str = "online",
) -> bool:
    """
    Initialize Weights & Biases run with error handling.

    Args:
        project_name: W&B project name
        experiment_name: Name of this experiment
        config: Configuration dictionary to log
        mode: 'online', 'offline', or 'disabled'

    Returns:
        True if W&B initialized successfully, False otherwise
    """
    try:
        # Auto-detect Lightning AI and use offline mode if not logged in
        if is_lightning_ai() and mode == "online":
            try:
                # Try to check if logged in by accessing API
                api = wandb.Api()
                # If we get here, we're logged in
            except Exception:
                print("⚠️  W&B not logged in on Lightning AI. Switching to offline mode.")
                mode = "offline"

        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            mode=mode,
        )
        return True
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize W&B: {e}")
        print("   Continuing without W&B logging...")
        try:
            # Try offline mode as fallback
            if mode != "offline":
                print("   Attempting offline mode...")
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    mode="offline",
                )
                return True
        except Exception:
            pass
        return False


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """
    Log metrics to W&B.

    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number
        prefix: Optional prefix to add to metric names
    """
    try:
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    except Exception:
        # Silently fail if W&B is not initialized
        pass


def finish_wandb() -> None:
    """Finish W&B run."""
    wandb.finish()

