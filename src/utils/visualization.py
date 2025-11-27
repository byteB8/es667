"""Visualization utilities using Matplotlib and Seaborn."""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot training and validation curves.

    Args:
        train_metrics: Dictionary of training metrics (e.g., {'loss': [...], 'acc': [...]})
        val_metrics: Dictionary of validation metrics
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]

    for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
        ax = axes[idx]
        epochs = range(1, len(train_values) + 1)

        ax.plot(epochs, train_values, label=f"Train {metric_name}", marker="o")
        if metric_name in val_metrics:
            ax.plot(
                epochs,
                val_metrics[metric_name],
                label=f"Val {metric_name}",
                marker="s",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{metric_name.capitalize()} Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = "acc",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot comparison of multiple experiments.

    Args:
        results: Dictionary mapping experiment names to their metrics
        metric: Metric to compare (e.g., 'acc', 'loss')
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    for exp_name, metrics in results.items():
        if metric in metrics:
            epochs = range(1, len(metrics[metric]) + 1)
            plt.plot(epochs, metrics[metric], label=exp_name, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Comparison: {metric.capitalize()}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_loss_landscape(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot loss landscape showing overfitting/underfitting.

    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="s")

    # Calculate gap
    gap = np.array(val_losses) - np.array(train_losses)
    plt.fill_between(epochs, train_losses, val_losses, alpha=0.3, label="Gap")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Landscape (Overfitting Analysis)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

