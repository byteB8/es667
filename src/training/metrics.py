"""Metrics tracking utilities."""

from typing import Dict, List
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class MetricsTracker:
    """Track training and validation metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []
        self.train_precisions: List[float] = []
        self.train_recalls: List[float] = []
        self.val_precisions: List[float] = []
        self.val_recalls: List[float] = []

    def update_train(
        self, loss: float, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """
        Update training metrics.

        Args:
            loss: Training loss
            predictions: Model predictions
            targets: Ground truth labels
        """
        acc = self._calculate_accuracy(predictions, targets)
        precision, recall, _, _ = self._calculate_precision_recall(
            predictions, targets
        )

        self.train_losses.append(loss)
        self.train_accs.append(acc)
        self.train_precisions.append(precision)
        self.train_recalls.append(recall)

    def update_val(
        self, loss: float, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """
        Update validation metrics.

        Args:
            loss: Validation loss
            predictions: Model predictions
            targets: Ground truth labels
        """
        acc = self._calculate_accuracy(predictions, targets)
        precision, recall, _, _ = self._calculate_precision_recall(
            predictions, targets
        )

        self.val_losses.append(loss)
        self.val_accs.append(acc)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)

    def _calculate_accuracy(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Calculate accuracy."""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == targets).float().sum()
        return (correct / len(targets)).item()

    def _calculate_precision_recall(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, float, float, float]:
        """Calculate precision and recall."""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)

        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(
            target_np, pred_np, average="weighted", zero_division=0
        )

        return float(precision), float(recall), float(f1), 0.0

    def get_latest_train_metrics(self) -> Dict[str, float]:
        """Get latest training metrics."""
        if not self.train_losses:
            return {}
        return {
            "loss": self.train_losses[-1],
            "acc": self.train_accs[-1],
            "precision": self.train_precisions[-1],
            "recall": self.train_recalls[-1],
        }

    def get_latest_val_metrics(self) -> Dict[str, float]:
        """Get latest validation metrics."""
        if not self.val_losses:
            return {}
        return {
            "loss": self.val_losses[-1],
            "acc": self.val_accs[-1],
            "precision": self.val_precisions[-1],
            "recall": self.val_recalls[-1],
        }

    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all metrics as dictionaries."""
        return {
            "train_loss": self.train_losses,
            "train_acc": self.train_accs,
            "val_loss": self.val_losses,
            "val_acc": self.val_accs,
            "train_precision": self.train_precisions,
            "train_recall": self.train_recalls,
            "val_precision": self.val_precisions,
            "val_recall": self.val_recalls,
        }

