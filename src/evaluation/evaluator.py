"""Model evaluation utilities."""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


class Evaluator:
    """Model evaluator."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize evaluator.

        Args:
            model: PyTorch model
            device: Device to evaluate on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate(
        self, data_loader: DataLoader, class_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.

        Args:
            data_loader: Data loader
            class_names: Optional list of class names

        Returns:
            Dictionary of metrics
        """
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions = torch.argmax(output, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        if class_names:
            report = classification_report(
                all_targets, all_predictions, target_names=class_names, output_dict=True
            )
            metrics["classification_report"] = report

        return metrics

    def get_predictions(
        self, data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions.

        Args:
            data_loader: Data loader

        Returns:
            Tuple of (predictions, targets)
        """
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions = torch.argmax(output, dim=1)

                all_predictions.append(predictions.cpu())
                all_targets.append(target.cpu())

        return torch.cat(all_predictions), torch.cat(all_targets)

