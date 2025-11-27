"""Main training loop."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from ..regularization.l1_l2 import get_regularization_loss
from .metrics import MetricsTracker
from ..utils.logger import log_metrics


class Trainer:
    """Trainer class for model training."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        use_wandb: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            lambda_l1: L1 regularization coefficient
            lambda_l2: L2 regularization coefficient
            use_wandb: Whether to log to W&B
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.use_wandb = use_wandb

        self.metrics = MetricsTracker()
        self.model.to(device)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            # Add L1 regularization (L2 is handled via weight_decay in optimizer)
            if self.lambda_l1 > 0:
                reg_loss = get_regularization_loss(
                    self.model, lambda_l1=self.lambda_l1, lambda_l2=0.0
                )
                loss = loss + reg_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_predictions.append(output.detach())
            all_targets.append(target.detach())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / len(self.train_loader)

        self.metrics.update_train(avg_loss, all_predictions, all_targets)

        return self.metrics.get_latest_train_metrics()

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                all_predictions.append(output)
                all_targets.append(target)

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        self.metrics.update_val(avg_loss, all_predictions, all_targets)

        return self.metrics.get_latest_val_metrics()

    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """
        Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model

        Returns:
            Dictionary with training history
        """
        best_val_acc = 0.0

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log metrics
            if self.use_wandb:
                log_metrics(train_metrics, step=epoch, prefix="train")
                log_metrics(val_metrics, step=epoch, prefix="val")

            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")

            # Save best model
            if save_best and val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                if save_dir:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_acc": val_metrics["acc"],
                        },
                        Path(save_dir) / "best_model.pth",
                    )

        return self.metrics.get_all_metrics()

