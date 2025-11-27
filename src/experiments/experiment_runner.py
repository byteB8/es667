"""Experiment runner for orchestrating multiple experiments."""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.resnet import create_resnet
from ..optimizers.optimizer_factory import create_optimizer
from ..training.trainer import Trainer
from ..evaluation.evaluator import Evaluator
from ..utils.device import get_device, set_seed
from ..utils.logger import setup_wandb, finish_wandb
from ..regularization.l1_l2 import get_regularization_loss


class ExperimentRunner:
    """Run multiple experiments with different configurations."""

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        """
        Initialize experiment runner.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes
            device: Device to run on
            seed: Random seed
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device or get_device()
        self.seed = seed

    def run_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        use_wandb: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            experiment_name: Name of experiment
            config: Experiment configuration
            use_wandb: Whether to log to W&B
            save_dir: Directory to save results

        Returns:
            Dictionary with experiment results
        """
        # Set seed for reproducibility
        set_seed(self.seed)

        # Setup W&B
        if use_wandb:
            setup_wandb(
                project_name=config.get("project_name", "dl667"),
                experiment_name=experiment_name,
                config=config,
            )

        # Create model
        model = create_resnet(
            num_classes=self.num_classes,
            variant=config.get("model_variant", "resnet18"),
            use_batch_norm=config.get("use_batch_norm", True),
            dropout_rate=config.get("dropout_rate", 0.0),
        )

        # Create optimizer
        optimizer = create_optimizer(
            model,
            optimizer_name=config.get("optimizer", "adam"),
            learning_rate=config.get("learning_rate", 0.001),
            **config.get("optimizer_kwargs", {}),
        )

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            lambda_l1=config.get("lambda_l1", 0.0),
            lambda_l2=config.get("lambda_l2", 0.0),
            use_wandb=use_wandb,
        )

        # Train
        history = trainer.train(
            num_epochs=config.get("num_epochs", 10),
            save_dir=save_dir,
            save_best=True,
        )

        # Evaluate
        evaluator = Evaluator(model, self.device)
        final_metrics = evaluator.evaluate(self.val_loader)

        # Finish W&B
        if use_wandb:
            finish_wandb()

        return {
            "experiment_name": experiment_name,
            "config": config,
            "history": history,
            "final_metrics": final_metrics,
        }

    def run_multiple_experiments(
        self,
        experiment_configs: List[Dict[str, Any]],
        use_wandb: bool = True,
        results_dir: str = "results",
    ) -> List[Dict[str, Any]]:
        """
        Run multiple experiments.

        Args:
            experiment_configs: List of experiment configurations
            use_wandb: Whether to log to W&B
            results_dir: Directory to save results

        Returns:
            List of experiment results
        """
        results = []

        for exp_config in experiment_configs:
            exp_name = exp_config.get("name", "experiment")
            exp_dir = f"{results_dir}/{exp_name}" if results_dir else None

            result = self.run_experiment(
                experiment_name=exp_name,
                config=exp_config,
                use_wandb=use_wandb,
                save_dir=exp_dir,
            )

            results.append(result)

        return results

