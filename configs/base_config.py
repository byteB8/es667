"""Base configuration classes."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class BaseConfig:
    """Base configuration class for experiments."""

    # Experiment metadata
    name: str = "experiment"
    project_name: str = "dl667"
    description: str = ""

    # Data configuration
    dataset_path: str = ""
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    train_ratio: float = 0.8

    # Model configuration
    model_variant: str = "resnet18"  # resnet18 or resnet34
    use_batch_norm: bool = True
    dropout_rate: float = 0.0

    # Optimizer configuration
    optimizer: str = "adam"  # sgd, momentum, adam, rmsprop, adagrad, gd
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Regularization configuration
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0

    # Data augmentation configuration
    use_augmentation: bool = True
    augmentation_strength: str = "medium"  # light, medium, strong

    # Training configuration
    num_epochs: int = 20
    seed: int = 42

    # Output configuration
    output_dir: str = "results"

    # W&B configuration
    use_wandb: bool = True
    wandb_mode: str = "online"  # online, offline, disabled

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def update(self, **kwargs: Any) -> "BaseConfig":
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration values to update

        Returns:
            New config instance with updated values
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return BaseConfig(**config_dict)

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        if self.optimizer not in ["sgd", "momentum", "adam", "rmsprop", "adagrad", "gd"]:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        if self.model_variant not in ["resnet18", "resnet34"]:
            raise ValueError(f"Unknown model variant: {self.model_variant}")

        if self.augmentation_strength not in ["light", "medium", "strong"]:
            raise ValueError(f"Unknown augmentation strength: {self.augmentation_strength}")

        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")

        if self.lambda_l1 < 0 or self.lambda_l2 < 0:
            raise ValueError("Regularization coefficients must be non-negative")

    def get_wandb_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for W&B logging."""
        return self.to_dict()


@dataclass
class ExperimentConfig(BaseConfig):
    """Extended configuration for experiments with additional metadata."""

    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

