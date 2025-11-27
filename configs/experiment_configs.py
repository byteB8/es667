"""Predefined experiment configurations."""

from typing import List, Dict, Any
from .base_config import BaseConfig, ExperimentConfig


def get_baseline_config(dataset_path: str) -> BaseConfig:
    """
    Get baseline configuration.

    Args:
        dataset_path: Path to dataset

    Returns:
        Baseline configuration
    """
    return BaseConfig(
        name="baseline",
        description="Baseline experiment with Adam optimizer, no regularization, medium augmentation",
        dataset_path=dataset_path,
        optimizer="adam",
        learning_rate=0.001,
        use_augmentation=True,
        augmentation_strength="medium",
        num_epochs=20,
    )


def get_optimizer_configs(dataset_path: str) -> List[BaseConfig]:
    """
    Get configurations for comparing different optimizers.

    Args:
        dataset_path: Path to dataset

    Returns:
        List of optimizer configurations
    """
    base = BaseConfig(
        dataset_path=dataset_path,
        use_augmentation=True,
        augmentation_strength="medium",
        num_epochs=20,
    )

    configs = []

    # Adam
    configs.append(
        base.update(
            name="optimizer_adam",
            description="Adam optimizer",
            optimizer="adam",
            learning_rate=0.001,
        )
    )

    # SGD
    configs.append(
        base.update(
            name="optimizer_sgd",
            description="SGD optimizer",
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.0,
        )
    )

    # SGD with Momentum
    configs.append(
        base.update(
            name="optimizer_momentum",
            description="SGD with Momentum",
            optimizer="momentum",
            learning_rate=0.01,
            momentum=0.9,
        )
    )

    # RMSProp
    configs.append(
        base.update(
            name="optimizer_rmsprop",
            description="RMSProp optimizer",
            optimizer="rmsprop",
            learning_rate=0.001,
        )
    )

    # Adagrad
    configs.append(
        base.update(
            name="optimizer_adagrad",
            description="Adagrad optimizer",
            optimizer="adagrad",
            learning_rate=0.01,
        )
    )

    # Gradient Descent
    configs.append(
        base.update(
            name="optimizer_gd",
            description="Gradient Descent (SGD with momentum=0)",
            optimizer="gd",
            learning_rate=0.01,
        )
    )

    return configs


def get_regularization_configs(dataset_path: str) -> List[BaseConfig]:
    """
    Get configurations for comparing different regularization techniques.

    Args:
        dataset_path: Path to dataset

    Returns:
        List of regularization configurations
    """
    base = BaseConfig(
        dataset_path=dataset_path,
        optimizer="adam",
        learning_rate=0.001,
        use_augmentation=True,
        augmentation_strength="medium",
        num_epochs=20,
    )

    configs = []

    # Baseline (no regularization)
    configs.append(
        base.update(
            name="reg_baseline",
            description="Baseline with no regularization",
        )
    )

    # L1 Regularization
    configs.append(
        base.update(
            name="reg_l1",
            description="L1 regularization",
            lambda_l1=0.0001,
        )
    )

    configs.append(
        base.update(
            name="reg_l1_strong",
            description="L1 regularization (strong)",
            lambda_l1=0.001,
        )
    )

    # L2 Regularization (weight decay)
    configs.append(
        base.update(
            name="reg_l2",
            description="L2 regularization (weight decay)",
            weight_decay=0.0001,
        )
    )

    configs.append(
        base.update(
            name="reg_l2_strong",
            description="L2 regularization (strong)",
            weight_decay=0.001,
        )
    )

    # Dropout
    configs.append(
        base.update(
            name="reg_dropout",
            description="Dropout regularization",
            dropout_rate=0.3,
        )
    )

    configs.append(
        base.update(
            name="reg_dropout_strong",
            description="Dropout regularization (strong)",
            dropout_rate=0.5,
        )
    )

    # No Batch Normalization
    configs.append(
        base.update(
            name="reg_no_bn",
            description="Without batch normalization",
            use_batch_norm=False,
        )
    )

    # Combined: L1 + Dropout
    configs.append(
        base.update(
            name="reg_l1_dropout",
            description="L1 + Dropout",
            lambda_l1=0.0001,
            dropout_rate=0.3,
        )
    )

    # Combined: L2 + Dropout
    configs.append(
        base.update(
            name="reg_l2_dropout",
            description="L2 + Dropout",
            weight_decay=0.0001,
            dropout_rate=0.3,
        )
    )

    return configs


def get_augmentation_configs(dataset_path: str) -> List[BaseConfig]:
    """
    Get configurations for comparing different data augmentation strategies.

    Args:
        dataset_path: Path to dataset

    Returns:
        List of augmentation configurations
    """
    base = BaseConfig(
        dataset_path=dataset_path,
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=20,
    )

    configs = []

    # No augmentation
    configs.append(
        base.update(
            name="aug_none",
            description="No data augmentation",
            use_augmentation=False,
        )
    )

    # Light augmentation
    configs.append(
        base.update(
            name="aug_light",
            description="Light data augmentation",
            use_augmentation=True,
            augmentation_strength="light",
        )
    )

    # Medium augmentation
    configs.append(
        base.update(
            name="aug_medium",
            description="Medium data augmentation",
            use_augmentation=True,
            augmentation_strength="medium",
        )
    )

    # Strong augmentation
    configs.append(
        base.update(
            name="aug_strong",
            description="Strong data augmentation",
            use_augmentation=True,
            augmentation_strength="strong",
        )
    )

    return configs


def get_combined_configs(dataset_path: str) -> List[BaseConfig]:
    """
    Get configurations for combined techniques.

    Args:
        dataset_path: Path to dataset

    Returns:
        List of combined configurations
    """
    base = BaseConfig(
        dataset_path=dataset_path,
        optimizer="adam",
        learning_rate=0.001,
        use_augmentation=True,
        augmentation_strength="medium",
        num_epochs=20,
    )

    configs = []

    # Best optimizer + Dropout
    configs.append(
        base.update(
            name="combined_adam_dropout",
            description="Adam + Dropout",
            dropout_rate=0.3,
        )
    )

    # Best optimizer + L1 + Dropout
    configs.append(
        base.update(
            name="combined_adam_l1_dropout",
            description="Adam + L1 + Dropout",
            lambda_l1=0.0001,
            dropout_rate=0.3,
        )
    )

    # Best optimizer + L2 + Dropout
    configs.append(
        base.update(
            name="combined_adam_l2_dropout",
            description="Adam + L2 + Dropout",
            weight_decay=0.0001,
            dropout_rate=0.3,
        )
    )

    # Best optimizer + Strong Augmentation
    configs.append(
        base.update(
            name="combined_adam_aug_strong",
            description="Adam + Strong Augmentation",
            augmentation_strength="strong",
        )
    )

    return configs


def get_all_experiment_configs(dataset_path: str) -> List[BaseConfig]:
    """
    Get all predefined experiment configurations.

    Args:
        dataset_path: Path to dataset

    Returns:
        List of all experiment configurations
    """
    configs = []

    # Baseline
    configs.append(get_baseline_config(dataset_path))

    # Optimizers
    configs.extend(get_optimizer_configs(dataset_path))

    # Regularization
    configs.extend(get_regularization_configs(dataset_path))

    # Augmentation
    configs.extend(get_augmentation_configs(dataset_path))

    # Combined
    configs.extend(get_combined_configs(dataset_path))

    return configs


def get_config_by_name(name: str, dataset_path: str) -> BaseConfig:
    """
    Get a specific configuration by name.

    Args:
        name: Configuration name
        dataset_path: Path to dataset

    Returns:
        Configuration with the specified name

    Raises:
        ValueError: If configuration name not found
    """
    all_configs = get_all_experiment_configs(dataset_path)
    for config in all_configs:
        if config.name == name:
            return config

    raise ValueError(f"Configuration '{name}' not found")


def create_custom_config(
    dataset_path: str,
    name: str,
    optimizer: str = "adam",
    learning_rate: float = 0.001,
    **kwargs: Any,
) -> BaseConfig:
    """
    Create a custom experiment configuration.

    Args:
        dataset_path: Path to dataset
        name: Experiment name
        optimizer: Optimizer to use
        learning_rate: Learning rate
        **kwargs: Additional configuration parameters

    Returns:
        Custom configuration
    """
    base = BaseConfig(
        dataset_path=dataset_path,
        name=name,
        optimizer=optimizer,
        learning_rate=learning_rate,
    )

    return base.update(**kwargs)

