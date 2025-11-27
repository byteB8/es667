"""Train model from configuration file."""

import argparse
import json
from pathlib import Path
import sys

# Add src and configs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import (
    BaseConfig,
    get_baseline_config,
    get_optimizer_configs,
    get_regularization_configs,
    get_all_experiment_configs,
    get_config_by_name,
    create_custom_config,
)
from src.experiments.experiment_runner import ExperimentRunner
from src.data import StanfordDogsDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
from src.utils.device import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model from configuration"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file or config name from predefined configs",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--config-type",
        type=str,
        choices=["file", "name", "baseline", "optimizers", "regularization", "all"],
        default="file",
        help="Type of configuration to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Use Weights & Biases",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
    )

    return parser.parse_args()


def load_config_from_file(config_path: str, dataset_path: str) -> BaseConfig:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Override dataset_path
    config_dict["dataset_path"] = dataset_path

    return BaseConfig(**config_dict)


def main():
    """Main function."""
    args = parse_args()

    # Get configuration(s)
    if args.config_type == "file":
        if not args.config:
            raise ValueError("--config required when --config-type is 'file'")
        config = load_config_from_file(args.config, args.dataset_path)
        configs = [config]

    elif args.config_type == "name":
        if not args.config:
            raise ValueError("--config required when --config-type is 'name'")
        config = get_config_by_name(args.config, args.dataset_path)
        configs = [config]

    elif args.config_type == "baseline":
        configs = [get_baseline_config(args.dataset_path)]

    elif args.config_type == "optimizers":
        configs = get_optimizer_configs(args.dataset_path)

    elif args.config_type == "regularization":
        configs = get_regularization_configs(args.dataset_path)

    elif args.config_type == "all":
        configs = get_all_experiment_configs(args.dataset_path)

    # Validate all configs
    for config in configs:
        config.validate()

    print(f"Running {len(configs)} experiment(s)")

    # Create data loaders (same for all experiments)
    train_transform = get_train_transforms(
        image_size=configs[0].image_size,
        use_augmentation=configs[0].use_augmentation,
        augmentation_strength=configs[0].augmentation_strength,
    )
    val_transform = get_val_transforms(image_size=configs[0].image_size)

    train_dataset = StanfordDogsDataset(
        dataset_path=args.dataset_path,
        split="train",
        transform=train_transform,
        train_ratio=configs[0].train_ratio,
    )
    val_dataset = StanfordDogsDataset(
        dataset_path=args.dataset_path,
        split="val",
        transform=val_transform,
        train_ratio=configs[0].train_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=configs[0].batch_size,
        shuffle=True,
        num_workers=configs[0].num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs[0].batch_size,
        shuffle=False,
        num_workers=configs[0].num_workers,
        pin_memory=True,
    )

    num_classes = train_dataset.get_num_classes()

    # Create experiment runner
    runner = ExperimentRunner(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=get_device(),
        seed=configs[0].seed,
    )

    # Run experiments
    experiment_configs = []
    for config in configs:
        exp_config = {
            "name": config.name,
            "project_name": config.project_name,
            "model_variant": config.model_variant,
            "use_batch_norm": config.use_batch_norm,
            "dropout_rate": config.dropout_rate,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "optimizer_kwargs": {
                "momentum": config.momentum,
                "weight_decay": config.weight_decay,
                **config.optimizer_kwargs,
            },
            "lambda_l1": config.lambda_l1,
            "lambda_l2": config.lambda_l2,
            "num_epochs": config.num_epochs,
        }
        experiment_configs.append(exp_config)

    results = runner.run_multiple_experiments(
        experiment_configs=experiment_configs,
        use_wandb=args.use_wandb,
        results_dir=args.output_dir,
    )

    print(f"\nCompleted {len(results)} experiment(s)")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

