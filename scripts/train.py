"""Main training script."""

from src.utils.logger import setup_wandb, finish_wandb
from src.utils.device import get_device, set_seed
from src.training.trainer import Trainer
from src.optimizers.optimizer_factory import create_optimizer
from src.models.resnet import create_resnet
from src.data import StanfordDogsDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
import torch
import argparse
import json
from pathlib import Path
import sys

# Add src to path BEFORE importing from src
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model on Stanford Dogs dataset")

    # Data arguments
    parser.add_argument("--dataset-path", type=str,
                        required=True, help="Path to dataset")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--image-size", type=int,
                        default=224, help="Image size")

    # Model arguments
    parser.add_argument("--model-variant", type=str,
                        default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--use-batch-norm", action="store_true",
                        default=True, help="Use batch normalization")
    parser.add_argument("--no-batch-norm",
                        dest="use_batch_norm", action="store_false")
    parser.add_argument("--dropout-rate", type=float,
                        default=0.0, help="Dropout rate")

    # Optimizer arguments
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["sgd", "momentum", "adam",
                                 "rmsprop", "adagrad", "gd"],
                        help="Optimizer to use")
    parser.add_argument("--learning-rate", type=float,
                        default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum (for SGD/Momentum)")
    parser.add_argument("--weight-decay", type=float,
                        default=0.0, help="Weight decay (L2 regularization)")

    # Regularization arguments
    parser.add_argument("--lambda-l1", type=float, default=0.0,
                        help="L1 regularization coefficient")
    parser.add_argument("--lambda-l2", type=float, default=0.0,
                        help="L2 regularization coefficient (alternative to weight-decay)")

    # Data augmentation
    parser.add_argument("--use-augmentation", action="store_true",
                        default=True, help="Use data augmentation")
    parser.add_argument("--no-augmentation",
                        dest="use_augmentation", action="store_false")
    parser.add_argument("--augmentation-strength", type=str, default="medium",
                        choices=["light", "medium", "strong"], help="Augmentation strength")

    # Training arguments
    parser.add_argument("--num-epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output arguments
    parser.add_argument("--output-dir", type=str,
                        default="results", help="Output directory")
    parser.add_argument("--experiment-name", type=str,
                        default="experiment", help="Experiment name")

    # W&B arguments
    parser.add_argument("--use-wandb", action="store_true",
                        default=True, help="Use Weights & Biases")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", type=str,
                        default="dl667", help="W&B project name")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])

    return parser.parse_args()


def create_data_loaders(args):
    """Create data loaders."""
    # Transforms
    train_transform = get_train_transforms(
        image_size=args.image_size,
        use_augmentation=args.use_augmentation,
        augmentation_strength=args.augmentation_strength,
    )
    val_transform = get_val_transforms(image_size=args.image_size)

    # Datasets
    train_dataset = StanfordDogsDataset(
        dataset_path=args.dataset_path,
        split="train",
        transform=train_transform,
    )
    val_dataset = StanfordDogsDataset(
        dataset_path=args.dataset_path,
        split="val",
        transform=val_transform,
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.get_num_classes()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, num_classes = create_data_loaders(args)
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print(f"Creating {args.model_variant} model...")
    model = create_resnet(
        num_classes=num_classes,
        variant=args.model_variant,
        use_batch_norm=args.use_batch_norm,
        dropout_rate=args.dropout_rate,
    )

    # Move model to device BEFORE creating optimizer
    # This ensures optimizer state is on the correct device (important for Adagrad)
    model = model.to(device)

    # Create optimizer
    optimizer_kwargs = {}
    if args.optimizer in ["sgd", "momentum"]:
        optimizer_kwargs["momentum"] = args.momentum
    optimizer_kwargs["weight_decay"] = args.weight_decay

    optimizer = create_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        **optimizer_kwargs,
    )

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Setup W&B
    wandb_initialized = False
    if args.use_wandb:
        config = vars(args)
        config["num_classes"] = num_classes
        wandb_initialized = setup_wandb(
            project_name=args.wandb_project,
            experiment_name=args.experiment_name,
            config=config,
            mode=args.wandb_mode,
        )
        if not wandb_initialized:
            # Disable W&B if initialization failed
            args.use_wandb = False
            print("   Training will continue without W&B logging")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        use_wandb=args.use_wandb,
    )

    # Train
    print("\nStarting training...")
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=str(output_dir),
        save_best=True,
    )

    # Save training history
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    # Finish W&B
    if args.use_wandb and wandb_initialized:
        finish_wandb()

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
