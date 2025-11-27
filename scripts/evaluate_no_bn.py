"""Evaluation script for models trained without batch normalization."""

import argparse
import json
from pathlib import Path
import sys

# Add src to path BEFORE importing from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data import StanfordDogsDataset, get_val_transforms
from src.models.resnet import create_resnet
from src.evaluation.evaluator import Evaluator
from src.utils.device import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model (no batch norm) on dataset"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34"],
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0, help="Dropout rate"
    )

    # Data arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="Dataset split ('all' uses all data without splitting)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="results/evaluation", help="Output directory"
    )
    parser.add_argument(
        "--use-checkpoint-name",
        action="store_true",
        default=True,
        help="Use checkpoint directory name for output file",
    )
    parser.add_argument(
        "--no-use-checkpoint-name",
        dest="use_checkpoint_name",
        action="store_false",
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint first to get model config
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Available checkpoints in results/: {list(Path('results').glob('*/best_model.pth')) if Path('results').exists() else 'No results directory'}"
        )

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Get model config from checkpoint if available
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        print(f"Found model config in checkpoint: {model_config}")
        num_classes = model_config.get("num_classes", None)
        dropout_rate = model_config.get("dropout_rate", args.dropout_rate)
        model_variant = model_config.get("model_variant", args.model_variant)
    else:
        # Fallback: try to infer from checkpoint or use defaults
        print("Warning: No model config in checkpoint, using defaults")
        num_classes = None  # Will be determined from dataset
        dropout_rate = args.dropout_rate
        model_variant = args.model_variant

    # Load dataset to get num_classes if not in checkpoint
    val_transform = get_val_transforms(image_size=args.image_size)

    if args.split == "all":
        dataset = StanfordDogsDataset(
            dataset_path=args.dataset_path,
            split="train",
            transform=val_transform,
            train_ratio=1.0,
        )
    else:
        dataset = StanfordDogsDataset(
            dataset_path=args.dataset_path,
            split=args.split,
            transform=val_transform,
        )

    dataset_num_classes = dataset.get_num_classes()
    class_names = dataset.get_class_names()

    # Use checkpoint num_classes if available, otherwise use dataset
    if num_classes is None:
        num_classes = dataset_num_classes
    elif num_classes != dataset_num_classes:
        print(
            f"Warning: Checkpoint has {num_classes} classes, but dataset has {dataset_num_classes}"
        )
        num_classes = dataset_num_classes

    print(f"Number of classes: {num_classes}")
    if args.split == "all":
        print(f"Evaluating on all data: {len(dataset)} samples")
    else:
        print(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Create model WITHOUT batch normalization
    print("Creating model WITHOUT batch normalization...")
    model = create_resnet(
        num_classes=num_classes,
        variant=model_variant,
        use_batch_norm=False,  # Always False for this script
        dropout_rate=dropout_rate,
    )

    # Load checkpoint weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model config: use_batch_norm=False, dropout_rate={dropout_rate}")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Evaluate
    evaluator = Evaluator(model, device)
    print("\nEvaluating model...")
    metrics = evaluator.evaluate(data_loader, class_names=class_names)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("=" * 50)

    # Save results
    if args.use_checkpoint_name:
        checkpoint_dir_name = checkpoint_path.parent.name
        results_filename = f"{checkpoint_dir_name}.json"
    else:
        results_filename = "evaluation_results.json"

    results_path = output_dir / results_filename

    # Convert numpy types to native Python types for JSON serialization
    results_to_save = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
    }
    if "classification_report" in metrics:
        results_to_save["classification_report"] = metrics["classification_report"]

    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

