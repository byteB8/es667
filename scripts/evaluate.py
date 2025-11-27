"""Evaluation script."""

from src.utils.device import get_device
from src.evaluation.evaluator import Evaluator
from src.models.resnet import create_resnet
from src.data import StanfordDogsDataset, get_val_transforms
from torch.utils.data import DataLoader
import torch
import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on Stanford Dogs dataset")

    # Model arguments
    parser.add_argument("--checkpoint", type=str,
                        required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-path", type=str,
                        required=True, help="Path to dataset")
    parser.add_argument("--model-variant", type=str,
                        default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--use-batch-norm", action="store_true",
                        default=True, help="Use batch normalization")
    parser.add_argument("--no-batch-norm",
                        dest="use_batch_norm", action="store_false")
    parser.add_argument("--dropout-rate", type=float,
                        default=0.0, help="Dropout rate")

    # Data arguments
    parser.add_argument("--batch-size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--image-size", type=int,
                        default=224, help="Image size")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"],
                        help="Dataset split ('all' uses all data without splitting)")

    # Output arguments
    parser.add_argument("--output-dir", type=str,
                        default="results/evaluation", help="Output directory")
    parser.add_argument("--use-checkpoint-name", action="store_true", default=True,
                        help="Use checkpoint directory name for output file")
    parser.add_argument("--no-use-checkpoint-name",
                        dest="use_checkpoint_name", action="store_false")

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

    # Load dataset to get num_classes
    val_transform = get_val_transforms(image_size=args.image_size)

    # For "all" split, use train_ratio=1.0 to get all samples
    if args.split == "all":
        dataset = StanfordDogsDataset(
            dataset_path=args.dataset_path,
            split="train",  # Use train split but with ratio=1.0
            transform=val_transform,
            train_ratio=1.0,  # Use all data
        )
    else:
        dataset = StanfordDogsDataset(
            dataset_path=args.dataset_path,
            split=args.split,
            transform=val_transform,
        )

    num_classes = dataset.get_num_classes()
    class_names = dataset.get_class_names()

    print(f"Number of classes: {num_classes}")
    if args.split == "all":
        print(f"Evaluating on all data: {len(dataset)} samples")
    else:
        print(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Create model
    model = create_resnet(
        num_classes=num_classes,
        variant=args.model_variant,
        use_batch_norm=args.use_batch_norm,
        dropout_rate=args.dropout_rate,
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Available checkpoints in results/: {list(Path('results').glob('*/best_model.pth')) if Path('results').exists() else 'No results directory'}"
        )

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

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
        # Extract checkpoint directory name from checkpoint path
        checkpoint_path = Path(args.checkpoint)
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
