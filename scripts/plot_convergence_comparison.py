"""Generate comparison plot for epochs to reach 80% validation accuracy."""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np


def load_history(history_path: Path) -> Dict:
    """Load training history from JSON file."""
    with open(history_path, "r") as f:
        return json.load(f)


def find_epoch_to_threshold(val_accs: list, threshold: float = 0.80) -> Optional[int]:
    """
    Find the first epoch where validation accuracy reaches threshold.
    
    Args:
        val_accs: List of validation accuracies per epoch
        threshold: Target accuracy threshold (default: 0.80)
    
    Returns:
        Epoch number (1-indexed) or None if never reached
    """
    for epoch, acc in enumerate(val_accs, start=1):
        if acc >= threshold:
            return epoch
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot epochs to reach 80% validation accuracy"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="my_plots/convergence_comparison.png",
        help="Path to save the plot",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Accuracy threshold (default: 0.80)",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Map optimizer names to experiment directories
    optimizer_experiments = {
        "Adam": "adam_baseline",
        "Adagrad": "adagrad_baseline",
        "RMSProp": "rmsprop_baseline",
        "SGD": "sgd_baseline",
        "GD": "batch_gd_baseline",
        "Momentum": "momentum_gd_baseline",
    }
    
    results_dir = Path(args.results_dir)
    results = {}
    
    # Load histories and find epochs to threshold
    for opt_name, exp_name in optimizer_experiments.items():
        history_path = results_dir / exp_name / "history.json"
        
        if not history_path.exists():
            print(f"Warning: Could not find {history_path}")
            continue
        
        history = load_history(history_path)
        val_accs = history.get("val_acc", [])
        
        if not val_accs:
            print(f"Warning: No validation accuracy data for {exp_name}")
            continue
        
        epoch_to_threshold = find_epoch_to_threshold(val_accs, args.threshold)
        final_val_acc = val_accs[-1] if val_accs else None
        
        results[opt_name] = {
            "epoch_to_threshold": epoch_to_threshold,
            "final_val_acc": final_val_acc,
        }
        
        if epoch_to_threshold:
            print(f"{opt_name}: Reached {args.threshold*100}% at epoch {epoch_to_threshold}, Final: {final_val_acc:.2%}")
        else:
            print(f"{opt_name}: Never reached {args.threshold*100}%, Final: {final_val_acc:.2%}")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Epochs to reach threshold
    optimizers = list(results.keys())
    epochs_to_threshold = [
        results[opt]["epoch_to_threshold"] if results[opt]["epoch_to_threshold"] else None
        for opt in optimizers
    ]
    final_accs = [results[opt]["final_val_acc"] for opt in optimizers]
    
    # Create bar plot for epochs to threshold
    colors = ['green' if e is not None else 'red' for e in epochs_to_threshold]
    bars = ax1.bar(optimizers, 
                   [e if e is not None else 0 for e in epochs_to_threshold],
                   color=colors,
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1.5)
    
    # Add "Never reached" labels for optimizers that didn't reach threshold
    max_epoch = max([e for e in epochs_to_threshold if e is not None], default=20)
    for i, (opt, epoch) in enumerate(zip(optimizers, epochs_to_threshold)):
        if epoch is None:
            ax1.text(i, max_epoch * 0.5, "Never\nreached", 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_ylabel(f'Epochs to Reach {args.threshold*100}% Val Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax1.set_title(f'Convergence Speed: Epochs to {args.threshold*100}% Validation Accuracy', 
                 fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max_epoch * 1.2)
    
    # Add value labels on bars
    for i, (bar, epoch) in enumerate(zip(bars, epochs_to_threshold)):
        if epoch is not None:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(epoch)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Final validation accuracy comparison
    bars2 = ax2.bar(optimizers, [acc * 100 for acc in final_accs],
                   color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Final Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax2.set_title('Final Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars2, final_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also create a summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Optimizer':<15} {'Epochs to 80%':<20} {'Final Val Acc':<15}")
    print("-" * 70)
    for opt in optimizers:
        epoch_str = f"{results[opt]['epoch_to_threshold']}" if results[opt]['epoch_to_threshold'] else "Never reached"
        final_str = f"{results[opt]['final_val_acc']*100:.2f}%"
        print(f"{opt:<15} {epoch_str:<20} {final_str:<15}")
    print("=" * 70)


if __name__ == "__main__":
    main()


