"""Generate comparison plots and analysis."""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.visualization import plot_training_curves, plot_comparison, plot_loss_landscape


def load_history(history_path: Path) -> Dict[str, List[float]]:
    """Load training history from JSON file."""
    with open(history_path, "r") as f:
        return json.load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze and compare experiment results")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="List of experiment names to compare (if not provided, uses all)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["acc", "loss"],
        help="Metrics to plot (default: acc, loss)",
    )

    return parser.parse_args()


def collect_experiment_results(results_dir: Path, experiment_names: List[str] = None) -> Dict[str, Dict]:
    """Collect results from multiple experiments."""
    results_dir = Path(results_dir)
    experiments = {}

    if experiment_names is None:
        # Find all experiments
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                history_path = exp_dir / "history.json"
                if history_path.exists():
                    experiments[exp_dir.name] = {
                        "name": exp_dir.name,
                        "history": load_history(history_path),
                        "path": exp_dir,
                    }
    else:
        # Load specified experiments
        for exp_name in experiment_names:
            exp_dir = results_dir / exp_name
            history_path = exp_dir / "history.json"
            if history_path.exists():
                experiments[exp_name] = {
                    "name": exp_name,
                    "history": load_history(history_path),
                    "path": exp_dir,
                }
            else:
                print(f"Warning: Could not find history for experiment: {exp_name}")

    return experiments


def create_comparison_plots(experiments: Dict[str, Dict], output_dir: Path, metrics: List[str]):
    """Create comparison plots for multiple experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot each metric
    for metric in metrics:
        results = {}
        for exp_name, exp_data in experiments.items():
            history = exp_data["history"]
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"

            if val_key in history:
                results[exp_name] = {metric: history[val_key]}

        if results:
            plot_comparison(
                results,
                metric=metric,
                save_path=str(output_dir / f"comparison_{metric}.png"),
            )
            print(f"Saved comparison plot: {output_dir / f'comparison_{metric}.png'}")

    # Plot loss landscape for each experiment
    for exp_name, exp_data in experiments.items():
        history = exp_data["history"]
        if "train_loss" in history and "val_loss" in history:
            plot_loss_landscape(
                history["train_loss"],
                history["val_loss"],
                save_path=str(output_dir / f"{exp_name}_loss_landscape.png"),
            )
            print(f"Saved loss landscape: {output_dir / f'{exp_name}_loss_landscape.png'}")


def create_summary_table(experiments: Dict[str, Dict], output_dir: Path):
    """Create summary table of all experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp_name, exp_data in experiments.items():
        history = exp_data["history"]
        row = {
            "Experiment": exp_name,
            "Final Train Loss": history.get("train_loss", [])[-1] if history.get("train_loss") else None,
            "Final Train Acc": history.get("train_acc", [])[-1] if history.get("train_acc") else None,
            "Final Val Loss": history.get("val_loss", [])[-1] if history.get("val_loss") else None,
            "Final Val Acc": history.get("val_acc", [])[-1] if history.get("val_acc") else None,
            "Best Val Acc": max(history.get("val_acc", [0])) if history.get("val_acc") else None,
            "Overfitting Gap": (
                history.get("val_loss", [])[-1] - history.get("train_loss", [])[-1]
                if history.get("val_loss") and history.get("train_loss")
                else None
            ),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Best Val Acc", ascending=False, na_last=True)

    # Save as CSV
    csv_path = output_dir / "experiment_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table: {csv_path}")

    # Print table
    print("\n" + "=" * 100)
    print("Experiment Summary")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    return df


def create_individual_plots(experiments: Dict[str, Dict], output_dir: Path):
    """Create individual training curves for each experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, exp_data in experiments.items():
        history = exp_data["history"]

        train_metrics = {}
        val_metrics = {}

        for key in ["loss", "acc"]:
            train_key = f"train_{key}"
            val_key = f"val_{key}"

            if train_key in history:
                train_metrics[key] = history[train_key]
            if val_key in history:
                val_metrics[key] = history[val_key]

        if train_metrics and val_metrics:
            plot_training_curves(
                train_metrics,
                val_metrics,
                save_path=str(output_dir / f"{exp_name}_curves.png"),
            )
            print(f"Saved training curves: {output_dir / f'{exp_name}_curves.png'}")


def main():
    """Main analysis function."""
    args = parse_args()

    print("Collecting experiment results...")
    experiments = collect_experiment_results(
        Path(args.results_dir),
        experiment_names=args.experiments,
    )

    if not experiments:
        print("No experiments found!")
        return

    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create individual plots
    print("\nCreating individual training curves...")
    create_individual_plots(experiments, output_dir)

    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(experiments, output_dir, args.metrics)

    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(experiments, output_dir)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

