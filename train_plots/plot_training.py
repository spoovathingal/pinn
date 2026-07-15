"""Plot PINN training and validation losses from a checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
(PROJECT_ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_CHECKPOINT = PROJECT_ROOT / "best_model.pt"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "train_plots"


def load_loss_history(checkpoint_path: Path) -> list[dict[str, float]]:
    """Load loss history saved by train.py."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get("loss_history", [])
    if not history:
        raise ValueError(f"No loss_history found in {checkpoint_path}")
    return history


def plot_losses(history: list[dict[str, float]], output_dir: Path) -> None:
    """Create linear and log-scale training-loss figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]
    loss_names = [name for name in history[0] if name != "epoch"]

    for log_scale, suffix in ((False, "linear"), (True, "log")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in loss_names:
            values = [entry[name] for entry in history if name in entry]
            valid_epochs = [entry["epoch"] for entry in history if name in entry]
            ax.plot(valid_epochs, values, label=name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"training_losses_{suffix}.png", dpi=200)
        plt.close(fig)

    if "validation" in loss_names:
        best_entry = min(history, key=lambda entry: entry.get("validation", float("inf")))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, [entry["validation"] for entry in history], label="validation")
        ax.axvline(best_entry["epoch"], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "validation_loss.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_loss_history(args.checkpoint)
    plot_losses(history, args.output_dir)
    print(f"Saved training plots to {args.output_dir}")


if __name__ == "__main__":
    main()
