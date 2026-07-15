"""Plot PINN predictions against the validation-grid exact solution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
(PROJECT_ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.model import EulerPINN

DEFAULT_CHECKPOINT = PROJECT_ROOT / "best_model.pt"
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "generated" / "validation_grid.npz"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "predict_plots"


def load_model(checkpoint_path: Path, device: torch.device) -> EulerPINN:
    """Rebuild and load a trained EulerPINN checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = EulerPINN(
        hidden_dim=config.get("hidden_dim", 64),
        num_hidden_layers=config.get("num_hidden_layers", 4),
        rho_min=config.get("rho_min", 1.0e-6),
        p_min=config.get("p_min", 1.0e-6),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_on_validation(
    model: EulerPINN,
    validation_path: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Evaluate the model at validation-grid points."""
    data = np.load(validation_path)
    x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
    t = torch.as_tensor(data["t"], dtype=torch.float32, device=device)
    with torch.no_grad():
        rho_pred, u_pred, p_pred = model(x, t)

    return {
        "x": data["x"].reshape(-1),
        "t": data["t"].reshape(-1),
        "rho_exact": data["rho"].reshape(-1),
        "u_exact": data["u"].reshape(-1),
        "p_exact": data["p"].reshape(-1),
        "rho_pred": rho_pred.cpu().numpy().reshape(-1),
        "u_pred": u_pred.cpu().numpy().reshape(-1),
        "p_pred": p_pred.cpu().numpy().reshape(-1),
    }


def infer_grid(state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Infer sorted unique x and t coordinates from flattened validation data."""
    x_values = np.unique(state["x"])
    t_values = np.unique(state["t"])
    return x_values, t_values


def reshape_field(values: np.ndarray, nt: int, nx: int) -> np.ndarray:
    """Reshape flattened field values onto the generated t-x grid."""
    return values.reshape(nt, nx)


def plot_profiles(
    state: dict[str, np.ndarray],
    output_dir: Path,
    time_slices: list[float],
) -> None:
    """Plot predicted and exact rho, u, p profiles at selected times."""
    x_values, t_values = infer_grid(state)
    nx = len(x_values)
    nt = len(t_values)
    variables = ("rho", "u", "p")

    for variable in variables:
        exact = reshape_field(state[f"{variable}_exact"], nt, nx)
        pred = reshape_field(state[f"{variable}_pred"], nt, nx)

        fig, axes = plt.subplots(len(time_slices), 1, figsize=(8, 2.2 * len(time_slices)), sharex=True)
        if len(time_slices) == 1:
            axes = [axes]

        for ax, requested_time in zip(axes, time_slices):
            tidx = int(np.argmin(np.abs(t_values - requested_time)))
            ax.plot(x_values, exact[tidx], color="black", linewidth=1.5, label="exact")
            ax.plot(x_values, pred[tidx], color="tab:blue", linestyle="--", label="PINN")
            ax.set_ylabel(variable)
            ax.set_title(f"t = {t_values[tidx]:.3f}")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("x")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{variable}_profiles.png", dpi=200)
        plt.close(fig)


def plot_error_heatmaps(state: dict[str, np.ndarray], output_dir: Path) -> None:
    """Plot absolute error heatmaps over the x-t domain."""
    x_values, t_values = infer_grid(state)
    nx = len(x_values)
    nt = len(t_values)

    for variable in ("rho", "u", "p"):
        exact = reshape_field(state[f"{variable}_exact"], nt, nx)
        pred = reshape_field(state[f"{variable}_pred"], nt, nx)
        error = np.abs(pred - exact)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        image = ax.imshow(
            error,
            origin="lower",
            aspect="auto",
            extent=[x_values.min(), x_values.max(), t_values.min(), t_values.max()],
        )
        fig.colorbar(image, ax=ax, label=f"|{variable} error|")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"{variable} absolute error")
        fig.tight_layout()
        fig.savefig(output_dir / f"{variable}_error_heatmap.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--times", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    state = predict_on_validation(model, args.validation, device)
    plot_profiles(state, args.output_dir, args.times)
    plot_error_heatmaps(state, args.output_dir)
    print(f"Saved prediction plots to {args.output_dir}")


if __name__ == "__main__":
    main()
