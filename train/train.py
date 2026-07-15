"""Train a PINN for the smooth 1D Euler entropy-wave problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from data_io import load_npz_dataset
from losses.assembly import total_loss
from model import EulerPINN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "generated"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT


def validation_loss(model: EulerPINN, validation: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return primitive-variable MSE on the held-out validation grid."""
    model.eval()
    with torch.no_grad():
        rho_pred, u_pred, p_pred = model(validation["x"], validation["t"])
        return (
            F.mse_loss(rho_pred, validation["rho"])
            + F.mse_loss(u_pred, validation["u"])
            + F.mse_loss(p_pred, validation["p"])
        )


def save_checkpoint(
    path: Path,
    model: EulerPINN,
    optimizer: torch.optim.Optimizer,
    config: dict,
    loss_history: list[dict[str, float]],
    best_validation_loss: float,
    epoch: int,
) -> None:
    """Save a training checkpoint for later plotting or continued training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "loss_history": loss_history,
            "best_validation_loss": best_validation_loss,
            "epoch": epoch,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    """Run PINN training."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data_dir
    collocation = load_npz_dataset(data_dir / "collocation.npz", device=device)
    initial_condition = load_npz_dataset(data_dir / "initial_condition.npz", device=device)
    boundary = load_npz_dataset(data_dir / "boundary_condition.npz", device=device)
    validation = load_npz_dataset(data_dir / "validation_grid.npz", device=device)

    sensor_path = data_dir / "sensor_data.npz"
    sensor_data = load_npz_dataset(sensor_path, device=device) if sensor_path.exists() else None

    config = {
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_hidden_layers,
        "rho_min": args.rho_min,
        "p_min": args.p_min,
        "gamma": args.gamma,
    }
    model = EulerPINN(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        rho_min=args.rho_min,
        p_min=args.p_min,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    weights = {
        "physics": args.physics_weight,
        "initial": args.initial_weight,
        "boundary": args.boundary_weight,
        "data": args.data_weight,
    }

    checkpoint_path = args.checkpoint_dir / "best_model.pt"
    final_checkpoint_path = args.checkpoint_dir / "final_model.pt"
    loss_history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = total_loss(
            model,
            collocation,
            initial_condition,
            boundary,
            sensor_data=sensor_data,
            gamma=args.gamma,
            weights=weights,
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            val = validation_loss(model, validation)
            record = {
                "epoch": float(epoch),
                **{name: float(value.cpu()) for name, value in parts.items()},
                "validation": float(val.cpu()),
            }
            loss_history.append(record)

            print(
                f"epoch={epoch:06d} "
                f"total={record['total']:.6e} "
                f"physics={record['physics']:.6e} "
                f"initial={record['initial']:.6e} "
                f"boundary={record['boundary']:.6e} "
                f"validation={record['validation']:.6e}"
            )

            if record["validation"] < best_val:
                best_val = record["validation"]
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    config,
                    loss_history,
                    best_val,
                    epoch,
                )

    save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        config,
        loss_history,
        best_val,
        args.epochs,
    )
    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Saved final checkpoint to {final_checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--rho-min", type=float, default=1.0e-6)
    parser.add_argument("--p-min", type=float, default=1.0e-6)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--initial-weight", type=float, default=10.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
