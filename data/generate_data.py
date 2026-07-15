"""Generate file-backed data for the 1D Euler entropy-wave PINN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from data.exact_solution import exact_solution
except ModuleNotFoundError:
    from exact_solution import exact_solution


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor and return a float32 numpy array."""
    return tensor.detach().cpu().numpy().astype(np.float32)


def save_supervised_dataset(
    path: Path,
    x: torch.Tensor,
    t: torch.Tensor,
    epsilon: float,
    u0: float,
    p0: float,
) -> None:
    """Save x, t, rho, u, p arrays to an .npz file."""
    rho, u, p = exact_solution(x, t, epsilon=epsilon, u0=u0, p0=p0)
    np.savez(
        path,
        x=to_numpy(x),
        t=to_numpy(t),
        rho=to_numpy(rho),
        u=to_numpy(u),
        p=to_numpy(p),
    )


def generate_entropy_wave_data(
    output_dir: Path,
    n_initial: int = 128,
    n_boundary: int = 128,
    n_sensor: int = 50,
    n_collocation: int = 1000,
    n_validation_x: int = 200,
    n_validation_t: int = 100,
    t_final: float = 1.0,
    epsilon: float = 0.2,
    u0: float = 1.0,
    p0: float = 1.0,
    seed: int = 1234,
) -> None:
    """Generate initial, boundary, sensor, and collocation .npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    x_initial = torch.linspace(0.0, 1.0, n_initial).reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    save_supervised_dataset(
        output_dir / "initial_condition.npz",
        x_initial,
        t_initial,
        epsilon,
        u0,
        p0,
    )

    t_boundary = torch.linspace(0.0, t_final, n_boundary).reshape(-1, 1)
    np.savez(output_dir / "boundary_condition.npz", t=to_numpy(t_boundary))

    x_collocation = torch.rand((n_collocation, 1), generator=generator)
    t_collocation = t_final * torch.rand((n_collocation, 1), generator=generator)
    np.savez(
        output_dir / "collocation.npz",
        x=to_numpy(x_collocation),
        t=to_numpy(t_collocation),
    )

    if n_sensor > 0:
        x_sensor = torch.rand((n_sensor, 1), generator=generator)
        t_sensor = t_final * torch.rand((n_sensor, 1), generator=generator)
        save_supervised_dataset(
            output_dir / "sensor_data.npz",
            x_sensor,
            t_sensor,
            epsilon,
            u0,
            p0,
        )

    x_validation = torch.linspace(0.0, 1.0, n_validation_x)
    t_validation = torch.linspace(0.0, t_final, n_validation_t)
    tt, xx = torch.meshgrid(t_validation, x_validation, indexing="ij")
    save_supervised_dataset(
        output_dir / "validation_grid.npz",
        xx.reshape(-1, 1),
        tt.reshape(-1, 1),
        epsilon,
        u0,
        p0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-initial", type=int, default=128)
    parser.add_argument("--n-boundary", type=int, default=128)
    parser.add_argument("--n-sensor", type=int, default=50)
    parser.add_argument("--n-collocation", type=int, default=1000)
    parser.add_argument("--n-validation-x", type=int, default=200)
    parser.add_argument("--n-validation-t", type=int, default=100)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--p0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_entropy_wave_data(
        output_dir=args.output_dir,
        n_initial=args.n_initial,
        n_boundary=args.n_boundary,
        n_sensor=args.n_sensor,
        n_collocation=args.n_collocation,
        n_validation_x=args.n_validation_x,
        n_validation_t=args.n_validation_t,
        t_final=args.t_final,
        epsilon=args.epsilon,
        u0=args.u0,
        p0=args.p0,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
