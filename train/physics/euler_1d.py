"""One-dimensional inviscid Euler equations for the entropy-wave PINN."""

from __future__ import annotations

import torch
from torch import nn


def primitive_to_conservative(
    rho: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    gamma: float = 1.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert primitive variables rho, u, p to rho, rho*u, E."""
    momentum = rho * u
    energy = p / (gamma - 1.0) + 0.5 * rho * u**2
    return rho, momentum, energy


def euler_fluxes(
    rho: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    gamma: float = 1.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mass, momentum, and energy fluxes for 1D inviscid Euler."""
    _, momentum, energy = primitive_to_conservative(rho, u, p, gamma)
    return momentum, rho * u**2 + p, u * (energy + p)


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx for batched scalar network outputs."""
    (grad,) = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )
    return grad


def euler_residuals(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    gamma: float = 1.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute residuals for q_t + f(q)_x = 0."""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    rho, u, p = model(x, t)
    q1, q2, q3 = primitive_to_conservative(rho, u, p, gamma)
    f1, f2, f3 = euler_fluxes(rho, u, p, gamma)

    return (
        gradient(q1, t) + gradient(f1, x),
        gradient(q2, t) + gradient(f2, x),
        gradient(q3, t) + gradient(f3, x),
    )
