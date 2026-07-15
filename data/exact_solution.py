"""Analytical entropy-wave solution for the 1D Euler equations."""

from __future__ import annotations

import math

import torch


def exact_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    epsilon: float = 0.2,
    u0: float = 1.0,
    p0: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rho, u, p for a smooth periodic entropy wave.

    The density perturbation is advected at constant velocity u0, while
    pressure and velocity remain spatially uniform.
    """
    phase = 2.0 * math.pi * (x - u0 * t)
    rho = 1.0 + epsilon * torch.sin(phase)
    u = u0 * torch.ones_like(x)
    p = p0 * torch.ones_like(x)
    return rho, u, p


def exact_conservative_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    gamma: float = 1.4,
    epsilon: float = 0.2,
    u0: float = 1.0,
    p0: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rho, rho*u, E for the entropy-wave solution."""
    rho, u, p = exact_solution(x, t, epsilon=epsilon, u0=u0, p0=p0)
    momentum = rho * u
    energy = p / (gamma - 1.0) + 0.5 * rho * u**2
    return rho, momentum, energy
