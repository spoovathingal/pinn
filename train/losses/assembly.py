"""Assemble weighted PINN training losses."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

try:
    from ..conditions.boundary import periodic_boundary_loss
    from ..conditions.initial import initial_condition_loss
    from ..conditions.sensors import sensor_data_loss
    from ..physics.euler_1d import euler_residuals
except ImportError:
    from conditions.boundary import periodic_boundary_loss
    from conditions.initial import initial_condition_loss
    from conditions.sensors import sensor_data_loss
    from physics.euler_1d import euler_residuals


def physics_loss(
    model: nn.Module,
    collocation: Mapping[str, torch.Tensor],
    gamma: float = 1.4,
) -> torch.Tensor:
    """Mean-squared Euler residual loss at interior collocation points."""
    r_mass, r_momentum, r_energy = euler_residuals(
        model,
        collocation["x"],
        collocation["t"],
        gamma=gamma,
    )
    return torch.mean(r_mass**2) + torch.mean(r_momentum**2) + torch.mean(r_energy**2)


def total_loss(
    model: nn.Module,
    collocation: Mapping[str, torch.Tensor],
    initial_condition: Mapping[str, torch.Tensor],
    boundary: Mapping[str, torch.Tensor],
    sensor_data: Mapping[str, torch.Tensor] | None = None,
    gamma: float = 1.4,
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return weighted total loss and individual detached components."""
    weights = {
        "physics": 1.0,
        "initial": 10.0,
        "boundary": 1.0,
        "data": 1.0,
        **(weights or {}),
    }

    components = {
        "physics": physics_loss(model, collocation, gamma=gamma),
        "initial": initial_condition_loss(model, initial_condition),
        "boundary": periodic_boundary_loss(model, boundary),
    }
    if sensor_data is not None:
        components["data"] = sensor_data_loss(model, sensor_data)

    loss = sum(weights[name] * value for name, value in components.items())
    detached = {name: value.detach() for name, value in components.items()}
    detached["total"] = loss.detach()
    return loss, detached
