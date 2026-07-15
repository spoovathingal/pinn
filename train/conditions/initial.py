"""Initial-condition loss for PINN training."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def initial_condition_loss(
    model: nn.Module,
    initial_condition: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return MSE between model primitives and initial-condition labels."""
    rho_pred, u_pred, p_pred = model(initial_condition["x"], initial_condition["t"])
    return (
        F.mse_loss(rho_pred, initial_condition["rho"])
        + F.mse_loss(u_pred, initial_condition["u"])
        + F.mse_loss(p_pred, initial_condition["p"])
    )
