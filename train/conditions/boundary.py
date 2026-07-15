"""Boundary-condition losses for PINN training."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def periodic_boundary_loss(
    model: nn.Module,
    boundary: Mapping[str, torch.Tensor],
    x_left: float = 0.0,
    x_right: float = 1.0,
) -> torch.Tensor:
    """Enforce periodic primitive variables at x_left and x_right.

    The boundary file only needs a column named ``t``.
    """
    t = boundary["t"]
    left = torch.full_like(t, x_left)
    right = torch.full_like(t, x_right)

    rho_l, u_l, p_l = model(left, t)
    rho_r, u_r, p_r = model(right, t)
    return (
        F.mse_loss(rho_l, rho_r)
        + F.mse_loss(u_l, u_r)
        + F.mse_loss(p_l, p_r)
    )
