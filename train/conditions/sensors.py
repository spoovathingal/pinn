"""Sparse sensor-data loss for PINN training."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def sensor_data_loss(
    model: nn.Module,
    sensor_data: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return MSE between model primitives and sparse sensor labels."""
    rho_pred, u_pred, p_pred = model(sensor_data["x"], sensor_data["t"])
    return (
        F.mse_loss(rho_pred, sensor_data["rho"])
        + F.mse_loss(u_pred, sensor_data["u"])
        + F.mse_loss(p_pred, sensor_data["p"])
    )
