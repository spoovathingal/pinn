"""Neural-network model for the 1D Euler PINN."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EulerPINN(nn.Module):
    """MLP that maps space-time coordinates to primitive Euler variables."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_hidden_layers: int = 4,
        rho_min: float = 1.0e-6,
        p_min: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 3))

        self.net = nn.Sequential(*layers)
        self.rho_min = rho_min
        self.p_min = p_min

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return rho, u, p at coordinates x and t.

        x and t should both have shape (N, 1). Density and pressure are passed
        through softplus so the Euler state remains physically admissible.
        """
        coords = torch.cat((x, t), dim=1)
        raw_rho, raw_u, raw_p = self.net(coords).split(1, dim=1)

        rho = self.rho_min + F.softplus(raw_rho)
        u = raw_u
        p = self.p_min + F.softplus(raw_p)
        return rho, u, p
