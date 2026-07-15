"""Loss assembly utilities for PINN training."""

from .assembly import physics_loss, total_loss

__all__ = ["physics_loss", "total_loss"]
