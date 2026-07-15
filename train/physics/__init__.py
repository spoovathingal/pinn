"""Physics residual modules for PINN training."""

from .euler_1d import euler_residuals, primitive_to_conservative

__all__ = ["euler_residuals", "primitive_to_conservative"]
