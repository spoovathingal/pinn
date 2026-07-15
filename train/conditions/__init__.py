"""Training condition losses for the 1D Euler PINN."""

from .boundary import periodic_boundary_loss
from .initial import initial_condition_loss
from .sensors import sensor_data_loss

__all__ = ["initial_condition_loss", "periodic_boundary_loss", "sensor_data_loss"]
