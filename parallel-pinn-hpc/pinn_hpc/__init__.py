"""Parallel (multi-GPU) training of a Navier-Stokes PINN with PyTorch DDP and mpi4py."""
from .model import PINN
from .physics import taylor_green, navier_stokes_residuals, pinn_loss, relative_l2
from .data import sample_collocation, sample_initial, test_grid

__all__ = [
    "PINN", "taylor_green", "navier_stokes_residuals", "pinn_loss",
    "relative_l2", "sample_collocation", "sample_initial", "test_grid",
]
__version__ = "1.0.0"
