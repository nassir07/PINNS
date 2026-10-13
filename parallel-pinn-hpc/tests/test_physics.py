"""Sanity tests: run with `pytest -q`."""
import math

import torch

from pinn_hpc import PINN, navier_stokes_residuals, pinn_loss, sample_collocation, sample_initial, taylor_green
from pinn_hpc.analysis import amdahl, fit_amdahl


def test_exact_solution_has_zero_residual():
    """The residual code is correct iff the analytical Taylor-Green vortex gives ~0."""
    torch.manual_seed(0)
    X = torch.rand(2000, 3, dtype=torch.float64)
    X[:, :2] *= 2 * math.pi
    X.requires_grad_(True)
    for r in navier_stokes_residuals(taylor_green(X, nu=0.1), X, nu=0.1):
        assert r.abs().max().item() < 1e-10


def test_model_is_periodic_in_space():
    model = PINN(width=16, depth=2).double()
    X = torch.rand(100, 3, dtype=torch.float64)
    shift = torch.tensor([2 * math.pi, -2 * math.pi, 0.0], dtype=torch.float64)
    assert torch.allclose(model(X), model(X + shift), atol=1e-10)


def test_loss_backpropagates_to_every_parameter():
    model = PINN(width=16, depth=2)
    X_f = sample_collocation(256, 1.0, seed=0, rank=0, world=1, device="cpu")
    X_0, U_0 = sample_initial(64, 0.1, seed=0, rank=0, world=1, device="cpu")
    pinn_loss(model, X_f, X_0, U_0, nu=0.1)["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_shards_partition_the_global_set():
    """Strong scaling fairness: the union of all shards equals the single-worker point set."""
    full = sample_collocation(1000, 1.0, seed=3, rank=0, world=1, device="cpu")
    shards = [sample_collocation(1000, 1.0, seed=3, rank=r, world=4, device="cpu") for r in range(4)]
    assert sum(s.shape[0] for s in shards) == full.shape[0]
    merged = torch.cat(shards)
    key = lambda A: A[torch.argsort(A[:, 0])]
    assert torch.equal(key(merged), key(full))


def test_amdahl_fit_recovers_serial_fraction():
    N = [1, 2, 4, 8]
    assert abs(fit_amdahl(N, amdahl(N, 0.07)) - 0.07) < 1e-3
