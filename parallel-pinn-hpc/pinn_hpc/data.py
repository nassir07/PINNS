"""Point sampling.

Every rank generates the SAME global point set from a shared seed and keeps only
its own shard (rank::world). In strong scaling the union of all shards is exactly
the single-GPU point set, so runs with different GPU counts solve the same problem
and can be compared fairly.
"""
import math

import torch

from .physics import taylor_green

TWO_PI = 2.0 * math.pi


def _shard(X: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    return X[rank::world].contiguous()


def sample_collocation(n_global: int, t_max: float, seed: int, rank: int, world: int, device):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n_global, 3, generator=g)
    X[:, :2] *= TWO_PI
    X[:, 2] *= t_max
    return _shard(X, rank, world).to(device)


def sample_initial(n_global: int, nu: float, seed: int, rank: int, world: int, device):
    g = torch.Generator().manual_seed(seed + 1)
    X = torch.rand(n_global, 3, generator=g)
    X[:, :2] *= TWO_PI
    X[:, 2] = 0.0
    X = _shard(X, rank, world)
    return X.to(device), taylor_green(X, nu).to(device)


def test_grid(nu: float, t_max: float, n_xy: int = 64, n_t: int = 5, device="cpu"):
    xs = torch.linspace(0.0, TWO_PI, n_xy)
    ts = torch.linspace(0.0, t_max, n_t)
    x, y, t = torch.meshgrid(xs, xs, ts, indexing="ij")
    X = torch.stack([x.reshape(-1), y.reshape(-1), t.reshape(-1)], dim=1)
    return X.to(device), taylor_green(X, nu).to(device)
