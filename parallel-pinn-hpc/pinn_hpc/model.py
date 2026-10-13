"""Fully connected PINN with a periodic Fourier input embedding."""
import torch
import torch.nn as nn


class PINN(nn.Module):
    """Maps (x, y, t) -> (u, v, p).

    x and y enter through [cos(kx), sin(kx), cos(ky), sin(ky)] for k = 1..n_freq,
    so the network is 2*pi-periodic in space *by construction*: the periodic
    boundary conditions of the Taylor-Green vortex hold exactly and need no
    boundary loss term. Time is rescaled to [-1, 1].
    """

    def __init__(self, width: int = 64, depth: int = 5, n_freq: int = 2, t_max: float = 1.0):
        super().__init__()
        self.t_max = float(t_max)
        # Constant buffer. DDP is built with broadcast_buffers=False, so it is never re-sent.
        self.register_buffer("freqs", torch.arange(1, n_freq + 1, dtype=torch.float32))

        layers, d_in = [], 4 * n_freq + 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 3))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        k = self.freqs.to(X.dtype)
        kx, ky = x * k, y * k
        t_n = 2.0 * t / self.t_max - 1.0
        return torch.cat([torch.cos(kx), torch.sin(kx), torch.cos(ky), torch.sin(ky), t_n], dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(X))
