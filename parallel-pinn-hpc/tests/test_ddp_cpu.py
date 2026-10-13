"""DDP correctness test on CPU (gloo backend, no GPU needed).

Claim being tested: with equal-size shards, the gradient DDP produces on 2 workers is
IDENTICAL to the single-process gradient on the full point set. This is the reason
data-parallel training is mathematically equivalent to large-batch training.
"""
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from pinn_hpc import PINN, pinn_loss, sample_collocation, sample_initial

NU, N_F, N_IC = 0.1, 512, 128


def _grads(model, rank, world):
    X_f = sample_collocation(N_F, 1.0, seed=0, rank=rank, world=world, device="cpu").double()
    X_0, U_0 = sample_initial(N_IC, NU, seed=0, rank=rank, world=world, device="cpu")
    pinn_loss(model, X_f, X_0.double(), U_0.double(), nu=NU)["total"].backward()


def _worker(rank, world, port, out_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    model = PINN(width=16, depth=2).double()
    ddp = DDP(model, broadcast_buffers=False)
    _grads(ddp, rank, world)
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    if rank == 0:
        torch.save(g, out_path)
    dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo unavailable")
def test_ddp_gradient_equals_full_batch_gradient(tmp_path):
    torch.manual_seed(0)
    ref_model = PINN(width=16, depth=2).double()
    _grads(ref_model, rank=0, world=1)
    g_ref = torch.cat([p.grad.reshape(-1) for p in ref_model.parameters()])

    out = tmp_path / "g.pt"
    mp.spawn(_worker, args=(2, _free_port(), str(out)), nprocs=2, join=True)
    g_ddp = torch.load(out)

    assert torch.allclose(g_ddp, g_ref, rtol=1e-9, atol=1e-12)
