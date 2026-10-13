"""Launcher-agnostic distributed setup.

Supported launches
------------------
* torchrun / python -m torch.distributed.run
      RANK, WORLD_SIZE and LOCAL_RANK are read from the environment.
* mpirun -np N python train.py
      mpi4py discovers rank and size, derives the local rank from hostnames and
      broadcasts a rendezvous address/port so PyTorch can build its process group.
      This is how jobs are typically launched on HPC clusters.
* python train.py
      single process, no DDP.
"""
import os
import socket
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist


@dataclass
class DistContext:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    backend: Optional[str]
    launcher: str
    comm: Any = None  # mpi4py communicator when launched with mpirun

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world > 1


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _mpi_bootstrap():
    """Fill in what torch.distributed needs using MPI. Returns (comm, rank, world, local_rank)."""
    try:
        from mpi4py import MPI
    except Exception:  # mpi4py or an MPI runtime is not installed
        return None, 0, 1, 0

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    if world == 1:
        return comm, 0, 1, 0

    hosts = comm.allgather(socket.gethostname())
    local_rank = hosts[:rank].count(hosts[rank])  # index among the ranks on this node

    payload = None
    if rank == 0:
        addr = "127.0.0.1" if len(set(hosts)) == 1 else socket.gethostbyname(socket.gethostname())
        payload = (addr, _free_port())
    addr, port = comm.bcast(payload, root=0)

    os.environ.update(
        MASTER_ADDR=addr, MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
        LOCAL_RANK=str(local_rank), LOCAL_WORLD_SIZE=str(hosts.count(hosts[rank])),
    )
    return comm, rank, world, local_rank


def setup_distributed(device: str = "auto") -> DistContext:
    comm = None
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        launcher = "torchrun"
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        comm, rank, world, local_rank = _mpi_bootstrap()
        launcher = "mpi" if world > 1 else "single"

    use_cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
    if use_cuda:
        n_gpu = torch.cuda.device_count()
        if local_rank >= n_gpu:
            raise RuntimeError(f"local rank {local_rank} needs a GPU but only {n_gpu} are visible")
        torch.cuda.set_device(local_rank)
        dev, backend = torch.device("cuda", local_rank), "nccl"
    else:
        dev, backend = torch.device("cpu"), "gloo"
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // local_world))

    if world > 1:
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world)
    else:
        backend = None
    return DistContext(rank, world, local_rank, dev, backend, launcher, comm)


def all_gather_object(ctx: DistContext, obj):
    """Gather a picklable object from every rank (via MPI when available)."""
    if not ctx.distributed:
        return [obj]
    if ctx.comm is not None:
        return ctx.comm.allgather(obj)
    out = [None] * ctx.world
    dist.all_gather_object(out, obj)
    return out


def barrier(ctx: DistContext):
    if ctx.distributed:
        dist.barrier()


def synchronize(ctx: DistContext):
    """Wait for queued GPU kernels so wall-clock timings are real (CUDA runs asynchronously)."""
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)


def cleanup(ctx: DistContext):
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()
