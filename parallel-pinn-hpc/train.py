"""Distributed training of a Navier-Stokes (Taylor-Green vortex) PINN.

Examples
--------
Single process:        python train.py --steps 3000
4 GPUs with torchrun:  torchrun --standalone --nproc_per_node=4 train.py --scaling strong
4 GPUs with MPI:       mpirun -np 4 python train.py --scaling strong
CPU smoke test:        torchrun --standalone --nproc_per_node=2 train.py --device cpu --steps 50
"""
import argparse
import json
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pinn_hpc import PINN, pinn_loss, relative_l2, sample_collocation, sample_initial, test_grid
from pinn_hpc.distributed import (all_gather_object, barrier, cleanup, setup_distributed,
                                  synchronize)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # physics
    p.add_argument("--nu", type=float, default=0.1, help="kinematic viscosity")
    p.add_argument("--t_max", type=float, default=1.0)
    # model
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--n_freq", type=int, default=2)
    # data
    p.add_argument("--n_colloc", type=int, default=32768,
                   help="strong scaling: GLOBAL collocation points; weak scaling: points PER RANK")
    p.add_argument("--n_ic", type=int, default=4096, help="initial-condition points (same convention)")
    p.add_argument("--scaling", choices=["strong", "weak"], default="strong")
    p.add_argument("--resample_every", type=int, default=0, help="redraw points every k steps (0 = fixed)")
    # optimisation
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3, help="base learning rate for ONE worker")
    p.add_argument("--lr_scaling", choices=["none", "linear", "sqrt"], default="sqrt",
                   help="scale the LR with the number of workers (effective batch grows with N)")
    p.add_argument("--warmup", type=int, default=200, help="linear LR warm-up steps")
    p.add_argument("--w_ic", type=float, default=10.0, help="weight of the initial-condition loss")
    # measurement
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--target_l2", type=float, default=1e-2, help="accuracy target for time-to-accuracy")
    p.add_argument("--timing_warmup", type=int, default=20, help="first steps excluded from mean step time")
    p.add_argument("--probe_steps", type=int, default=20, help="steps used to estimate communication cost")
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--no_save_model", action="store_true")
    return p.parse_args(argv)


def lr_factor(world: int, mode: str) -> float:
    return {"none": 1.0, "linear": float(world), "sqrt": math.sqrt(world)}[mode]


def make_schedule(warmup: int, steps: int):
    """Linear warm-up, then cosine decay to 10% of the peak learning rate."""
    def f(s):
        w = min(1.0, (s + 1) / warmup) if warmup > 0 else 1.0
        return w * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(s, steps) / steps)))
    return f


def communication_probe(ctx, ddp, batch, args):
    """Estimate communication cost: time forward+backward with and without gradient all-reduce.

    DDP.no_sync() skips the all-reduce, so (t_sync - t_nosync) approximates the part of
    a step spent communicating that is NOT hidden behind computation.
    """
    if not ctx.distributed or args.probe_steps <= 0:
        return {"t_sync": None, "t_nosync": None, "comm_fraction": 0.0}
    res = {}
    for mode in ("nosync", "sync"):
        times = []
        for i in range(args.probe_steps + 3):
            ddp.zero_grad(set_to_none=True)
            barrier(ctx)
            synchronize(ctx)
            t0 = time.perf_counter()
            with ddp.no_sync() if mode == "nosync" else nullcontext():
                pinn_loss(ddp, *batch, nu=args.nu, w_ic=args.w_ic)["total"].backward()
            synchronize(ctx)
            if i >= 3:  # discard a few warm-up iterations
                times.append(time.perf_counter() - t0)
        res[f"t_{mode}"] = sum(times) / len(times)
    ddp.zero_grad(set_to_none=True)
    res["comm_fraction"] = max(0.0, (res["t_sync"] - res["t_nosync"]) / res["t_sync"])
    return res


def params_in_sync(ctx, model) -> float:
    """Max absolute difference between this rank's weights and rank 0's (should be 0)."""
    if not ctx.distributed:
        return 0.0
    vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    ref = vec.clone()
    dist.broadcast(ref, src=0)
    diff = (vec - ref).abs().max()
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    return diff.item()


def main(argv=None):
    args = parse_args(argv)
    ctx = setup_distributed(args.device)
    torch.manual_seed(args.seed)  # same init everywhere (DDP also broadcasts rank 0's weights)

    # ---- problem size per rank --------------------------------------------------------
    if args.scaling == "strong":
        n_f_global, n_ic_global = args.n_colloc, args.n_ic
    else:
        n_f_global, n_ic_global = args.n_colloc * ctx.world, args.n_ic * ctx.world

    X_f = sample_collocation(n_f_global, args.t_max, args.seed, ctx.rank, ctx.world, ctx.device)
    X_0, U_0 = sample_initial(n_ic_global, args.nu, args.seed, ctx.rank, ctx.world, ctx.device)

    # ---- model, DDP wrapper, optimiser -------------------------------------------------
    model = PINN(args.width, args.depth, args.n_freq, args.t_max).to(ctx.device)
    if ctx.distributed:
        ddp = DDP(model, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                  broadcast_buffers=False)
    else:
        ddp = model

    peak_lr = args.lr * lr_factor(ctx.world, args.lr_scaling)
    opt = torch.optim.Adam(ddp.parameters(), lr=peak_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_schedule(args.warmup, args.steps))

    X_test, U_test = test_grid(args.nu, args.t_max, device=ctx.device) if ctx.is_main else (None, None)

    if ctx.is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{args.scaling}] world={ctx.world} launcher={ctx.launcher} backend={ctx.backend} "
              f"device={ctx.device} params={n_params} points/rank={X_f.shape[0]} "
              f"global points={n_f_global} peak_lr={peak_lr:.2e}", flush=True)

    # ---- training loop -----------------------------------------------------------------
    step_times, history = [], []
    train_time, time_to_target, step_to_target = 0.0, None, None

    for step in range(1, args.steps + 1):
        if args.resample_every and step % args.resample_every == 0:
            X_f = sample_collocation(n_f_global, args.t_max, args.seed + 1000 * step,
                                     ctx.rank, ctx.world, ctx.device)

        synchronize(ctx)
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        losses = pinn_loss(ddp, X_f, X_0, U_0, nu=args.nu, w_ic=args.w_ic)
        losses["total"].backward()  # DDP all-reduces (averages) gradients here
        opt.step()
        sched.step()

        synchronize(ctx)
        dt = time.perf_counter() - t0
        step_times.append(dt)
        train_time += dt

        # ---- logging (excluded from the timers) ----------------------------------------
        if step % args.log_every == 0 or step == args.steps:
            vals = torch.stack([losses[k].detach() for k in ("total", "pde", "ic")])
            if ctx.distributed:
                dist.all_reduce(vals)
                vals /= ctx.world
            if ctx.is_main:
                # Evaluate the unwrapped module: a DDP forward on rank 0 alone could deadlock.
                err = relative_l2(model, X_test, U_test)
                rec = {"step": step, "train_time": train_time, "lr": sched.get_last_lr()[0],
                       "loss": vals[0].item(), "loss_pde": vals[1].item(), "loss_ic": vals[2].item(),
                       "l2_u": err["u"], "l2_v": err["v"], "l2_p": err["p"]}
                history.append(rec)
                if time_to_target is None and max(err["u"], err["v"]) < args.target_l2:
                    time_to_target, step_to_target = train_time, step
                print(f"step {step:6d} | t={train_time:8.2f}s | loss={rec['loss']:.3e} "
                      f"| L2 u={err['u']:.2e} v={err['v']:.2e} p={err['p']:.2e}", flush=True)
            barrier(ctx)  # keep ranks aligned so evaluation time never leaks into step timings

    # ---- post-training measurements ----------------------------------------------------
    probe = communication_probe(ctx, ddp, (X_f, X_0, U_0), args)
    sync_diff = params_in_sync(ctx, model)
    k = min(args.timing_warmup, len(step_times) - 1)
    mean_step = sum(step_times[k:]) / len(step_times[k:])
    per_rank_step = all_gather_object(ctx, mean_step)

    if ctx.is_main:
        final = relative_l2(model, X_test, U_test)
        result = {
            "config": vars(args),
            "env": {"world": ctx.world, "launcher": ctx.launcher, "backend": ctx.backend,
                    "device": str(ctx.device),
                    "device_name": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else platform.processor() or "cpu",
                    "torch": torch.__version__, "host": platform.node()},
            "points_per_rank": int(X_f.shape[0]), "global_points": int(n_f_global),
            "peak_lr": peak_lr,
            "mean_step_time": mean_step,
            "mean_step_time_per_rank": per_rank_step,
            "train_time": train_time,
            "throughput_pts_per_s": n_f_global / max(per_rank_step),
            "time_to_target": time_to_target, "step_to_target": step_to_target,
            "final_l2": final, "final_loss": history[-1]["loss"],
            "comm_probe": probe,
            "param_max_abs_diff_across_ranks": sync_diff,
            "history": history,
        }
        out_dir = Path(args.out) / args.tag / args.scaling
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"N{ctx.world}_seed{args.seed}"
        (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
        if not args.no_save_model:
            torch.save(model.state_dict(), out_dir / f"{stem}.pt")
        print(f"done: mean step {mean_step * 1e3:.2f} ms | train {train_time:.1f}s | "
              f"L2 u={final['u']:.2e} v={final['v']:.2e} | comm~{probe['comm_fraction']:.0%} "
              f"| weights in sync: max diff={sync_diff:.1e} -> {out_dir / stem}.json", flush=True)

    cleanup(ctx)


if __name__ == "__main__":
    main()
