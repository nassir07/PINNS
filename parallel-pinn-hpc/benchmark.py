"""Run a scaling sweep: train.py for several worker counts, scaling modes and seeds.

Every argument this script does not recognise is forwarded to train.py.

Examples
--------
python benchmark.py --nprocs 1 2 4 --scaling strong weak --repeats 3 --steps 3000
python benchmark.py --launcher mpirun --nprocs 1 2 4 --mpi_args "--oversubscribe"
python benchmark.py --nprocs 1 2 --device cpu --steps 100          # CPU smoke test
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = str(ROOT / "train.py")


def default_nprocs():
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if n == 0:
        return [1, 2]
    out, k = [], 1
    while k <= n:
        out.append(k)
        k *= 2
    if out[-1] != n:
        out.append(n)
    return out


def build_cmd(launcher, n, mpi_args, train_args):
    if launcher == "torchrun":
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={n}", TRAIN, *train_args]
    return ["mpirun", "-np", str(n), *shlex.split(mpi_args), sys.executable, TRAIN, *train_args]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nprocs", type=int, nargs="+", default=None, help="worker counts (default: 1,2,4.. up to #GPUs)")
    p.add_argument("--scaling", nargs="+", choices=["strong", "weak"], default=["strong", "weak"])
    p.add_argument("--repeats", type=int, default=1, help="seeds per configuration")
    p.add_argument("--launcher", choices=["torchrun", "mpirun"], default="torchrun")
    p.add_argument("--mpi_args", default="", help='extra mpirun flags, e.g. "--oversubscribe"')
    p.add_argument("--tag", default="bench")
    p.add_argument("--out", default=str(ROOT / "results"))
    args, passthrough = p.parse_known_args()

    nprocs = args.nprocs or default_nprocs()
    runs = [(s, n, r) for s in args.scaling for n in nprocs for r in range(args.repeats)]
    print(f"{len(runs)} runs | nprocs={nprocs} | scaling={args.scaling} | launcher={args.launcher}", flush=True)

    t_start = time.time()
    for i, (scaling, n, seed) in enumerate(runs, 1):
        train_args = [*passthrough, "--scaling", scaling, "--seed", str(seed),
                      "--tag", args.tag, "--out", args.out]
        cmd = build_cmd(args.launcher, n, args.mpi_args, train_args)
        print(f"\n=== [{i}/{len(runs)}] {scaling} scaling | N={n} | seed={seed}\n$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"\nAll runs finished in {(time.time() - t_start) / 60:.1f} min. Results in {args.out}/{args.tag}/")


if __name__ == "__main__":
    main()
