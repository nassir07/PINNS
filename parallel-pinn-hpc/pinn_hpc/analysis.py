"""Helpers to load benchmark results and compute scaling metrics (used by the notebook)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(result_dir) -> list:
    """Load every JSON result under result_dir (recursively)."""
    return [json.loads(f.read_text()) for f in sorted(Path(result_dir).rglob("*.json"))]


def runs_to_frame(runs: list) -> pd.DataFrame:
    rows = []
    for r in runs:
        c = r["config"]
        rows.append({
            "scaling": c["scaling"], "N": r["env"]["world"], "seed": c["seed"],
            "step_ms": 1e3 * max(r["mean_step_time_per_rank"]),  # slowest rank sets the pace
            "train_s": r["train_time"],
            "throughput": r["throughput_pts_per_s"],
            "tta_s": r["time_to_target"],
            "l2_u": r["final_l2"]["u"], "l2_v": r["final_l2"]["v"], "l2_p": r["final_l2"]["p"],
            "final_loss": r["final_loss"],
            "comm_frac": r["comm_probe"]["comm_fraction"],
            "sync_diff": r["param_max_abs_diff_across_ranks"],
            "peak_lr": r["peak_lr"],
            "pts_per_rank": r["points_per_rank"],
            "device": r["env"]["device_name"],
        })
    return pd.DataFrame(rows).sort_values(["scaling", "N", "seed"]).reset_index(drop=True)


def scaling_table(df: pd.DataFrame, scaling: str) -> pd.DataFrame:
    """Mean/std per worker count plus speedup, efficiency and time-to-accuracy speedup.

    strong scaling: speedup = T(1) / T(N), efficiency = speedup / N
    weak scaling:   efficiency = T(1) / T(N)  (ideal: step time stays constant)
    """
    d = df[df.scaling == scaling]
    if d.empty:
        return d
    g = d.groupby("N").agg(
        step_ms=("step_ms", "mean"), step_ms_std=("step_ms", "std"),
        throughput=("throughput", "mean"), comm_frac=("comm_frac", "mean"),
        tta_s=("tta_s", "mean"), l2_u=("l2_u", "mean"), l2_v=("l2_v", "mean"), l2_p=("l2_p", "mean"),
        sync_diff=("sync_diff", "max"), runs=("seed", "count"),
    ).reset_index()
    base = g.loc[g.N == g.N.min()].iloc[0]
    n0 = base.N
    if scaling == "strong":
        g["speedup"] = base.step_ms / g.step_ms * n0
        g["efficiency"] = g.speedup / g.N
        g["tta_speedup"] = base.tta_s / g.tta_s * n0 if pd.notna(base.tta_s) else np.nan
    else:
        g["efficiency"] = base.step_ms / g.step_ms
        g["scaled_speedup"] = g.efficiency * g.N / n0
    return g


def amdahl(N, f):
    """Amdahl's law: speedup with serial (non-parallelisable) fraction f."""
    N = np.asarray(N, dtype=float)
    return 1.0 / (f + (1.0 - f) / N)


def fit_amdahl(N, speedup) -> float:
    """Least-squares estimate of the serial fraction f (grid search, robust and dependency-free)."""
    fs = np.linspace(0.0, 1.0, 10001)
    N, S = np.asarray(N, float), np.asarray(speedup, float)
    errs = [np.sum((amdahl(N, f) - S) ** 2) for f in fs]
    return float(fs[int(np.argmin(errs))])
