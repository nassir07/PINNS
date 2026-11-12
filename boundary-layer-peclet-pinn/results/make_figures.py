"""Rebuilds the summary figures from results_495_runs.csv (no training needed).

    python results/make_figures.py
"""
import os
from math import erf, sqrt, pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "images")
os.makedirs(IMG, exist_ok=True)
df = pd.read_csv(os.path.join(HERE, "results_495_runs.csv"))


def thickness(flow, Pe):
    """Slope thickness delta* = (jump in c) / (max slope at the wall)."""
    if flow == "outflow layer":                       # layer ~ 1/Pe
        return 2.0 / (Pe + np.sqrt(Pe ** 2 + 4 * np.pi ** 2))
    a = sqrt(Pe / 2.0)                                # stagnation flow: layer ~ 1/sqrt(Pe)
    return 1.0 / (2.0 * a / sqrt(pi) / erf(a))


df["h_over_delta"] = [(1 / n) / thickness(f, Pe) for f, n, Pe in zip(df.flow, df.n, df.Pe)]
df["cell_Pe"] = [(1.0 if f == "outflow layer" else np.sqrt(2)) * (1 / n) * Pe for f, n, Pe in zip(df.flow, df.n, df.Pe)]
df["success"] = df.wall_err < 0.05

plain = df[df.method == "plain"]
colors = {"outflow layer": "#2a78d6", "stagnation flow": "#eb6834"}

# ---- Figure 1: which rule predicts failure?
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, col, label, title in [
        (axes[0], "h_over_delta", "point spacing / layer thickness   (h / δ*)", "Points per layer: both flows switch at ~1.5"),
        (axes[1], "cell_Pe", "cell Péclet number   (|u|max · h · Pe)", "CFD cell Péclet: thresholds ~100× apart")]:
    for flow, g in plain.groupby("flow"):
        cases = g.groupby(["Pe", "n"]).agg(x=(col, "first"), rate=("success", "mean")).reset_index()
        ax.scatter(cases.x, cases.rate, s=55, color=colors[flow], alpha=0.8, label=flow)
        fit = LogisticRegression(C=1e6).fit(np.log10(g[[col]].values), g.success.values)
        xs = np.logspace(np.log10(g[col].min()) - 0.3, np.log10(g[col].max()) + 0.3, 200)
        ax.plot(xs, fit.predict_proba(np.log10(xs)[:, None])[:, 1], color=colors[flow])
        x50 = 10 ** (-fit.intercept_[0] / fit.coef_[0, 0])
        ax.axvline(x50, color=colors[flow], ls=":")
        ax.text(x50 * 1.08, 0.52 if flow == "outflow layer" else 0.42, f"{x50:.2f}", color=colors[flow])
    ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("fraction of seeds with wall-flux error < 5%")
axes[0].legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "rule_points_per_layer_vs_cell_peclet.png"), dpi=130)

# ---- Figure 2: how the plain PINN ends, and what the fix changes
def outcome(r):
    if r.success:
        return "success"
    return "overfit between points" if r.gap > 1 else "smooth wrong state"

order = ["success", "overfit between points", "smooth wrong state"]
cols = {"success": "#1c7c4a", "overfit between points": "#eb6834", "smooth wrong state": "#9a9a9a"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, flow in zip(axes, ["stagnation flow", "outflow layer"]):
    g = df[df.flow == flow].copy()
    g["outcome"] = g.apply(outcome, axis=1)
    methods = ["plain", "zoom only", "curriculum only", "zoom + curriculum"]
    tab = g.groupby(["method", "outcome"]).size().unstack(fill_value=0).reindex(methods).reindex(columns=order, fill_value=0)
    frac = tab.div(tab.sum(axis=1), axis=0)
    left = np.zeros(len(methods))
    for o in order:
        ax.barh(methods, frac[o], left=left, color=cols[o], label=o, height=0.6)
        left += frac[o].values
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title(flow, loc="left")
    ax.set_xlabel("fraction of runs")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig(os.path.join(IMG, "outcomes_by_method.png"), dpi=130)
print("figures written to", os.path.abspath(IMG))
