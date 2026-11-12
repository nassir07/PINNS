"""
Figures for the collocation-Péclet study.

    python plots.py --results results_dir [results_dir2 ...] --out figs
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
C_OK, C_OVER, C_OPT = "#1baf7a", "#eb6834", "#2a78d6"     # success / overfitting / optimisation failure
SUCCESS = 0.05       # rel L2 threshold
GAP_OVER = 1.0       # log10(test/train residual) above this = overfitting between points

NAMES = {"vanilla": "Vanilla PINN", "hard": "Hard BCs", "gradreg": "Gradient penalty",
         "rad": "RAD sampling", "xi": "Layer coordinate ξ", "xi_hard": "ξ + hard BCs"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 10, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def load(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "results.jsonl"), recursive=True):
            for line in open(f):
                r = json.loads(line); r.pop("hist", None); rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates(["method", "prec", "N", "Pe", "seed"])
    df["ok"] = df.relL2 < SUCCESS
    df["regime"] = np.where(df.ok, 0, np.where(df.gap > GAP_OVER, 1, 2))
    return df


def agg(df):
    g = df.groupby(["method", "prec", "Pe", "N"])
    a = g.agg(relL2=("relL2", "median"), gap=("gap", "median"), flux=("flux_err", "median"),
              bl=("blL2", "median"), succ=("ok", "mean"), n=("seed", "count")).reset_index()
    a["regime"] = np.where(a.relL2 < SUCCESS, 0, np.where(a.gap > GAP_OVER, 1, 2))
    return a


def edges(v):
    lv = np.log10(v); d = np.diff(lv) / 2
    return 10 ** np.concatenate([[lv[0] - d[0]], lv[:-1] + d, [lv[-1] + d[-1]]])


def panel_setup(ax, pes, ns, title):
    ax.set_xscale("log"); ax.set_yscale("log", base=2)
    ax.set_xlim(edges(pes)[0], edges(pes)[-1]); ax.set_ylim(edges(ns)[0], edges(ns)[-1])
    ax.set_xticks(pes); ax.set_xticklabels([f"{p:g}" for p in pes], fontsize=7)
    ax.set_yticks(ns); ax.set_yticklabels([str(n) for n in ns], fontsize=7)
    ax.minorticks_off()
    ax.set_title(title, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def pe_h_lines(ax, pes, ns):
    x = np.array([edges(pes)[0], edges(pes)[-1]])
    ax.plot(x, x, color=INK, lw=1.4, ls="--")            # Pe_h = Pe/N = 1  <=>  N = Pe
    ax.plot(x, x / 10, color=INK, lw=0.9, ls=":")         # Pe_h = 10


def phase_grid(a, key_list, out, value="relL2"):
    pes = np.sort(a.Pe.unique()); ns = np.sort(a.N.unique())
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ncol = min(3, len(key_list)); nrow = int(np.ceil(len(key_list) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol + 0.9, 3.0 * nrow + 0.5), squeeze=False)
    for ax, (m, p) in zip(axs.flat, key_list):
        d = a[(a.method == m) & (a.prec == p)]
        Z = np.full((len(ns), len(pes)), np.nan)
        for _, r in d.iterrows():
            Z[list(ns).index(r.N), list(pes).index(r.Pe)] = np.log10(max(r.relL2, 1e-6))
        im = ax.pcolormesh(edges(pes), edges(ns), Z, cmap=cmap, vmin=-4, vmax=0.5, shading="flat",
                           edgecolors=SURF, linewidth=1.0)
        # 5% contour (success boundary), hatched cells = failure
        for i, n in enumerate(ns):
            for j, pe in enumerate(pes):
                if np.isfinite(Z[i, j]) and Z[i, j] >= np.log10(SUCCESS):
                    ax.text(pe, n, "✕", ha="center", va="center", fontsize=6.5,
                            color="white" if Z[i, j] > -1.2 else INK)
        pe_h_lines(ax, pes, ns)
        panel_setup(ax, pes, ns, f"{NAMES.get(m, m)} · {p.upper()}")
    for ax in axs.flat[len(key_list):]: ax.axis("off")
    for ax in axs[-1]: ax.set_xlabel("Péclet number Pe")
    for ax in axs[:, 0]: ax.set_ylabel("collocation points N")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    cb.set_label("median relative L2 error (log10)"); cb.outline.set_visible(False)
    fig.text(0.01, 0.005, "✕ = fails (error ≥ 5%).  dashed: Pe_h = Pe/N = 1   dotted: Pe_h = 10",
             color=INK2, fontsize=8)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def regime_grid(a, key_list, out):
    pes = np.sort(a.Pe.unique()); ns = np.sort(a.N.unique())
    cmap = ListedColormap([C_OK, C_OVER, C_OPT]); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    ncol = min(3, len(key_list)); nrow = int(np.ceil(len(key_list) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow + 0.6), squeeze=False)
    for ax, (m, p) in zip(axs.flat, key_list):
        d = a[(a.method == m) & (a.prec == p)]
        Z = np.full((len(ns), len(pes)), np.nan)
        for _, r in d.iterrows():
            Z[list(ns).index(r.N), list(pes).index(r.Pe)] = r.regime
        ax.pcolormesh(edges(pes), edges(ns), Z, cmap=cmap, norm=norm, edgecolors=SURF, linewidth=1.0)
        for i, n in enumerate(ns):
            for j, pe in enumerate(pes):
                if np.isfinite(Z[i, j]):
                    ax.text(pe, n, "✓O×"[int(Z[i, j])] if False else ["✓", "O", "·"][int(Z[i, j])],
                            ha="center", va="center", fontsize=7, color="white")
        pe_h_lines(ax, pes, ns)
        panel_setup(ax, pes, ns, f"{NAMES.get(m, m)} · {p.upper()}")
    for ax in axs.flat[len(key_list):]: ax.axis("off")
    for ax in axs[-1]: ax.set_xlabel("Péclet number Pe")
    for ax in axs[:, 0]: ax.set_ylabel("collocation points N")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_OK, label="✓ success (error < 5%)"),
                        Patch(color=C_OVER, label="O overfitting: test residual ≫ train residual"),
                        Patch(color=C_OPT, label="· optimisation failure: training loss itself stuck")],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def collapse(a, out, method="vanilla", prec="f64"):
    """Does Pe_h collapse the data? error vs Pe_h, one sequential shade per N."""
    d = a[(a.method == method) & (a.prec == prec)].copy(); d["Pe_h"] = d.Pe / d.N
    ns = np.sort(d.N.unique())
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2))
    shades = [SEQ[min(1 + int(round(i * (len(SEQ) - 2) / max(len(ns) - 1, 1))), len(SEQ) - 1)] for i in range(len(ns))]
    for n, c in zip(ns, shades):
        dn = d[d.N == n].sort_values("Pe")
        axs[0].plot(dn.Pe, dn.relL2, "-o", color=c, lw=2, ms=4, label=f"N={n}")
        axs[1].plot(dn.Pe_h, dn.relL2, "-o", color=c, lw=2, ms=4)
    for ax, xl in zip(axs, ["Péclet number Pe", "collocation Péclet  Pe_h = Pe / N"]):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl)
        ax.axhline(SUCCESS, color=INK2, lw=1, ls="--")
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axs[1].axvline(1, color=INK, lw=1, ls=":")
    axs[0].set_ylabel("median relative L2 error")
    axs[0].set_title(f"{NAMES[method]} ({prec.upper()}): error vs Pe", loc="left")
    axs[1].set_title("same runs vs Pe_h", loc="left")
    axs[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="figs")
    a_ = ap.parse_args()
    os.makedirs(a_.out, exist_ok=True)
    df = load(a_.results); a = agg(df)
    a.to_csv(os.path.join(a_.out, "summary.csv"), index=False)
    keys = [k for k in [("vanilla", "f32"), ("vanilla", "f64"), ("hard", "f64"), ("gradreg", "f64"),
                        ("rad", "f64"), ("xi", "f64"), ("xi_hard", "f64")]
            if ((a.method == k[0]) & (a.prec == k[1])).any()]
    phase_grid(a, keys, os.path.join(a_.out, "phase_diagram.png"))
    regime_grid(a, keys, os.path.join(a_.out, "regime_map.png"))
    if ((a.method == "vanilla") & (a.prec == "f64")).any():
        collapse(a, os.path.join(a_.out, "collapse_vanilla_f64.png"))
    print(a.groupby(["method", "prec"]).agg(success=("succ", "mean"), med=("relL2", "median")))
