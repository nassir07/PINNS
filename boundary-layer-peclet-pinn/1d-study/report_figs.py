"""Static figures for the PDF report."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from plots import INK, INK2, MUTED, GRID, SURF

BLUE, GREY, ORANGE = "#2a78d6", "#b9b7b0", "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
OUT = "figs/report"
import os; os.makedirs(OUT, exist_ok=True)


def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def exact(x, Pe):
    return (np.exp(Pe * (x - 1)) - np.exp(-Pe)) / (1 - np.exp(-Pe))


# ---------- Fig 1: the boundary layer vs the point spacing
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
x = np.linspace(0, 1, 20001)
for Pe, c in [(10, SEQ[2]), (100, SEQ[4]), (1000, SEQ[6])]:
    axs[0].plot(x, exact(x, Pe), color=c, lw=2, label=f"Pe = {Pe}")
axs[0].set_xlabel("x"); axs[0].set_ylabel("concentration c(x)")
axs[0].legend(frameon=False, loc="upper left"); clean(axs[0])
axs[0].set_title("Exact solution: a layer of thickness ≈ 1/Pe at x = 1", loc="left", fontsize=9.5, color=INK)
# zoom: Pe=1000 with N=32 and N=256 points
xz = np.linspace(0.95, 1, 4000)
axs[1].plot(xz, exact(xz, 1000), color=INK, lw=2)
for N, yv, col in [(32, -0.12, ORANGE), (256, -0.24, BLUE)]:
    p = (np.arange(N) + 0.5) / N; p = p[p > 0.95]
    axs[1].plot(p, np.full_like(p, yv), "|", color=col, ms=10, mew=1.6)
    axs[1].text(0.9505, yv + 0.04, f"N = {N}: {len(p)} points shown, spacing h = 1/{N}", color=col, fontsize=7.5)
axs[1].axvspan(1 - 1 / 1000 * 5, 1, color=SEQ[0], alpha=0.6, lw=0)
axs[1].text(0.9935, 0.55, "layer\n(~5/Pe)", ha="right", fontsize=7.5, color=INK2)
axs[1].set_xlim(0.95, 1.0005); axs[1].set_ylim(-0.32, 1.08)
axs[1].set_xlabel("x (zoom near x = 1)"); clean(axs[1])
axs[1].set_title("Pe = 1000: the points barely see the layer", loc="left", fontsize=9.5, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_layer.png", dpi=200); plt.close(fig)

# ---------- Fig 2: PINN loss schematic
fig, ax = plt.subplots(figsize=(8.2, 2.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2.2)
boxes = [(0.1, "input x", "a position\nin [0, 1]"),
         (2.35, "neural network", "4 layers × 64,\ntanh"),
         (4.6, "output c(x)", "+ its derivatives\nc′, c″ by autodiff"),
         (6.85, "loss", "PDE residual² at N points\n+ boundary errors²")]
for x0, t, s in boxes:
    ax.add_patch(FancyBboxPatch((x0, 0.35), 2.0, 1.45, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#f4f3f0", ec=MUTED, lw=1))
    ax.text(x0 + 1.0, 1.45, t, ha="center", fontsize=10, color=INK, weight="bold")
    ax.text(x0 + 1.0, 0.85, s, ha="center", va="center", fontsize=8, color=INK2)
for x0 in (2.1, 4.35, 6.6):
    ax.annotate("", xy=(x0 + 0.25, 1.07), xytext=(x0, 1.07), arrowprops=dict(arrowstyle="->", color=INK2))
ax.annotate("", xy=(3.35, 0.3), xytext=(7.85, 0.3),
            arrowprops=dict(arrowstyle="->", color=BLUE, connectionstyle="arc3,rad=0.25"))
ax.text(5.6, 0.0, "optimiser (Adam, then L-BFGS) adjusts the weights to push the loss to zero",
        ha="center", fontsize=8, color=BLUE)
fig.savefig(f"{OUT}/fig2_pinn.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: phase map from the local sweep
rows = [json.loads(l) for l in open("local_results/results.jsonl")]
pes = sorted({r["Pe"] for r in rows}); ns = sorted({r["N"] for r in rows})
fig, ax = plt.subplots(figsize=(8.2, 4.0))
for r in rows:
    i, j = pes.index(r["Pe"]), ns.index(r["N"])
    ok = r["relL2"] < 0.05
    kind = "ok" if ok else ("overfit" if r["gap"] > 1 else "stuck")
    a = 1.0 if ok else min(1, 0.35 + 0.3 * np.log10(r["relL2"] / 0.05))
    fc = "#efeeea" if ok else BLUE
    ax.add_patch(FancyBboxPatch((i + 0.04, j + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, alpha=a if not ok else 1, ec="none"))
    e = r["relL2"] * 100
    txt = f"{e:,.0f}%" if e >= 10 else f"{e:.2g}%"
    dark = (not ok) and a > 0.6
    ax.text(i + 0.5, j + 0.58, txt, ha="center", va="center", fontsize=9.5, color="white" if dark else INK)
    sub = {"ok": f"Pe_h={r['Pe']/r['N']:.2g}", "overfit": "overfits", "stuck": "straight line"}[kind]
    ax.text(i + 0.5, j + 0.3, sub, ha="center", va="center", fontsize=7.2, color="white" if dark else INK2)
# Pe_h = 2 line in index coordinates
lp = np.log10(pes)
fi = lambda pe: np.interp(np.log10(pe), lp, np.arange(len(pes))) + 0.5
fj = lambda n: np.interp(np.log2(n), np.log2(ns), np.arange(len(ns))) + 0.5
pp = np.geomspace(2 * ns[0] * 0.75, 2 * ns[-1] * 1.3, 50)
ax.plot([fi(p) for p in pp], [fj(p / 2) if p / 2 <= ns[-1] else fj(ns[-1]) + (np.log2(p / 2) - np.log2(ns[-1])) for p in pp],
        color=INK, lw=1.8, ls="--")
ax.text(3.1, 4.08, "Pe_h = Pe/N = 2  (the CFD cell-Péclet limit)", fontsize=8.5, color=INK)
ax.set_xlim(0, len(pes)); ax.set_ylim(0, len(ns) + 0.25)
ax.set_xticks(np.arange(len(pes)) + 0.5); ax.set_xticklabels([f"{p:,.0f}" for p in pes])
ax.set_yticks(np.arange(len(ns)) + 0.5); ax.set_yticklabels(ns)
ax.set_xlabel("Péclet number Pe  (thinner layer →)"); ax.set_ylabel("collocation points N")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_phase.png", dpi=200); plt.close(fig)

# ---------- Fig 4: median error vs N (GPU run 1 log)
N = [32, 64, 128, 256, 512, 1024]; plain = [1.93, 0.104, 0.00792, 0.00103, 0.00126, 0.00456]
Ng = [32, 64, 128, 256]; grad = [0.256, 0.0293, 0.00316, 0.00114]
fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.plot(Ng, np.array(grad) * 100, "-o", color=GREY, lw=2.2, ms=5, label="gradient penalty")
ax.plot(N, np.array(plain) * 100, "-o", color=BLUE, lw=2.2, ms=5, label="plain PINN")
ax.axhline(5, color=INK2, lw=1, ls="--"); ax.text(1024, 6.2, "5% success threshold", ha="right", fontsize=8, color=INK2)
for n, v in [(32, 1.93), (256, 0.00103), (1024, 0.00456)]:
    ax.annotate(f"{v*100:.2g}%" if v < 0.1 else f"{v*100:.0f}%", (n, v * 100), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=INK)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(N); ax.set_xticklabels(N); ax.minorticks_off()
ax.set_xlabel("collocation points N"); ax.set_ylabel("median relative error (%)")
ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); clean(ax)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_error_vs_N.png", dpi=200); plt.close(fig)
print("ok")
