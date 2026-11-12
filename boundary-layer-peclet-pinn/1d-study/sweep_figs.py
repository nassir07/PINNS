"""Figures from the Kaggle Version-2 sweep (567 runs)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from plots import INK, INK2, MUTED, GRID

BLUE, GREY, ORANGE, GREEN = "#2a78d6", "#b9b7b0", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
OUT = "figs/report"

df = pd.read_csv("kaggle_v2/all.csv")
df["ok"] = df.relL2 < 0.05
df["Peh"] = df.Pe / df.N


def cls(r):
    if r.ok: return "success"
    tgt = 1.0 if r.method in ("hard", "xi_hard") else 1 / 3
    if abs(r.loss_final - tgt) < 0.05 * tgt: return "trivial"
    return "overfit" if r.gap > 1 else "other"


df["cls"] = df.apply(cls, axis=1)
df.to_csv("kaggle_v2/all_classified.csv", index=False)
NAMES = {("vanilla", "f64"): "Plain PINN · FP64", ("vanilla", "f32"): "Plain PINN · FP32",
         ("gradreg", "f32"): "Gradient penalty · FP32", ("hard", "f32"): "Hard BCs · FP32",
         ("xi_hard", "f32"): "ξ input + hard BCs · FP32"}
KEYS = list(NAMES)
PES = sorted(df.Pe.unique()); NS = sorted(df.N.unique())

# ---------------- Fig S1: success maps
fig, axs = plt.subplots(1, 5, figsize=(13.5, 3.9), sharey=True)
for ax, k in zip(axs, KEYS):
    g = df[(df.method == k[0]) & (df.prec == k[1])]
    for i, pe in enumerate(PES):
        for j, n in enumerate(NS):
            c = g[(g.Pe == pe) & (g.N == n)]
            if c.empty:
                ax.add_patch(FancyBboxPatch((i + .06, j + .06), .88, .88, boxstyle="round,pad=0,rounding_size=.08",
                                            fc="white", ec="#e4e3df", lw=.8, ls=":"))
                continue
            s = int(c.ok.sum()); tot = len(c)
            fc = BLUE if s == 0 else ("#efeeea" if s == tot else SEQ[1])
            ax.add_patch(FancyBboxPatch((i + .06, j + .06), .88, .88, boxstyle="round,pad=0,rounding_size=.08",
                                        fc=fc, ec="none"))
            ax.text(i + .5, j + .5, f"{s}/{tot}", ha="center", va="center", fontsize=7.5,
                    color="white" if s == 0 else INK)
    # Pe_h = 1 line: N = Pe
    xs = np.linspace(-0.2, len(PES) + 0.2, 200)
    pe_x = 10 ** np.interp(xs - 0.5, range(len(PES)), np.log10(PES), left=np.nan, right=np.nan)
    lp = np.log10(PES)
    ext = lambda x: 10 ** (lp[0] + (x - .5) * (lp[1] - lp[0])) if x < .5 else (10 ** (lp[-1] + (x - .5 - (len(PES) - 1)) * (lp[-1] - lp[-2])) if x > len(PES) - .5 else 10 ** np.interp(x - .5, range(len(PES)), lp))
    pe_x = np.array([ext(x) for x in xs])
    yj = np.log2(pe_x / NS[0]) + .5
    m = (yj > 0) & (yj < len(NS))
    ax.plot(xs[m], yj[m], color=INK, lw=1.4, ls="--")
    ax.set_xlim(0, len(PES)); ax.set_ylim(0, len(NS))
    ax.set_xticks(np.arange(len(PES)) + .5); ax.set_xticklabels([f"{p:g}" if p < 1000 else f"{p/1000:g}k" for p in PES], fontsize=7.5)
    ax.set_yticks(np.arange(len(NS)) + .5); ax.set_yticklabels(NS, fontsize=7.5)
    ax.set_title(f"{NAMES[k]}\n{int(g.ok.sum())}/{len(g)} runs succeed", fontsize=9, color=INK, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.set_xlabel("Péclet number Pe")
axs[0].set_ylabel("collocation points N")
fig.legend(handles=[Patch(color="#efeeea", label="all 3 seeds succeed"), Patch(color=SEQ[1], label="1–2 of 3 succeed"),
                    Patch(color=BLUE, label="0 of 3 succeed"), Patch(fc="white", ec="#c9c8c3", ls=":", label="not run (time limit)"),
                    plt.Line2D([], [], color=INK, ls="--", label="Pe_h = Pe/N = 1")],
           loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, -0.04))
fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(f"{OUT}/s1_success_maps.png", dpi=200); plt.close(fig)

# ---------------- Fig S2: success vs Pe_h, one line per Pe (plain PINN, FP32 + FP64 pooled)
v = df[df.method == "vanilla"]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
shades = dict(zip(PES, SEQ))
for pe in PES:
    g = v[v.Pe == pe].groupby("Peh").ok.mean().reset_index().sort_values("Peh")
    hot = pe >= 1000
    ax.plot(g.Peh, g.ok * 100, "-o", color=shades[pe] if not hot else ORANGE if pe == 1000 else ("#b8471c" if pe == 3000 else "#7a2c10"),
            lw=2, ms=4.5, label=f"Pe = {pe:g}")
ax.axvline(1, color=INK2, lw=1, ls=":"); ax.text(1.08, 103, "Pe_h = 1", fontsize=8, color=INK2)
ax.set_xscale("log"); ax.set_ylim(-5, 112)
ax.set_xlabel("collocation Péclet number  Pe_h = Pe / N"); ax.set_ylabel("runs succeeding (%)")
ax.grid(True, axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.0, .5))
ax.text(0.012, 50, "Pe ≤ 300: curves line up\n→ Pe_h decides", fontsize=8, color=SEQ[5])
ax.text(3, 70, "Pe ≥ 1000: fail even at\nPe_h ≈ 1 → a second limit", fontsize=8, color="#b8471c")
fig.tight_layout(); fig.savefig(f"{OUT}/s2_success_vs_peh.png", dpi=200); plt.close(fig)

# ---------------- Fig S3: what kind of failure, per method
order = ["success", "overfit", "trivial", "other"]
cols = {"success": "#d9d8d3", "overfit": BLUE, "trivial": ORANGE, "other": "#8a8984"}
lab = {"success": "success", "overfit": "overfits between points", "trivial": "trivial solution (straight line / c = x)", "other": "other"}
fig, ax = plt.subplots(figsize=(7.6, 2.7))
for i, k in enumerate(KEYS[::-1]):
    g = df[(df.method == k[0]) & (df.prec == k[1])]
    left = 0
    for c in order:
        w = (g.cls == c).mean() * 100
        ax.barh(i, w, left=left, color=cols[c], height=.62, edgecolor="white", lw=1.5)
        if w >= 7: ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                           color="white" if c in ("overfit", "trivial") else INK)
        left += w
ax.set_yticks(range(len(KEYS))); ax.set_yticklabels([NAMES[k] for k in KEYS[::-1]], fontsize=8.5)
ax.set_xlim(0, 100); ax.set_xlabel("share of runs (%)")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=cols[c], label=lab[c]) for c in order], frameon=False, fontsize=8, ncol=4,
          loc="upper center", bbox_to_anchor=(.42, 1.22))
fig.tight_layout(); fig.savefig(f"{OUT}/s3_failure_types.png", dpi=200); plt.close(fig)

# ---------------- Fig S4: median error vs N per Pe (plain PINN FP64)
g = df[(df.method == "vanilla") & (df.prec == "f64")].groupby(["Pe", "N"]).relL2.median().reset_index()
fig, ax = plt.subplots(figsize=(7.6, 3.4))
for pe in PES:
    s = g[g.Pe == pe]
    c = shades[pe] if pe < 1000 else (ORANGE if pe == 1000 else ("#b8471c" if pe == 3000 else "#7a2c10"))
    ax.plot(s.N, s.relL2 * 100, "-o", color=c, lw=2, ms=4)
    ax.text(s.N.iloc[-1] * 1.12, s.relL2.iloc[-1] * 100, f"Pe={pe:g}", fontsize=7.5, color=c, va="center")
ax.axhline(5, color=INK2, lw=1, ls="--"); ax.text(33, 6.3, "5% success threshold", fontsize=7.5, color=INK2)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(NS[:6]); ax.set_xticklabels(NS[:6]); ax.minorticks_off()
ax.set_xlim(26, 1900)
ax.set_xlabel("collocation points N"); ax.set_ylabel("median relative error (%)")
ax.grid(True, axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/s4_error_vs_N_perPe.png", dpi=200); plt.close(fig)

# numbers for the text
summ = df.groupby(["method", "prec"]).agg(n=("ok", "size"), succ=("ok", "mean")).round(3)
print(summ)
print(df.groupby(["method", "prec", "cls"]).size().unstack(fill_value=0))
b = pd.cut(v.Peh, [0, .5, 1, 2, 5, 1e4])
print(v[v.Pe <= 300].groupby(b[v.Pe <= 300], observed=True).ok.agg(["size", "mean"]).round(2))
print(v[v.Pe >= 1000].groupby(b[v.Pe >= 1000], observed=True).ok.agg(["size", "mean"]).round(2))
