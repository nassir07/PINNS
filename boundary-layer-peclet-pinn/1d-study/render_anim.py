"""Render a snapshot file from animate.py into a GIF (and a final-frame PNG).
    python render_anim.py anim/vanilla_pe1000_n64.npz [anim/gradreg_pe1000_n64.npz] --out figs/comb.gif
Two files -> side-by-side comparison."""
import argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from plots import INK, INK2, MUTED, GRID, SURF, NAMES

BLUE, ORANGE = "#2a78d6", "#eb6834"
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=12)
a = ap.parse_args()
D = [np.load(f, allow_pickle=True) for f in a.files]
nF = min(len(d["it"]) for d in D)

fig, axs = plt.subplots(2, len(D), figsize=(4.6 * len(D), 5.6), squeeze=False,
                        gridspec_kw=dict(height_ratios=[1, 1.1]))
art = []
for k, d in enumerate(D):
    x, pts, Pe, N = d["x"], d["pts"], float(d["Pe"]), int(d["N"])
    ax, bx = axs[0, k], axs[1, k]
    ax.plot(x, d["exact"], color=INK, lw=1.2, ls="--", label="exact")
    lc, = ax.plot([], [], color=BLUE, lw=2, label="PINN")
    ax.set_xlim(0, 1); ax.set_ylim(-0.8, 1.3)
    ax.set_ylabel("c(x)"); ax.legend(frameon=False, loc="upper left", fontsize=8)
    ttl = ax.set_title("", loc="left", fontsize=9.5)
    # zoom into the layer
    ins = ax.inset_axes([0.52, 0.12, 0.44, 0.5])
    ins.plot(x, d["exact"], color=INK, lw=1, ls="--"); lci, = ins.plot([], [], color=BLUE, lw=1.8)
    ins.set_xlim(1 - 8 / Pe, 1); ins.set_ylim(-0.1, 1.1); ins.tick_params(labelsize=6)
    ins.set_title(f"boundary layer (δ≈1/Pe)", fontsize=7, color=INK2)
    for xp in pts[pts > 1 - 8 / Pe]: ins.axvline(xp, color=ORANGE, lw=0.8)
    lR, = bx.semilogy([], [], color=BLUE, lw=1.2, label="|R(x)| between points")
    sR, = bx.semilogy([], [], "o", color=ORANGE, ms=4, label=f"|R| at the {N} collocation points")
    bx.set_xlim(0, 1); bx.set_ylim(1e-7, 1e5); bx.set_xlabel("x")
    bx.set_ylabel("PDE residual |R|"); bx.legend(frameon=False, fontsize=7.5, loc="upper left")
    bx.grid(True, color=GRID, lw=0.5)
    for axx in (ax, bx):
        for s in ("top", "right"): axx.spines[s].set_visible(False)
    art.append((d, lc, lci, lR, sR, ttl, Pe, N))


def upd(i):
    out = []
    for d, lc, lci, lR, sR, ttl, Pe, N in art:
        x = d["x"]
        lc.set_data(x, d["c"][i]); lci.set_data(x, d["c"][i])
        lR.set_data(x, np.abs(d["R"][i]) + 1e-12); sR.set_data(d["pts"], np.abs(d["Rpts"][i]) + 1e-12)
        e = np.sqrt(np.trapezoid((d["c"][i] - d["exact"]) ** 2, x) / np.trapezoid(d["exact"] ** 2, x))
        ph = "Adam" if d["phase"][i] == 0 else "L-BFGS"
        ttl.set_text(f"{NAMES.get(str(d['method']), d['method'])} · Pe={Pe:g}, N={N} (Pe_h={Pe/N:.0f})\n"
                     f"iter {d['it'][i]} ({ph}) · train loss {d['loss'][i]:.1e} · error {100*e:.1f}%")
        out += [lc, lci, lR, sR, ttl]
    return out


fig.tight_layout()
anim = FuncAnimation(fig, upd, frames=list(range(nF)) + [nF - 1] * 18, blit=False)
anim.save(a.out, writer=PillowWriter(fps=a.fps), dpi=90)
upd(nF - 1); fig.savefig(a.out.replace(".gif", "_final.png"), dpi=170)
print("saved", a.out)
