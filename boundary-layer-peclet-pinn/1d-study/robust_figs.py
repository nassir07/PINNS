import json, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from plots import INK, INK2, MUTED, GRID
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b7b0"
rng = np.random.default_rng(1)
v2 = pd.read_csv("kaggle_v2/all.csv"); v3 = pd.read_csv("kaggle_v2/v3.csv")
van = v2[v2.method == "vanilla"].copy(); van["Peh"] = van.Pe / van.N; van["l"] = np.log10(van.Peh)
lo = van[van.Pe <= 300]; hi = van[van.Pe >= 1000]

# ---------- R1: logistic fit
fig, ax = plt.subplots(figsize=(7.6, 3.6))
xx = np.linspace(-2.6, 2.6, 300)
for th, ls, a in [(0.01, ":", .7), (0.05, "-", 1), (0.10, "--", .7)]:
    y = (lo.relL2 < th).astype(int)
    r = sm.Logit(y.values, sm.add_constant(lo[["l"]].values)).fit(disp=0, method="bfgs", maxiter=500)
    p = r.predict(sm.add_constant(xx[:, None]))
    ax.plot(10 ** xx, p * 100, ls, color=BLUE, lw=2 if th == .05 else 1.4, alpha=a, label=f"logistic fit, success = error < {th*100:g}%")
bins = [0, .5, 1, 2, 3, 5, 1e4]
lo2 = lo.assign(b=pd.cut(lo.Peh, bins))
for b, g in lo2.groupby("b", observed=True):
    k = (g.relL2 < .05).sum(); n = len(g); l_, h_ = proportion_confint(k, n, method="wilson")
    x = np.exp(np.mean(np.log(g.Peh)))
    ax.errorbar(x, k / n * 100, yerr=[[100 * (k / n - l_)], [100 * (h_ - k / n)]], fmt="o", color=INK, ms=5, capsize=3, lw=1.2)
hb = hi.assign(b=pd.cut(hi.Peh, [0, 1, 2, 5, 20, 1e4]))
first = True
for b, g in hb.groupby("b", observed=True):
    k = (g.relL2 < .05).sum(); n = len(g)
    ax.plot(np.exp(np.mean(np.log(g.Peh))), k / n * 100, "s", color=ORANGE, ms=6, label="observed, Pe ≥ 1000" if first else None); first = False
ax.plot([], [], "o", color=INK, label="observed, Pe ≤ 300 (95% Wilson CI)")
ax.axvline(2, color=INK2, lw=.8, ls=":"); ax.text(2.1, 3, "Pe_h = 2", fontsize=7.5, color=INK2)
ax.set_xscale("log"); ax.set_ylim(-4, 106); ax.set_xlim(3e-3, 400)
ax.set_xlabel("collocation Péclet number  Pe_h = Pe / N"); ax.set_ylabel("runs succeeding (%)")
ax.grid(True, axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1, .5))
fig.tight_layout(); fig.savefig("figs/report/r1_logistic.png", dpi=200); plt.close(fig)

# ---------- R2: forest plot of paired effects
f64 = v2[(v2.method == "vanilla") & (v2.prec == "f64")]; f32 = v2[(v2.method == "vanilla") & (v2.prec == "f32")]
base3 = f64[(f64.Pe >= 300) & (f64.N.between(64, 512))]
comps = [("FP64 vs FP32 (plain PINN)", f32, f64, "main"),
         ("Gradient penalty vs plain (FP32, N ≤ 128)", f32[f32.N <= 128], v2[v2.method == "gradreg"], "main"),
         ("Hard BCs vs plain (FP32)", f32, v2[v2.method == "hard"], "main"),
         ("ξ + hard BCs vs plain (FP32, N ≤ 256)", f32[f32.N <= 256], v2[v2.method == "xi_hard"], "main"),
         ("Curriculum vs plain (FP64)", base3, v3[v3.method == "curr"], "follow-up"),
         ("ξ input vs plain (FP64)", base3, v3[v3.method == "xi"], "follow-up"),
         ("ξ + curriculum vs plain (FP64)", base3, v3[v3.method == "xi_curr"], "follow-up"),
         ("Adaptive sampling vs plain (FP64)", base3, v3[v3.method == "rad"], "follow-up")]
rows = []
for name, a, b, run in comps:
    m = a.merge(b, on=["Pe", "N", "seed"], suffixes=("_a", "_b"))
    d = (np.log10(m.relL2_b) - np.log10(m.relL2_a)).values
    boot = [np.median(rng.choice(d, len(d))) for _ in range(4000)]
    rows.append((name, np.median(d), *np.percentile(boot, [2.5, 97.5]), len(d), run))
json.dump([[r[0], round(10 ** r[1], 3), round(10 ** r[2], 3), round(10 ** r[3], 3), r[4], r[5]] for r in rows],
          open("forest.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(7.6, 3.5))
for i, (name, med, l_, h_, n, run) in enumerate(rows[::-1]):
    c = BLUE if h_ < 0 else (ORANGE if l_ > 0 else GREY)
    ax.plot([10 ** l_, 10 ** h_], [i, i], color=c, lw=2.2); ax.plot(10 ** med, i, "o", color=c, ms=6)
    ax.text(1.6e3, i, f"×{10**med:.2g}", va="center", fontsize=8, color=INK)
ax.axvline(1, color=INK2, lw=1)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r[0]}  (n={r[4]})" for r in rows[::-1]], fontsize=8)
ax.set_xscale("log"); ax.set_xlim(1e-3, 3e3)
ax.set_xlabel("median ratio of error, new method / baseline (paired by Pe, N, seed), 95% bootstrap CI")
ax.text(1.2e-3, len(rows) - .35, "← lower error", fontsize=8, color=BLUE); ax.text(8, len(rows) - .35, "higher error →", fontsize=8, color=ORANGE)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0); ax.grid(True, axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figs/report/r2_forest.png", dpi=200); plt.close(fig)
print(open("forest.json").read())
