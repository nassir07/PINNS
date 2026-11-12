"""Statistical analysis of the existing sweeps (no new runs). Writes stats.json + figures."""
import json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

rng = np.random.default_rng(0)
v2 = pd.read_csv("kaggle_v2/all.csv"); v3 = pd.read_csv("kaggle_v2/v3.csv")
v2["run"] = "main"; v3["run"] = "followup"
df = pd.concat([v2, v3], ignore_index=True)
df["Peh"] = df.Pe / df.N
df["lpeh"] = np.log10(df.Peh)
out = {}


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return [round(k / n, 3), round(lo, 3), round(hi, 3)]


def trivial(r):
    t = 1.0 if r.method in ("hard", "xi_hard") else 1 / 3
    return abs(r.loss_final - t) < 0.05 * t


df["trivial"] = df.apply(trivial, axis=1)

# ---- 1. success rates with Wilson CIs (main sweep)
tab = {}
for th in (0.01, 0.05, 0.10):
    for (m, p), g in df[df.run == "main"].groupby(["method", "prec"]):
        k = int((g.relL2 < th).sum()); tab.setdefault(f"{m}_{p}", {})[str(th)] = wilson(k, len(g)) + [k, len(g)]
for m, g in df[df.run == "followup"].groupby("method"):
    for th in (0.01, 0.05, 0.10):
        k = int((g.relL2 < th).sum()); tab.setdefault(f"{m}_followup", {})[str(th)] = wilson(k, len(g)) + [k, len(g)]
out["success"] = tab

# ---- 2. logistic fit: does Pe_h alone decide success? (plain PINN, Pe <= 300, both precisions)
van = df[(df.run == "main") & (df.method == "vanilla")]
lo_pe = van[van.Pe <= 300].copy()


def fit_threshold(d, th):
    y = (d.relL2 < th).astype(int).values
    X = sm.add_constant(d[["lpeh"]].values)
    r = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=500)
    b0, b1 = r.params
    return r, 10 ** (-b0 / b1)


lr = {}
for th in (0.01, 0.05, 0.10):
    r, p50 = fit_threshold(lo_pe, th)
    # bootstrap over runs, stratified by cell
    boots = []
    cells = [g for _, g in lo_pe.groupby(["Pe", "N", "prec"])]
    for _ in range(2000):
        bs = pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1e9)) for g in cells])
        try:
            _, b = fit_threshold(bs, th)
            if np.isfinite(b) and 1e-3 < b < 1e3: boots.append(b)
        except Exception:
            pass
    q = np.percentile(boots, [2.5, 97.5])
    lr[str(th)] = dict(peh50=round(p50, 2), ci=[round(q[0], 2), round(q[1], 2)], slope=round(r.params[1], 2),
                       n=len(lo_pe), nboot=len(boots))
out["logit_peh50"] = lr

# nested model test: does log Pe (or log N) add information beyond log Pe_h?
y = (lo_pe.relL2 < 0.05).astype(int).values
X1 = sm.add_constant(lo_pe[["lpeh"]].values)
X2 = sm.add_constant(np.column_stack([lo_pe.lpeh.values, np.log10(lo_pe.Pe.values)]))
X3 = sm.add_constant(np.column_stack([lo_pe.lpeh.values, (lo_pe.prec == "f64").astype(float).values]))
r1 = sm.Logit(y, X1).fit(disp=0, method="bfgs", maxiter=500)
r2 = sm.Logit(y, X2).fit(disp=0, method="bfgs", maxiter=500)
r3 = sm.Logit(y, X3).fit(disp=0, method="bfgs", maxiter=500)
lrt2 = 2 * (r2.llf - r1.llf); lrt3 = 2 * (r3.llf - r1.llf)
out["nested"] = dict(add_logPe=dict(LR=round(lrt2, 2), p=float(stats.chi2.sf(lrt2, 1)), coef=round(r2.params[2], 2)),
                     add_precision=dict(LR=round(lrt3, 2), p=float(stats.chi2.sf(lrt3, 1)), coef=round(r3.params[2], 2)),
                     auc_peh_only=None)
from sklearn.metrics import roc_auc_score
out["nested"]["auc_peh_only"] = round(roc_auc_score(y, -lo_pe.lpeh.values), 3)
# same model applied to Pe >= 1000: predicted vs observed successes
hi_pe = van[van.Pe >= 1000]
pred = r1.predict(sm.add_constant(hi_pe[["lpeh"]].values, has_constant="add"))
out["second_limit"] = dict(n=len(hi_pe), observed=int((hi_pe.relL2 < 0.05).sum()), predicted=round(float(pred.sum()), 1),
                           trivial_frac=round(hi_pe.trivial.mean(), 3))

# ---- 3. empirical success by Pe_h bin with CIs (Pe<=300)
bins = [0, .5, 1, 2, 3, 5, 1e4]; lab = ["≤0.5", "0.5–1", "1–2", "2–3", "3–5", ">5"]
lo_pe["bin"] = pd.cut(lo_pe.Peh, bins, labels=lab)
bt = []
for b, g in lo_pe.groupby("bin", observed=True):
    k = int((g.relL2 < .05).sum()); bt.append([b, len(g)] + wilson(k, len(g)))
out["peh_bins"] = bt

# ---- 4. paired tests
def paired(a, b, keys=("Pe", "N", "seed")):
    m = a.merge(b, on=list(keys), suffixes=("_a", "_b"))
    sa, sb = m.relL2_a < .05, m.relL2_b < .05
    t = [[int((sa & sb).sum()), int((sa & ~sb).sum())], [int((~sa & sb).sum()), int((~sa & ~sb).sum())]]
    mc = mcnemar(t, exact=True)
    d = np.log10(m.relL2_b) - np.log10(m.relL2_a)
    w = stats.wilcoxon(d) if (d != 0).any() else None
    return dict(n=len(m), a_only=t[0][1], b_only=t[1][0], both=t[0][0], neither=t[1][1], mcnemar_p=float(mc.pvalue),
                median_log10_ratio=round(float(np.median(d)), 3), wilcoxon_p=float(w.pvalue) if w else None)

M = df[df.run == "main"]
f64 = M[(M.method == "vanilla") & (M.prec == "f64")]; f32 = M[(M.method == "vanilla") & (M.prec == "f32")]
out["paired"] = {}
out["paired"]["fp32_vs_fp64"] = paired(f32, f64)
out["paired"]["plain_vs_gradreg_N<=128"] = paired(f32[f32.N <= 128], M[(M.method == "gradreg")])
out["paired"]["plain_vs_hard"] = paired(f32, M[M.method == "hard"])
out["paired"]["plain_vs_xi_hard_N<=256"] = paired(f32[f32.N <= 256], M[M.method == "xi_hard"])
F = df[df.run == "followup"]
base3 = f64[(f64.Pe >= 300) & (f64.N.between(64, 512))]
for m in ("curr", "xi", "xi_curr", "rad"):
    out["paired"][f"plain_vs_{m}_followup"] = paired(base3, F[F.method == m])

# ---- 5. seed-to-seed spread (std of log10 error within a cell)
sp = df.groupby(["run", "method", "prec", "Pe", "N"]).relL2.apply(lambda s: np.std(np.log10(s))).reset_index()
sp["region"] = np.where(sp.Pe / sp.N <= 1, "Pe_h≤1", np.where(sp.Pe / sp.N <= 5, "1<Pe_h≤5", "Pe_h>5"))
out["seed_spread"] = sp.groupby(["method", "region"]).relL2.median().round(2).reset_index().values.tolist()
# fraction of cells whose seeds disagree on success
cell = df.groupby(["run", "method", "prec", "Pe", "N"]).relL2.apply(lambda s: ((s < .05).sum(), len(s))).reset_index()
mixed = cell.relL2.apply(lambda t: 0 < t[0] < t[1])
out["mixed_cells"] = dict(n_cells=len(cell), mixed=int(mixed.sum()))

# ---- 6. trivial-solution exactness
tr = df[df.trivial]
out["trivial"] = dict(n=len(tr), max_rel_dev=float((abs(tr.loss_final - np.where(tr.method.isin(["hard", "xi_hard"]), 1, 1/3))
                                                   / np.where(tr.method.isin(["hard", "xi_hard"]), 1, 1/3)).max()),
                      median_gap=float(tr.gap.median()))
# ---- 7. flux conservation vs error (does flux error track accuracy?)
ok = df.relL2 < .05
out["flux"] = dict(spearman=round(float(stats.spearmanr(df.flux_err, df.relL2).correlation), 3),
                   median_flux_success=float(df[ok].flux_err.median()), median_flux_fail=float(df[~ok].flux_err.median()))
# ---- 8. residual-gap diagnostic: how well does gap>1 flag failure among runs with tiny loss?
small = df[df.loss_final < 1e-4]
out["gap_diag"] = dict(n=len(small), fail_rate=round(float((small.relL2 >= .05).mean()), 3),
                       auc=round(float(roc_auc_score((small.relL2 >= .05).astype(int), small.gap)), 3))
out["totals"] = dict(main=int((df.run == "main").sum()), followup=int((df.run == "followup").sum()))
json.dump(out, open("stats.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:6000])
