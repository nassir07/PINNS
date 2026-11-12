"""
Collocation-Péclet phase diagram for PINNs on 1D steady advection-diffusion.

    c'(x) = (1/Pe) c''(x),  x in (0,1),  c(0)=0, c(1)=1
    exact: c(x) = (exp(Pe(x-1)) - exp(-Pe)) / (1 - exp(-Pe))
    boundary layer at x=1, thickness delta ~ 1/Pe
    total flux J = c - c'/Pe is constant: J_exact = -1/(exp(Pe)-1)

Collocation Péclet number:  Pe_h = Pe * h = Pe / N   (h = 1/N, uniform points)

Methods
  vanilla : soft BCs, tanh MLP, Adam -> L-BFGS
  hard    : hard BCs  c = x + x(1-x) NN(x)
  gradreg : + (h^2) * mean (dR/dx)^2   ("double PINN"/gPINN-style gradient penalty,
            weight h^2 = first-order Taylor estimate of the residual between points)
  rad     : residual-based adaptive distribution (Wu et al. 2023), N kept fixed
  xi      : layer-aware input feature  xi = (1-x) Pe  (inner coordinate from matched asymptotics)
  xi_hard : xi + hard BCs
Precision is chosen per run ('f32' or 'f64').

Everything is vmapped over (Pe, seed) so one compiled call runs a whole row of the phase diagram.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time, json, argparse, itertools
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

METHODS = ["vanilla", "hard", "gradreg", "rad", "xi", "xi_hard"]
DT = {"f32": jnp.float32, "f64": jnp.float64}


# ----------------------------------------------------------------------------- exact solution
def exact(x, Pe):
    return (jnp.exp(Pe * (x - 1.0)) - jnp.exp(-Pe)) / (1.0 - jnp.exp(-Pe))


def exact_flux(Pe):
    return -jnp.exp(-Pe) / (1.0 - jnp.exp(-Pe))      # = -1/(e^Pe - 1), overflow-safe


# ----------------------------------------------------------------------------- model
def init_mlp(key, sizes, dtype):
    params = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (i, o), dtype) * jnp.sqrt(2.0 / (i + o)).astype(dtype)
        params.append((W, jnp.zeros((o,), dtype)))
    return params


def mlp(params, h):
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]


def n_inputs(method):
    return 2 if method.startswith("xi") else 1


def model(params, x, Pe, method):
    """scalar x -> scalar c"""
    feats = [2.0 * x - 1.0]
    if method.startswith("xi"):
        feats.append((1.0 - x) * Pe)
    n = mlp(params, jnp.stack(feats))
    if method in ("hard", "xi_hard"):
        return x + x * (1.0 - x) * n
    return n


def derivs(params, x, Pe, method):
    f = lambda s: model(params, s, Pe, method)
    fx = jax.grad(f)
    fxx = jax.grad(fx)
    return f(x), fx(x), fxx(x)


def residual(params, x, Pe, method):
    _, cx, cxx = derivs(params, x, Pe, method)
    return cx - cxx / Pe


# ----------------------------------------------------------------------------- loss
def make_loss(Pe, pts, method, h):
    R_v = jax.vmap(residual, in_axes=(None, 0, None, None))
    Rx_v = jax.vmap(jax.grad(residual, argnums=1), in_axes=(None, 0, None, None))
    one = jnp.ones((), pts.dtype)

    def loss(params):
        R = R_v(params, pts, Pe, method)
        L = jnp.mean(R ** 2)
        if method not in ("hard", "xi_hard"):
            c0 = model(params, 0.0 * one, Pe, method)
            c1 = model(params, one, Pe, method)
            L = L + c0 ** 2 + (c1 - 1.0) ** 2
        if method == "gradreg":
            L = L + (h ** 2) * jnp.mean(Rx_v(params, pts, Pe, method) ** 2)
        return L
    return loss


# ----------------------------------------------------------------------------- metrics
def eval_grid(Pe, dtype, n_uni=20001, n_layer=2001):
    """uniform grid + grid stretched into the layer (x = 1 - s/Pe, s in [0, 30]); sorted."""
    xu = jnp.linspace(0.0, 1.0, n_uni, dtype=dtype)
    s = jnp.linspace(0.0, 30.0, n_layer, dtype=dtype)
    xl = jnp.clip(1.0 - s / Pe, 0.0, 1.0)
    return jnp.sort(jnp.concatenate([xu, xl]))


def trapz(y, x):
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


def metrics(params, Pe, pts, method, dtype):
    xg = eval_grid(Pe, dtype)
    D = jax.vmap(derivs, in_axes=(None, 0, None, None))
    c, cx, cxx = D(params, xg, Pe, method)
    ce = exact(xg, Pe)
    err = c - ce
    relL2 = jnp.sqrt(trapz(err ** 2, xg) / trapz(ce ** 2, xg))
    # boundary layer: x > 1 - 5/Pe
    s = jnp.linspace(0.0, 5.0, 2001, dtype=dtype)
    xl = 1.0 - s[::-1] / Pe
    cl = jax.vmap(model, in_axes=(None, 0, None, None))(params, xl, Pe, method)
    cle = exact(xl, Pe)
    blL2 = jnp.sqrt(trapz((cl - cle) ** 2, xl) / trapz(cle ** 2, xl))
    # residual gap: integral mean of R^2 over the domain vs mean at training points
    R = cx - cxx / Pe
    R2_test = trapz(R ** 2, xg)                      # domain length 1 -> mean
    R2_train = jnp.mean(jax.vmap(residual, in_axes=(None, 0, None, None))(params, pts, Pe, method) ** 2)
    gap = jnp.log10(R2_test + 1e-30) - jnp.log10(R2_train + 1e-30)
    # flux conservation (non-dimensionalised by U*dc = 1)
    J = c - cx / Pe
    flux_err = jnp.max(jnp.abs(J - exact_flux(Pe)))
    bc_err = jnp.abs(c[0]) + jnp.abs(c[-1] - 1.0)
    return dict(relL2=relL2, blL2=blL2, R2_test=R2_test, R2_train=R2_train, gap=gap,
                flux_err=flux_err, bc_err=bc_err)


# ----------------------------------------------------------------------------- training
def uniform_pts(N, dtype):
    return ((jnp.arange(N) + 0.5) / N).astype(dtype)


def rad_resample(key, params, Pe, method, N, dtype, pool=32768, k=1.0, c=1.0):
    """Wu et al. (2023) RAD: p(x) ∝ |R|^k / mean|R|^k + c ; draw N points from a random pool."""
    k1, k2 = jax.random.split(key)
    xp = jax.random.uniform(k1, (pool,), dtype)
    R = jnp.abs(jax.vmap(residual, in_axes=(None, 0, None, None))(params, xp, Pe, method))
    w = R ** k / jnp.mean(R ** k) + c
    w = jnp.where(jnp.isfinite(w), w, 1.0)
    idx = jax.random.choice(k2, pool, (N,), replace=False, p=w / jnp.sum(w))
    return xp[idx]


def train_one(key, Pe, *, N, method, prec, n_adam, n_lbfgs, width=64, depth=4, rad_every=1000):
    dtype = DT[prec]
    Pe = Pe.astype(dtype)
    h = jnp.asarray(1.0 / N, dtype)
    kinit, krad = jax.random.split(key)
    params = init_mlp(kinit, [n_inputs(method)] + [width] * depth + [1], dtype)
    pts = uniform_pts(N, dtype)

    sched = optax.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9)
    adam = optax.adam(sched)

    def adam_phase(params, opt_state, pts, n):
        loss = make_loss(Pe, pts, method, h)

        def step(carry, _):
            p, s = carry
            v, g = jax.value_and_grad(loss)(p)
            u, s = adam.update(g, s, p)
            return (optax.apply_updates(p, u), s), v
        (params, opt_state), hist = jax.lax.scan(step, (params, opt_state), None, length=n)
        return params, opt_state, hist

    opt_state = adam.init(params)
    if method == "rad":
        n_rounds = max(n_adam // rad_every, 1)

        def round_(carry, kr):
            p, s, x = carry
            p, s, hist = adam_phase(p, s, x, rad_every)
            x = rad_resample(kr, p, Pe, method, N, dtype)
            return (p, s, x), hist
        (params, opt_state, pts), hist = jax.lax.scan(
            round_, (params, opt_state, pts), jax.random.split(krad, n_rounds))
        hist_adam = hist.reshape(-1)
    else:
        params, opt_state, hist_adam = adam_phase(params, opt_state, pts, n_adam)

    # ---- L-BFGS
    loss = make_loss(Pe, pts, method, h)
    lb = optax.lbfgs()
    vg = optax.value_and_grad_from_state(loss)

    def lstep(carry, _):
        p, s = carry
        v, g = vg(p, state=s)
        u, s = lb.update(g, s, p, value=v, grad=g, value_fn=loss)
        p_new = optax.apply_updates(p, u)
        ok = jax.tree_util.tree_reduce(lambda a, b: a & b,
                                       jax.tree_util.tree_map(lambda t: jnp.all(jnp.isfinite(t)), p_new))
        p_new = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), p_new, p)
        return (p_new, s), v
    (params, _), hist_lb = jax.lax.scan(lstep, (params, lb.init(params)), None, length=n_lbfgs)

    m = metrics(params, Pe, pts, method, dtype)
    m["loss_final"] = loss(params)
    m = {k: v.astype(jnp.float64) for k, v in m.items()}
    return params, pts, m, jnp.concatenate([hist_adam, hist_lb]).astype(jnp.float64)


@partial(jax.jit, static_argnames=("N", "method", "prec", "n_adam", "n_lbfgs"))
def run_batch(keys, Pes, *, N, method, prec, n_adam, n_lbfgs):
    f = partial(train_one, N=N, method=method, prec=prec, n_adam=n_adam, n_lbfgs=n_lbfgs)
    return jax.vmap(f)(keys, Pes)


# ----------------------------------------------------------------------------- sweep
def sweep(pes, ns, seeds, methods, precs, n_adam, n_lbfgs, out, keep_params=False):
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done.add((r["method"], r["prec"], r["N"], r["Pe"], r["seed"]))
    for method, prec, N in itertools.product(methods, precs, ns):
        grid = [(pe, s) for pe in pes for s in range(seeds) if (method, prec, N, pe, s) not in done]
        if not grid:
            continue
        Pes = jnp.array([g[0] for g in grid], jnp.float64)
        keys = jnp.stack([jax.random.PRNGKey(1000 * s + 7) for _, s in grid])
        t0 = time.time()
        params, pts, m, hist = run_batch(keys, Pes, N=N, method=method, prec=prec,
                                         n_adam=n_adam, n_lbfgs=n_lbfgs)
        jax.block_until_ready(m)
        dt = time.time() - t0
        m = {k: np.asarray(v) for k, v in m.items()}
        hist = np.asarray(hist)
        with open(res_path, "a") as f:
            for i, (pe, s) in enumerate(grid):
                r = dict(method=method, prec=prec, N=N, Pe=pe, seed=s, Pe_h=pe / N,
                         n_adam=n_adam, n_lbfgs=n_lbfgs,
                         **{k: float(v[i]) for k, v in m.items()},
                         hist=[float(x) for x in hist[i, ::max(1, hist.shape[1] // 200)]])
                f.write(json.dumps(r) + "\n")
        if keep_params:
            import pickle
            with open(os.path.join(out, f"params_{method}_{prec}_N{N}.pkl"), "wb") as f:
                pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), pts=np.asarray(pts),
                                 grid=grid), f)
        print(f"{method:8s} {prec} N={N:5d}  {len(grid):3d} runs  {dt:7.1f}s  "
              f"median relL2={np.median(m['relL2']):.2e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", type=float, nargs="+", default=[10, 30, 100, 300, 1000, 3000, 10000])
    ap.add_argument("--n", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--precs", nargs="+", default=["f32", "f64"])
    ap.add_argument("--n_adam", type=int, default=10000)
    ap.add_argument("--n_lbfgs", type=int, default=5000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--keep_params", action="store_true")
    a = ap.parse_args()
    print("devices:", jax.devices())
    sweep(a.pe, a.n, a.seeds, a.methods, a.precs, a.n_adam, a.n_lbfgs, a.out, a.keep_params)
