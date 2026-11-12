"""
Training snapshots for the "comb" animation: residual R(x) on a fine grid vs the collocation points,
plus the solution c(x), recorded during Adam + L-BFGS for one (Pe, N, method) configuration.

    python animate.py --pe 1000 --n 64 --method vanilla --out anim_pe1000_n64.npz
"""
import argparse, numpy as np, jax, jax.numpy as jnp, optax
from pinn_pe import (init_mlp, n_inputs, make_loss, residual, model, uniform_pts, exact, DT)

ap = argparse.ArgumentParser()
ap.add_argument("--pe", type=float, default=1000.0)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--method", default="vanilla")
ap.add_argument("--prec", default="f64")
ap.add_argument("--n_adam", type=int, default=10000)
ap.add_argument("--n_lbfgs", type=int, default=5000)
ap.add_argument("--every", type=int, default=250)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dt = DT[a.prec]
Pe = jnp.asarray(a.pe, dt)
N = a.n
h = jnp.asarray(1.0 / N, dt)
pts = uniform_pts(N, dt)
params = init_mlp(jax.random.PRNGKey(1000 * a.seed + 7), [n_inputs(a.method)] + [64] * 4 + [1], dt)
loss = make_loss(Pe, pts, a.method, h)

# fine grid: uniform + dense in the layer
xg = np.unique(np.concatenate([np.linspace(0, 1, 8001), 1 - np.linspace(0, 12, 1201) / a.pe]).clip(0, 1))
xg_j = jnp.asarray(xg, dt)
Rv = jax.jit(jax.vmap(residual, in_axes=(None, 0, None, None)), static_argnums=3)
Cv = jax.jit(jax.vmap(model, in_axes=(None, 0, None, None)), static_argnums=3)

adam = optax.adam(optax.exponential_decay(1e-3, 1000, 0.9))
s = adam.init(params)


@jax.jit
def adam_chunk(p, s):
    def step(c, _):
        p, s = c
        v, g = jax.value_and_grad(loss)(p)
        u, s = adam.update(g, s, p)
        return (optax.apply_updates(p, u), s), v
    (p, s), hist = jax.lax.scan(step, (p, s), None, length=a.every)
    return p, s, hist


lb = optax.lbfgs()
vg = optax.value_and_grad_from_state(loss)


@jax.jit
def lb_chunk(p, s):
    def step(c, _):
        p, s = c
        v, g = vg(p, state=s)
        u, s = lb.update(g, s, p, value=v, grad=g, value_fn=loss)
        return (optax.apply_updates(p, u), s), v
    (p, s), hist = jax.lax.scan(step, (p, s), None, length=a.every)
    return p, s, hist


frames = dict(it=[], R=[], Rpts=[], c=[], loss=[], phase=[])


def snap(it, phase, lval):
    frames["it"].append(it); frames["phase"].append(phase); frames["loss"].append(float(lval))
    frames["R"].append(np.asarray(Rv(params, xg_j, Pe, a.method)))
    frames["Rpts"].append(np.asarray(Rv(params, pts, Pe, a.method)))
    frames["c"].append(np.asarray(Cv(params, xg_j, Pe, a.method)))


snap(0, 0, loss(params))
it = 0
for _ in range(a.n_adam // a.every):
    params, s, hist = adam_chunk(params, s)
    it += a.every
    snap(it, 0, hist[-1])
    print(it, float(hist[-1]), flush=True)
sl = lb.init(params)
for _ in range(a.n_lbfgs // a.every):
    params, sl, hist = lb_chunk(params, sl)
    it += a.every
    snap(it, 1, hist[-1])
    print(it, float(hist[-1]), flush=True)

np.savez_compressed(a.out, x=xg, pts=np.asarray(pts), exact=np.asarray(exact(xg_j, Pe)), Pe=a.pe, N=N,
                    method=a.method, **{k: np.array(v) for k, v in frames.items()})
print("saved", a.out)
