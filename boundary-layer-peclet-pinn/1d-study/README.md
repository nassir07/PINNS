# A collocation Péclet number for PINNs

Where do physics-informed neural networks (PINNs) break on boundary layers? Is each failure *overfitting* or an *optimisation* failure? Which fix helps in which regime?

## Problem

1D steady advection–diffusion. This is the simplest boundary-layer problem, and it has an exact solution:

```
c'(x) = (1/Pe) c''(x),   x ∈ (0,1),   c(0)=0, c(1)=1
c(x)  = (exp(Pe(x−1)) − exp(−Pe)) / (1 − exp(−Pe))        layer at x=1, thickness δ ≈ 1/Pe
J(x)  = c − c'/Pe = −1/(e^Pe − 1)   (constant flux, used as a conservation check)
```

**Collocation Péclet number.** With N uniform collocation points (spacing h = 1/N):

```
Pe_h = Pe · h = Pe / N  ≈  h / δ
```

It says how many collocation spacings fit inside the boundary layer. It is the PINN analogue of the cell Péclet number in finite differences, where Pe_h > 2 triggers oscillations in central schemes.

## Pre-registered hypotheses

- **H1:** the failure boundary in the (Pe, N) plane follows a line of constant Pe_h, of order 1–10.
- **H2:** the alternative (Arzani et al., JCP 2023): adding points does not fix boundary-layer failure, so the boundary does not move with N.
- **H3:** failures at Pe_h ≫ 1 are *overfitting*: a small residual at the collocation points and a large one between them (gap G ≫ 1). Failures at Pe_h ≲ 1 are *optimisation* failures: the training loss itself is stuck.
- **H4:** FP64 helps only in the optimisation-failure region. Regularisation and adaptive sampling help only in the overfitting region.

## Methods (tanh MLP 4×64, Adam 10k steps then L-BFGS 5k)

| key | what |
|---|---|
| `vanilla` | soft BCs, FP32 and FP64 |
| `hard` | hard BCs: c = x + x(1−x)·NN(x) |
| `gradreg` | adds h²·mean(dR/dx)². The weight h² is a first-order Taylor estimate of the residual between points |
| `rad` | residual-based adaptive distribution (Wu et al. 2023), N held fixed |
| `xi` | extra input ξ = (1−x)·Pe, the inner coordinate of matched asymptotics |
| `xi_hard` | ξ + hard BCs |

## Metrics

- relative L2 error on a grid refined inside the layer
- boundary-layer L2 error (x > 1 − 5/Pe)
- **residual gap** G = log10(∫R² dx / mean R² at training points)
- max flux error |J − J_exact|

A run succeeds when its relative L2 error is below 5%.

**Regime labels:**
- success
- *overfitting*: fails and G > 1
- *optimisation failure*: fails and G ≤ 1

## Run

```bash
pip install "jax[cuda12]" optax matplotlib pandas
python pinn_pe.py --methods vanilla --precs f64 f32 --out results      # sweep (vmapped over Pe × seeds)
python plots.py --results results --out figs                            # phase diagram, regime map, collapse plot
python animate.py --pe 1000 --n 64 --method vanilla --out anim/v.npz    # training snapshots
python render_anim.py anim/v.npz --out figs/comb.gif                    # the "comb" animation
```

`kaggle_cell.py` is the one-cell driver used on Kaggle (2× T4). It splits the methods across both GPUs.
