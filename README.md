# Physics-Informed Neural Networks for Fluid Flow

Physics-informed neural network (PINN) implementations that solve the incompressible Navier–Stokes equations directly from the governing PDEs, with no CFD training data — only the physics, boundary conditions, and (where available) an analytical solution for validation.

| Project | Problem | Method |
|---|---|---|
| [`taylor-green-vortex-pinn/`](./taylor-green-vortex-pinn) | 2D unsteady decaying vortex, Re = 100 | Time-dependent PINN, SOAP + L-BFGS |
| [`cylinder-flow-pinn/`](./cylinder-flow-pinn) | 2D steady flow past a cylinder, Re = 20 | Steady-state PINN, Adam + L-BFGS |
| [`boundary-layer-peclet-pinn/`](./boundary-layer-peclet-pinn) | Advection–diffusion boundary layers, Pe up to 10⁶ (1D + 2D) | 1,254-run JAX study: when PINNs fail, and a fix |

Both networks take spatial (and, for the vortex case, temporal) coordinates as input and predict velocity and pressure fields `(u, v, p)` directly, trained by minimizing the residual of the Navier–Stokes equations at collocation points plus boundary/initial-condition losses — the standard PINN formulation (Raissi et al.).

## Taylor-Green Vortex

<p align="center">
  <img src="taylor-green-vortex-pinn/images/tgv_04_cell23.png" width="45%"/>
  <img src="taylor-green-vortex-pinn/images/tgv_03_cell20.png" width="45%"/>
</p>

An exact solution exists for this case, which makes it a clean benchmark for PINN accuracy. The trained network reaches a relative L2 error of **~1.6%** on both velocity components at the final time, and tracks the exact kinetic-energy decay curve closely. Details, equations, and all training curves: [`taylor-green-vortex-pinn/README.md`](./taylor-green-vortex-pinn/README.md).

## Flow Past a Cylinder

<p align="center">
  <img src="cylinder-flow-pinn/images/cyl_06_cell28.png" width="45%"/>
  <img src="cylinder-flow-pinn/images/cyl_05_cell27.png" width="45%"/>
</p>

A steady, low-Reynolds-number (Re = 20) channel flow around a circular obstacle, with a parabolic inlet profile and no-slip walls/cylinder. The network recovers the expected recirculation bubble behind the cylinder and the pressure buildup on its upstream face. Details: [`cylinder-flow-pinn/README.md`](./cylinder-flow-pinn/README.md).

## When does a PINN miss a boundary layer?

<p align="center">
  <img src="boundary-layer-peclet-pinn/images/rule_points_per_layer_vs_cell_peclet.png" width="90%"/>
</p>

A controlled study (1,254 trainings, JAX) of PINNs on thin advection–diffusion boundary layers. A plain PINN gets the wall flux wrong once the spacing between collocation points exceeds about 1.5 layer thicknesses, with the same threshold in two different flows; the CFD cell-Péclet criterion is off by a factor of ~100. A zoomed boundary-layer input plus a Péclet curriculum cuts the wall-flux error from 67% to 5% on the hardest showcase case. Details: [`boundary-layer-peclet-pinn/README.md`](./boundary-layer-peclet-pinn/README.md).

## Stack

PyTorch (Navier–Stokes projects) and JAX (boundary-layer study), automatic differentiation for the PDE residuals, Adam/SOAP for first-stage optimization, L-BFGS for fine-tuning convergence.
