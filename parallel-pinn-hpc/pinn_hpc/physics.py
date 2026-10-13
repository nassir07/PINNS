"""Incompressible 2D Navier-Stokes: exact solution, residuals, loss and error metric."""
import torch


def taylor_green(X: torch.Tensor, nu: float) -> torch.Tensor:
    """Exact 2D Taylor-Green vortex (density = 1) on [0, 2pi]^2. Returns (u, v, p)."""
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    F = torch.exp(-2.0 * nu * t)
    u = -torch.cos(x) * torch.sin(y) * F
    v = torch.sin(x) * torch.cos(y) * F
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * F**2
    return torch.cat([u, v, p], dim=1)


def _grad(f: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # create_graph=True: these derivatives enter the loss, so they must remain
    # differentiable with respect to the network weights.
    return torch.autograd.grad(f, X, grad_outputs=torch.ones_like(f), create_graph=True)[0]


def navier_stokes_residuals(out: torch.Tensor, X: torch.Tensor, nu: float):
    """PDE residuals of the prediction out = (u, v, p) at points X (X.requires_grad must be True).

    momentum-x : u_t + u u_x + v u_y + p_x - nu (u_xx + u_yy)
    momentum-y : v_t + u v_x + v v_y + p_y - nu (v_xx + v_yy)
    continuity : u_x + v_y
    """
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    du, dv, dp = _grad(u, X), _grad(v, X), _grad(p, X)
    u_x, u_y, u_t = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    v_x, v_y, v_t = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]

    u_xx = _grad(u_x, X)[:, 0:1]
    u_yy = _grad(u_y, X)[:, 1:2]
    v_xx = _grad(v_x, X)[:, 0:1]
    v_yy = _grad(v_y, X)[:, 1:2]

    r_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    r_c = u_x + v_y
    return r_u, r_v, r_c


def pinn_loss(model, X_f: torch.Tensor, X_0: torch.Tensor, U_0: torch.Tensor,
              nu: float, w_ic: float = 10.0) -> dict:
    """PDE residual loss on collocation points + initial-condition loss.

    Collocation and initial points share ONE forward pass: with DDP, exactly one
    forward per backward keeps gradient synchronisation simple and correct.
    """
    n_f = X_f.shape[0]
    X = torch.cat([X_f, X_0], dim=0).detach().requires_grad_(True)
    out = model(X)

    r_u, r_v, r_c = navier_stokes_residuals(out, X, nu)
    l_pde = (r_u[:n_f] ** 2).mean() + (r_v[:n_f] ** 2).mean() + (r_c[:n_f] ** 2).mean()
    l_ic = ((out[n_f:] - U_0) ** 2).mean()  # u, v, p at t = 0 (p also fixes the pressure gauge)
    return {"total": l_pde + w_ic * l_ic, "pde": l_pde, "ic": l_ic}


@torch.no_grad()
def relative_l2(model, X: torch.Tensor, U: torch.Tensor) -> dict:
    """Relative L2 errors of u, v, p. Pressure is compared after removing its mean (gauge freedom)."""
    pred = model(X)
    err = {}
    for i, name in enumerate(("u", "v", "p")):
        a, b = pred[:, i], U[:, i]
        if name == "p":
            a, b = a - a.mean(), b - b.mean()
        err[name] = (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()
    return err
