"""Deterministic RK4 ODE sampler over the three manifolds.

Integrates dz/dt = v_theta(z_t, t, c) from the prior (t=0) to data (t=1) in N
steps. Each manifold uses its own update/retraction:
  - lattice: Euclidean RK4 in (log-volume, shape) param space, decoded to a
    3x3 matrix for each field evaluation
  - centroid: torus update + re-wrap into [0,1)
  - orient: SO(3) exponential-map retraction (stays on the manifold), and the
    so(3) velocities are re-expressed across the half/full steps.
For SO(3) we use a manifold RK4 in the body frame: average the four tangent
estimates and apply a single exp-map update (a standard Munthe-Kaas style step).
"""
from __future__ import annotations
import torch

from .flow import CrystalState
from . import manifolds as M


@torch.no_grad()
def rk4_sample(model, mol_emb, init: CrystalState, sg, steps: int = 50, symmetrize: bool = False):
    """Returns a CrystalState at t=1. `model` is a SymMCFlow."""
    mask = init.mask
    n = mask.sum(-1).clamp_min(1)
    kL = M.lattice_to_param(init.lattice, n)
    x = init.centroid.clone()
    R = init.orient.clone()
    B = kL.shape[0]
    dt = 1.0 / steps
    dev, dtp = kL.device, kL.dtype

    def field(k_, x_, R_, t_scalar):
        t = torch.full((B,), float(t_scalar), device=dev, dtype=dtp)
        L_ = M.param_to_lattice(k_, n)
        if symmetrize:
            return model.symmetrized_velocity(mol_emb, L_, x_, R_, t, sg, mask)
        return model.forward(mol_emb, L_, x_, R_, t, sg, mask)

    for i in range(steps):
        t0 = i * dt
        # --- lattice & centroid: classic RK4 (param-space / torus tangent) ----
        # --- orient: collect tangent estimates, apply one exp-map step --------
        k1L, k1x, k1R = field(kL, x, R, t0)
        kL2 = kL + 0.5 * dt * k1L
        x2 = M.wrap(x + 0.5 * dt * k1x)
        R2 = R @ M.so3_exp(0.5 * dt * k1R)
        k2L, k2x, k2R = field(kL2, x2, R2, t0 + 0.5 * dt)
        kL3 = kL + 0.5 * dt * k2L
        x3 = M.wrap(x + 0.5 * dt * k2x)
        R3 = R @ M.so3_exp(0.5 * dt * k2R)
        k3L, k3x, k3R = field(kL3, x3, R3, t0 + 0.5 * dt)
        kL4 = kL + dt * k3L
        x4 = M.wrap(x + dt * k3x)
        R4 = R @ M.so3_exp(dt * k3R)
        k4L, k4x, k4R = field(kL4, x4, R4, t0 + dt)

        kL = kL + (dt / 6.0) * (k1L + 2 * k2L + 2 * k3L + k4L)
        x = M.wrap(x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x))
        omega = (dt / 6.0) * (k1R + 2 * k2R + 2 * k3R + k4R)
        R = R @ M.so3_exp(omega)
        R = M.project_so3(R)  # guard against numerical drift

    m = mask.unsqueeze(-1).float()
    return CrystalState(M.param_to_lattice(kL, n), x * m, R, mask)
