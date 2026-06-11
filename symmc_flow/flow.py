"""Conditional flow-matching paths and the joint CFM loss.

Builds an interpolated state z_t and its target velocity u_t for each of the
three manifolds, then the loss is the squared error between the network field
v_theta and u_t, summed with per-manifold weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from . import manifolds as M


@dataclass
class CrystalState:
    """A batch of (rigid-body) molecular-crystal states.
    lattice:(B,3,3)  centroid:(B,Mmax,3) in [0,1)  orient:(B,Mmax,3,3)  mask:(B,Mmax)."""
    lattice: torch.Tensor
    centroid: torch.Tensor
    orient: torch.Tensor
    mask: torch.Tensor


def sample_prior(like: CrystalState, lattice_scale: float = 1.0) -> CrystalState:
    B, Mmax = like.mask.shape
    dev, dt = like.lattice.device, like.lattice.dtype
    return CrystalState(
        lattice=lattice_scale * M.prior_lattice((B,), dev, dt),
        centroid=M.prior_centroid((B, Mmax), dev, dt),
        orient=M.prior_orientation((B, Mmax), dev, dt),
        mask=like.mask,
    )


def ot_couple(z0: CrystalState, z1: CrystalState) -> CrystalState:
    """Optimal-transport coupling: per structure, permute the (exchangeable) prior
    atoms to the minimum-cost assignment against the data atoms using torus distance.
    Removes the arbitrary prior->data pairing that makes the centroid target
    unlearnable. Permutes prior centroids + orientations together."""
    from scipy.optimize import linear_sum_assignment
    x0, x1 = z0.centroid, z1.centroid
    R0 = z0.orient
    mask = z1.mask
    B = x0.shape[0]
    new_x0 = x0.clone()
    new_R0 = R0.clone()
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        a, c = x0[b, idx], x1[b, idx]                       # (n,3) prior, data
        d = a.unsqueeze(1) - c.unsqueeze(0)                 # (n_prior, n_data, 3)
        d = d - d.round()                                  # minimum image on torus
        cost = (d ** 2).sum(-1)                             # (n_prior, n_data)
        ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
        ri = torch.as_tensor(ri, device=x0.device)
        ci = torch.as_tensor(ci, device=x0.device)
        na, nR = a.clone(), R0[b, idx].clone()
        na[ci] = a[ri]                                      # align prior to data order
        nR[ci] = R0[b, idx][ri]
        new_x0[b, idx] = na
        new_R0[b, idx] = nR
    return CrystalState(z0.lattice, new_x0, new_R0, z0.mask)


def interpolate(z0: CrystalState, z1: CrystalState, t: torch.Tensor):
    """Return (z_t, targets) where targets = (u_lattice, u_centroid, u_orient).
    t:(B,) in [0,1]."""
    tL = t.view(-1, 1, 1)
    tx = t.view(-1, 1, 1)
    L_t = M.lattice_geodesic(z0.lattice, z1.lattice, tL)
    u_L = M.lattice_velocity(z0.lattice, z1.lattice)

    x_t = M.torus_geodesic(z0.centroid, z1.centroid, tx)
    u_x = M.torus_velocity(z0.centroid, z1.centroid)

    R_t = M.so3_geodesic(z0.orient, z1.orient, t.view(-1, 1))
    u_R = M.so3_velocity(z0.orient, z1.orient)

    z_t = CrystalState(L_t, x_t, R_t, z1.mask)
    return z_t, (u_L, u_x, u_R)


def cfm_loss(pred, targets, mask, weights=(1.0, 1.0, 1.0)):
    """pred = (v_L (B,3,3), v_x (B,M,3), v_R (B,M,3)). targets same shapes.
    Returns scalar loss and a dict of components."""
    v_L, v_x, v_R = pred
    u_L, u_x, u_R = targets
    wL, wx, wR = weights
    m = mask.unsqueeze(-1).float()                  # (B,M,1)
    denom = m.sum().clamp_min(1.0)

    loss_L = ((v_L - u_L) ** 2).mean()
    loss_x = (((v_x - u_x) ** 2) * m).sum() / denom
    loss_R = (((v_R - u_R) ** 2) * m).sum() / denom
    total = wL * loss_L + wx * loss_x + wR * loss_R
    return total, {"lattice": loss_L.detach(), "centroid": loss_x.detach(),
                   "orient": loss_R.detach(), "total": total.detach()}
