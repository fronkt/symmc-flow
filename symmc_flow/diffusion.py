"""DiffCSP-style diffusion baseline for SymMC-Flow.

Reuses the SymMCFlow network UNCHANGED so a flow-vs-diffusion comparison isolates the
*objective*, not the architecture. The network's two carbon-relevant heads are reread:
  - lattice head (10-d)   -> DDPM noise prediction eps over the lattice param-space k.
  - centroid head (3-d)   -> the *scaled score* (sigma * score) of a wrapped-normal
                             diffusion of the fractional coordinates on the torus.
Orientation is not diffused (carbon = single atoms, lambda_orient = 0).

Two coupled processes share a discrete time t in {1..T}:
  - lattice k in R^10: variance-preserving DDPM with a cosine alpha-bar (eps-prediction).
  - fractional f on T^3: variance-exploding wrapped-normal, geometric sigma schedule
    (DiffCSP: sigma 0.005 -> 0.5), denoising score matching.
Sampling: lattice by deterministic DDIM, fractional by the (tuning-free) ancestral SMLD
reverse chain on the torus. Prior: k ~ N(0, I), f ~ Uniform[0,1).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .flow import CrystalState, n_atoms
from . import manifolds as M


def _cosine_alpha_bar(T: int, device, dtype, s: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal cosine schedule -> alpha_bar[0..T], alpha_bar[0]=1."""
    steps = torch.arange(T + 1, device=device, dtype=dtype)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1.0)


@dataclass
class DiffusionProcess:
    T: int = 1000
    sigma_min: float = 0.005
    sigma_max: float = 0.5

    def alpha_bar(self, device, dtype) -> torch.Tensor:
        return _cosine_alpha_bar(self.T, device, dtype)

    def sigmas(self, device, dtype) -> torch.Tensor:
        """Geometric sigma_t increasing with t; index 0..T (sigma_0 = sigma_min)."""
        t = torch.arange(self.T + 1, device=device, dtype=dtype) / self.T
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t


def wrapped_normal_score(delta: torch.Tensor, sigma: torch.Tensor,
                         n_images: int = 4) -> torch.Tensor:
    """d/d(delta) log sum_j N(delta + j; 0, sigma^2) on the unit torus.
    delta wrapped to (-0.5, 0.5]; sigma broadcastable to delta."""
    var = sigma ** 2
    num = torch.zeros_like(delta)
    den = torch.zeros_like(delta)
    for j in range(-n_images, n_images + 1):
        y = delta + j
        w = torch.exp(-0.5 * y * y / var)
        num = num + w * (-y / var)
        den = den + w
    return num / den.clamp_min(1e-20)


def _wrap_pm(x: torch.Tensor) -> torch.Tensor:
    """Wrap to (-0.5, 0.5]."""
    return x - torch.round(x)


def diffusion_targets(proc: DiffusionProcess, z1: CrystalState, t_idx: torch.Tensor):
    """Forward-diffuse (lattice k, fractional f) to step t_idx:(B,) ints in {1..T}.
    Returns (L_t, f_t, t_cont (B,), eps (B,10), score_target (B,M,3), sigma_t (B,))."""
    dev, dtype = z1.lattice.device, z1.lattice.dtype
    n = n_atoms(z1)
    abar = proc.alpha_bar(dev, dtype)[t_idx]                     # (B,)
    sig = proc.sigmas(dev, dtype)[t_idx]                         # (B,)

    k0 = M.lattice_to_param(z1.lattice, n)                       # (B,10)
    eps = torch.randn_like(k0)
    k_t = abar.sqrt().unsqueeze(-1) * k0 + (1 - abar).sqrt().unsqueeze(-1) * eps
    L_t = M.param_to_lattice(k_t, n)

    f0 = z1.centroid
    z = torch.randn_like(f0)
    f_t = M.wrap(f0 + sig.view(-1, 1, 1) * z)
    delta = _wrap_pm(f_t - f0)
    score_target = sig.view(-1, 1, 1) * wrapped_normal_score(delta, sig.view(-1, 1, 1))

    t_cont = t_idx.to(dtype) / proc.T
    return L_t, f_t, t_cont, eps, score_target, sig


def diffusion_loss(model, batch, proc: DiffusionProcess, device,
                   w_lat: float = 1.0, w_frac: float = 1.0):
    """DDPM eps-loss on the lattice + scaled-score DSM on the fractional coords."""
    from .data import batch_to_state
    z1 = batch_to_state(batch)
    B = z1.lattice.shape[0]
    t_idx = torch.randint(1, proc.T + 1, (B,), device=device)
    L_t, f_t, t_cont, eps, score_tgt, _ = diffusion_targets(proc, z1, t_idx)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    eps_pred, score_pred, _ = model(mol_emb, L_t, f_t, z1.orient, t_cont,
                                    batch["sg"], z1.mask)
    loss_lat = ((eps_pred - eps) ** 2).mean()
    m = z1.mask.unsqueeze(-1).float()
    loss_frac = (((score_pred - score_tgt) ** 2) * m).sum() / m.sum().clamp_min(1.0)
    total = w_lat * loss_lat + w_frac * loss_frac
    return total, {"lattice": loss_lat.detach(), "frac": loss_frac.detach(),
                   "total": total.detach()}


@torch.no_grad()
def diffusion_sample(model, mol_emb, init: CrystalState, sg, proc: DiffusionProcess,
                     steps: int = 250) -> CrystalState:
    """Reverse diffusion to a clean CrystalState. `init` supplies n / mask / orient.
    Lattice: deterministic DDIM. Fractional: ancestral SMLD reverse chain on the torus."""
    dev, dtype = init.lattice.device, init.lattice.dtype
    n = n_atoms(init)
    mask = init.mask
    B, Mm = mask.shape
    abar = proc.alpha_bar(dev, dtype)
    sig = proc.sigmas(dev, dtype)
    m = mask.unsqueeze(-1).float()

    # grid of integer timesteps from T down to 1
    ts = torch.linspace(proc.T, 1, steps, device=dev).round().long().clamp(1, proc.T)

    k = torch.randn(B, 10, device=dev, dtype=dtype)              # VP prior ~ N(0,I)
    f = torch.rand(B, Mm, 3, device=dev, dtype=dtype) * m        # VE prior ~ Uniform
    R = init.orient

    for i in range(steps):
        t = int(ts[i])
        t_next = int(ts[i + 1]) if i + 1 < steps else 0
        t_cont = torch.full((B,), t / proc.T, device=dev, dtype=dtype)
        abar_t, sig_t = abar[t], sig[t]
        abar_next = abar[t_next] if t_next > 0 else torch.ones((), device=dev, dtype=dtype)
        sig_next = sig[t_next] if t_next > 0 else proc.sigma_min

        L = M.param_to_lattice(k, n)
        eps_pred, score_scaled, _ = model(mol_emb, L, f, R, t_cont, sg, mask)

        # lattice: DDIM (eta=0) -- works for any subsampled grid, deterministic.
        # Clamp the x0 prediction (static thresholding): without it the divide by the
        # tiny sqrt(abar_t) at high noise blows the log-volume up and exp() overflows.
        k0 = (k - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt().clamp_min(1e-3)
        k0 = k0.clamp(-10.0, 10.0)
        k0[..., 0] = k0[..., 0].clamp(-2.0, 6.0)  # per-atom log-volume stays physical
        k = abar_next.sqrt() * k0 + (1 - abar_next).sqrt() * eps_pred

        # fractional: ancestral SMLD reverse step on the torus
        score = score_scaled / sig_t
        drift = (sig_t ** 2 - sig_next ** 2) * score
        if t_next > 0:
            std = (sig_next ** 2 * (sig_t ** 2 - sig_next ** 2) / sig_t ** 2).clamp_min(0).sqrt()
            noise = std * torch.randn_like(f)
        else:
            noise = 0.0
        f = M.wrap(f + drift + noise) * m

    return CrystalState(M.param_to_lattice(k, n), f * m, R, mask)
