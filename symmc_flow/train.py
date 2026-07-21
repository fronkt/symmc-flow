"""CUDA-aware training loop for SymMC-Flow on the synthetic harness."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import SyntheticCrystalDataset, collate, batch_to_state
from .flow import sample_prior, interpolate, cfm_loss, ot_couple, PriorCache
from .model import SymMCFlow
from . import manifolds as M


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _step_loss(model, batch, weights, device, ot=False, vol_per_atom=10.0,
               centroid_prior_std=None, prior_cache=None, cond_clean_packing=False,
               so3_avg_k=1, logvol_std=0.3, dev_std=0.3):
    """One CFM forward pass. Returns (loss_tensor, parts).

    cond_clean_packing (diagnostic): condition the field on the TRUE lattice+centroid (z1)
    instead of the noised z_t, while still noising orientation and keeping the orient target.
    Tests whether the SO(3) head floors only because it must read a half-noised packing. No
    label leakage (z1.orient is never shown). NOTE: under this flag the lattice/centroid heads
    see a state inconsistent with their own t-velocity targets, so ONLY the orient loss is
    meaningful here.

    so3_avg_k>1 (C5): the SO(3)-AVERAGED orientation objective -- estimate the orient loss as a
    K-sample mean over independent prior rotations (each re-noises orientation, re-interpolates
    the geodesic, and re-evaluates the orient head), reducing the variance of the SO(3) flow
    target while lattice/centroid conditioning is held at the single-sample state (cf. the
    averaged flow-matching of arXiv:2507.09785). k=1 is the default and unchanged."""
    z1 = batch_to_state(batch)
    repr = getattr(model.cfg, "lattice_repr", "shape10")
    fam = getattr(model.cfg, "lattice_family_mask", False)
    sg = batch["sg"]
    if prior_cache is not None:
        z0 = prior_cache.assemble(z1, batch["idx"], sg=sg)
    else:
        z0 = sample_prior(z1, vol_per_atom=vol_per_atom, centroid_prior_std=centroid_prior_std,
                          sg=sg, lattice_repr=repr, family_mask=fam,
                          logvol_std=logvol_std, dev_std=dev_std)
    if ot:
        z0 = ot_couple(z0, z1)
    t = torch.rand(z1.lattice.shape[0], device=device)
    z_t, targets = interpolate(z0, z1, t, sg=sg, lattice_repr=repr, family_mask=fam)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    cond_L = z1.lattice if cond_clean_packing else z_t.lattice
    cond_x = z1.centroid if cond_clean_packing else z_t.centroid
    coset = batch.get("coset")
    pred = model(mol_emb, cond_L, cond_x, z_t.orient, t, batch["sg"], z1.mask, coset=coset)
    if so3_avg_k <= 1:
        return cfm_loss(pred, targets, z1.mask, weights)

    # --- SO(3)-averaged orientation objective (only the orient term is re-sampled) ----------
    v_L, v_x, v_R = pred
    u_L, u_x, u_R = targets
    wL, wx, wR = weights
    m = z1.mask.unsqueeze(-1).float()
    denom = m.sum().clamp_min(1.0)
    loss_L = ((v_L - u_L) ** 2).mean()
    loss_x = (((v_x - u_x) ** 2) * m).sum() / denom
    accR = (((v_R - u_R) ** 2) * m).sum() / denom            # the primary draw
    B, Mm = z1.mask.shape
    for _ in range(so3_avg_k - 1):
        R0k = M.random_so3((B * Mm,)).reshape(B, Mm, 3, 3).to(device=device, dtype=z1.orient.dtype)
        R_tk = M.so3_geodesic(R0k, z1.orient, t.view(-1, 1))
        u_Rk = M.so3_velocity(R0k, z1.orient)
        v_Rk = model(mol_emb, cond_L, cond_x, R_tk, t, batch["sg"], z1.mask, coset=coset)[2]
        accR = accR + (((v_Rk - u_Rk) ** 2) * m).sum() / denom
    loss_R = accR / so3_avg_k
    total = wL * loss_L + wx * loss_x + wR * loss_R
    return total, {"lattice": loss_L.detach(), "centroid": loss_x.detach(),
                   "orient": loss_R.detach(), "total": total.detach()}


@torch.no_grad()
def evaluate(model, loader, weights, device, max_batches=20, ot=False, vol_per_atom=10.0,
             centroid_prior_std=None, prior_cache=None, cond_clean_packing=False,
             logvol_std=0.3, dev_std=0.3):
    model.eval()
    tot = 0.0
    n = 0
    for batch in loader:
        if n >= max_batches:
            break
        batch = move_batch(batch, device)
        _, parts = _step_loss(model, batch, weights, device, ot, vol_per_atom,
                              centroid_prior_std, prior_cache, cond_clean_packing,
                              logvol_std=logvol_std, dev_std=dev_std)
        tot += float(parts["total"])
        n += 1
    model.train()
    return tot / max(n, 1)


def train(model_cfg: ModelConfig | None = None, train_cfg: TrainConfig | None = None,
          verbose: bool = True, train_dataset=None, val_dataset=None):
    """Train SymMC-Flow. If `train_dataset` is None, uses the synthetic harness."""
    mcfg = model_cfg or ModelConfig()
    tcfg = train_cfg or TrainConfig()
    torch.manual_seed(tcfg.seed)
    device = resolve_device(tcfg.device)

    ds = train_dataset or SyntheticCrystalDataset(
        tcfg.n_train, tcfg.max_mols, tcfg.max_atoms_per_mol, tcfg.seed)
    dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate)
    val_dl = (DataLoader(val_dataset, batch_size=tcfg.batch_size, shuffle=False,
                         collate_fn=collate) if val_dataset is not None else None)

    model = SymMCFlow(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)

    # fixed per-structure priors: separate caches per split (idx spaces overlap)
    _pc = dict(lattice_repr=getattr(mcfg, "lattice_repr", "shape10"),
               family_mask=getattr(mcfg, "lattice_family_mask", False),
               logvol_std=getattr(tcfg, "prior_logvol_std", 0.3),
               dev_std=getattr(tcfg, "prior_dev_std", 0.3))
    train_cache = (PriorCache(tcfg.prior_vol_per_atom, tcfg.centroid_prior_std, **_pc)
                   if tcfg.fixed_prior else None)
    val_cache = (PriorCache(tcfg.prior_vol_per_atom, tcfg.centroid_prior_std, **_pc)
                 if tcfg.fixed_prior else None)

    history, step, skipped = [], 0, 0
    model.train()
    while step < tcfg.steps:
        for batch in dl:
            if step >= tcfg.steps:
                break
            batch = move_batch(batch, device)
            loss, parts = _step_loss(model, batch, weights, device,
                                     tcfg.use_ot_coupling, tcfg.prior_vol_per_atom,
                                     tcfg.centroid_prior_std, train_cache,
                                     tcfg.cond_clean_packing, so3_avg_k=tcfg.so3_avg_k,
                                     logvol_std=getattr(tcfg, "prior_logvol_std", 0.3),
                                     dev_std=getattr(tcfg, "prior_dev_std", 0.3))

            opt.zero_grad()
            loss.backward()
            # clip_grad_norm_ returns the total grad norm -> non-finite iff some grad is NaN/inf.
            # Skip the optimizer step on a non-finite loss/grad so one bad batch (e.g. a
            # near-singular config in a higher-capacity model) can't poison the weights;
            # forward behavior is unchanged, so existing checkpoints stay comparable.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            if torch.isfinite(loss) and torch.isfinite(gnorm):
                opt.step()
            else:
                skipped += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                continue

            history.append(parts["total"].item())
            if verbose and step % tcfg.log_every == 0:
                msg = (f"step {step:5d}  loss {parts['total']:.4f}  "
                       f"(L {parts['lattice']:.3f}  x {parts['centroid']:.3f}  "
                       f"R {parts['orient']:.3f})")
                if val_dl is not None and step % (tcfg.log_every * 5) == 0 and step > 0:
                    msg += f"  | val {evaluate(model, val_dl, weights, device, ot=tcfg.use_ot_coupling, vol_per_atom=tcfg.prior_vol_per_atom, centroid_prior_std=tcfg.centroid_prior_std, prior_cache=val_cache, cond_clean_packing=tcfg.cond_clean_packing):.4f}"
                print(msg)
            step += 1

    if verbose and skipped:
        print(f"  (skipped {skipped} non-finite step(s) to protect weights)")
    return model, history


if __name__ == "__main__":
    train()
