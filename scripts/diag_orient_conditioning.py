"""DIAGNOSTIC: is the SO(3) orientation flow floored by NOISED-PACKING conditioning?

On the real CSD corpus the orientation head sits at its predict-zero floor (E||u_R||^2 ~ 5.24)
while lattice + centroid learn. Leading hypothesis: orientation is determined by the CLEAN
packing (true lattice + true centroids), but the network is only ever shown a half-noised
packing (in train._step_loss the field reads z_t.lattice/z_t.centroid at the same flow time t).

This run sets `cond_clean_packing=True`: the field is conditioned on the TRUE lattice+centroid
while orientation is still noised (z_t.orient) and the orient target u_R is unchanged. There is
NO label leakage -- z1.orient is never shown; the head must infer the rotation velocity from
the clean packing geometry. This is literally STAGE 2 of the eventual two-stage model
("orientation | true packing").

Interpretation (ONLY the orient number is meaningful here -- under clean conditioning the
lattice/centroid heads see a state inconsistent with their own t-velocity targets):
  * orient val loss DESCENDS clearly below the ~5.2 floor + generalizes -> noised conditioning
    IS the cause; the two-stage architecture is validated -> build it next.
  * orient still floors -> the cause is deeper (orientation not packing-determined in this
    corpus, or a target/gauge issue) -> paper-fallback framing.

Control = scripts/train_csd_molcrystal.py (same setup, noised conditioning), which stays at the
floor. Same seed/split/prior, so the only difference is the conditioning.

    python scripts/diag_orient_conditioning.py --cache data/csd_mol/ds.pt --steps 800
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.molcrystal import MolCrystalDataset
from symmc_flow.model import SymMCFlow
from symmc_flow.train import train, _step_loss, move_batch, resolve_device
from symmc_flow.data import collate


def corpus_vol_per_atom(ds):
    """Mean cell volume per real atom over the corpus -> matches the lattice prior to data."""
    vols, atoms = 0.0, 0
    for i in range(len(ds)):
        it = ds[i]
        vols += abs(float(torch.det(it["lattice"])))
        atoms += int(it["atom_mask"].sum())
    return vols / max(atoms, 1)


@torch.no_grad()
def per_head_val(model, ds, device, weights, vol_per_atom, cond_clean_packing,
                 seed=0, batch_size=16, passes=4):
    """Average per-head CFM loss over the split with the SAME conditioning used in training
    (cond_clean_packing). `passes` independent (t, prior) draws average out target noise;
    the seed is fixed so untrained and trained measurements see identical random conditions."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    acc = {"lattice": 0.0, "centroid": 0.0, "orient": 0.0, "total": 0.0}
    n = 0
    was_training = model.training
    model.eval()
    for p in range(passes):
        torch.manual_seed(seed + p)
        for batch in dl:
            batch = move_batch(batch, device)
            _, parts = _step_loss(model, batch, weights, device, vol_per_atom=vol_per_atom,
                                  cond_clean_packing=cond_clean_packing)
            for k in acc:
                acc[k] += float(parts[k])
            n += 1
    if was_training:
        model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt", help="factorized dataset cache")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--lambda-orient", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/diag_orient_cleanpack.pt")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        sys.exit(f"no cache at {args.cache}; run csd_export.py + factorize_cifs.py first")

    full = MolCrystalDataset(cache_path=args.cache)
    n = len(full)
    print(f"corpus: {n} real CSD molecular crystals (cache {args.cache})")
    print("DIAGNOSTIC: cond_clean_packing=True (orientation conditioned on TRUE lattice+centroid)\n")

    # deterministic train/val split -- IDENTICAL to train_csd_molcrystal.py (same seed)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(4, int(round(args.val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    print(f"  split: {len(train_ds)} train / {len(val_ds)} val")

    vpa = corpus_vol_per_atom(full)
    print(f"  lattice prior vol/atom matched to corpus: {vpa:.1f} A^3")

    mcfg = ModelConfig(lambda_orient=args.lambda_orient)
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       seed=args.seed, log_every=50, prior_vol_per_atom=vpa,
                       cond_clean_packing=True)
    device = resolve_device(tcfg.device)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)
    print(f"  device={device}  lambda_orient={args.lambda_orient}  steps={args.steps}\n")

    # untrained baseline on val (same seed/conditions as the post measurement)
    torch.manual_seed(args.seed)
    base = SymMCFlow(mcfg).to(device)
    pre = per_head_val(base, val_ds, device, weights, vpa, cond_clean_packing=True, seed=12345)

    model, history = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    post = per_head_val(model, val_ds, device, weights, vpa, cond_clean_packing=True, seed=12345)

    print("\n==== held-out validation, per head (untrained -> trained) ====")
    print("     [only ORIENT is meaningful under clean-packing conditioning]")
    for k in ("lattice", "centroid", "orient", "total"):
        drop = 100 * (1 - post[k] / pre[k]) if pre[k] else 0.0
        tag = "  <-- ORIENTATION (thesis head)" if k == "orient" else ""
        print(f"  {k:9s}: {pre[k]:.4f} -> {post[k]:.4f}  ({drop:+.1f}%){tag}")

    floor = pre["orient"]
    verdict = ("CONFIRMED: orientation learns when conditioned on the true packing -> "
               "noised conditioning is the cause; build the two-stage model."
               if post["orient"] < 0.9 * floor else
               "NULL: orientation still at floor even on the true packing -> cause is deeper; "
               "paper-fallback framing.")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__,
                "vol_per_atom": vpa, "val_idx": val_idx, "train_idx": train_idx,
                "pre_val": pre, "post_val": post, "cond_clean_packing": True}, args.ckpt)
    print(f"checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
