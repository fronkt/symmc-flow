"""Orientation-ON training on the REAL CSD rigid-body molecular-crystal corpus.

THE thesis experiment. The headline novelty (rigid-body conformers + SO(3) orientation flow)
was previously validated only on synthetic data and on single-atom benchmarks with
lambda_orient=0. This trains the orientation flow (lambda_orient>0) on real molecular
crystals exported from the CSD (csd_export.py -> factorize_cifs.py cache), with a held-out
validation split, and reports the per-head loss (lattice / centroid / ORIENTATION) before vs
after training. The claim is proven if the orientation loss DESCENDS and GENERALIZES to the
val split -- i.e. the SO(3) head learns real, packing-determined molecular orientation.

    python scripts/train_csd_molcrystal.py --cache data/csd_mol/ds.pt --steps 1500

Reproducible from the manifest + seed; no synthetic data anywhere in this run.
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
    """Mean cell volume per real atom over the corpus -> matches the lattice prior to data
    scale (organic crystals are far less dense than the synthetic default of 10 A^3/atom)."""
    vols, atoms = 0.0, 0
    for i in range(len(ds)):
        it = ds[i]
        vols += abs(float(torch.det(it["lattice"])))
        atoms += int(it["atom_mask"].sum())
    return vols / max(atoms, 1)


@torch.no_grad()
def per_head_val(model, ds, device, weights, vol_per_atom, seed=0, batch_size=16, passes=4):
    """Average per-head CFM loss over the split. `passes` independent draws of (t, prior)
    average out the CFM target stochasticity so pre/post comparison is stable; the seed is
    fixed so the untrained and trained measurements see the SAME random conditions."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    acc = {"lattice": 0.0, "centroid": 0.0, "orient": 0.0, "total": 0.0}
    n = 0
    was_training = model.training
    model.eval()
    for p in range(passes):
        torch.manual_seed(seed + p)
        for batch in dl:
            batch = move_batch(batch, device)
            _, parts = _step_loss(model, batch, weights, device, vol_per_atom=vol_per_atom)
            for k in acc:
                acc[k] += float(parts[k])
            n += 1
    if was_training:
        model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt", help="factorized dataset cache")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="3e-4 default: the intrinsic-frame orient targets give larger early "
                         "gradients that diverge under Adam at 1e-3 (NaN ~step 17)")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--lambda-orient", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/csd_molcrystal.pt")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        sys.exit(f"no cache at {args.cache}; run csd_export.py + factorize_cifs.py first")

    full = MolCrystalDataset(cache_path=args.cache)
    n = len(full)
    print(f"corpus: {n} real CSD molecular crystals (cache {args.cache})")
    if n < 40:
        print("  [warn] small corpus; scale csd_export --n/--sample for a stronger result")

    # deterministic train/val split
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
                       seed=args.seed, log_every=50, prior_vol_per_atom=vpa)
    device = resolve_device(tcfg.device)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)
    print(f"  device={device}  lambda_orient={args.lambda_orient}  steps={args.steps}\n")

    # untrained baseline on val (same seed/conditions as the post measurement)
    torch.manual_seed(args.seed)
    base = SymMCFlow(mcfg).to(device)
    pre = per_head_val(base, val_ds, device, weights, vpa, seed=12345)

    model, history = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    post = per_head_val(model, val_ds, device, weights, vpa, seed=12345)

    print("\n==== held-out validation, per head (untrained -> trained) ====")
    for k in ("lattice", "centroid", "orient", "total"):
        drop = 100 * (1 - post[k] / pre[k]) if pre[k] else 0.0
        tag = "  <-- ORIENTATION (thesis head)" if k == "orient" else ""
        print(f"  {k:9s}: {pre[k]:.4f} -> {post[k]:.4f}  ({drop:+.1f}%){tag}")

    assert post["orient"] < pre["orient"], "orientation flow did NOT generalize on real data"
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__,
                "vol_per_atom": vpa, "val_idx": val_idx, "train_idx": train_idx,
                "pre_val": pre, "post_val": post}, args.ckpt)
    print(f"\nOK: SO(3) orientation flow trains + generalizes on REAL CSD molecular crystals.")
    print(f"checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
