"""DIAGNOSTIC: does the SO(3) flow learn the WITHIN-CRYSTAL RELATIVE orientation?

The absolute per-molecule target R_m bundles two parts: rot(g_m) from the space-group op that
generates copy m (learnable) and the asymmetric unit's FREE orientation R_asym (per-crystal,
gauge-arbitrary, unlearnable) -- and R_asym dominates, pinning the head at its predict-zero
floor (confirmed even on the true packing by diag_orient_conditioning.py). This run re-gauges
each crystal to the RELATIVE convention (`relative_gauge_item`): the first copy of each species
is the reference (orient := I) and the rest carry R'_m = R_m R0^{-1}, cancelling R_asym so the
only remaining signal is the symmetry-induced relative rotation. The corpus is restricted to
crystals with >=2 copies of a species (so a non-trivial relative target exists).

To avoid a trivial "predict identity" positive, the orient loss is reported split by
reference copies (target I, trivial) vs NON-reference copies (the genuinely symmetry-determined
targets). The claim is proven iff the NON-REFERENCE orient loss descends + generalizes.

  python scripts/diag_orient_relative.py --cache data/csd_mol/ds.pt --steps 800 [--clean-packing]

--clean-packing also conditions on the true lattice+centroid (best case = relative target AND
clean packing). If even that floors, orientation is unlearnable here -> reframe is final.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item,
                                    species_multiplicity)
from symmc_flow.model import SymMCFlow
from symmc_flow.train import train, _step_loss, move_batch, resolve_device
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior, interpolate


def corpus_vol_per_atom(items):
    vols, atoms = 0.0, 0
    for it in items:
        vols += abs(float(torch.det(it["lattice"])))
        atoms += int(it["atom_mask"].sum())
    return vols / max(atoms, 1)


@torch.no_grad()
def per_head_val(model, ds, device, weights, vol_per_atom, cond_clean_packing,
                 seed=12345, batch_size=16, passes=4):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    acc = {"lattice": 0.0, "centroid": 0.0, "orient": 0.0, "total": 0.0}
    n = 0
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
    return {k: v / max(n, 1) for k, v in acc.items()}


@torch.no_grad()
def orient_split_val(model, ds, device, vol_per_atom, cond_clean_packing,
                     seed=12345, batch_size=16, passes=4):
    """Per-molecule orient CFM loss split into reference (target I) vs non-reference copies.
    Same scale as cfm_loss's `orient` term (mean over molecules of ||v_R - u_R||^2)."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    s = {"ref": 0.0, "nonref": 0.0}
    c = {"ref": 0.0, "nonref": 0.0}
    model.eval()
    for p in range(passes):
        torch.manual_seed(seed + p)
        for batch in dl:
            batch = move_batch(batch, device)
            z1 = batch_to_state(batch)
            z0 = sample_prior(z1, vol_per_atom=vol_per_atom)
            t = torch.rand(z1.lattice.shape[0], device=device)
            z_t, targets = interpolate(z0, z1, t)
            mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
            cond_L = z1.lattice if cond_clean_packing else z_t.lattice
            cond_x = z1.centroid if cond_clean_packing else z_t.centroid
            _, _, v_R = model(mol_emb, cond_L, cond_x, z_t.orient, t, batch["sg"], z1.mask)
            se = ((v_R - targets[2]) ** 2).sum(-1)          # (B,M) per-molecule ||.||^2
            ref = batch["is_ref"] & z1.mask
            non = (~batch["is_ref"]) & z1.mask
            s["ref"] += float((se * ref).sum()); c["ref"] += float(ref.sum())
            s["nonref"] += float((se * non).sum()); c["nonref"] += float(non.sum())
    return {k: s[k] / max(c[k], 1.0) for k in s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-packing", action="store_true",
                    help="also condition the field on the TRUE lattice+centroid (best case)")
    ap.add_argument("--ckpt", default="checkpoints/diag_orient_relative.pt")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        sys.exit(f"no cache at {args.cache}; run csd_export.py + factorize_cifs.py first")

    full = MolCrystalDataset(cache_path=args.cache)
    n0 = len(full)
    # re-gauge to relative orientation + keep only multi-copy crystals
    rel_items = [relative_gauge_item(full.items[i]) for i in range(n0)]
    mults = [species_multiplicity(it) for it in rel_items]
    keep = [i for i, m in enumerate(mults) if m >= 2]
    full.items = [rel_items[i] for i in keep]
    n = len(full)
    print(f"corpus: {n0} crystals -> {n} multi-copy (>=2 copies of a species) after relative re-gauge")
    print(f"DIAGNOSTIC: relative orientation target; clean_packing={args.clean_packing}")
    hist = {}
    for i in keep:
        hist[mults[i]] = hist.get(mults[i], 0) + 1
    print("  max-species-multiplicity histogram:",
          {k: hist[k] for k in sorted(hist)})
    if n < 40:
        sys.exit("too few multi-copy crystals for a meaningful run")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(4, int(round(args.val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    print(f"  split: {len(train_ds)} train / {len(val_ds)} val")

    vpa = corpus_vol_per_atom(full.items)
    print(f"  lattice prior vol/atom matched to corpus: {vpa:.1f} A^3\n")

    mcfg = ModelConfig(lambda_orient=1.0)
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       seed=args.seed, log_every=50, prior_vol_per_atom=vpa,
                       cond_clean_packing=args.clean_packing)
    device = resolve_device(tcfg.device)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)
    print(f"  device={device}  steps={args.steps}\n")

    torch.manual_seed(args.seed)
    base = SymMCFlow(mcfg).to(device)
    pre = per_head_val(base, val_ds, device, weights, vpa, args.clean_packing)
    pre_split = orient_split_val(base, val_ds, device, vpa, args.clean_packing)

    model, _ = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    post = per_head_val(model, val_ds, device, weights, vpa, args.clean_packing)
    post_split = orient_split_val(model, val_ds, device, vpa, args.clean_packing)

    print("\n==== held-out validation, per head (untrained -> trained) ====")
    for k in ("lattice", "centroid", "orient", "total"):
        drop = 100 * (1 - post[k] / pre[k]) if pre[k] else 0.0
        print(f"  {k:9s}: {pre[k]:.4f} -> {post[k]:.4f}  ({drop:+.1f}%)")
    print("  orient split (the decisive numbers):")
    for k in ("ref", "nonref"):
        drop = 100 * (1 - post_split[k] / pre_split[k]) if pre_split[k] else 0.0
        tag = "  <-- NON-REFERENCE (symmetry-determined; THIS is the claim)" if k == "nonref" else ""
        print(f"    {k:7s}: {pre_split[k]:.4f} -> {post_split[k]:.4f}  ({drop:+.1f}%){tag}")

    nonref_drop = 1 - post_split["nonref"] / pre_split["nonref"] if pre_split["nonref"] else 0.0
    if nonref_drop > 0.50:
        verdict = ("CONFIRMED (strong): the SO(3) flow learns within-crystal RELATIVE "
                   "orientation -> the floor was the free asymmetric-unit orientation.")
    elif nonref_drop > 0.15:
        verdict = ("PARTIAL: the symmetry-determined relative orientation is partially "
                   "learnable + generalizes (the absolute target gave ~0%) -> the floor was "
                   "dominated by the free asymmetric-unit orientation, now characterized.")
    else:
        verdict = ("NULL: even the relative (symmetry-determined) orientation stays at floor "
                   "-> orientation is unlearnable in this pipeline; the reframe is final.")
    print(f"\nVERDICT ({100 * nonref_drop:+.1f}% non-ref): {verdict}")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__, "vol_per_atom": vpa,
                "val_idx": val_idx, "train_idx": train_idx, "pre": pre, "post": post,
                "pre_split": pre_split, "post_split": post_split,
                "clean_packing": args.clean_packing}, args.ckpt)
    print(f"checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
