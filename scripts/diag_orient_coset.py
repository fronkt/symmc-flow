"""2c: does conditioning on the space-group COSET id make the relative orientation exact?

The relative-orientation diagnostic showed the SO(3) flow learns the symmetry-induced relative
rotation only partially (+27% non-ref) -- the head must INFER which symmetry op generated each
copy from the (noised) packing. This run hands the model that information directly as a discrete
per-molecule coset id (`assign_cosets`: clusters the non-reference relative rotations into a
per-space-group codebook). If the non-reference orient loss now COLLAPSES toward 0, the SO(3)
head can represent rot(g_m) exactly and the +27% ceiling is purely the difficulty of inferring
the coset from packing -- a clean upper bound on the decomposition. If it stays at +27%, even
the discrete op identity is not enough and the residual is representational.

    python scripts/diag_orient_coset.py --cache data/csd_mol/ds.pt --steps 800 [--clean-packing]

`--no-coset` reruns with the embedding disabled (n_cosets=0) for a paired control on the SAME
cosetted corpus, isolating the effect of the conditioning alone.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Subset

from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item,
                                    species_multiplicity, assign_cosets,
                                    assign_symmetry_cosets)
from symmc_flow.model import SymMCFlow
from symmc_flow.train import train, _step_loss, resolve_device, move_batch
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
    from torch.utils.data import DataLoader
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
    """orient CFM loss split ref vs non-ref, passing the per-molecule coset id to the field."""
    from torch.utils.data import DataLoader
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
            _, _, v_R = model(mol_emb, cond_L, cond_x, z_t.orient, t, batch["sg"], z1.mask,
                              coset=batch.get("coset"))
            se = ((v_R - targets[2]) ** 2).sum(-1)
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
    ap.add_argument("--angle-tol", type=float, default=20.0, help="coset clustering tolerance (deg)")
    ap.add_argument("--deployable", action="store_true",
                    help="use the LEAK-FREE symmetry-derived coset (assign_symmetry_cosets: the "
                         "generating space-group operation, available from a template at sampling "
                         "time) instead of the observed-rotation clustering (assign_cosets)")
    ap.add_argument("--clean-packing", action="store_true")
    ap.add_argument("--so3-avg-k", type=int, default=1,
                    help="C5: K for the SO(3)-averaged orientation objective (1 = standard CFM)")
    ap.add_argument("--no-coset", action="store_true", help="paired control: disable the embedding")
    ap.add_argument("--lattice-logmetric", action="store_true",
                    help="Phase F: use the O(3)-invariant log-metric k6 lattice repr")
    ap.add_argument("--family-mask", action="store_true",
                    help="Phase F: freeze crystal-family-constrained lattice DOF (needs "
                         "--lattice-logmetric); the deployable lattice analogue of the coset")
    ap.add_argument("--logvol-std", type=float, default=0.3,
                    help="Phase F: informed lattice prior ln(V) std (~0.11 for molecular crystals)")
    ap.add_argument("--dev-std", type=float, default=0.3,
                    help="Phase F: deviatoric log-metric prior std (logmetric6)")
    ap.add_argument("--ckpt", default="checkpoints/diag_orient_coset.pt")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        sys.exit(f"no cache at {args.cache}")

    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    keep = [i for i, it in enumerate(rel) if species_multiplicity(it) >= 2]
    items = [rel[i] for i in keep]
    if args.deployable:
        items, n_cosets = assign_symmetry_cosets(items)
    else:
        items, n_cosets = assign_cosets(items, angle_tol_deg=args.angle_tol)
    full.items = items
    n = len(full)
    scheme = "DEPLOYABLE symmetry-op" if args.deployable else f"clustered (tol {args.angle_tol} deg)"
    print(f"corpus: {len(rel)} -> {n} multi-copy after relative re-gauge")
    print(f"cosets: {n_cosets} distinct space-group cosets [{scheme}]; "
          f"coset conditioning: {'OFF (control)' if args.no_coset else 'ON'}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(4, int(round(args.val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    print(f"  split: {len(train_ds)} train / {len(val_ds)} val")

    vpa = corpus_vol_per_atom(full.items)
    print(f"  lattice prior vol/atom: {vpa:.1f} A^3\n")

    mcfg = ModelConfig(lambda_orient=1.0, n_cosets=0 if args.no_coset else n_cosets,
                       lattice_repr="logmetric6" if args.lattice_logmetric else "shape10",
                       lattice_family_mask=args.family_mask)
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       seed=args.seed, log_every=50, prior_vol_per_atom=vpa,
                       cond_clean_packing=args.clean_packing, so3_avg_k=args.so3_avg_k,
                       prior_logvol_std=args.logvol_std, prior_dev_std=args.dev_std)
    device = resolve_device(tcfg.device)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)
    print(f"  device={device}  steps={args.steps}  n_cosets(model)={mcfg.n_cosets}\n")

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
    print("  orient split:")
    for k in ("ref", "nonref"):
        drop = 100 * (1 - post_split[k] / pre_split[k]) if pre_split[k] else 0.0
        tag = "  <-- NON-REFERENCE (coset-conditioned target)" if k == "nonref" else ""
        print(f"    {k:7s}: {pre_split[k]:.4f} -> {post_split[k]:.4f}  ({drop:+.1f}%){tag}")

    nonref_drop = 1 - post_split["nonref"] / pre_split["nonref"] if pre_split["nonref"] else 0.0
    print(f"\nNON-REF drop ({100 * nonref_drop:+.1f}%): vs +27% without coset conditioning -> "
          f"{'coset id collapses the residual (representational head, inference-limited ceiling)' if nonref_drop > 0.6 else 'coset id helps but residual remains' if nonref_drop > 0.35 else 'coset id does not materially help (residual is representational)'}")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__, "vol_per_atom": vpa,
                "val_idx": val_idx, "train_idx": train_idx, "pre_split": pre_split,
                "post_split": post_split, "n_cosets": n_cosets,
                "deployable": args.deployable, "clean_packing": args.clean_packing,
                "no_coset": args.no_coset,
                "prior_logvol_std": args.logvol_std, "prior_dev_std": args.dev_std}, args.ckpt)
    print(f"checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
