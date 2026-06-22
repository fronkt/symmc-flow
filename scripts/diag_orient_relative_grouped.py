"""T1.5 / Reviewer M1: headline relative-orientation result under a SPECIES-GROUPED split.

The 3-seed error bars in the paper vary the model/sampling seed on ONE fixed random 964/131
split. That measures optimization noise, not generalization, and a random split can place the
same molecular species (or a polymorph of it) in both train and val -- the 'glitter' leakage
critique. Refcode-family grouping needs the CSD API (unavailable here), so we group by
MOLECULAR SPECIES instead: crystals sharing any species (identified by the conformer registry
reference: element list + rounded body-frame coordinates) are unioned and assigned WHOLLY to
train or val, so no species appears on both sides. We re-run the relative-orientation diagnostic
over >=1 such grouped splits and report the held-out NON-REFERENCE loss drop -- the headline
learnability number -- to show it survives leakage control.

    python scripts/diag_orient_relative_grouped.py --cache data/csd_mol/ds.pt --steps 800 \
        --split-seed 0 --ckpt checkpoints/rel_grouped_s0.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Subset

from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item,
                                    species_multiplicity, _species_groups)
from symmc_flow.model import SymMCFlow
from symmc_flow.train import train, resolve_device
# reuse the evaluation helpers from the ungrouped diagnostic
from diag_orient_relative import corpus_vol_per_atom, per_head_val, orient_split_val


def species_sigs(item):
    """Set of species signatures in a crystal: (elements, rounded body-frame coords) of each
    species' reference conformer. Stable across crystals because the conformer registry is
    shared at dataset build, so the same species has identical `local`."""
    sigs = set()
    for g in _species_groups(item):
        m = g[0]
        am = item["atom_mask"][m]
        Z = tuple(item["Z"][m][am].tolist())
        loc = tuple(map(tuple, torch.round(item["local"][m][am] * 50).to(torch.int32).tolist()))
        sigs.add((Z, loc))
    return sigs


def grouped_split(orig_items, val_frac, seed):
    """Union-find crystals that share any species; assign whole components to train/val so no
    species straddles the split. Returns (train_idx, val_idx) into orig_items order."""
    n = len(orig_items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sig_to_crystal = {}
    crystal_sigs = [species_sigs(it) for it in orig_items]
    for i, sigs in enumerate(crystal_sigs):
        for s in sigs:
            if s in sig_to_crystal:
                union(i, sig_to_crystal[s])
            else:
                sig_to_crystal[s] = i

    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    comp_lists = list(comps.values())

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(comp_lists), generator=g).tolist()
    target_val = int(round(val_frac * n))
    val_idx, train_idx, nval = [], [], 0
    for ci in order:
        members = comp_lists[ci]
        if nval < target_val:
            val_idx += members
            nval += len(members)
        else:
            train_idx += members
    return sorted(train_idx), sorted(val_idx), len(comp_lists)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/rel_grouped_s0.pt")
    args = ap.parse_args()

    full = MolCrystalDataset(cache_path=args.cache)
    rel_items = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    keep = [i for i, it in enumerate(rel_items) if species_multiplicity(it) >= 2]
    orig_keep = [full.items[i] for i in keep]      # for species signatures (un-gauged)
    rel_keep = [rel_items[i] for i in keep]        # for training (relative gauge)
    full.items = rel_keep
    n = len(rel_keep)

    train_idx, val_idx, n_comp = grouped_split(orig_keep, args.val_frac, args.split_seed)
    print(f"corpus {len(rel_items)} -> {n} multi-copy; {n_comp} species-components")
    print(f"SPECIES-GROUPED split (seed {args.split_seed}): {len(train_idx)} train / "
          f"{len(val_idx)} val; no species shared across the split")

    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    vpa = corpus_vol_per_atom(full.items)

    mcfg = ModelConfig(lambda_orient=1.0)
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       seed=args.seed, log_every=100, prior_vol_per_atom=vpa)
    device = resolve_device(tcfg.device)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)

    torch.manual_seed(args.seed)
    base = SymMCFlow(mcfg).to(device)
    pre = per_head_val(base, val_ds, device, weights, vpa, False)
    pre_split = orient_split_val(base, val_ds, device, vpa, False)

    model, _ = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    post = per_head_val(model, val_ds, device, weights, vpa, False)
    post_split = orient_split_val(model, val_ds, device, vpa, False)

    print("\n==== held-out (species-grouped) per head (untrained -> trained) ====")
    for k in ("lattice", "centroid", "orient"):
        drop = 100 * (1 - post[k] / pre[k]) if pre[k] else 0.0
        print(f"  {k:9s}: {pre[k]:.4f} -> {post[k]:.4f}  ({drop:+.1f}%)")
    for k in ("ref", "nonref"):
        drop = 100 * (1 - post_split[k] / pre_split[k]) if pre_split[k] else 0.0
        tag = "  <-- NON-REFERENCE (the claim)" if k == "nonref" else ""
        print(f"  orient/{k:7s}: {pre_split[k]:.4f} -> {post_split[k]:.4f}  ({drop:+.1f}%){tag}")

    nonref_drop = 100 * (1 - post_split["nonref"] / pre_split["nonref"])
    print(f"\nGROUPED-SPLIT non-ref drop: {nonref_drop:+.1f}%  "
          f"(ungrouped 3-seed baseline was +27.5+-2.7%)")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__, "vol_per_atom": vpa,
                "val_idx": val_idx, "train_idx": train_idx,
                "pre_split": pre_split, "post_split": post_split}, args.ckpt)
    print(f"checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
