"""C / reviewer M1: end-to-end DE-NOVO joint-generation match rate on molecular crystals.

Unlike eval_orient_matchrate.py, which fixes the true lattice+centroid+conformer and samples
only SO(3), this samples ALL THREE manifolds (lattice, centroid, orientation) from the prior
for each held-out crystal's molecules -- i.e. full rigid-body crystal structure prediction given
the molecular composition and conformer. Reconstructed crystals are matched to the ground truth
with StructureMatcher (best-of-k). This gives the joint generation match rate that contextualizes
the orientation-isolated number.

    python scripts/eval_denovo_matchrate.py --ckpt checkpoints/diag_orient_relative_noised.pt \
        --match-k 20 --sampler-steps 50 [--workers 16] [--limit N]

The checkpoint is a diag_orient_relative-style file (model, model_cfg, vol_per_atom, val_idx).
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher

from symmc_flow.config import ModelConfig
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item,
                                    rigid_to_structure, species_multiplicity)
from symmc_flow.model import SymMCFlow
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample
from symmc_flow import manifolds as M

RAD2DEG = 180.0 / 3.141592653589793


def _match_one(args, ltol, stol, angle_tol):
    """Worker: (ref_struct, [k gen_structs]) -> 1 if any gen matches ref else 0.
    Structures are passed as pymatgen dicts to be picklable across processes."""
    from pymatgen.core import Structure
    ref_d, gen_ds = args
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    ref = Structure.from_dict(ref_d)
    for gd in gen_ds:
        if sm.fit(Structure.from_dict(gd), ref):
            return 1
    return 0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/diag_orient_relative_noised.pt")
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--match-k", type=int, default=20)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8, help="matcher processes (cap on big boxes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="denovo")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vpa = ck["vol_per_atom"]
    val_idx = ck["val_idx"]
    mcfg = ModelConfig(**ck["model_cfg"])
    device = resolve_device("auto")

    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    keep = [i for i, it in enumerate(rel) if species_multiplicity(it) >= 2]
    full.items = [rel[i] for i in keep]
    val_items = [full.items[i] for i in val_idx]
    if args.limit:
        val_items = val_items[:args.limit]
    print(f"[{args.tag}] corpus {len(rel)} -> {len(keep)} multi-copy; val {len(val_items)}; "
          f"device {device}; k={args.match_k} steps={args.sampler_steps}", flush=True)

    model = SymMCFlow(mcfg).to(device).eval()
    model.load_state_dict(ck["model"])

    # accumulate per-crystal best-of-k generated structures + component errors
    refs, gens_per_crystal = [], []
    lat_err, cen_err, ori_err = [], [], []
    torch.manual_seed(args.seed)
    for s in range(0, len(val_items), args.batch_size):
        chunk = val_items[s:s + args.batch_size]
        batch = move_batch(collate([{**it, "idx": torch.tensor(0)} for it in chunk]), device)
        z1 = batch_to_state(batch)
        B = z1.lattice.shape[0]
        mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        coset = batch.get("coset")
        # reference structures (rebuilt from truth in the same representation)
        for b in range(B):
            refs.append(rigid_to_structure(
                z1.lattice[b], batch["Z"][b], batch["local"][b], z1.centroid[b],
                z1.orient[b], batch["atom_mask"][b], batch["mol_mask"][b]).as_dict())
        draws = [[] for _ in range(B)]
        best_lat = torch.full((B,), 1e9); best_cen = torch.full((B,), 1e9)
        best_ori = torch.full((B,), 1e9)
        for _ in range(args.match_k):
            z0 = sample_prior(z1, vol_per_atom=vpa)
            samp = rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps)
            for b in range(B):
                draws[b].append(rigid_to_structure(
                    samp.lattice[b], batch["Z"][b], batch["local"][b], samp.centroid[b],
                    samp.orient[b], batch["atom_mask"][b], batch["mol_mask"][b]).as_dict())
            # component errors (per crystal, best over draws)
            n = z1.mask.sum(-1).clamp_min(1)
            kL = M.lattice_to_param(samp.lattice, n) - M.lattice_to_param(z1.lattice, n)
            le = kL.norm(dim=-1).cpu()
            m = z1.mask.float()
            ce = (((M.torus_diff(samp.centroid, z1.centroid)) ** 2).sum(-1).sqrt()
                  * m).sum(-1) / m.sum(-1).clamp_min(1)
            oe = (M.so3_angle(samp.orient, z1.orient) * m).sum(-1) / m.sum(-1).clamp_min(1)
            best_lat = torch.minimum(best_lat, le)
            best_cen = torch.minimum(best_cen, ce.cpu())
            best_ori = torch.minimum(best_ori, (oe * RAD2DEG).cpu())
        gens_per_crystal.extend(draws)
        lat_err += best_lat.tolist(); cen_err += best_cen.tolist(); ori_err += best_ori.tolist()
        print(f"  sampled {len(refs)}/{len(val_items)}", flush=True)

    # match (parallel over crystals)
    work = list(zip(refs, gens_per_crystal))
    fn = partial(_match_one, ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            hits = pool.map(fn, work)
    else:
        hits = [fn(w) for w in work]

    n = len(hits)
    mr = sum(hits) / max(n, 1)
    # Wilson 95% CI
    import math
    z = 1.96
    phat = mr
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom

    def med(x):
        x = sorted(x); return x[len(x) // 2] if x else float("nan")

    print(f"\n==== DE-NOVO joint generation, match@{args.match_k} (n={n}) ====", flush=True)
    print(f"  match rate: {100*mr:.1f}%  (Wilson 95% CI {100*(centre-half):.1f}-{100*(centre+half):.1f}%)")
    print(f"  median best-of-k component error: lattice-param {med(lat_err):.3f}  "
          f"centroid(frac) {med(cen_err):.3f}  orient {med(ori_err):.1f} deg")
    print(f"  TAG {args.tag} k {args.match_k} match {100*mr:.2f}")


if __name__ == "__main__":
    main()
