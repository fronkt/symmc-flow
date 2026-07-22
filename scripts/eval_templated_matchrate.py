"""Template-based (symmetry-conditioned) end-to-end generation match rate -- the strengthening
headline.

The de-novo eval samples all three manifolds with NO symmetry information (the MolCrystalFlow /
MOFFlow regime: rigid bodies, no space-group conditioning). This script adds the piece that
line of work omits: at generation time it supplies the DEPLOYABLE per-molecule coset -- the
generating space-group operation, exactly what a template-based CSP pipeline (space group +
Wyckoff + copy ordering) provides -- to a coset-conditioned model, then measures how much that
symmetry template improves reconstruction over the unconditioned baseline on the SAME held-out
crystals, corpus, and split.

Run it twice and compare (the honest, scale-immune delta):

    # unconditioned baseline (a no-coset / relative checkpoint, MolCrystalFlow-style)
    python scripts/eval_templated_matchrate.py --ckpt checkpoints/diag_orient_relative_noised.pt \
        --tag unconditioned
    # symmetry-conditioned (a coset checkpoint trained with --deployable)
    python scripts/eval_templated_matchrate.py --ckpt checkpoints/coset_deployable_seed0.pt \
        --tag templated

`--ablate-no-coset` forces coset=None on a coset model (same-model ablation). All other
machinery (matcher, component errors, Wilson CI) mirrors eval_denovo_matchrate.py.
"""
import argparse
import math
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher

from symmc_flow.config import ModelConfig
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item,
                                    rigid_to_structure, species_multiplicity,
                                    assign_symmetry_cosets)
from symmc_flow.model import SymMCFlow
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample
from symmc_flow import manifolds as M

RAD2DEG = 180.0 / 3.141592653589793


def _match_one(args, ltol, stol, angle_tol):
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
    ap.add_argument("--ckpt", default="checkpoints/coset_deployable_seed0.pt")
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--match-k", type=int, default=20)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ablate-no-coset", action="store_true",
                    help="force coset=None on a coset model (same-model ablation)")
    ap.add_argument("--tag", default="templated")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vpa = ck["vol_per_atom"]
    val_idx = ck["val_idx"]
    mcfg = ModelConfig(**ck["model_cfg"])
    device = resolve_device("auto")
    use_coset = (mcfg.n_cosets > 0) and not args.ablate_no_coset

    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    keep = [i for i, it in enumerate(rel) if species_multiplicity(it) >= 2]
    full.items = [rel[i] for i in keep]
    if mcfg.n_cosets > 0:
        # assign on the FULL multi-copy corpus so ids match the trained model
        assign_symmetry_cosets(full.items)
    val_items = [full.items[i] for i in val_idx]
    if args.limit:
        val_items = val_items[:args.limit]

    scheme = ("symmetry-conditioned (deployable coset template)" if use_coset
              else "UNCONDITIONED (no symmetry template)")
    print(f"[{args.tag}] {scheme}", flush=True)
    print(f"  corpus {len(rel)} -> {len(keep)} multi-copy; val {len(val_items)}; device {device}; "
          f"k={args.match_k} steps={args.sampler_steps}; n_cosets(model)={mcfg.n_cosets}", flush=True)

    model = SymMCFlow(mcfg).to(device).eval()
    model.load_state_dict(ck["model"])

    refs, gens_per_crystal = [], []
    lat_err, cen_err, ori_err = [], [], []
    torch.manual_seed(args.seed)
    for s in range(0, len(val_items), args.batch_size):
        chunk = val_items[s:s + args.batch_size]
        batch = move_batch(collate([{**it, "idx": torch.tensor(0)} for it in chunk]), device)
        z1 = batch_to_state(batch)
        B = z1.lattice.shape[0]
        mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        coset = batch.get("coset") if use_coset else None
        for b in range(B):
            refs.append(rigid_to_structure(
                z1.lattice[b], batch["Z"][b], batch["local"][b], z1.centroid[b],
                z1.orient[b], batch["atom_mask"][b], batch["mol_mask"][b]).as_dict())
        draws = [[] for _ in range(B)]
        best_lat = torch.full((B,), 1e9); best_cen = torch.full((B,), 1e9)
        best_ori = torch.full((B,), 1e9)
        for _ in range(args.match_k):
            z0 = sample_prior(z1, vol_per_atom=vpa)
            samp = rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps,
                              coset=coset)
            for b in range(B):
                draws[b].append(rigid_to_structure(
                    samp.lattice[b], batch["Z"][b], batch["local"][b], samp.centroid[b],
                    samp.orient[b], batch["atom_mask"][b], batch["mol_mask"][b]).as_dict())
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

    work = list(zip(refs, gens_per_crystal))
    work1 = [(r, g[:1]) for r, g in work]
    fn = partial(_match_one, ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)
    if args.workers > 1:
        import multiprocessing as mp
        torch.cuda.empty_cache()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            hits = pool.map(fn, work)
            hits1 = pool.map(fn, work1)
    else:
        hits = [fn(w) for w in work]
        hits1 = [fn(w) for w in work1]

    n = len(hits)
    mr, mr1 = sum(hits) / max(n, 1), sum(hits1) / max(n, 1)
    z = 1.96
    denom = 1 + z * z / n
    centre = (mr + z * z / (2 * n)) / denom
    half = z * math.sqrt(mr * (1 - mr) / n + z * z / (4 * n * n)) / denom

    def med(x):
        x = sorted(x); return x[len(x) // 2] if x else float("nan")

    print(f"\n==== {args.tag}: end-to-end template-based generation, match@{args.match_k} (n={n}) ====",
          flush=True)
    print(f"  scheme: {scheme}")
    print(f"  match@1 (single draw): {100*mr1:.1f}%")
    print(f"  match rate: {100*mr:.1f}%  (Wilson 95% CI {100*(centre-half):.1f}-{100*(centre+half):.1f}%)")
    print(f"  median best-of-k component error: lattice-param {med(lat_err):.3f}  "
          f"centroid(frac) {med(cen_err):.3f}  orient {med(ori_err):.1f} deg")
    print(f"  TAG {args.tag} coset {int(use_coset)} k {args.match_k} match {100*mr:.2f}")


if __name__ == "__main__":
    main()
