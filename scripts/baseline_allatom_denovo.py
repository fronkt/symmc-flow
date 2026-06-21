"""T1.3 / Reviewer M2: a NON-RIGID ALL-ATOM baseline for de-novo molecular-crystal CSP.

Our headline de-novo result is 0% match@20 with the rigid-body SO(3) flow. Reviewers note
that 0% is uninterpretable without a baseline: is the corpus brutal, or is the rigid method
weak? This script answers it on the IDENTICAL corpus and split by dropping the rigid-body
factorization entirely. Each molecular crystal is exploded into single-atom blocks (the exact
representation our inorganic MP-20/carbon-24 path uses: lambda_orient=0, one atom per block),
and a plain all-atom flow over lattice + fractional coordinates is trained -- a DiffCSP-style
all-atom generator built inside our own infrastructure, so the matcher, sampler, and split are
the same. We report de-novo match@1 / match@20 for:

  * all-atom flow : the trained non-rigid baseline.
  * random prior  : draw lattice+coords from the prior, no model -- the no-information floor.

The same 964/131 multi-copy split as diag_orient_relative.py (seed 0) is reproduced exactly,
and references are the rigid_to_structure ground truth, so the number is directly comparable to
the rigid de-novo 0%.

    python scripts/baseline_allatom_denovo.py --cache data/csd_mol/ds.pt --steps 8000 \
        --match-k 20 --workers 16
"""
import argparse
import math
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, Subset

from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.molcrystal import (MolCrystalDataset, rigid_to_structure,
                                    species_multiplicity)
from symmc_flow.model import SymMCFlow
from symmc_flow.train import train, resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample


def allatom_item(it, max_atoms):
    """Explode a rigid-body crystal item into single-atom blocks (one atom per 'molecule').
    centroid := atom fractional coords; local := 0; orient := I (inactive). Padded to
    max_atoms blocks."""
    st = rigid_to_structure(it["lattice"], it["Z"], it["local"], it["centroid"],
                            it["orient"], it["atom_mask"], it["mol_mask"])
    Znums = torch.tensor([s.Z for s in st.species], dtype=torch.long)
    frac = torch.tensor(st.frac_coords % 1.0, dtype=torch.float32)
    na = len(Znums)
    Z = torch.zeros(max_atoms, 1, dtype=torch.long)
    local = torch.zeros(max_atoms, 1, 3)
    atom_mask = torch.zeros(max_atoms, 1, dtype=torch.bool)
    mol_mask = torch.zeros(max_atoms, dtype=torch.bool)
    centroid = torch.rand(max_atoms, 3)
    orient = torch.eye(3).expand(max_atoms, 3, 3).clone()
    Z[:na, 0] = Znums
    atom_mask[:na, 0] = True
    mol_mask[:na] = True
    centroid[:na] = frac
    return {"Z": Z, "local": local, "atom_mask": atom_mask, "mol_mask": mol_mask,
            "lattice": it["lattice"].clone().float(), "centroid": centroid,
            "orient": orient, "sg": it["sg"].clone()}


class AllAtomDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {**self.items[i], "idx": torch.tensor(i, dtype=torch.long)}


def to_structures(state, Z, mask):
    from pymatgen.core import Structure, Lattice
    from pymatgen.core.periodic_table import Element
    out = []
    for b in range(state.lattice.shape[0]):
        idx = mask[b].nonzero(as_tuple=True)[0]
        try:
            species = [Element.from_Z(int(z)) for z in Z[b, idx]]
            st = Structure(Lattice(state.lattice[b].cpu().numpy()), species,
                           state.centroid[b, idx].cpu().numpy())
        except Exception:
            st = None
        out.append(st)
    return out


def _match_one(args, ltol, stol, angle_tol):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    ref_d, gen_ds = args
    if ref_d is None:
        return 0
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    ref = Structure.from_dict(ref_d)
    for gd in gen_ds:
        if gd is not None and sm.fit(Structure.from_dict(gd), ref):
            return 1
    return 0


@torch.no_grad()
def evaluate_denovo(model, val_items, max_atoms, vpa, device, args, tag, random_floor=False):
    refs, gens = [], []
    torch.manual_seed(args.seed)
    for s in range(0, len(val_items), args.batch_size):
        chunk = val_items[s:s + args.batch_size]
        batch = move_batch(collate([{**it, "idx": torch.tensor(0)} for it in chunk]), device)
        z1 = batch_to_state(batch)
        B = z1.lattice.shape[0]
        Z = batch["Z"][..., 0]
        mol_emb = None if random_floor else model.encode_molecules(
            batch["Z"], batch["local"], batch["atom_mask"])
        for st in to_structures(z1, Z, z1.mask):
            refs.append(st.as_dict() if st is not None else None)
        draws = [[] for _ in range(B)]
        for _ in range(args.match_k):
            z0 = sample_prior(z1, vol_per_atom=vpa)
            samp = z0 if random_floor else rk4_sample(model, mol_emb, z0, batch["sg"],
                                                      steps=args.sampler_steps)
            for b, st in enumerate(to_structures(samp, Z, z1.mask)):
                draws[b].append(st.as_dict() if st is not None else None)
        gens.extend(draws)
        print(f"  [{tag}] sampled {len(refs)}/{len(val_items)}", flush=True)

    work = list(zip(refs, gens))
    work1 = [(r, g[:1]) for r, g in work]
    fn = partial(_match_one, ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)
    if args.workers > 1:
        import multiprocessing as mp
        torch.cuda.empty_cache()  # release before forking matcher workers
        # spawn (not fork) so workers do NOT inherit this process's CUDA context
        with mp.get_context("spawn").Pool(args.workers) as pool:
            hits, hits1 = pool.map(fn, work), pool.map(fn, work1)
    else:
        hits, hits1 = [fn(w) for w in work], [fn(w) for w in work1]
    n = len([r for r in refs if r is not None])
    mr, mr1 = sum(hits) / max(n, 1), sum(hits1) / max(n, 1)
    z = 1.96
    denom = 1 + z * z / n
    centre = (mr + z * z / (2 * n)) / denom
    half = z * math.sqrt(mr * (1 - mr) / n + z * z / (4 * n * n)) / denom
    print(f"\n==== {tag}: de-novo all-atom, n={n} ====")
    print(f"  match@1: {100*mr1:.1f}%   match@{args.match_k}: {100*mr:.1f}%  "
          f"(Wilson 95% CI {100*(centre-half):.1f}-{100*(centre+half):.1f}%)")
    return mr1, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--attn-layers", type=int, default=8)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--match-k", type=int, default=20)
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--ckpt", default="checkpoints/baseline_allatom.pt")
    args = ap.parse_args()

    full = MolCrystalDataset(cache_path=args.cache)
    keep = [i for i in range(len(full)) if species_multiplicity(full.items[i]) >= 2]
    orig = [full.items[i] for i in keep]
    n = len(orig)
    # reproduce the EXACT diag_orient_relative split (seed, multi-copy corpus order)
    g = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(4, int(round(args.val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    max_atoms = max(int(rigid_to_structure(
        it["lattice"], it["Z"], it["local"], it["centroid"], it["orient"],
        it["atom_mask"], it["mol_mask"]).num_sites) for it in orig)
    print(f"corpus {len(full)} -> {n} multi-copy; split {len(train_idx)}/{len(val_idx)}; "
          f"max atoms/cell {max_atoms}")

    aa = [allatom_item(it, max_atoms) for it in orig]
    vols = sum(abs(float(torch.det(it["lattice"]))) for it in aa)
    atoms = sum(int(it["mol_mask"].sum()) for it in aa)
    vpa = vols / max(atoms, 1)
    print(f"all-atom vol/atom {vpa:.1f} A^3; building datasets...")

    ds = AllAtomDS(aa)
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)
    val_items = [aa[i] for i in val_idx]
    device = resolve_device("auto")

    # random-prior floor first (no training needed)
    evaluate_denovo(None, val_items, max_atoms, vpa, device, args, "random-prior",
                    random_floor=True)

    mcfg = ModelConfig(d_model=args.d_model, n_heads=8, n_attn_layers=args.attn_layers,
                       egnn_layers=2, atom_embed_dim=64,
                       lambda_lattice=1.0, lambda_centroid=1.0, lambda_orient=0.0)
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       seed=args.seed, log_every=200, prior_vol_per_atom=vpa,
                       use_ot_coupling=True)
    print(f"\ntraining all-atom flow: d={args.d_model} attn={args.attn_layers} "
          f"steps={args.steps} device={device}\n")
    model, hist = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    first, last = sum(hist[:20]) / 20, sum(hist[-20:]) / 20
    print(f"loss first-20 {first:.4f} -> last-20 {last:.4f} ({100*(1-last/first):.1f}% drop)")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__, "vol_per_atom": vpa,
                "val_idx": val_idx, "max_atoms": max_atoms}, args.ckpt)
    evaluate_denovo(model, val_items, max_atoms, vpa, device, args, "all-atom-flow")


if __name__ == "__main__":
    main()
