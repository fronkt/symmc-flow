"""R2 (re-review): best-of-k component-error breakdown for the all-atom de-novo baseline.

The all-atom baseline (baseline_allatom_denovo.py) reports only match@1/@20. To show the
0% reflects the SAME joint-packing bottleneck as the rigid-body de-novo run (lattice + centroid),
this loads the saved all-atom checkpoint and reports median best-of-k lattice-parameter and
fractional-coordinate errors, using the IDENTICAL metric as eval_denovo_matchrate.py (orientation
is inactive for the all-atom model, so only the two packing components are meaningful).

Also times the sampler (50 RK steps) to report wall-clock per structure (re-review R4).

    python scripts/eval_allatom_components.py --ckpt checkpoints/baseline_allatom.pt --match-k 20
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from symmc_flow.config import ModelConfig
from symmc_flow.molcrystal import MolCrystalDataset, rigid_to_structure, species_multiplicity
from symmc_flow.model import SymMCFlow
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample
from symmc_flow import manifolds as M

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_allatom_denovo import allatom_item


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/baseline_allatom.pt")
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--match-k", type=int, default=20)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vpa = ck["vol_per_atom"]
    val_idx = ck["val_idx"]
    max_atoms = ck["max_atoms"]
    mcfg = ModelConfig(**ck["model_cfg"])
    device = resolve_device("auto")

    # reproduce the EXACT baseline split + all-atom explosion
    full = MolCrystalDataset(cache_path=args.cache)
    keep = [i for i in range(len(full)) if species_multiplicity(full.items[i]) >= 2]
    orig = [full.items[i] for i in keep]
    aa = [allatom_item(it, max_atoms) for it in orig]
    val_items = [aa[i] for i in val_idx]
    print(f"[allatom-components] corpus {len(full)} -> {len(keep)} multi-copy; "
          f"val {len(val_items)}; device {device}; k={args.match_k}", flush=True)

    model = SymMCFlow(mcfg).to(device).eval()
    model.load_state_dict(ck["model"])

    lat_err, cen_err = [], []
    n_struct, t_sample = 0, 0.0
    torch.manual_seed(args.seed)
    for s in range(0, len(val_items), args.batch_size):
        chunk = val_items[s:s + args.batch_size]
        batch = move_batch(collate([{**it, "idx": torch.tensor(0)} for it in chunk]), device)
        z1 = batch_to_state(batch)
        B = z1.lattice.shape[0]
        mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        best_lat = torch.full((B,), 1e9); best_cen = torch.full((B,), 1e9)
        for _ in range(args.match_k):
            z0 = sample_prior(z1, vol_per_atom=vpa)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            samp = rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps)
            if device == "cuda":
                torch.cuda.synchronize()
            t_sample += time.time() - t0
            n_struct += B
            n = z1.mask.sum(-1).clamp_min(1)
            le = (M.lattice_to_param(samp.lattice, n) - M.lattice_to_param(z1.lattice, n)).norm(dim=-1).cpu()
            m = z1.mask.float()
            ce = (((M.torus_diff(samp.centroid, z1.centroid)) ** 2).sum(-1).sqrt() * m).sum(-1) / m.sum(-1).clamp_min(1)
            best_lat = torch.minimum(best_lat, le)
            best_cen = torch.minimum(best_cen, ce.cpu())
        lat_err += best_lat.tolist(); cen_err += best_cen.tolist()
        print(f"  sampled {min(s + args.batch_size, len(val_items))}/{len(val_items)}", flush=True)

    def med(x):
        x = sorted(x); return x[len(x) // 2] if x else float("nan")

    print(f"\n==== all-atom de-novo: best-of-{args.match_k} component error (n={len(lat_err)}) ====")
    print(f"  median lattice-param {med(lat_err):.3f}   median centroid(frac) {med(cen_err):.3f}")
    print(f"  [wall-clock] {1000*t_sample/max(n_struct,1):.1f} ms/structure "
          f"({args.sampler_steps} RK steps, batch {args.batch_size}, {n_struct} samples)")


if __name__ == "__main__":
    main()
