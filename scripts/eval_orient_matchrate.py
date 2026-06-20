"""2a: turn the relative-orientation loss into a PHYSICAL StructureMatcher match rate.

The +27% non-reference orient-loss drop (diag_orient_relative.py) is a loss-curve claim.
This converts it into reconstruction: with the TRUE lattice + centroid + conformer held fixed
(the clean-packing conditioning the diagnostic showed is equivalent), we *only* sample the SO(3)
orientation field and rebuild a pymatgen Structure (`rigid_to_structure`), then ask
StructureMatcher whether it matches the ground truth. Four conditions on the same held-out
crystals isolate orientation's contribution:

    oracle    : orient = TRUE relative pose            -> ceiling (validates matcher tolerances)
    identity  : every copy un-rotated (R = I)          -> naive "ignore relative orientation"
    untrained : SO(3) field of a fresh (random) net    -> the predict-floor reconstruction
    trained   : SO(3) field of the trained net         -> THE result

A single deterministic ODE draw per crystal (orientation-only RK4, lattice+centroid frozen at
truth) keeps the comparison about orientation alone. Reports best/mean StructureMatcher.fit
rate and mean min RMSD (get_rms_dist) per condition.

    python scripts/eval_orient_matchrate.py --ckpt checkpoints/diag_orient_relative_noised.pt
"""
import argparse
import os
import sys

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
from symmc_flow import manifolds as M


@torch.no_grad()
def sample_orient_only(model, mol_emb, L_true, x_true, R0, sg, mask, steps):
    """Orientation-only RK4 on SO(3): integrate dR/dt = v_R with lattice+centroid FROZEN at
    their true values (clean-packing conditioning). Mirrors the orient branch of rk4_sample so
    the result is exactly the flow's orientation given the true packing."""
    B = R0.shape[0]
    R = R0.clone()
    dt = 1.0 / steps
    dev, dtp = L_true.device, L_true.dtype

    def vR(R_, t_scalar):
        t = torch.full((B,), float(t_scalar), device=dev, dtype=dtp)
        return model.forward(mol_emb, L_true, x_true, R_, t, sg, mask)[2]

    for i in range(steps):
        t0 = i * dt
        k1 = vR(R, t0)
        k2 = vR(R @ M.so3_exp(0.5 * dt * k1), t0 + 0.5 * dt)
        k3 = vR(R @ M.so3_exp(0.5 * dt * k2), t0 + 0.5 * dt)
        k4 = vR(R @ M.so3_exp(dt * k3), t0 + dt)
        omega = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        R = M.project_so3(R @ M.so3_exp(omega))
    return R


def structures_for(batch, z1, orient):
    """Build one pymatgen Structure per crystal in the batch from a given orientation tensor
    (true lattice/centroid/conformer). Returns a list (len B)."""
    out = []
    for b in range(z1.lattice.shape[0]):
        out.append(rigid_to_structure(
            z1.lattice[b], batch["Z"][b], batch["local"][b], z1.centroid[b],
            orient[b], batch["atom_mask"][b], batch["mol_mask"][b]))
    return out


RAD2DEG = 180.0 / 3.141592653589793


@torch.no_grad()
def per_crystal_err_deg(R_pred, R_true, nonref):
    """Per-crystal mean SO(3) geodesic error (deg) over non-ref copies -> (B,). Always defined
    (unlike StructureMatcher RMSD); the direct physical reading of the orient loss."""
    ang = M.so3_angle(R_pred, R_true)                       # (B,M) radians
    sel = nonref.float()
    return (ang * sel).sum(1) / sel.sum(1).clamp_min(1.0) * RAD2DEG


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/diag_orient_relative_noised.pt")
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--steps", type=int, default=50, help="orient-only ODE steps")
    ap.add_argument("--match-k", type=int, default=8,
                    help="orientation prior draws per crystal; best-of-k (multimodal SO(3))")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle-tol", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap #val crystals (0 = all)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"no checkpoint at {args.ckpt}; run scripts/diag_orient_relative.py first")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vpa = ck["vol_per_atom"]
    val_idx = ck["val_idx"]
    mcfg = ModelConfig(**ck["model_cfg"])
    device = resolve_device("auto")

    # rebuild the SAME relative-gauge multi-copy corpus + split the checkpoint trained on
    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    keep = [i for i, it in enumerate(rel) if species_multiplicity(it) >= 2]
    full.items = [rel[i] for i in keep]
    val_items = [full.items[i] for i in val_idx]
    if args.limit:
        val_items = val_items[:args.limit]
    print(f"corpus: {len(rel)} -> {len(keep)} multi-copy; val crystals: {len(val_items)}")
    print(f"ckpt {args.ckpt}  clean_packing(train)={ck.get('clean_packing')}  vol/atom {vpa:.1f}\n")

    trained = SymMCFlow(mcfg).to(device).eval()
    trained.load_state_dict(ck["model"])
    torch.manual_seed(args.seed)
    untrained = SymMCFlow(mcfg).to(device).eval()  # fresh net = the predict-floor field

    sm = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)
    keys = ("oracle", "identity", "untrained", "trained")
    matched = {k: [] for k in keys}    # per-crystal best-of-k match (0/1)
    err = {k: [] for k in keys}        # per-crystal best-of-k non-ref orient error (deg)
    K = max(args.match_k, 1)

    def fit_any(gens, refs):
        return [1 if (r is not None and any(sm.fit(g, r) for g in gset)) else 0
                for gset, r in zip(gens, refs)]

    torch.manual_seed(args.seed)
    emb_un = None
    for s in range(0, len(val_items), args.batch_size):
        chunk = val_items[s:s + args.batch_size]
        batch = move_batch(collate([{**it, "idx": torch.tensor(0)} for it in chunk]), device)
        z1 = batch_to_state(batch)
        B = z1.lattice.shape[0]
        mol_emb = trained.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        emb_un = untrained.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        nonref = (~batch["is_ref"]) & z1.mask          # the symmetry-determined targets
        refs = structures_for(batch, z1, z1.orient)
        I = torch.eye(3, device=device).expand_as(z1.orient).contiguous()

        # deterministic conditions (k=1)
        for k, R in (("oracle", z1.orient), ("identity", I)):
            matched[k] += fit_any([[g] for g in structures_for(batch, z1, R)], refs)
            err[k] += per_crystal_err_deg(R, z1.orient, nonref).tolist()

        # stochastic flow conditions: best-of-k over orientation priors
        for k, net, emb in (("untrained", untrained, emb_un), ("trained", trained, mol_emb)):
            draws_struct = [[] for _ in range(B)]      # per crystal: list of k structures
            best_err = torch.full((B,), 1e9, device=device)
            for _ in range(K):
                R = sample_orient_only(net, emb, z1.lattice, z1.centroid,
                                       sample_prior(z1, vol_per_atom=vpa).orient,
                                       batch["sg"], z1.mask, args.steps)
                gs = structures_for(batch, z1, R)
                for b in range(B):
                    draws_struct[b].append(gs[b])
                best_err = torch.minimum(best_err, per_crystal_err_deg(R, z1.orient, nonref))
            matched[k] += fit_any(draws_struct, refs)
            err[k] += best_err.tolist()

    n = len(matched["oracle"])
    print(f"==== orientation-isolated reconstruction, best-of-{K} (true lattice+centroid+conformer) ====")
    print(f"  matcher: ltol={args.ltol} stol={args.stol} angle_tol={args.angle_tol}\n")
    print(f"  {'condition':10s}  {'match-rate':>10s}  {'non-ref orient err':>19s}")
    for k in keys:
        mr = sum(matched[k]) / max(n, 1)
        me = sum(err[k]) / max(len(err[k]), 1)
        tag = {"oracle": "  (ceiling)", "identity": "  (naive R=I)",
               "untrained": "  (floor)", "trained": "  <-- RESULT"}[k]
        print(f"  {k:10s}  {100*mr:9.1f}%  {me:16.1f} deg{tag}")
    print(f"\n  (n={n} crystals; orient err = best-of-{K} mean SO(3) angle on non-ref copies)")


if __name__ == "__main__":
    main()
