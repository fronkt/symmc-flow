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
                                    rigid_to_structure, species_multiplicity,
                                    assign_symmetry_cosets)
from symmc_flow.model import SymMCFlow
from symmc_flow.coset_predictor import CosetPredictor
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow import manifolds as M


@torch.no_grad()
def sample_orient_only(model, mol_emb, L_true, x_true, R0, sg, mask, steps, coset=None):
    """Orientation-only RK4 on SO(3): integrate dR/dt = v_R with lattice+centroid FROZEN at
    their true values (clean-packing conditioning). Mirrors the orient branch of rk4_sample so
    the result is exactly the flow's orientation given the true packing. `coset` (B,M) supplies
    the per-molecule space-group coset id to the field when the model is coset-conditioned."""
    B = R0.shape[0]
    R = R0.clone()
    dt = 1.0 / steps
    dev, dtp = L_true.device, L_true.dtype

    def vR(R_, t_scalar):
        t = torch.full((B,), float(t_scalar), device=dev, dtype=dtp)
        return model.forward(mol_emb, L_true, x_true, R_, t, sg, mask, coset=coset)[2]

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
    ap.add_argument("--coset", action="store_true",
                    help="supply the DEPLOYABLE symmetry-op coset to the trained field (needs a "
                         "coset-conditioned checkpoint); the template-based orientation read")
    ap.add_argument("--predictor-ckpt", default="",
                    help="a trained CosetPredictor (C4); use its PREDICTED coset (from packing "
                         "only, no template) instead of the ground-truth coset -- the de-novo read")
    ap.add_argument("--sweep", action="store_true",
                    help="also report match rate + median matched RMSD across a stol sweep")
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
    if args.coset or args.predictor_ckpt:
        if mcfg.n_cosets <= 0:
            sys.exit("--coset/--predictor-ckpt need a coset-conditioned checkpoint (n_cosets>0)")
        # assign on the FULL multi-copy corpus (train+val) so ids match the trained model
        assign_symmetry_cosets(full.items)
    val_items = [full.items[i] for i in val_idx]
    if args.limit:
        val_items = val_items[:args.limit]
    print(f"corpus: {len(rel)} -> {len(keep)} multi-copy; val crystals: {len(val_items)}")
    print(f"ckpt {args.ckpt}  clean_packing(train)={ck.get('clean_packing')}  vol/atom {vpa:.1f}\n")

    trained = SymMCFlow(mcfg).to(device).eval()
    trained.load_state_dict(ck["model"])
    torch.manual_seed(args.seed)
    untrained = SymMCFlow(mcfg).to(device).eval()  # fresh net = the predict-floor field

    predictor = None
    if args.predictor_ckpt:
        pk = torch.load(args.predictor_ckpt, map_location="cpu", weights_only=False)
        predictor = CosetPredictor(ModelConfig(**pk["model_cfg"])).to(device).eval()
        predictor.load_state_dict(pk["model"])
        print(f"coset PREDICTOR {args.predictor_ckpt} (val acc {100*pk.get('acc', float('nan')):.1f}%)"
              f" -> using PREDICTED coset (de-novo, no template)\n")

    keys = ("oracle", "identity", "untrained", "trained")
    err = {k: [] for k in keys}        # per-crystal best-of-k non-ref orient error (deg)
    all_refs = []                      # per-crystal ref Structure (len n)
    all_gens = {k: [] for k in keys}   # per-crystal list of draw Structures (k=1 for det.)
    K = max(args.match_k, 1)

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
        all_refs += structures_for(batch, z1, z1.orient)
        I = torch.eye(3, device=device).expand_as(z1.orient).contiguous()

        # deterministic conditions (k=1)
        for k, R in (("oracle", z1.orient), ("identity", I)):
            for g in structures_for(batch, z1, R):
                all_gens[k].append([g])
            err[k] += per_crystal_err_deg(R, z1.orient, nonref).tolist()

        if predictor is not None:                       # de-novo: coset PREDICTED from packing
            pemb = predictor.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
            cs = predictor.predict(pemb, z1.lattice, z1.centroid, batch["sg"], z1.mask)
        elif args.coset:                                 # template: ground-truth deployable coset
            cs = batch.get("coset")
        else:
            cs = None

        # stochastic flow conditions: store all k draws (draw 0 = the match@1 draw)
        for k, net, emb in (("untrained", untrained, emb_un), ("trained", trained, mol_emb)):
            draws_struct = [[] for _ in range(B)]      # per crystal: list of k structures
            best_err = torch.full((B,), 1e9, device=device)
            for _ in range(K):
                R = sample_orient_only(net, emb, z1.lattice, z1.centroid,
                                       sample_prior(z1, vol_per_atom=vpa).orient,
                                       batch["sg"], z1.mask, args.steps, coset=cs)
                gs = structures_for(batch, z1, R)
                for b in range(B):
                    draws_struct[b].append(gs[b])
                best_err = torch.minimum(best_err, per_crystal_err_deg(R, z1.orient, nonref))
            all_gens[k] += draws_struct
            err[k] += best_err.tolist()

    n = len(all_refs)

    def match_stats(gens, refs, ltol, stol, angle_tol, kcap=None):
        """Return (match@1, match@kcap, median matched best-RMSD) at a tolerance.
        The match DECISION uses StructureMatcher.fit (max-displacement <= stol, the strict
        criterion the paper's headline rate uses); get_rms_dist is used ONLY to report the
        RMSD of structures that already passed fit (it applies a more lenient rms-based
        criterion, so 'get_rms_dist is not None' would over-count borderline matches).
        Draw 0 of each crystal is the single-draw (match@1) candidate."""
        sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        hit1, hitk, rmsds = 0, 0, []
        for gset, r in zip(gens, refs):
            if r is None:
                continue
            cap = len(gset) if kcap is None else min(kcap, len(gset))
            try:
                if sm.fit(gset[0], r):
                    hit1 += 1
            except Exception:
                pass
            matched, best = False, None
            for g in gset[:cap]:
                try:
                    if sm.fit(g, r):
                        matched = True
                        rms = sm.get_rms_dist(g, r)
                        if rms is not None and (best is None or rms[0] < best):
                            best = rms[0]
                except Exception:
                    pass
            if matched:
                hitk += 1
                if best is not None:
                    rmsds.append(best)
        rmsds.sort()
        med = rmsds[len(rmsds) // 2] if rmsds else float("nan")
        return hit1 / max(n, 1), hitk / max(n, 1), med

    print(f"==== orientation-isolated reconstruction (true lattice+centroid+conformer) ====")
    print(f"  default matcher: ltol={args.ltol} stol={args.stol} angle_tol={args.angle_tol}; "
          f"best-of-{K}\n")
    print(f"  {'condition':10s}  {'match@1':>8s}  {'match@k':>8s}  {'med RMSD':>9s}  "
          f"{'non-ref err':>12s}")
    for k in keys:
        m1, mk, med = match_stats(all_gens[k], all_refs, args.ltol, args.stol, args.angle_tol)
        me = sum(err[k]) / max(len(err[k]), 1)
        tag = {"oracle": "  (ceiling)", "identity": "  (naive R=I)",
               "untrained": "  (floor)", "trained": "  <-- RESULT"}[k]
        print(f"  {k:10s}  {100*m1:7.1f}%  {100*mk:7.1f}%  {med:9.3f}  {me:9.1f} deg{tag}")
    print(f"\n  (n={n} crystals; match@1 = single deterministic-prior draw, "
          f"match@k = best of {K}; med RMSD = median matched StructureMatcher RMS)")

    if args.sweep:
        print(f"\n==== StructureMatcher tolerance sweep (trained; best-of-{K}) ====")
        print(f"  {'ltol':>5s} {'stol':>5s} {'angle':>5s}  {'match@1':>8s}  {'match@k':>8s}  "
              f"{'med RMSD':>9s}")
        for (lt, st, at) in [(0.2, 0.2, 8.0), (0.3, 0.3, 10.0), (0.3, 0.5, 10.0),
                             (0.5, 0.7, 15.0)]:
            m1, mk, med = match_stats(all_gens["trained"], all_refs, lt, st, at)
            o1, ok, _ = match_stats(all_gens["oracle"], all_refs, lt, st, at)
            print(f"  {lt:5.2f} {st:5.2f} {at:5.1f}  {100*m1:7.1f}%  {100*mk:7.1f}%  "
                  f"{med:9.3f}   (oracle@k {100*ok:.0f}%)")


if __name__ == "__main__":
    main()
