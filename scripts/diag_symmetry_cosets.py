"""Sanity/validation of the DEPLOYABLE symmetry-derived coset (`assign_symmetry_cosets`).

Contrasts the leak-free, symmetry-grounded coset with the old clustered codebook
(`assign_cosets`) on the real multi-copy CSD corpus, and checks that the centroid-derived
generating operation is geometrically consistent with the observed relative rotation (so the
label genuinely carries the target signal without ever looking at the orientation).

    python scripts/diag_symmetry_cosets.py --cache data/csd_mol/ds.pt

Reports, over the non-reference copies: distinct-coset count (new vs old), the min-image
centroid residual distribution (small => confident op assignment; large => special position /
disorder), the proper/improper operation split, and, for proper-op copies, the geodesic angle
between the operation's Cartesian rotation R_cart and the observed relative rotation R'_m."""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from symmc_flow import manifolds as M
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item, species_multiplicity,
                                    assign_cosets, assign_symmetry_cosets, _species_groups)
from symmc_flow.space_group import get_ops, cartesian_rotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--angle-tol", type=float, default=20.0, help="old-codebook clustering tol (deg)")
    args = ap.parse_args()
    if not os.path.exists(args.cache):
        sys.exit(f"no cache at {args.cache}")

    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    items = [it for it in rel if species_multiplicity(it) >= 2]
    print(f"multi-copy crystals: {len(items)}")

    old = copy.deepcopy(items)
    _, n_old = assign_cosets(old, angle_tol_deg=args.angle_tol)
    _, n_new = assign_symmetry_cosets(items)
    print(f"distinct cosets:  OLD clustered = {n_old}   NEW symmetry-derived = {n_new}")

    resid, ang_proper, n_proper, n_improper, n_nonref = [], [], 0, 0, 0
    for it in items:
        sg = int(it["sg"])
        ops = get_ops(sg)
        Rc = cartesian_rotations(sg, it["lattice"])
        for grp in _species_groups(it):
            c0 = it["centroid"][grp[0]]
            cand = torch.einsum("kij,j->ki", ops.W, c0) + ops.t
            cand = cand - torch.floor(cand)
            for m in grp[1:]:
                n_nonref += 1
                d = it["centroid"][m].unsqueeze(0) - cand
                d = d - torch.round(d)
                k = int(d.norm(dim=-1).argmin())
                resid.append(float(it["coset_resid"][m]))
                if float(torch.det(Rc[k])) > 0:
                    n_proper += 1
                    ang_proper.append(float(M.so3_angle(Rc[k], it["orient"][m])) * 180 / 3.14159265)
                else:
                    n_improper += 1

    resid = torch.tensor(resid)
    ang = torch.tensor(ang_proper)
    print(f"non-ref copies: {n_nonref}  (proper-op {n_proper}, improper {n_improper})")
    print(f"centroid residual: median={resid.median():.4f}  "
          f"frac<0.05={float((resid < 0.05).float().mean()):.3f}  "
          f"frac<0.10={float((resid < 0.10).float().mean()):.3f}")
    print(f"proper-op copies: geodesic(R_cart, observed R'_m) median={ang.median():.1f} deg  "
          f"frac<15deg={float((ang < 15).float().mean()):.3f}  "
          f"(=> the deployable label equals the target rotation for these)")


if __name__ == "__main__":
    main()
