"""Is the absolute molecular orientation distributed uniformly (Haar) on SO(3)?

Reviewer point M2: the claim that the free asymmetric-unit pose R_asym carries no
learnable signal is supported iff the absolute per-molecule orientations are
(close to) Haar-uniform on SO(3) -- i.e. there is no preferred orientation for the
field to predict. For Haar-uniform rotations the geodesic angle theta from the
identity has density p(theta) = (1 - cos theta)/pi on [0, pi] and CDF
F(theta) = (theta - sin theta)/pi; the mean angle is ~126.5 deg.

We take the reference copy of each species in each crystal (its absolute pose in
the molecule-intrinsic gauge, which is exactly R_asym up to the generating op of
the first-detected copy), compute the geodesic angle from the identity, and
compare the empirical distribution to the Haar density via a one-sample KS test.
A non-significant deviation + mean ~126.5 deg supports M2.

    python scripts/analyze_rasym_uniformity.py --cache data/csd_mol/ds.pt
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from symmc_flow.molcrystal import MolCrystalDataset, _species_groups
from symmc_flow import manifolds as M


def haar_cdf(theta):
    return (theta - np.sin(theta)) / math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--out", default="paper/figures/figS_rasym.pdf")
    args = ap.parse_args()

    ds = MolCrystalDataset(cache_path=args.cache)
    I = torch.eye(3)
    ref_angles, all_angles = [], []
    for it in ds.items:
        for g in _species_groups(it):
            # reference copy = first-detected copy of the species: absolute pose ~ R_asym
            ref_angles.append(float(M.so3_angle(it["orient"][g[0]], I)))
            for m in g:
                all_angles.append(float(M.so3_angle(it["orient"][m], I)))
    ref = np.array(ref_angles)
    allv = np.array(all_angles)

    _t = np.linspace(0, math.pi, 20001)
    mean_haar = math.degrees(  # numeric mean of Haar angle (= pi/2 + 1/pi rad ~ 126.5 deg)
        np.trapezoid(_t * (1 - np.cos(_t)) / math.pi, _t))

    print(f"reference-copy poses: n={len(ref)}")
    print(f"  mean geodesic angle from I: {math.degrees(ref.mean()):.1f} deg "
          f"(Haar expectation {mean_haar:.1f} deg)")
    print(f"all real-molecule poses: n={len(allv)}; "
          f"mean {math.degrees(allv.mean()):.1f} deg")

    try:
        from scipy.stats import kstest
        ks = kstest(ref, haar_cdf)
        print(f"  KS test vs Haar(SO(3)): D={ks.statistic:.4f}, p={ks.pvalue:.4f}")
        verdict = ("consistent with Haar-uniform (no preferred absolute orientation) "
                   "-> absolute target has no learnable signal, supporting M2"
                   if ks.pvalue > 0.05 else
                   "deviates from Haar; absolute orientation carries some structure")
        print(f"  VERDICT: {verdict}")
    except Exception as e:
        ks = None
        print(f"  [scipy unavailable: {e}]")

    # figure: empirical histogram vs Haar density
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        th = np.linspace(0, math.pi, 400)
        dens_per_deg = (1 - np.cos(th)) / 180.0   # Haar density expressed per degree
        fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=150)
        ax.hist(np.degrees(ref), bins=30, density=True, color="#0072B2", alpha=0.65,
                label=f"reference poses (n={len(ref)})")
        ax.plot(np.degrees(th), dens_per_deg,
                color="#D55E00", lw=2, label="Haar SO(3) density")
        ax.set_xlabel("geodesic angle from identity (deg)")
        ax.set_ylabel("density (1/deg)")
        ax.set_xlim(0, 180)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight")
        fig.savefig(args.out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"  figure -> {args.out}")
    except Exception as e:
        print(f"  [figure skipped: {e}]")


if __name__ == "__main__":
    main()
