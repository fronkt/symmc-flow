"""T1.2 / Devil's-Advocate P2: is R_asym CONDITIONALLY predictable from packing?

The Haar-uniformity test (analyze_rasym_uniformity.py, Fig S1) shows the *marginal*
distribution of the asymmetric-unit pose R_asym is ~uniform on SO(3). A uniform marginal
does NOT by itself prove conditional independence: a weak p(R_asym | packing) can coexist
with a Haar marginal. This script tests the conditional directly. We train a supervised
PROBE network to predict the reference-copy absolute orientation (= R_asym up to the
generating op) from composition-and-packing features ONLY -- lattice shape, the molecule's
fractional centroid, the cell's centroid arrangement, the conformer's intrinsic shape, and
the space group -- and compare its held-out geodesic error to:

  * constant : the single best rotation (Fréchet/chordal mean of the train targets); the
               error any feature-free predictor incurs.
  * haar     : a random-rotation guess; the no-information ceiling.

If the probe cannot beat the constant predictor, then under this conditioning R_asym carries
no exploitable signal -- the operational claim the paper makes -- independently of the
marginal test.

    python scripts/probe_rasym_conditional.py --cache data/csd_mol/ds.pt --epochs 300
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from symmc_flow.molcrystal import MolCrystalDataset, _species_groups
from symmc_flow import manifolds as M

RAD2DEG = 180.0 / math.pi


def cell_params(L):
    """3x3 lattice (rows = vectors) -> (a,b,c, cos alpha, cos beta, cos gamma)."""
    a, b, c = L[0], L[1], L[2]
    la, lb, lc = a.norm(), b.norm(), c.norm()
    ca = (b @ c) / (lb * lc).clamp_min(1e-6)
    cb = (a @ c) / (la * lc).clamp_min(1e-6)
    cg = (a @ b) / (la * lb).clamp_min(1e-6)
    return torch.tensor([la, lb, lc, ca, cb, cg])


def gyration_eig(local, mask):
    """Intrinsic conformer shape: sorted eigenvalues of the gyration tensor (3,)."""
    x = local[mask]
    if x.shape[0] < 2:
        return torch.zeros(3)
    x = x - x.mean(0)
    cov = (x.t() @ x) / x.shape[0]
    ev = torch.linalg.eigvalsh(cov)
    return torch.sort(ev, descending=True).values


def build_samples(items):
    """One sample per species group: (features, target R0, sg). Target is the absolute pose
    of the reference (first) copy -- R_asym in the molecule-intrinsic gauge."""
    feats, targets, sgs = [], [], []
    for it in items:
        L = it["lattice"]
        cps = cell_params(L)
        cents = it["centroid"][it["mol_mask"]]
        cmean = cents.mean(0) if cents.shape[0] else torch.zeros(3)
        cstd = cents.std(0) if cents.shape[0] > 1 else torch.zeros(3)
        nmol = float(it["mol_mask"].sum())
        sg = int(it["sg"])
        for g in _species_groups(it):
            m = g[0]
            own_c = it["centroid"][m]
            shape = gyration_eig(it["local"][m], it["atom_mask"][m])
            natoms = float(it["atom_mask"][m].sum())
            f = torch.cat([cps, own_c, cmean, cstd,
                           torch.tensor([nmol, natoms]), shape])  # 6+3+3+3+2+3 = 20
            feats.append(f)
            targets.append(it["orient"][m])
            sgs.append(sg)
    return torch.stack(feats), torch.stack(targets), torch.tensor(sgs, dtype=torch.long)


def sixd_to_R(x):
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns -> rotation matrix


class Probe(nn.Module):
    def __init__(self, n_feat, n_sg, d=128, emb=16):
        super().__init__()
        self.sg_emb = nn.Embedding(n_sg, emb)
        self.net = nn.Sequential(
            nn.Linear(n_feat + emb, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, 6))

    def forward(self, f, sg):
        h = torch.cat([f, self.sg_emb(sg)], dim=-1)
        return sixd_to_R(self.net(h))


def frechet_mean(R):
    """Chordal mean of a set of rotations -> nearest SO(3) matrix (approx geodesic mean for
    concentrated data)."""
    return M.project_so3(R.mean(0, keepdim=True))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = MolCrystalDataset(cache_path=args.cache)
    F, T, SG = build_samples(ds.items)
    n = F.shape[0]
    print(f"samples (reference copies): {n}; feature dim {F.shape[1]}")

    # normalize features; remap sg to contiguous ids
    F = (F - F.mean(0)) / F.std(0).clamp_min(1e-6)
    uniq = {int(s): i for i, s in enumerate(sorted(set(SG.tolist())))}
    SGc = torch.tensor([uniq[int(s)] for s in SG], dtype=torch.long)

    perm = torch.randperm(n)
    nv = int(round(args.val_frac * n))
    vi, ti = perm[:nv], perm[nv:]
    Ftr, Ttr, Str = F[ti], T[ti], SGc[ti]
    Fva, Tva, Sva = F[vi], T[vi], SGc[vi]
    print(f"split: {len(ti)} train / {len(vi)} val\n")

    # ---- baselines ----
    Rconst = frechet_mean(Ttr)
    const_err = M.so3_angle(Rconst.expand_as(Tva), Tva).mean() * RAD2DEG
    train_const_err = M.so3_angle(Rconst.expand_as(Ttr), Ttr).mean() * RAD2DEG
    Rhaar = M.random_so3((Tva.shape[0],))
    haar_err = M.so3_angle(Rhaar, Tva).mean() * RAD2DEG

    # ---- probe ----
    probe = Probe(F.shape[1], len(uniq))
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=1e-5)
    best_va = 1e9
    for ep in range(args.epochs):
        probe.train()
        opt.zero_grad()
        Rp = probe(Ftr, Str)
        # chordal (Frobenius) loss -- smooth surrogate for geodesic
        loss = ((Rp - Ttr) ** 2).sum((-1, -2)).mean()
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            probe.eval()
            with torch.no_grad():
                tr_err = M.so3_angle(probe(Ftr, Str), Ttr).mean() * RAD2DEG
                va_err = M.so3_angle(probe(Fva, Sva), Tva).mean() * RAD2DEG
            best_va = min(best_va, float(va_err))
            print(f"  ep {ep+1:4d}  train geo {tr_err:6.1f} deg   val geo {va_err:6.1f} deg")

    probe.eval()
    with torch.no_grad():
        probe_va = M.so3_angle(probe(Fva, Sva), Tva).mean() * RAD2DEG

    print("\n==== R_asym conditional-predictability probe (held-out geodesic error) ====")
    print(f"  haar guess (no info)        : {haar_err:6.1f} deg")
    print(f"  constant (Fréchet mean)     : {const_err:6.1f} deg   "
          f"(train {train_const_err:.1f})")
    print(f"  probe (packing+sg features) : {probe_va:6.1f} deg   (best {best_va:.1f})")
    gain = 100 * (1 - probe_va / const_err) if const_err else 0.0
    print(f"\n  probe vs constant: {gain:+.1f}% error reduction")
    if gain < 5:
        print("  VERDICT: probe does NOT beat the feature-free constant -> R_asym carries no")
        print("           exploitable conditional signal under composition+packing here.")
    else:
        print("  VERDICT: probe beats constant -> some conditional signal exists; revisit claim.")


if __name__ == "__main__":
    main()
