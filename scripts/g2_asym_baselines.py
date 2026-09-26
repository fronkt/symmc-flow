"""G2 (Omega rebuild), part 1: asymmetric-unit crystals + the physics-only baselines.

The rebuilt model learns ONE free orientation per crystal: the asymmetric-unit pose R_asym. Every
other copy is generated exactly by the space-group operations applied to atoms, including the
improper ones (det -1), so mirror/inversion copies need no learned orientation at all. This script

  1. converts the mirror-aware G1 dataset (`ds_o3.pt`) into asymmetric-unit items for Z'=1
     general-position crystals (molecule count == number of space-group operations), and checks the
     exact expansion reconstructs each crystal (oracle, must be ~100%);
  2. scores the baselines a learned R_asym must beat, holding the TRUE lattice and asym centroid
     fixed (orientation-isolated, the same setting as the paper's orientation reads):
       haar          random R_asym, no relaxation (floor)
       haar+press    random R_asym, then the rigid-press packing relaxer
       bestK+press   K random R_asym, each relaxed, keep the lowest packing energy (classical search)
       anyK+press    oracle upper bound: did ANY of the K relaxed draws match?
       true(+press)  the true R_asym, with/without relaxation (checks the relaxer keeps a right answer)
Match = pymatgen StructureMatcher on the expanded structure vs the ground truth.

    python scripts/g2_asym_baselines.py --data data/csd_mol/ds_o3.pt --out results/g2_baselines.json \
        --n 200 --k 16 --workers 3
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def crystal_ops(it):
    """Fractional symmetry ops (W, t) in the crystal's OWN setting, from spglib on the item itself.
    The standard-setting table (`space_group.get_ops`) is wrong for alternative settings the CSD
    uses (e.g. P2_1/n translates by (1/2,1/2,1/2), P2_1/c by (0,1/2,1/2)), so never assume it."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    st = to_structure(it["lattice"], it["centroid"], it["orient"], it["local"], it["Z"],
                      it["atom_mask"], it["mol_mask"])
    ops = SpacegroupAnalyzer(st, symprec=0.1).get_symmetry_operations(cartesian=False)
    W = torch.tensor(np.array([op.rotation_matrix for op in ops]), dtype=torch.float64)
    t = torch.tensor(np.array([op.translation_vector for op in ops]), dtype=torch.float64)
    t = t - torch.floor(t + 1e-6)
    ident = [k for k in range(len(ops)) if torch.allclose(W[k], torch.eye(3, dtype=W.dtype))
             and float(t[k].abs().max()) < 1e-6]
    order = ident + [k for k in range(len(ops)) if k not in ident]  # identity first
    return W[order], t[order]


def asym_item(it, ops=None):
    """ds_o3 item -> asym-unit dict (or None if not Z'=1 general position)."""
    real = [m for m in range(it["mol_mask"].shape[0]) if bool(it["mol_mask"][m])]
    na = [int(it["atom_mask"][m].sum()) for m in real]
    if len(set(na)) != 1:
        return None
    try:
        W, t = ops if ops is not None else crystal_ops(it)
    except Exception:
        return None
    K = W.shape[0]
    if len(real) != K:
        return None
    m0 = real[0]
    A = na[0]
    return {"sg": int(it["sg"]), "L": it["lattice"].double(), "W": W, "t": t,
            "c0": it["centroid"][m0].double(), "R0": it["orient"][m0].double(),
            "local": it["local"][m0, :A].double(), "Z": it["Z"][m0, :A], "K": K,
            "refcode": it.get("refcode", "?"), "orig": it}


def cart_ops(a):
    Lt = a["L"].transpose(-1, -2)
    return Lt @ a["W"] @ torch.linalg.inv(Lt)                  # (K,3,3), det +-1


def expand(a, R0, c0=None):
    """Asym pose -> full-cell tensors (every copy from the space-group ops, det +-1 allowed)."""
    c0 = a["c0"] if c0 is None else c0
    Rc = cart_ops(a)
    cent = torch.einsum("kij,j->ki", a["W"], c0) + a["t"]
    cent = cent - torch.floor(cent)
    orient = Rc @ R0
    K, A = a["K"], a["local"].shape[0]
    return (a["L"], cent, orient, a["local"].expand(K, A, 3).clone(),
            a["Z"].expand(K, A).clone(), torch.ones(K, A, dtype=torch.bool),
            torch.ones(K, dtype=torch.bool))


def to_structure(L, cent, orient, local, Z, am, mm):
    from symmc_flow.molcrystal import rigid_to_structure
    return rigid_to_structure(L.float(), Z, local.float(), cent.float(), orient.float(), am, mm)


def energy(L, cent, orient, local, Z, am, mm, contact=0.90, cutoff=6.0):
    from symmc_flow.rigid_press import _radii, build_neighbor_list, packing_energy_nbr
    Lf = L.float()
    # keep each molecule whole (centroid + R @ local, NOT per-atom wrapped): wrapping splits a
    # molecule across the cell edge and its own bonds then count as intermolecular clashes
    X = (cent.float() @ Lf).unsqueeze(1) + torch.einsum("mij,maj->mai", orient.float(), local.float())
    X = X[am]
    mid = torch.arange(am.shape[0]).unsqueeze(1).expand_as(am)[am]
    r = _radii(Z[am])
    nbr = build_neighbor_list(X, r, mid, Lf, contact, cutoff)
    return float(packing_energy_nbr(X, Lf, *nbr, cutoff=cutoff))


def press(a, R0, steps, lr=0.02, contact=0.90, cutoff=6.0):
    """Orientation-only packing relaxation of the asym pose. Lattice and asym centroid are the
    given (true) template in this orientation-isolated read, so only R_asym = exp(delta) R0 moves;
    every copy is regenerated from it by the space-group ops each step, so symmetry stays exact.
    (`rigid_press.finish_structure` also moves the centroid with fractional-coordinate steps sized
    for rough generated cells; from the TRUE pose it drifted 1-3 A / 6-18 deg, so it cannot serve
    as a baseline that must keep a right answer right.)"""
    from symmc_flow import manifolds as M
    from symmc_flow.rigid_press import _radii, build_neighbor_list, packing_energy_nbr
    L, cent, orient, local, Z, am, mm = expand(a, R0)
    Lf = L.float()
    Rc = cart_ops(a).float()                                  # (K,3,3), det +-1
    base = (cent.float() @ Lf).unsqueeze(1)                   # (K,1,3) copy centroids, fixed
    loc = local.float()
    r = _radii(Z[am])
    mid = torch.arange(am.shape[0]).unsqueeze(1).expand_as(am)[am]
    R0f = R0.float()
    delta = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    def build():
        R = Rc @ (M.so3_exp(delta) @ R0f)                     # (K,3,3)
        return (base + torch.einsum("kij,kaj->kai", R, loc))[am]

    nbr = None
    for s in range(steps):
        if s % 15 == 0:
            with torch.no_grad():
                nbr = build_neighbor_list(build(), r, mid, Lf, contact, cutoff)
        opt.zero_grad()
        e = packing_energy_nbr(build(), Lf, *nbr, cutoff=cutoff)
        if not torch.isfinite(e):
            break
        e.backward()
        torch.nn.utils.clip_grad_norm_([delta], 1.0)
        opt.step()
    with torch.no_grad():
        R0n = (M.so3_exp(delta) @ R0f).double()
    return expand(a, R0n)


def _work(task):
    warnings.filterwarnings("ignore")
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from symmc_flow import manifolds as M
    it, k, steps, seed = task
    a = asym_item(it)
    if a is None:
        return None
    torch.manual_seed(seed)
    sm = StructureMatcher()
    o = a["orig"]
    truth = to_structure(o["lattice"], o["centroid"], o["orient"], o["local"], o["Z"],
                         o["atom_mask"], o["mol_mask"])
    fit = lambda cell: bool(sm.fit(to_structure(*cell), truth))
    t0 = time.time()
    res = {"refcode": a["refcode"], "sg": a["sg"], "K": a["K"], "A": int(a["local"].shape[0])}
    res["oracle"] = fit(expand(a, a["R0"]))
    true_rel = press(a, a["R0"], steps)
    res["true_press_match"] = fit(true_rel)               # force-field offset: relaxed truth vs truth
    res["true_press_dR"] = math.degrees(float(M.so3_angle(true_rel[2][0].float(),
                                                          expand(a, a["R0"])[2][0].float())))
    basin_ref = to_structure(*true_rel)
    basin = lambda cell: bool(sm.fit(to_structure(*cell), basin_ref))
    draws = M.random_so3((k,), dtype=torch.float64)
    res["haar"] = fit(expand(a, draws[0]))                # unrelaxed exact match (floor)
    pressed = [press(a, draws[i], steps) for i in range(k)]
    es = [energy(*c) for c in pressed]
    inb = [basin(c) for c in pressed]
    best = min(range(k), key=lambda i: es[i])
    res["haar_press_basin"] = inb[0]                      # one relaxed random draw finds the basin
    res["bestK_press_basin"] = inb[best]                  # classical search: lowest-energy of K
    res["anyK_press_basin"] = any(inb)                    # upper bound for K draws
    res["e_true_press"] = energy(*true_rel)
    res["e_best"] = es[best]
    res["sec"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="crystals to score (seeded sample of Z'=1)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    items = torch.load(args.data, weights_only=False)["items"]
    elig = [it for it in items if asym_item(it) is not None]
    print(f"{len(items)} crystals, {len(elig)} Z'=1 general-position", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    pick = [elig[i] for i in torch.randperm(len(elig), generator=g)[:args.n].tolist()]

    from multiprocessing import Pool, TimeoutError as MPTimeout
    pool = Pool(args.workers)
    pend = [pool.apply_async(_work, ((it, args.k, args.steps, args.seed + i),))
            for i, it in enumerate(pick)]
    rows = []
    for i, ar in enumerate(pend):
        try:
            r = ar.get(timeout=args.timeout)
        except MPTimeout:
            r = {"refcode": pick[i].get("refcode", "?"), "timeout": True}
        rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(pick)}", flush=True)
    pool.terminate(); pool.join()

    done = [r for r in rows if r and not r.get("timeout")]
    keys = ["oracle", "true_press_match", "haar", "haar_press_basin", "bestK_press_basin", "anyK_press_basin"]
    summ = {k: sum(r[k] for r in done) / max(len(done), 1) for k in keys}
    summ["n"] = len(done)
    summ["timeouts"] = sum(1 for r in rows if r and r.get("timeout"))
    summ["n_eligible"] = len(elig)
    summ["median_true_press_dR"] = sorted(r["true_press_dR"] for r in done)[len(done) // 2] if done else None
    summ["true_below_best_energy"] = sum(r["e_true_press"] <= r["e_best"] + 1e-3 for r in done) / max(len(done), 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "summary": summ, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
