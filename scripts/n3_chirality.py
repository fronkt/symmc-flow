"""N3: is each asym conformer superimposable on its mirror image? (protocol §2.6 CHIRAL; §1.1 WL de-leak)

Port of the protocol-review script g3_chiral2.py with only I/O changed. Deterministic (no RNG).

mirror_rmsd = min over proper rotations R and element-preserving atom permutations P of
RMSD(local, P . mirror(local) . R^T), estimated two ways:
  parser : graph automorphisms (JmolNN bond rule on the conformer; the first max_iso=200 from networkx
           GraphMatcher, as molcrystal._align_to_reference) + det+1 Kabsch
  refined: min(parser, element-blocked Hungarian ICP, 30 iterations, from the 4 sign-diagonal rotations
           and the best automorphism rotation)
CHIRAL(tol) := refined > tol (tol = conf_tol 0.3 A). The bond graph `graph(X, Z)` is also the graph whose
Weisfeiler-Lehman hash (`wl_hash`) drives the TEST-B de-leak.

    python scripts/n3_chirality.py --set devtest --out results/n3/private/labels_chiral_devtest.json
"""
import argparse
import json
import os
import sys
import warnings

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_RAD = None
_SYM = {}


def _radius(z):
    global _RAD
    if _RAD is None:
        from pymatgen.analysis.local_env import JmolNN
        _RAD = JmolNN().el_radius
    if z not in _SYM:
        from pymatgen.core.periodic_table import Element
        _SYM[z] = Element.from_Z(int(z)).symbol
    return _RAD[_SYM[z]]


def graph(X, Z):
    """Bond graph of one conformer: bond iff 0.4 A < d <= r_i + r_j + 0.45 A (JmolNN radii)."""
    X, Z = np.asarray(X, float), np.asarray(Z)
    r = np.array([_radius(int(z)) for z in Z])
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    B = (D > 0.4) & (D <= r[:, None] + r[None] + 0.45)
    g = nx.Graph()
    for i, z in enumerate(Z):
        g.add_node(i, element=int(z))
    ii, jj = np.nonzero(np.triu(B, 1))
    g.add_edges_from(zip(ii.tolist(), jj.tolist()))
    return g


def wl_hash(X, Z):
    """Molecule identity for the de-leak: WL hash of the bond graph (3 iterations, digest 16)."""
    return nx.weisfeiler_lehman_graph_hash(graph(X, Z), node_attr="element", iterations=3, digest_size=16)


def kabsch(X, Y):
    """det+1 R minimising ||X - Y R^T||; returns (rmsd, R)."""
    U, S, Vt = np.linalg.svd(Y.T @ X)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    res = X - Y @ R.T
    return float(np.sqrt((res ** 2).sum(-1).mean())), R


def icp(X, Y, Z, R, iters=30):
    """Align Y (the mirror) onto X with a free element-matched permutation, from start rotation R."""
    best = np.inf
    for _ in range(iters):
        Yr = Y @ R.T
        perm = np.empty(len(Z), dtype=int)
        for z in np.unique(Z):
            idx = np.nonzero(Z == z)[0]
            C = np.linalg.norm(X[idx][:, None] - Yr[idx][None], axis=-1) ** 2
            ri, ci = linear_sum_assignment(C)
            perm[idx[ri]] = idx[ci]
        rmsd, R = kabsch(X, Y[perm])
        if rmsd > best - 1e-9:
            break
        best = rmsd
    return best


def mirror_rmsd(X, Z, max_iso=200):
    """-> dict(parser, refined, n_iso, hash, connected)."""
    X, Z = np.asarray(X, float), np.asarray(Z)
    g = graph(X, Z)
    Y = X * np.array([1.0, 1.0, -1.0])
    gm = GraphMatcher(g, g, node_match=lambda a, b: a["element"] == b["element"])
    parser, bestR, n_iso = np.inf, None, 0
    for n, mp in enumerate(gm.isomorphisms_iter()):
        if n >= max_iso:
            break
        n_iso += 1
        perm = np.array([mp[k] for k in range(len(Z))])
        r, R = kabsch(X, Y[perm])
        if r < parser:
            parser, bestR = r, R
    starts = [np.diag(s) for s in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    if bestR is not None:
        starts.append(bestR)
    refined = min([parser] + [icp(X, Y, Z, R) for R in starts])
    return {"parser": float(parser), "refined": float(refined), "n_iso": n_iso,
            "hash": nx.weisfeiler_lehman_graph_hash(g, node_attr="element", iterations=3, digest_size=16),
            "connected": bool(nx.is_connected(g))}


def label_items(items):
    """Per-crystal chirality / handedness labels for a list of asym items."""
    import torch
    rows = []
    for a in items:
        m = mirror_rmsd(a["local"].numpy(), a["Z"].numpy())
        improper = bool((torch.linalg.det(a["W"]) < 0).any())
        rows.append({"refcode": a["refcode"], "sg": int(a["sg"]), "K": int(a["K"]), "improper": improper, **m,
                     "chiral": m["refined"] > 0.3, "mixed": improper and m["refined"] > 0.3,
                     "improper_achiral": improper and not m["refined"] > 0.3})
    return rows


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import n3_common as C
    rows = label_items(C.load_set(args.set))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=0)
    n = len(rows)
    print(f"{args.set}: n={n} chiral={sum(r['chiral'] for r in rows)} mixed={sum(r['mixed'] for r in rows)} "
          f"improper_achiral={sum(r['improper_achiral'] for r in rows)} "
          f"iso_cap_hit={sum(r['n_iso'] >= 200 for r in rows)}")


if __name__ == "__main__":
    main()
