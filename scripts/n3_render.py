"""Shared ray-traced rendering for the ACS figures that draw the LAJKOX crystal (Figure 1 and the TOC graphic).

One look for both: PyMOL ball-and-stick in outline ray-trace mode (scripts/toc_pymol_render.py, run in the PyMOL
venv with the scene on stdin, so no coordinates are written to disk), carbon atoms coloured by operation class
(teal = proper, det W_k = +1, a rotated copy; amber = improper, det W_k = -1, a mirror image), N, O and H in CPK
colours, Arial text (math too). crystal() returns the Figure 1 crystal, chosen and checked by the fixed rule in
scripts/fig_n3_worked_example.py, with every copy placed so that its centroid lies in the unit cell drawn with
its origin moved by -c/4 (any lattice translate of the cell is a unit cell; the copies are unchanged).
"""
from __future__ import annotations

import json
import os
import subprocess

import matplotlib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYMOL_PY = os.environ.get("PYMOL_PY", r"C:\Users\frank\.venvs\pymol\Scripts\python.exe")
RENDER = os.path.join(REPO, "scripts", "toc_pymol_render.py")
NO_WINDOW = 0x08000000

PROPER, IMPROPER = "#2A9D8F", "#E9A23B"      # carbon colour by operation class (teal / amber: clear of N blue, O red)
CPK = {1: "#FFFFFF", 6: "#909090", 7: "#3050F8", 8: "#FF0D0D"}
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
BALL = {1: 0.21, 6: 0.40, 7: 0.40, 8: 0.40}
STICK = 0.15
INK = "#222222"
CELL_SHIFT = np.array([0.0, 0.0, -0.25])     # fractional origin of the drawn cell
YAW, PITCH = np.radians(-18), np.radians(14)  # camera: view along b, turned to show the cell's depth

RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42, "ps.fonttype": 42,
      "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
      "mathtext.bf": "Arial:bold", "mathtext.default": "regular"}


def rgb(h):
    return [float(v) for v in matplotlib.colors.to_rgb(h)]


def run_pymol(scene, png):
    res = subprocess.run([PYMOL_PY, RENDER, png], input=json.dumps(scene), capture_output=True, text=True,
                         creationflags=NO_WINDOW)
    if res.returncode != 0 or not os.path.exists(png):
        raise RuntimeError(f"PyMOL render failed\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")


def frame(L, yaw=YAW, pitch=PITCH):
    """Camera basis (right, up, toward viewer) for the view along b, turned by yaw about up, then pitch about right."""
    bh = L[1] / np.linalg.norm(L[1])
    e1 = L[0] - (L[0] @ bh) * bh
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, bh)
    B = np.stack([e1, e2, -bh])                # rows: right, up, toward viewer (right-handed)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry @ B


def molecule_atoms(P, Zs, carbon):
    return [{"el": SYM[int(z)], "p": np.asarray(p).tolist(), "r": BALL[int(z)],
             "rgb": rgb(carbon if int(z) == 6 else CPK[int(z)])} for p, z in zip(P, Zs)]


def window_for(pts, pad, aspect):
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    if w / h < aspect:
        d = (aspect * h - w) / 2
        lo[0], hi[0] = lo[0] - d, hi[0] + d
    else:
        d = (w / aspect - h) / 2
        lo[1], hi[1] = lo[1] - d, hi[1] + d
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]


def cell_edges(L, cam, r=0.06, col="#6b6b6b"):
    """The 12 edges of the drawn cell (origin CELL_SHIFT) as CGO cylinders in camera coordinates."""
    o = CELL_SHIFT @ L
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    Cc = cam(o + corners @ L)
    edges = [(p, q) for p in range(8) for q in range(p + 1, 8) if np.abs(corners[p] - corners[q]).sum() == 1]
    return [{"a": Cc[p].tolist(), "b": Cc[q].tolist(), "r": r, "rgb": rgb(col)} for p, q in edges], Cc


def crystal():
    """The Figure 1 crystal: Figure 1's selection, split, CIF symmetry and expand-oracle checks, then expand(a, R0).

    Returns dict(ref, L, X (K, A, 3) whole copies in the drawn cell, Zs, bonds (heavy and H), W, t, kinds, dets,
    meta) with meta holding the selection facts the figure scripts print."""
    import fig_n3_worked_example as F
    from g2_asym_baselines import expand
    from n3_chirality import graph

    idx, a, train, mrmsd, n_cand = F.select()
    ref = a["refcode"]
    split = F.confirm_split(a, train)
    name, hall, cif_ops = F.cif_symmetry(ref)
    F.check_ops_against_cif(a, cif_ops)
    assert F.oracle(a), "expand(a, R0) does not reproduce the experimental structure"
    res, sep = F.orbit_check(a)

    L, cent, orient, local, Z, am, mm = expand(a, a["R0"])
    L = L.double().numpy()
    X = (cent.double().numpy() @ L)[:, None, :] + np.einsum("kij,kaj->kai", orient.double().numpy(),
                                                            local.double().numpy())
    fc = cent.double().numpy()
    shift = np.floor(fc - CELL_SHIFT)
    X = X - (shift @ L)[:, None, :]
    fc2 = fc - shift
    assert ((fc2 >= CELL_SHIFT) & (fc2 < CELL_SHIFT + 1)).all()
    Zs = Z[0].numpy().astype(int)
    bonds = [[int(i), int(j)] for i, j in graph(a["local"].double().numpy(), Zs).edges()]
    W, t = a["W"].numpy(), a["t"].numpy()
    kinds = [F.op_kind(w) for w in W]
    dets = [int(round(np.linalg.det(w))) for w in W]
    meta = dict(idx=idx, n_cand=n_cand, mirror_rmsd=mrmsd, split=split, hm=name, hall=hall, cif_ops=cif_ops,
                orbit_residual=res, orbit_sep=sep, frac_centroids=fc2)
    return dict(ref=ref, L=L, X=X, Zs=Zs, bonds=bonds, W=W, t=t, kinds=kinds, dets=dets, meta=meta)
