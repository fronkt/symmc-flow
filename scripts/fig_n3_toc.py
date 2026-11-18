"""Table-of-contents graphic for the ACS Omega resubmission (ACS TOC size: 3.25 x 1.75 in).

v2 (2026-11-17), redrawn after Frank found v1 cluttered and flat: ray-traced PyMOL ball-and-stick renders in the
STS-report style (outline ray trace mode, CPK colours), no results strip, no symmetry symbols, three short labels.

Left to right:
  1. the asymmetric-unit molecule of LAJKOX (the Figure 1 crystal, chosen by Figure 1's fixed selection rule) in
     its true orientation, inside a 3-D rotation arrow: the one quantity the model learns, R_asym;
  2. an arrow: exact space-group expansion (P2_1/c, four operations);
  3. the cell with the four copies made by expand(a, R0) (as Figure 1(a)). Carbon atoms carry the copy's class:
     teal = proper operation (identity, 2_1 screw: a rotated copy), amber = improper operation (inversion,
     c-glide, det W_k = -1: a mirror image); N, O, H keep CPK colours. The drawn cell is the lattice's unit cell
     with its origin moved by -c/4, so that every copy's centroid lies inside the box (any lattice translate of
     the cell is a unit cell); the copies are unchanged.

Selection, split check, CIF symmetry check and expand oracle are imported from scripts/fig_n3_worked_example.py.
The scene goes to the PyMOL process on stdin (scripts/toc_pymol_render.py, PyMOL venv); no coordinates are
written to any file, only rendered images. Outputs: PNG opaque RGB at 600 dpi (1950 x 1050 px) and a PDF of
exactly 3.25 x 1.75 in (vector text, embedded renders).

    python scripts/fig_n3_toc.py
writes paper/submissions/ACSOmega/figures/fig_n3_toc.{pdf,png}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import fig_n3_worked_example as F  # noqa: E402  (shared rcParams: Arial, mathtext regular, fonttype 42)

plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold"})   # math labels in Arial too: one embedded typeface
OUT = os.environ.get("TOC_OUT", F.OUT)        # TOC_OUT: draft directory while iterating
PYMOL_PY = os.environ.get("PYMOL_PY", r"C:\Users\frank\.venvs\pymol\Scripts\python.exe")
RENDER = os.path.join(REPO, "scripts", "toc_pymol_render.py")
NO_WINDOW = 0x08000000

W_IN, H_IN, DPI = 3.25, 1.75, 600
PROPER, IMPROPER = "#2A9D8F", "#E9A23B"      # carbon colour by operation class (teal / amber: clear of N blue, O red)
CPK = {1: "#FFFFFF", 6: "#909090", 7: "#3050F8", 8: "#FF0D0D"}
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
BALL = {1: 0.21, 6: 0.40, 7: 0.40, 8: 0.40}
STICK = 0.15
INK = "#222222"


def rgb(h):
    return [float(v) for v in matplotlib.colors.to_rgb(h)]


def run_pymol(scene, png):
    res = subprocess.run([PYMOL_PY, RENDER, png], input=json.dumps(scene), capture_output=True, text=True,
                         creationflags=NO_WINDOW)
    if res.returncode != 0 or not os.path.exists(png):
        raise RuntimeError(f"PyMOL render failed\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")


def frame(L, yaw, pitch):
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
    return [{"el": SYM[int(z)], "p": p.tolist(), "r": BALL[int(z)],
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


def rotation_arrow(u, v, radius, start_deg, sweep_deg, r, col, n=120):
    """3-D circular arrow r(cos th u + sin th v) about the origin (camera frame), as CGO cylinders, joint spheres
    and a cone head. u, v orthonormal; tilting v out of the page makes the ring pass behind and in front."""
    th = np.radians(start_deg + np.linspace(0, sweep_deg, n))
    pts = radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    head = 8.0 * r                              # cone length
    tang = pts[-1] - pts[-2]
    tang /= np.linalg.norm(tang)
    cyl = [{"a": pts[i].tolist(), "b": pts[i + 1].tolist(), "r": r, "rgb": rgb(col)} for i in range(n - 1)]
    sph = [{"p": q.tolist(), "r": r, "rgb": rgb(col)} for q in pts[1:-1]] + [{"p": pts[0].tolist(), "r": r,
                                                                             "rgb": rgb(col)}]
    cone = [{"a": pts[-1].tolist(), "b": (pts[-1] + head * tang).tolist(), "r": 3.0 * r, "rgb": rgb(col)}]
    return cyl, sph, cone, pts


ARC_START = float(os.environ.get("TOC_ARC_START", 115))   # deg; where the ring's tail starts (head = +290)


def _clears(s, t, a, cosb, margin, n=720):
    """Every atom (s, t) lies at least `margin` inside the ellipse with semi-axes a (along s) and a*cosb (along t)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    E = np.stack([a * np.cos(th), a * cosb * np.sin(th)], 1)
    D = np.hypot(s[:, None] - E[None, :, 0], t[:, None] - E[None, :, 1]).min(1)
    return bool((D >= margin).all())


def main():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    from g2_asym_baselines import expand
    from n3_chirality import graph

    # ---------------- crystal: Figure 1's selection, split, CIF symmetry and oracle checks, unchanged
    idx, a, train, mrmsd, n_cand = F.select()
    ref = a["refcode"]
    F.confirm_split(a, train)
    _, _, cif_ops = F.cif_symmetry(ref)
    F.check_ops_against_cif(a, cif_ops)
    assert F.oracle(a), "expand(a, R0) does not reproduce the experimental structure"

    L, cent, orient, local, Z, am, mm = expand(a, a["R0"])
    L = L.double().numpy()
    X = (cent.double().numpy() @ L)[:, None, :] + np.einsum("kij,kaj->kai", orient.double().numpy(),
                                                            local.double().numpy())
    Zs = Z[0].numpy().astype(int)
    bonds = [[int(i), int(j)] for i, j in graph(a["local"].double().numpy(), Zs).edges()]
    W = a["W"].numpy()
    kinds = [F.op_kind(w) for w in W]
    dets = [int(round(np.linalg.det(w))) for w in W]
    k_id = kinds.index("1")

    # copies whose centroid lies outside the shifted cell [0,1) x [0,1) x [-1/4, 3/4) move by a lattice vector
    fc = cent.double().numpy()
    shift = np.floor(fc - np.array([0.0, 0.0, -0.25]))
    X = X - (shift @ L)[:, None, :]
    fc2 = fc - shift
    assert ((fc2 >= [0, 0, -0.25]) & (fc2 < [1, 1, 0.75])).all()

    B = frame(L, yaw=np.radians(-18), pitch=np.radians(14))
    cam = lambda P: np.asarray(P) @ B.T                       # noqa: E731

    # ---------------- right: the cell with four copies
    o = np.array([0.0, 0.0, -0.25]) @ L
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    Cc = cam(o + corners @ L)
    edges = [(p, q) for p in range(8) for q in range(p + 1, 8) if np.abs(corners[p] - corners[q]).sum() == 1]
    atoms, bnd = [], []
    for k in range(len(X)):
        base = len(atoms)
        atoms += molecule_atoms(cam(X[k]), Zs, PROPER if dets[k] > 0 else IMPROPER)
        bnd += [[base + i, base + j] for i, j in bonds]
    cell_cyl = [{"a": Cc[p].tolist(), "b": Cc[q].tolist(), "r": 0.06, "rgb": rgb("#6b6b6b")} for p, q in edges]
    pts = np.vstack([np.array([at["p"] for at in atoms])[:, :2], Cc[:, :2]])
    RX_IN, RY_IN = 1.74, 1.24                   # size of the cell render on the page
    winR = window_for(pts, 0.6, RX_IN / RY_IN)
    sceneR = {"atoms": atoms, "bonds": bnd, "stick_r": STICK, "cyl": cell_cyl, "window": winR,
              "res_x": int(RX_IN * DPI), "res_y": int(RY_IN * DPI), "outline": True, "gain": 0.09,
              "fog_start": 0.55}

    # ---------------- left: the asymmetric-unit molecule inside a rotation arrow
    P0 = cam(X[k_id])
    c0 = P0.mean(0)
    P0 = P0 - c0
    # ring: an ellipse on the page along the molecule's long axis d, tilted out of the page about d by TILT, sized
    # so the projected ellipse clears every atom by MARGIN
    TILT, MARGIN = np.radians(38), 0.75
    ev, evec = np.linalg.eigh(np.cov(P0[:, :2].T))
    d = evec[:, -1]
    nrm = np.array([-d[1], d[0]])
    s, t = P0[:, :2] @ d, P0[:, :2] @ nrm
    rad = 1.0
    while np.any((s / (rad + 1e-9)) ** 2 + (t / (rad * np.cos(TILT))) ** 2 > 1.0) or \
            np.min(np.hypot(s, t) - 0) < 0 or not _clears(s, t, rad, np.cos(TILT), MARGIN):
        rad += 0.05
    u = np.array([d[0], d[1], 0.0])
    v = np.array([nrm[0] * np.cos(TILT), nrm[1] * np.cos(TILT), np.sin(TILT)])
    cyl, sph, cone, arc = rotation_arrow(u, v, rad, start_deg=ARC_START, sweep_deg=290, r=0.12, col=INK)
    LX_IN, LY_IN = 0.90, 1.16
    winL = window_for(np.vstack([P0[:, :2], arc[:, :2]]), 0.7, LX_IN / LY_IN)
    sceneL = {"atoms": molecule_atoms(P0, Zs, PROPER), "bonds": bonds, "stick_r": STICK, "cyl": cyl, "sph": sph,
              "cone": cone, "window": winL, "res_x": int(LX_IN * DPI), "res_y": int(LY_IN * DPI),
              "outline": True, "gain": 0.09, "fog": False}

    with tempfile.TemporaryDirectory() as td:
        pngL, pngR = os.path.join(td, "left.png"), os.path.join(td, "right.png")
        run_pymol(sceneL, pngL)
        run_pymol(sceneR, pngR)
        imL, imR = mpimg.imread(pngL), mpimg.imread(pngR)

    # ---------------- compose
    fig = plt.figure(figsize=(W_IN, H_IN))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W_IN)
    bg.set_ylim(0, H_IN)
    bg.axis("off")
    yL = 0.40                                   # bottom of the renders (room for the labels below)
    axL = fig.add_axes([0.02 / W_IN, yL / H_IN, LX_IN / W_IN, LY_IN / H_IN])
    axL.imshow(imL, interpolation="none")
    axL.axis("off")
    xR = W_IN - RX_IN - 0.02
    axR = fig.add_axes([xR / W_IN, (yL - 0.06) / H_IN, RX_IN / W_IN, RY_IN / H_IN])
    axR.imshow(imR, interpolation="none")
    axR.axis("off")

    fs = 7.2
    cxL = 0.02 + LX_IN / 2
    bg.text(cxL, 0.27, "learn one rotation", ha="center", va="center", fontsize=fs, color=INK)
    bg.text(cxL, 0.12, "$R_{\\mathrm{asym}}$", ha="center", va="center", fontsize=fs + 0.5, color=INK)
    x0, x1, ya = 0.02 + LX_IN + 0.02, xR + 0.06, yL + 0.55
    bg.add_patch(FancyArrowPatch((x0, ya), (x1, ya), arrowstyle="-|>", mutation_scale=9, lw=1.3, color=INK))
    bg.text((x0 + x1) / 2, ya + 0.09, "exact\nexpansion", ha="center", va="bottom", fontsize=fs, color=INK,
            linespacing=1.1)
    bg.text((x0 + x1) / 2, ya - 0.10, "$\\mathit{P}2_1/\\mathit{c}$", ha="center", va="top", fontsize=fs, color=INK)
    cxR = xR + RX_IN / 2
    for x, col, txt in ((cxR - 0.62, PROPER, "rotated copy"), (cxR + 0.22, IMPROPER, "mirror image")):
        bg.add_patch(Circle((x, 0.16), 0.045, facecolor=col, edgecolor=INK, lw=0.5))
        bg.text(x + 0.08, 0.16, txt, ha="left", va="center", fontsize=fs, color=INK)

    os.makedirs(OUT, exist_ok=True)
    pdf, png = os.path.join(OUT, "fig_n3_toc.pdf"), os.path.join(OUT, "fig_n3_toc.png")
    fig.savefig(pdf, dpi=DPI)
    fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)
    from PIL import Image
    im = Image.open(png).convert("RGB")
    im.save(png)
    assert im.size == (int(W_IN * DPI), int(H_IN * DPI)), im.size
    print(f"{ref}: copies {kinds} det {dets}; wrote {pdf} and {png} {im.size}")


if __name__ == "__main__":
    main()
