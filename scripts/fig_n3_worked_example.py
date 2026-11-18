"""Figure (main text): worked P2_1/c example of exact space-group expansion (referee request).

Crystal selection (fixed rule, applied below; no hand-picking). Among TRAIN crystals
(`n3_common.load_set("train")`, the validated cache of the seed-0 perm[300:] of data/csd_mol/g2_asym.pt) that
  1. have K = 4 operations {identity, 2-fold screw along b, inversion, glide plane normal to b with
     translation (0, 0, 1/2)}: P2_1/c in the P 1 21/c 1 setting (unique axis b, c-glide);
  2. pass the protocol's general-position orbit check (P section 1.1 build step 2(i)): every W_k c0 + t_k
     lies within 1e-3 of a stored copy centroid and no two coincide;
  3. contain C, H, N, O only, with <= 20 heavy atoms;
  4. have b as the shortest cell axis, so the view along b shows the four copies side by side rather than
     stacked (legibility);
  5. have a clearly chiral conformer, so that the mirrored copies differ visibly from the asymmetric unit:
     the protocol's mirror RMSD (scripts/n3_chirality.mirror_rmsd, "refined") >= 1.0 A (CHIRAL needs > 0.3 A),
take the one with the fewest heavy atoms; ties go to the earliest in TRAIN order. Rules 4 and 5 are
presentation choices for a worked example; nothing statistical depends on this crystal.
Split membership is asserted: the refcode is in TRAIN and in none of VAL, DEV-TEST (seed-0 split) or the frozen
SEL / TEST-B lists (tasks/n3_sets), and its 6-letter family is not in SEL / TEST-B. The script also checks that
the operations equal the CIF's own symmetry loop (data/csd_mol/cif/<refcode>.cif; only the space-group name and
symop lines are read, never atom records) and that expand(a, R0) reproduces the experimental structure
(StructureMatcher defaults, the protocol's expand oracle).

Expansion is OURS's `g2_asym_baselines.expand(a, R0)`. v2 (2026-11-18) draws it in the TOC graphic's style
(scripts/n3_render.py): ray-traced PyMOL ball-and-stick, all atoms including H, carbon teal for copies made by
proper operations (det W_k = +1, rotated copies) and amber for improper ones (det W_k = -1, mirror images); N, O
and H in CPK colours; Arial throughout. Both panels use one camera (view along b, turned 18 deg about the
vertical and 14 deg about the horizontal so the cell's depth shows), so each tile in (a) is oriented exactly as
its copy in (b). The scene goes to the PyMOL process on stdin; no coordinates are written to any file.
  (a) one tile per operation (identity, 2_1 screw rotation, inversion, c-glide reflection), each the copy
      expand() builds, with the operation in the crystal's CSD setting and det W_k;
  (b) the four copies in the unit cell drawn with its origin moved by -c/4 so every copy's centroid lies inside
      (n3_render.CELL_SHIFT), each tagged with its operation symbol; a and c as projected by the camera.

    python scripts/fig_n3_worked_example.py
writes paper/submissions/ACSOmega/figures/fig_n3_worked_example.{pdf,png}
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import n3_render as NR  # noqa: E402  (shared look with the TOC graphic)

OUT = os.path.join(REPO, "paper", "submissions", "ACSOmega", "figures")
CIF_DIR = os.path.join(REPO, "data", "csd_mol", "cif")
MINUS = "−"
TSYM = "t"            # translation symbol: the manuscript's eq (expand) writes (W_k, t_k)
CHIRAL_MIN = 1.0      # A, selection rule 4

ORDER = ["1", "2", "-1", "m"]                    # identity, 2_1 screw, inversion, c-glide (selection rule 1)

plt.rcParams.update(NR.RC)


# ------------------------------------------------------------------------------------------ selection
def op_kind(W):
    W = np.rint(np.asarray(W)).astype(int)
    if (W == np.eye(3, dtype=int)).all():
        return "1"
    if (W == -np.eye(3, dtype=int)).all():
        return "-1"
    return "2" if round(np.linalg.det(W)) == 1 else "m"


def is_p21c_b(a):
    """P2_1/c, unique axis b, glide translation (0, 0, 1/2) (the c-glide)."""
    if int(a["sg"]) != 14 or int(a["K"]) != 4:
        return False
    W, t = a["W"].numpy(), a["t"].numpy()
    kinds = [op_kind(w) for w in W]
    if sorted(kinds) != sorted(ORDER):
        return False
    k2, km = kinds.index("2"), kinds.index("m")
    if not (np.allclose(np.rint(W[k2]), np.diag([-1, 1, -1])) and np.allclose(np.rint(W[km]), np.diag([1, -1, 1]))):
        return False
    g = t[km].copy()
    g[1] = 0.0
    g = g - np.floor(g + 1e-6)
    return bool(np.allclose(g, [0.0, 0.0, 0.5], atol=1e-6))


def orbit_check(a):
    """Protocol P section 1.1 build step 2(i): (max residual to a stored centroid, min orbit separation)."""
    o = a["orig"]
    cents = o["centroid"][o["mol_mask"]].double().numpy()
    orb = np.einsum("kij,j->ki", a["W"].double().numpy(), a["c0"].double().numpy()) + a["t"].double().numpy()

    def md(u, v):
        d = u - v
        d -= np.round(d)
        return float(np.abs(d).max())

    res = max(min(md(p, q) for q in cents) for p in orb)
    sep = min(md(orb[i], orb[j]) for i in range(len(orb)) for j in range(i + 1, len(orb)))
    return res, sep


def mirror(a):
    from n3_chirality import mirror_rmsd
    return mirror_rmsd(a["local"].double().numpy(), a["Z"].numpy())["refined"]


def select():
    import n3_common as C
    train = C.load_set("train")
    cands = []
    for idx, a in enumerate(train):
        Z = a["Z"].numpy()
        heavy = int((Z > 1).sum())
        if not is_p21c_b(a) or heavy > 20 or not set(Z.tolist()) <= {1, 6, 7, 8}:
            continue
        res, sep = orbit_check(a)
        if res > 1e-3 or sep <= 1e-3:
            continue
        if int(np.argmin(np.linalg.norm(a["L"].double().numpy(), axis=1))) != 1:
            continue
        cands.append((heavy, idx, a))
    for heavy, idx, a in sorted(cands, key=lambda c: (c[0], c[1])):
        m = mirror(a)
        if m >= CHIRAL_MIN:
            return idx, a, train, m, len(cands)
    raise SystemExit("no TRAIN crystal satisfies the selection rule")


def confirm_split(a, train):
    import n3_common as C
    ref = a["refcode"]
    assert sum(x["refcode"] == ref for x in train) == 1
    for s in ("val", "devtest"):
        assert ref not in {x["refcode"] for x in C.load_set(s)}, f"{ref} in {s}"
    for s in ("sel", "testB"):
        refs = C.frozen_refcodes(s)
        assert ref not in set(refs), f"{ref} in {s}"
        assert ref[:6] not in {r[:6] for r in refs}, f"{ref[:6]} family in {s}"
    return "in TRAIN (seed-0 perm[300:] of g2_asym.pt); not in VAL, DEV-TEST, SEL or TEST-B"


def cif_symmetry(ref):
    """Space-group name, Hall symbol and symop strings from the CIF header; atom records are never read."""
    name = hall = None
    ops, in_loop = [], False
    with open(os.path.join(CIF_DIR, ref + ".cif"), encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("_symmetry_space_group_name_H-M") or s.startswith("_space_group_name_H-M_alt"):
                name = s.split(None, 1)[1].strip("'\"")
            elif s.startswith("_space_group_name_Hall") or s.startswith("_symmetry_space_group_name_Hall"):
                hall = s.split(None, 1)[1].strip("'\"")
            elif s in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
                in_loop = True
            elif in_loop:
                if not s or s.startswith("_") or s.startswith("loop_"):
                    if ops:
                        break
                    continue
                ops.append(re.sub(r"^\d+\s+", "", s).strip("'\""))
    return name, hall, ops


def check_ops_against_cif(a, ops):
    from pymatgen.core.operations import SymmOp
    W, t = a["W"].numpy(), a["t"].numpy()
    cif = [SymmOp.from_xyz_str(o) for o in ops]
    assert len(cif) == len(W)
    for k in range(len(W)):
        hit = [c for c in cif if np.allclose(c.rotation_matrix, W[k])
               and np.allclose((c.translation_vector - t[k] + 0.5) % 1.0 - 0.5, 0, atol=1e-6)]
        assert len(hit) == 1, f"operation {k} not in the CIF symmetry loop"


def oracle(a):
    import n3_common as C
    from g2_asym_baselines import expand, to_structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    return bool(StructureMatcher().fit(C.truth_structure(a), to_structure(*expand(a, a["R0"]))))


# ------------------------------------------------------------------------------------------ drawing helpers
def frac(x):
    f = Fraction(float(x)).limit_denominator(12)
    return {Fraction(1, 2): "½", Fraction(1, 4): "¼", Fraction(3, 4): "¾"}.get(f, str(f))


def op_string(W, t):
    """(W, t) -> 'x, y+1/2, -z+1/2' style string in the crystal's own setting."""
    out = []
    for i in range(3):
        terms = ""
        for j, v in enumerate("xyz"):
            c = int(round(W[i][j]))
            if c:
                terms += ("+" if c > 0 and terms else "") + (MINUS if c < 0 else "") + \
                         ("" if abs(c) == 1 else str(abs(c))) + v
        tt = float(t[i]) % 1.0
        if 1e-6 < tt < 1 - 1e-6:
            terms += "+" + frac(tt)
        out.append(terms)
    return ", ".join(out)


# ------------------------------------------------------------------------------------------ figure (v2)
TILE_ORDER = ["1", "2", "-1", "m"]              # identity, 2_1 screw, inversion, c-glide (row-major 2 x 2)
TAG = {"1": "1", "2": "$2_1$", "-1": "$\\bar{1}$", "m": "$\\mathit{c}$"}   # ITA: glide letter italic
TEXT_COL = {+1: "#1E7268", -1: "#A8691A"}       # darker teal / amber for text on white
W_IN, DPI = 7.0, 600                             # ACS double-column width


def tag_position(c2, others, centroids, lo, hi, need=1.1):
    """Label anchor for a copy whose projected centroid is c2: the nearest point (rings of growing radius, 48
    directions) that is at least `need` A from every projected atom, nearer this copy's centroid than any other
    copy's, and inside the window [lo, hi]; among equally near points, the one with the most clearance."""
    for r in np.arange(1.5, 9.0, 0.35):
        best = None
        for ang in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            p = c2 + r * np.array([np.cos(ang), np.sin(ang)])
            if not ((p > lo + 0.8).all() and (p < hi - 0.8).all()):
                continue
            if np.min(np.linalg.norm(centroids - p, axis=1)) < np.linalg.norm(c2 - p) - 1e-9:
                continue
            clear = np.min(np.linalg.norm(others - p, axis=1))
            if clear >= need and (best is None or clear > best[0]):
                best = (clear, p)
        if best is not None:
            return best[1]
    raise RuntimeError("no free spot for a copy tag")


def italic_xyz(s):
    """'-x, y+1/2, -z+1/2' with the coordinate letters in italics (ITA convention)."""
    return re.sub(r"([xyz])", lambda m: "$\\mathit{" + m.group(1) + "}$", s)


def main():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    import matplotlib.image as mpimg
    import tempfile
    from matplotlib.patches import Circle, FancyArrowPatch

    cr = NR.crystal()
    ref, L, X, Zs, bonds, W, t, kinds, dets, meta = (cr[k] for k in ("ref", "L", "X", "Zs", "bonds", "W", "t",
                                                                       "kinds", "dets", "meta"))
    B = NR.frame(L)
    cam = lambda P: np.asarray(P) @ B.T          # noqa: E731
    Pc = [cam(X[k]) for k in range(len(X))]       # every copy in camera coordinates (A)

    # ---------------- (a) the four copies, one tile each, same camera as (b)
    # each tile centres its copy's bounding box (in the page plane) and keeps its depth about the centroid
    cen = [np.r_[0.5 * (P[:, :2].min(0) + P[:, :2].max(0)), P[:, 2].mean()] for P in Pc]
    ext = np.max([np.ptp(P[:, :2], axis=0) for P in Pc], axis=0)
    tw, th = ext[0] + 2.0, ext[1] + 4.6            # tile size: molecule + room for the labels above and below
    atoms, bnd, tiles = [], [], {}
    for n, kd in enumerate(TILE_ORDER):
        k = kinds.index(kd)
        gx, gy = (n % 2 + 0.5) * tw, (1 - n // 2 + 0.5) * th - 0.25
        P = Pc[k] - cen[k] + np.array([gx, gy, 0.0])
        base = len(atoms)
        atoms += NR.molecule_atoms(P, Zs, NR.PROPER if dets[k] > 0 else NR.IMPROPER)
        bnd += [[base + i, base + j] for i, j in bonds]
        tiles[kd] = (k, gx, gy, P)
    winA = [0.0, 2 * tw, 0.0, 2 * th]
    AX_IN = 3.15
    AY_IN = AX_IN * (2 * th) / (2 * tw)
    sceneA = {"atoms": atoms, "bonds": bnd, "stick_r": NR.STICK, "window": winA, "res_x": int(AX_IN * DPI),
              "res_y": int(AY_IN * DPI), "outline": True, "gain": 0.09, "fog": False}

    # ---------------- (b) the cell
    atomsB, bndB = [], []
    for k in range(len(X)):
        base = len(atomsB)
        atomsB += NR.molecule_atoms(Pc[k], Zs, NR.PROPER if dets[k] > 0 else NR.IMPROPER)
        bndB += [[base + i, base + j] for i, j in bonds]
    cell_cyl, Cc = NR.cell_edges(L, cam)
    allB = np.vstack([np.array([a_["p"] for a_ in atomsB])[:, :2], Cc[:, :2]])
    BX_IN = W_IN - AX_IN - 0.35
    winB = NR.window_for(allB, 1.0, BX_IN / AY_IN)
    sceneB = {"atoms": atomsB, "bonds": bndB, "stick_r": NR.STICK, "cyl": cell_cyl, "window": winB,
              "res_x": int(BX_IN * DPI), "res_y": int(AY_IN * DPI), "outline": True, "gain": 0.09,
              "fog_start": 0.55}

    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "a.png"), os.path.join(td, "b.png")
        NR.run_pymol(sceneA, pa)
        NR.run_pymol(sceneB, pb)
        imA, imB = mpimg.imread(pa), mpimg.imread(pb)

    # ---------------- compose
    TOP, BOT = 0.30, 0.42                          # title band and legend band (in)
    H_IN = AY_IN + TOP + BOT
    fig = plt.figure(figsize=(W_IN, H_IN))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W_IN)
    bg.set_ylim(0, H_IN)
    bg.axis("off")
    axA = fig.add_axes([0.0, BOT / H_IN, AX_IN / W_IN, AY_IN / H_IN])
    axA.imshow(imA, extent=winA, interpolation="none")
    axA.set_xlim(winA[0], winA[1])
    axA.set_ylim(winA[2], winA[3])
    axA.axis("off")
    xB = W_IN - BX_IN
    axB = fig.add_axes([xB / W_IN, BOT / H_IN, BX_IN / W_IN, AY_IN / H_IN])
    axB.imshow(imB, extent=winB, interpolation="none")
    axB.set_xlim(winB[0], winB[1])
    axB.set_ylim(winB[2], winB[3])
    axB.axis("off")

    FS, FT = 7.5, 8.5
    bg.text(0.02, H_IN - 0.06, "(a)", fontsize=FT, fontweight="bold", va="top", ha="left", color=NR.INK)
    bg.text(0.32, H_IN - 0.06, "the four copies made by exact expansion", fontsize=FT, va="top", ha="left",
            color=NR.INK)
    bg.text(xB - 0.05, H_IN - 0.06, "(b)", fontsize=FT, fontweight="bold", va="top", ha="left", color=NR.INK)
    bg.text(xB + 0.25, H_IN - 0.06, "the same copies in the unit cell", fontsize=FT,
            va="top", ha="left", color=NR.INK)

    names = {"1": "identity", "2": "screw rotation", "-1": "inversion", "m": "glide reflection"}
    for kd, (k, gx, gy, P) in tiles.items():
        d = dets[k]
        col = TEXT_COL[d]
        ytop = gy + ext[1] / 2 + 0.55
        axA.text(gx, ytop, TAG[kd] + "   " + names[kd], fontsize=FS, ha="center", va="bottom", color=col,
                 fontweight="bold")
        ybot = gy - ext[1] / 2 - 0.45
        axA.text(gx, ybot, italic_xyz(op_string(W[k], t[k])), fontsize=FS, ha="center", va="top",
                 color=NR.INK)
        axA.text(gx, ybot - 1.15, "det $\\mathit{W}$ = " + ("+1" if d > 0 else MINUS + "1") +
                 ("" if d > 0 else ",  mirror image"), fontsize=FS, ha="center", va="top", color=col)

    # tags on the copies in (b)
    lo, hi = np.array([winB[0], winB[2]]), np.array([winB[1], winB[3]])
    allP = np.vstack([P[:, :2] for P in Pc])
    cents = np.array([P[:, :2].mean(0) for P in Pc])
    for k, P in enumerate(Pc):
        c2 = P[:, :2].mean(0)
        p = tag_position(c2, allP, cents, lo, hi)
        axB.text(p[0], p[1], TAG[kinds[k]], fontsize=FT, ha="center", va="center", color=TEXT_COL[dets[k]],
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none",
                                              alpha=0.85))
    # axis key (a and c as projected by the same camera; b points away from the viewer)
    origin = np.array([winB[0] + 1.3, winB[2] + 1.2])
    for vec, lab in ((L[0], "a"), (L[2], "c")):
        v2 = cam(vec)[:2]
        v2 = 2.6 * v2 / np.linalg.norm(v2)
        axB.add_patch(FancyArrowPatch(tuple(origin), tuple(origin + v2), arrowstyle="-|>", mutation_scale=7,
                                      lw=0.9, color=NR.INK, shrinkA=0, shrinkB=0))
        axB.text(*(origin + v2 * 1.22), "$\\mathit{" + lab + "}$", fontsize=FS, ha="center", va="center",
                 color=NR.INK)

    # legend
    y = BOT / 2
    x = W_IN / 2 - 2.05
    for col, txt in ((NR.PROPER, "proper operation (det $\\mathit{W}$ = +1): rotated copy"),
                     (NR.IMPROPER, "improper operation (det $\\mathit{W}$ = " + MINUS + "1): mirror image")):
        bg.add_patch(Circle((x, y), 0.05, facecolor=col, edgecolor=NR.INK, lw=0.5))
        tx = bg.text(x + 0.10, y, txt, fontsize=FS, ha="left", va="center", color=NR.INK)
        fig.canvas.draw()
        x = bg.transData.inverted().transform(tx.get_window_extent(fig.canvas.get_renderer()))[1, 0] + 0.35

    os.makedirs(OUT, exist_ok=True)
    pdf, png = (os.path.join(OUT, f"fig_n3_worked_example.{e}") for e in ("pdf", "png"))
    fig.savefig(pdf, dpi=DPI)
    fig.savefig(png, dpi=DPI, facecolor="white")
    plt.close(fig)
    from PIL import Image
    Image.open(png).convert("RGB").save(png)
    print("wrote", os.path.relpath(pdf, REPO), "and", os.path.relpath(png, REPO), f"({W_IN} x {H_IN:.2f} in)")
    print(f"crystal: {ref} (TRAIN index {meta['idx']}; {meta['n_cand']} TRAIN crystals pass rules 1-4); "
          f"{meta['split']}")
    print(f"heavy atoms {int((Zs > 1).sum())}, mirror RMSD {meta['mirror_rmsd']:.2f} A, orbit residual "
          f"{meta['orbit_residual']:.1e}, min separation {meta['orbit_sep']:.3f}")
    print(f"CIF: H-M '{meta['hm']}', Hall '{meta['hall']}', symops {meta['cif_ops']}")
    print("operations (CSD setting): " + "; ".join(f"{names[kd]} [{op_string(W[k], t[k])}] det {dets[k]:+d}"
                                                  for k, kd in enumerate(kinds)))


if __name__ == "__main__":
    main()
