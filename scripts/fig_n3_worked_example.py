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

Expansion is OURS's `g2_asym_baselines.expand(a, R0)`. No coordinates are written to any file; the outputs are
rendered images. H atoms are omitted from the drawing (they are expanded like every other atom).
  (a) the cell projected along b (b into the page): the four copies coloured by operation; copies made by
      improper operations (det W_k = -1) are hatched and labelled "mirrored"; inversion centres and 2_1 axes
      are drawn where the operations put them. Paler atoms lie further from the viewer.
  (b) schematic: copy k has centroid c_k = W_k c0 + t_k and orientation C_k R_asym (eq 1), each operation in the CSD setting with
      det W_k. Each glyph is the copy's conformer up to a proper rotation: l itself (det +1) or its mirror
      image (det -1).

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
from matplotlib.patches import Circle, Ellipse, Polygon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = os.path.join(REPO, "paper", "submissions", "ACSOmega", "figures")
CIF_DIR = os.path.join(REPO, "data", "csd_mol", "cif")
MINUS = "−"
TSYM = "t"            # translation symbol: the manuscript's eq (expand) writes (W_k, t_k)
CHIRAL_MIN = 1.0      # A, selection rule 4

# Okabe-Ito; proper operations cool, improper warm
OPS = {  # kind -> (colour, name)
    "1": ("#0072B2", "identity"),
    "2": ("#009E73", "$2_1$ screw"),
    "-1": ("#D55E00", "inversion"),
    "m": ("#CC79A7", "$c$-glide"),
}
ORDER = ["1", "2", "-1", "m"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
    "hatch.linewidth": 0.6,
})


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


def lighten(col, f):
    c = np.array(matplotlib.colors.to_rgb(col))
    return tuple(c + (1 - c) * f)


def darken(col, f):
    return tuple(np.array(matplotlib.colors.to_rgb(col)) * (1 - f))


def draw_molecule(ax, P2, d, Z, bonds, col, z0, r, lw):
    """Stick-and-ball in 2D, depth ordered. d in [0, 1]: 1 = nearest the viewer (drawn on top, full colour).
    C filled in the copy colour, N open, O dark."""
    for i, j in bonds:
        dd = 0.5 * (d[i] + d[j])
        ax.plot([P2[i, 0], P2[j, 0]], [P2[i, 1], P2[j, 1]], color=lighten(col, 0.5 * (1 - dd)), lw=lw,
                solid_capstyle="round", zorder=z0 + dd)
    for i in np.argsort(d):
        c = lighten(col, 0.5 * (1 - d[i]))
        z = int(Z[i])
        if z == 7:
            fc, ec, rr = "white", c, 1.15 * r
        elif z == 8:
            fc, ec, rr = darken(c, 0.45), darken(c, 0.45), 1.15 * r
        else:
            fc, ec, rr = c, c, r
        ax.add_patch(Circle(P2[i], rr, facecolor=fc, edgecolor=ec, lw=0.7, zorder=z0 + d[i] + 0.01))


def halo(ax, P2, col, improper, z, pad_pt):
    from scipy.spatial import ConvexHull
    H = P2[ConvexHull(P2).vertices]
    ax.add_patch(Polygon(H, closed=True, facecolor=lighten(col, 0.84), edgecolor=lighten(col, 0.84), lw=pad_pt,
                         joinstyle="round", zorder=z))
    if improper:
        ax.add_patch(Polygon(H, closed=True, facecolor="none", edgecolor=lighten(col, 0.3), lw=0,
                             hatch="//////", zorder=z + 0.001))


def screw_symbol(ax, xy, s, z):
    """2_1 axis normal to the page (ITA: lens with two tails), size s in data units."""
    x, y = xy
    ax.add_patch(Ellipse((x, y), 1.6 * s, 0.75 * s, facecolor="black", edgecolor="black", lw=0, zorder=z))
    ax.plot([x + 0.8 * s, x + 0.8 * s], [y, y + 0.5 * s], color="black", lw=0.7, zorder=z)
    ax.plot([x - 0.8 * s, x - 0.8 * s], [y, y - 0.5 * s], color="black", lw=0.7, zorder=z)


def inv_symbol(ax, xy, s, z):
    ax.add_patch(Circle(xy, s, facecolor="white", edgecolor="black", lw=0.6, zorder=z))


def special_points(W, t, kind):
    """Fractional (x, z) of inversion centres / 2_1 axes (along b) inside the closed cell, from (W, t)."""
    pts = []
    for k in range(len(W)):
        if op_kind(W[k]) != kind:
            continue
        for n in np.ndindex(2, 2):
            f = np.array([t[k][0] / 2 + n[0] / 2, t[k][2] / 2 + n[1] / 2]) % 1.0
            for sh in np.ndindex(2, 2):
                g = f + np.array(sh)
                if (g <= 1 + 1e-9).all():
                    pts.append((round(float(g[0]), 6), round(float(g[1]), 6)))
    return sorted(set(pts))


# ------------------------------------------------------------------------------------------ figure
def place_labels(fig, ax, items, discs, segments):
    """Greedy label placement: for each (anchor centre, copy atoms, text, text kwargs), try positions on rings
    around the copy and keep the one whose text box overlaps the fewest atoms / symbols / cell edges / earlier
    labels (box measured with the renderer, in data units)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    boxes = []

    def seg_hits(bb, p, q, n=24):
        s = np.linspace(0, 1, n)[:, None]
        P = p[None] * (1 - s) + q[None] * s
        return bool(((P[:, 0] > bb[0]) & (P[:, 0] < bb[2]) & (P[:, 1] > bb[1]) & (P[:, 1] < bb[3])).any())

    cents = [it[0] for it in items]
    for c, P2, txt, kw in items:
        best = None
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            u = np.array([np.cos(ang), np.sin(ang)])
            ext = np.max((P2 - c) @ u)
            for gap in (0.7, 1.2, 1.8, 2.5):
                pos = c + u * (ext + gap)
                t = ax.text(pos[0], pos[1], txt, ha="center", va="center", **kw)
                e = t.get_window_extent(rend)
                t.remove()
                (bx0, by0), (bx1, by1) = inv.transform([[e.x0, e.y0], [e.x1, e.y1]])
                bb = (bx0 - 0.15, by0 - 0.1, bx1 + 0.15, by1 + 0.1)
                pen = 0.0
                for (dx, dy, r, w) in discs:
                    qx, qy = min(max(dx, bb[0]), bb[2]), min(max(dy, bb[1]), bb[3])
                    if (qx - dx) ** 2 + (qy - dy) ** 2 < r * r:
                        pen += w
                pen += sum(3.0 for p, q in segments if seg_hits(bb, p, q))
                pen += sum(50.0 for b in boxes if not (bb[2] < b[0] or bb[0] > b[2] or bb[3] < b[1] or bb[1] > b[3]))
                if bb[0] < x0 or bb[2] > x1 or bb[1] < y0 or bb[3] > y1:
                    pen += 100.0
                pen += 0.3 * gap
                # a label must sit nearer its own copy than any other copy
                own = np.linalg.norm(pos - c)
                if any(np.linalg.norm(pos - o) < own for o in cents if o is not c):
                    pen += 6.0
                if best is None or pen < best[0]:
                    best = (pen, pos, bb)
        boxes.append(best[2])
        ax.text(best[1][0], best[1][1], txt, ha="center", va="center", zorder=50,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.1"), **kw)


def main():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    from g2_asym_baselines import expand
    from n3_chirality import graph
    idx, a, train, mrmsd, n_cand = select()
    ref = a["refcode"]
    split = confirm_split(a, train)
    name, hall, cif_ops = cif_symmetry(ref)
    check_ops_against_cif(a, cif_ops)
    assert oracle(a), "expand(a, R0) does not reproduce the experimental structure"
    res, sep = orbit_check(a)

    L, cent, orient, local, Z, am, mm = expand(a, a["R0"])
    L = L.double().numpy()
    X = (cent.double().numpy() @ L)[:, None, :] + np.einsum("kij,kaj->kai", orient.double().numpy(),
                                                            local.double().numpy())   # (K, A, 3) whole molecules
    Zs = Z[0].numpy()
    heavy = Zs > 1
    remap = {int(i): n for n, i in enumerate(np.nonzero(heavy)[0])}
    loc = a["local"].double().numpy()
    bonds = [(remap[i], remap[j]) for i, j in graph(loc, Zs).edges() if heavy[i] and heavy[j]]
    Zh = Zs[heavy]
    W, t = a["W"].numpy(), a["t"].numpy()
    kinds = [op_kind(w) for w in W]
    dets = [int(round(np.linalg.det(w))) for w in W]

    # projection along b, b into the page: e1 along a (perp. to b), e2 = e1 x b; nearer the viewer = smaller x.b
    bh = L[1] / np.linalg.norm(L[1])
    e1 = L[0] - (L[0] @ bh) * bh
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, bh)
    proj = lambda P: np.stack([P @ e1, P @ e2], -1)
    near = lambda P: -(P @ bh)
    Xh = X[:, heavy]
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], float) @ L
    C2 = proj(corners)
    allP = proj(Xh.reshape(-1, 3))
    m = 2.3                                            # A of room around the drawing for the copy labels
    lo = np.minimum(allP.min(0), C2.min(0)) - np.array([m, 0.8])     # labels go to the sides or above
    hi = np.maximum(allP.max(0), C2.max(0)) + np.array([m, 2.0])
    lo[1] -= 1.3                                       # key row
    hi[1] += 0.7                                       # title row
    W_IN, B_IN, GAP = 3.3, 1.9, 0.14                   # figure width; height of panel (b); gap between panels
    a_h = 0.98 * W_IN * (hi[1] - lo[1]) / (hi[0] - lo[0])
    H_IN = a_h + B_IN + GAP
    fig = plt.figure(figsize=(W_IN, H_IN))
    axA = fig.add_axes([0.01, (B_IN + GAP) / H_IN, 0.98, a_h / H_IN])
    axB = fig.add_axes([0.0, 0.0, 1.0, B_IN / H_IN])

    # ---------------- (a) the cell along b
    ax = axA
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Polygon(C2, closed=True, fill=False, edgecolor="#444444", lw=0.7, zorder=1.5))
    allN = near(Xh.reshape(-1, 3))
    nmin, nmax = allN.min(), allN.max()
    R_AT = 0.24
    for zr, k in enumerate(np.argsort([near(Xh[k]).mean() for k in range(len(Xh))])):   # far copies first
        col, _ = OPS[kinds[k]]
        P2 = proj(Xh[k])
        halo(ax, P2, col, dets[k] < 0, 1.0 + 0.01 * zr, pad_pt=6)
        draw_molecule(ax, P2, (near(Xh[k]) - nmin) / (nmax - nmin), Zh, bonds, col, 2 + 2 * zr, r=R_AT, lw=1.1)
    sym_pts = []
    for x, z in special_points(W, t, "-1"):
        p = proj((np.array([x, 0.0, z]) @ L)[None])[0]
        inv_symbol(ax, p, 0.22, 40)
        sym_pts.append(p)
    for x, z in special_points(W, t, "2"):
        p = proj((np.array([x, 0.0, z]) @ L)[None])[0]
        screw_symbol(ax, p, 0.38, 40)
        sym_pts.append(p)
    ax.text(lo[0] + 0.1, hi[1] - 0.05, f"(a) {ref}, $P2_1/c$, viewed along $b$", fontsize=7.5, ha="left", va="top")
    # key row: axes, symmetry symbols, glide plane
    ky = lo[1] + 0.55
    x = lo[0] + 0.35
    arrow = dict(arrowstyle="-|>", lw=0.7, color="black", mutation_scale=5, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(x + 1.6, ky - 0.35), xytext=(x, ky - 0.35), arrowprops=arrow)
    cdir = proj(L[2][None])[0]
    cdir = cdir / np.linalg.norm(cdir)
    ax.annotate("", xy=(x + 1.6 * cdir[0], ky - 0.35 + 1.6 * cdir[1]), xytext=(x, ky - 0.35), arrowprops=arrow)
    ax.text(x + 2.0, ky - 0.35, "$a$", fontsize=7, va="center", ha="left")
    ax.text(x + 1.6 * cdir[0] + 0.45, ky - 0.35 + 1.6 * cdir[1], "$c$", fontsize=7, va="center", ha="left")
    span = hi[0] - lo[0]
    x = lo[0] + 0.20 * span
    inv_symbol(ax, (x, ky), 0.22, 5)
    ax.text(x + 0.45, ky, "inversion center", fontsize=6.0, va="center")
    x = lo[0] + 0.50 * span
    screw_symbol(ax, (x, ky), 0.38, 5)
    ax.text(x + 0.6, ky, "$2_1$ axis $\\parallel$ $b$", fontsize=6.0, va="center")
    gm = kinds.index("m")
    ax.text(hi[0] - 0.1, ky, f"glide plane at $y$ = {frac((t[gm][1] / 2) % 1.0)}", fontsize=6.0, va="center",
            ha="right")
    # copy labels, placed clear of atoms, symbols, cell edges and each other
    discs = [(p[0], p[1], R_AT + 0.25, 1.0) for p in allP] + [(p[0], p[1], 0.55, 2.0) for p in sym_pts]
    segs = [(C2[i], C2[(i + 1) % 4]) for i in range(4)]
    items = []
    for k in range(len(Xh)):
        col, nm = OPS[kinds[k]]
        P2 = proj(Xh[k])
        items.append((P2.mean(0), P2, nm + ("\nmirrored" if dets[k] < 0 else ""),
                      dict(color=darken(col, 0.15), fontsize=6.6, fontweight="bold", linespacing=0.95)))
    place_labels(fig, ax, items, discs, segs)

    # ---------------- (b) schematic
    ax = axB
    top = 10 * B_IN / W_IN
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.axis("off")
    ax.text(0.12, top - 0.05, "(b) exact expansion: one $R_{asym}$ gives every copy", fontsize=7.5, ha="left",
            va="top")
    ax.text(5.0, top - 0.62, "copy $k$: centroid $c_k = W_k c_0 + " + TSYM + "_k$,  orientation $C_k R_{asym}$",
            fontsize=7.3, ha="center", va="center")
    # conformer glyph in its principal frame (largest two axes in the page)
    cl = loc[heavy] - loc[heavy].mean(0)
    _, _, Vt = np.linalg.svd(cl, full_matrices=False)
    G = cl @ Vt.T
    G2 = G[:, :2] / np.max(np.abs(G[:, :2]))
    Gd = (G[:, 2] - G[:, 2].min()) / max(np.ptp(G[:, 2]), 1e-9)
    gsize = 0.95                     # half-width of a glyph, data units

    def glyph(cx, cy, col, mirrored):
        P = G2 * gsize
        if mirrored:     # reflection through the plane of the view direction and the vertical axis
            P = P * np.array([-1.0, 1.0])
        P = P + np.array([cx, cy])
        halo(ax, P, col, mirrored, 5, pad_pt=5)
        draw_molecule(ax, P, Gd, Zh, bonds, col, 10, r=0.075, lw=0.9)

    gh = gsize * np.max(np.abs(G2[:, 1]))          # glyph half-height
    gx, gy = 1.6, top - 1.05 - gh
    glyph(gx, gy, OPS["1"][0], False)
    ax.text(gx + 1.35, gy + 0.2, "asymmetric unit: conformer $l$ rotated by $R_{asym}$, at $c_0$", fontsize=6.8, ha="left", va="center")
    ax.text(gx + 1.35, gy - 0.22, "$R_{asym}$ predicted (3 numbers);  $l$, $c_0$ given", fontsize=6.0,
            ha="left", va="center", color="#444444")
    xs = [1.25, 3.75, 6.25, 8.75]
    yg = gh + 1.75
    ybus = 0.5 * ((gy - gh - 0.2) + (yg + gh + 0.2))
    ax.plot([gx, gx], [gy - gh - 0.2, ybus], color="#444444", lw=0.7)
    ax.plot([xs[0], xs[-1]], [ybus, ybus], color="#444444", lw=0.7)
    for x, kd in zip(xs, ORDER):
        k = kinds.index(kd)
        col, nm = OPS[kd]
        ax.annotate("", xy=(x, yg + gh + 0.2), xytext=(x, ybus),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#444444", mutation_scale=6, shrinkA=0,
                                    shrinkB=0))
        glyph(x, yg, col, dets[k] < 0)
        tc = darken(col, 0.15)
        ax.text(x, yg - gh - 0.40, nm, fontsize=6.6, ha="center", va="center", color=tc, fontweight="bold")
        ax.text(x, yg - gh - 0.76, op_string(W[k], t[k]), fontsize=6.0, ha="center", va="center")
        ax.text(x, yg - gh - 1.09, "det $W_k$ = " + ("+1" if dets[k] > 0 else MINUS + "1"), fontsize=6.0,
                ha="center", va="center")
        if dets[k] < 0:
            ax.text(x, yg - gh - 1.40, "mirrored", fontsize=6.0, ha="center", va="center", color=tc,
                    fontweight="bold")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig_n3_worked_example.{ext}")
        fig.savefig(p, dpi=300)
        print("wrote", os.path.relpath(p, REPO))
    print(f"crystal: {ref} (TRAIN index {idx}; {n_cand} TRAIN crystals pass rules 1-4); {split}")
    print(f"heavy atoms {int(heavy.sum())}, mirror RMSD {mrmsd:.2f} A, orbit residual {res:.1e}, "
          f"min separation {sep:.3f}, figure {W_IN} x {H_IN:.2f} in")
    print(f"CIF: H-M '{name}', Hall '{hall}', symops {cif_ops}")
    print("operations (CSD setting): " + "; ".join(f"{OPS[kd][1]} [{op_string(W[k], t[k])}] det {dets[k]:+d}"
                                                  for k, kd in enumerate(kinds)))


if __name__ == "__main__":
    main()
