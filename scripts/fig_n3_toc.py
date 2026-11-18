"""Table-of-contents graphic for the ACS Omega resubmission (ACS TOC spec: 3.25 x 1.75 in, sans-serif, white).

Left to right:
  1. given: the LAJKOX cell viewed along b (the Figure 1 crystal, chosen by Figure 1's fixed selection rule), the
     inversion centres and 2_1 axes the operations put there, and the conformer at c0 NOT yet rotated: drawn in its
     principal-axis body frame (as the asymmetric-unit glyph of Figure 1(b)), centred where the identity copy sits.
     Its in-page angle has no physical meaning; it is fixed by a rule (multiples of 5 deg, the angle with the most
     clearance from the symmetry symbols that stays on the canvas; ties to the smallest |angle|);
  2. learned: one rotation R_asym (3 numbers), drawn as a rotation arrow on the expansion arrow;
  3. exact expansion: the four P2_1/c copies in their true orientations (expand(a, R0), as Figure 1(a)), the two
     improper (det W_k = -1) copies hatched;
  4. result strip: picked@1 on TEST-B, read from results/n3/n3_results.json -> sets.testB.report.table1:
     this model = OURS picked1_mean_pct; retrained learned baselines = max over MCF-R, MCF-R+G, iv-P of
     picked1_mean_pct, shown as an upper bound rounded UP to one decimal; physics search = iii-s picked1_mean_pct.
     Bars are drawn to scale on a 0-100% track.

Molecule rendering, selection, split check, CIF symmetry check and expand oracle are imported from
scripts/fig_n3_worked_example.py, so the drawing matches Figure 1. check_layout() asserts that no text box overlaps
another text, an atom, a symmetry symbol, a bar or the arrows, and that all text is on the canvas. The PNG is opaque
RGB at 600 dpi (1950 x 1050 px); the PDF is vector at exactly 3.25 x 1.75 in. No coordinates are written to any
file; the outputs are rendered images only.

    python scripts/fig_n3_toc.py
writes paper/submissions/ACSOmega/figures/fig_n3_toc.{pdf,png}
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.path import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import fig_n3_worked_example as F  # noqa: E402  (sets the shared rcParams: Arial, mathtext regular, fonttype 42)

OUT = F.OUT
RESULTS = os.path.join(REPO, "results", "n3", "n3_results.json")
W_IN, H_IN = 3.25, 1.75          # ACS TOC graphic, exact
DPI = 600
BASELINES = ("MCF-R", "MCF-R+G", "iv-P")
GREY = "#444444"


def results():
    """(label, value_pct, shown_text, colour, json key) for the three bars; values only from the JSON."""
    with open(RESULTS, encoding="utf-8") as f:
        t = json.load(f)["sets"]["testB"]["report"]
    n, t1 = t["n"], t["table1"]
    ours = t1["OURS"]["picked1_mean_pct"]
    base = {k: t1[k]["picked1_mean_pct"] for k in BASELINES}
    bmax = max(base.values())
    bound = math.ceil(round(bmax * 10, 9)) / 10          # upper bound at one decimal: rounded up, never down
    phys = t1["iii-s"]["picked1_mean_pct"]
    rows = [
        ("this model", ours, f"{ours:.1f}%", "#0072B2", "OURS.picked1_mean_pct"),
        ("retrained learned baselines", bmax, f"≤{bound:.1f}%", "#8C8C8C",
         "max(" + ", ".join(f"{k}.picked1_mean_pct" for k in BASELINES) + ")"),
        ("physics search", phys, f"{phys:.1f}%", "#4D4D4D", "iii-s.picked1_mean_pct"),
    ]
    return n, rows, base


def rotation_glyph(ax, c, r, col, lw):
    """Circular arrow (about 300 degrees, counter-clockwise) centred at c, radius r in data units."""
    th = np.radians(np.linspace(115, 405, 80))
    P = np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)]
    path = Path(P, [Path.MOVETO] + [Path.LINETO] * (len(P) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=7.5, lw=lw, color=col, zorder=20,
                                 shrinkA=0, shrinkB=0, capstyle="round", joinstyle="round"))


def check_layout(fig, ax, discs, boxes, edge=0.02):
    """Assert every text box is inside the canvas (edge in inches), overlaps no other text, no atom / symbol disc
    (x, y, r) and no bar / arrow box (x0, y0, x1, y1). All in inches (the axes spans the canvas in inch units)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    tb = []
    for t in ax.texts:
        e = t.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([[e.x0, e.y0], [e.x1, e.y1]])
        tb.append((" ".join(t.get_text().split()), (x0, y0, x1, y1)))
    hit = lambda b, c: not (b[2] <= c[0] or c[2] <= b[0] or b[3] <= c[1] or c[3] <= b[1])
    for i, (s1, b) in enumerate(tb):
        assert b[0] >= edge and b[1] >= edge and b[2] <= W_IN - edge and b[3] <= H_IN - edge, f"clipped: {s1} {b}"
        for s2, c in tb[i + 1:]:
            assert not hit(b, c), f"text overlap: {s1!r} / {s2!r}"
        for c in boxes:
            assert not hit(b, c), f"text {s1!r} overlaps box {c}"
        for (dx, dy, r) in discs:
            qx, qy = min(max(dx, b[0]), b[2]), min(max(dy, b[1]), b[3])
            assert (qx - dx) ** 2 + (qy - dy) ** 2 >= r * r, f"text {s1!r} overlaps a disc at ({dx:.2f}, {dy:.2f})"
    return min(min(b[0], b[1], W_IN - b[2], H_IN - b[3]) for _, b in tb)


def main():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    from g2_asym_baselines import expand
    from n3_chirality import graph

    # ---------------- crystal: Figure 1's selection, split, CIF symmetry and oracle checks, unchanged
    idx, a, train, mrmsd, n_cand = F.select()
    ref = a["refcode"]
    split = F.confirm_split(a, train)
    _, _, cif_ops = F.cif_symmetry(ref)
    F.check_ops_against_cif(a, cif_ops)
    assert F.oracle(a), "expand(a, R0) does not reproduce the experimental structure"

    L, cent, orient, local, Z, am, mm = expand(a, a["R0"])
    L = L.double().numpy()
    X = (cent.double().numpy() @ L)[:, None, :] + np.einsum("kij,kaj->kai", orient.double().numpy(),
                                                            local.double().numpy())
    Zs = Z[0].numpy()
    heavy = Zs > 1
    remap = {int(i): n for n, i in enumerate(np.nonzero(heavy)[0])}
    bonds = [(remap[i], remap[j]) for i, j in graph(a["local"].double().numpy(), Zs).edges() if heavy[i] and heavy[j]]
    Zh = Zs[heavy]
    W, t = a["W"].numpy(), a["t"].numpy()
    kinds = [F.op_kind(w) for w in W]
    dets = [int(round(np.linalg.det(w))) for w in W]
    k_id = kinds.index("1")

    # projection along b exactly as Figure 1(a)
    bh = L[1] / np.linalg.norm(L[1])
    e1 = L[0] - (L[0] @ bh) * bh
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, bh)
    proj = lambda P: np.stack([P @ e1, P @ e2], -1)
    near = lambda P: -(P @ bh)
    Xh = X[:, heavy]
    C2 = proj(np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], float) @ L)
    allP = proj(Xh.reshape(-1, 3))
    allN = near(Xh.reshape(-1, 3))
    nmin, nmax = allN.min(), allN.max()

    # ---------------- figure in inch units (one axes over the whole canvas)
    fig = plt.figure(figsize=(W_IN, H_IN))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_IN)
    ax.set_ylim(0, H_IN)
    ax.axis("off")

    # scale: the expanded drawing (cell + all copies) fills a panel box; panels 1 and 3 share box, scale and
    # placement, so the two cells sit at the same height and size
    lo3 = np.minimum(allP.min(0), C2.min(0))
    hi3 = np.maximum(allP.max(0), C2.max(0))
    BOX_W, BOX_H, MARGIN = 1.20, 0.98, 0.06                             # MARGIN: atom centres to canvas edge
    s = min(BOX_W / (hi3[0] - lo3[0]), BOX_H / (hi3[1] - lo3[1]))       # in per A
    Y_STRIP = 0.44                                                       # top of the result strip
    Y_CAP = Y_STRIP + 0.13                                               # caption block centre (2 lines)
    y_mid = H_IN - MARGIN - 0.5 * s * (hi3[1] - lo3[1])                  # drawing centre height
    x1c, x3c = MARGIN + 0.5 * BOX_W, W_IN - MARGIN - 0.5 * BOX_W

    def place(P, centre, lo, hi):
        return np.asarray(centre) + s * (P - 0.5 * (lo + hi))

    R_AT, LW, PAD = 0.30 * s, 0.55, 3.0
    discs, boxes = [], []                                              # obstacles for check_layout

    def cell(centre, lo, hi):
        ax.add_patch(Polygon(place(C2, centre, lo, hi), closed=True, fill=False, edgecolor=GREY, lw=0.5, zorder=1.5))

    def copy(k, centre, lo, hi, zr):
        col, _ = F.OPS[kinds[k]]
        P2 = place(proj(Xh[k]), centre, lo, hi)
        discs.extend((x, y, R_AT + PAD / 144.0) for x, y in P2)
        F.halo(ax, P2, col, dets[k] < 0, 1.0 + 0.01 * zr, pad_pt=PAD)
        F.draw_molecule(ax, P2, (near(Xh[k]) - nmin) / (nmax - nmin), Zh, bonds, col, 2 + 2 * zr, r=R_AT, lw=LW)

    # (1) given: the cell, the conformer at c0 NOT yet rotated (principal-axis body frame, as the asymmetric-unit
    # glyph of Figure 1(b); its orientation is what is learned), and the symmetry elements of the operations
    lo1, hi1 = lo3, hi3
    c1 = (x1c, y_mid)
    cell(c1, lo1, hi1)
    sym = {kd: [place(proj((np.array([x, 0.0, z]) @ L)[None]), c1, lo1, hi1)[0] for x, z in F.special_points(W, t, kd)]
           for kd in ("-1", "2")}
    lh = a["local"].double().numpy()[heavy]
    cl = lh - lh.mean(0)
    Vt = np.linalg.svd(cl, full_matrices=False)[2]
    G = cl @ Vt.T
    centre_id = place(proj(Xh[k_id]).mean(0)[None], c1, lo1, hi1)[0]  # where the identity copy sits
    halo_in = PAD / 72.0
    best = None
    for deg in sorted(range(-90, 91, 5), key=abs):     # in-page angle of the body frame: a presentation choice,
        th = np.radians(deg)                           # the most clearance from the symmetry symbols on canvas
        Pg = centre_id + s * G[:, :2] @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        if Pg[:, 1].max() + R_AT + halo_in > H_IN - 0.01 or Pg[:, 0].min() - R_AT - halo_in < 0.01:
            continue
        clear = min(np.linalg.norm(Pg - q, axis=1).min() for q in sym["-1"] + sym["2"])
        if best is None or clear > best[0] + 1e-9:
            best = (clear, deg, Pg)
    clear1, deg1, Pg = best
    col_id = F.OPS["1"][0]
    discs.extend((x, y, R_AT + PAD / 144.0) for x, y in Pg)
    discs.extend((x, y, 0.03) for x, y in sym["-1"] + sym["2"])
    F.halo(ax, Pg, col_id, False, 1.0, pad_pt=PAD)
    F.draw_molecule(ax, Pg, (G[:, 2] - G[:, 2].min()) / np.ptp(G[:, 2]), Zh, bonds, col_id, 2, r=R_AT, lw=LW)
    for p in sym["-1"]:
        F.inv_symbol(ax, p, 0.016, 40)
    for p in sym["2"]:
        F.screw_symbol(ax, p, 0.026, 40)

    # (3) exact expansion: all copies, far copies first, improper copies hatched
    c3 = (x3c, y_mid)
    cell(c3, lo3, hi3)
    for zr, k in enumerate(np.argsort([near(Xh[k]).mean() for k in range(len(Xh))])):
        copy(k, c3, lo3, hi3, zr)

    # (2) learned: R_asym, a rotation arrow riding on the expansion arrow
    xa0 = x1c + 0.5 * s * (hi1[0] - lo1[0]) + 0.05
    xa1 = x3c - 0.5 * s * (hi3[0] - lo3[0]) - 0.05
    xm = 0.5 * (xa0 + xa1)
    ya = place(C2, c3, lo3, hi3)[:, 1].mean()                           # arrow at the cells' mid-height
    ax.add_patch(FancyArrowPatch((xa0, ya), (xa1, ya), arrowstyle="-|>", mutation_scale=8, lw=1.0,
                                 color=GREY, shrinkA=0, shrinkB=0, zorder=5))
    rotation_glyph(ax, (xm, ya + 0.16), 0.09, "black", 0.9)
    boxes += [(xa0, ya - 0.03, xa1, ya + 0.03), (xm - 0.11, ya + 0.05, xm + 0.11, ya + 0.27)]

    # captions
    cap = dict(fontsize=7.0, ha="center", va="center", color="black", linespacing=1.05)
    ax.text(x1c, Y_CAP, "lattice, operations,\n$c_0$ given", **cap)
    ax.text(xm, ya - 0.15, "learned: $R_{asym}$\n(3 numbers)", **cap)
    ax.text(x3c, Y_CAP, "exact expansion:\n4 copies, 2 mirrored", **cap)

    # ---------------- (4) result strip
    n, rows, base = results()
    x_lab, x_bar0, x_bar1, x_val = 1.30, 1.36, 2.84, 2.89
    row_h, bar_h = 0.105, 0.054
    y0 = Y_STRIP - 0.155
    ax.text(x_bar0, Y_STRIP - 0.045, f"structures reproduced ({n} test crystals)", fontsize=6.5, ha="left",
            va="center", color=GREY)
    for i, (lab, v, txt, col, _) in enumerate(rows):
        y = y0 - i * row_h
        ax.text(x_lab, y, lab, fontsize=7.0, ha="right", va="center")
        boxes.append((x_bar0, y - bar_h / 2, x_bar1, y + bar_h / 2))
        ax.add_patch(Rectangle((x_bar0, y - bar_h / 2), x_bar1 - x_bar0, bar_h, facecolor="#EEEEEE",
                               edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((x_bar0, y - bar_h / 2), (x_bar1 - x_bar0) * v / 100.0, bar_h, facecolor=col,
                               edgecolor="none", zorder=2))
        ax.text(x_val, y, txt, fontsize=7.0, ha="left", va="center", color=col if i == 0 else "black",
                fontweight="bold" if i == 0 else "normal")

    text_margin = check_layout(fig, ax, discs, boxes)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig_n3_toc.{ext}")
        fig.savefig(p, dpi=DPI, facecolor="white")
        if ext == "png":                                   # opaque RGB, no alpha channel
            from PIL import Image
            im = Image.open(p)
            assert im.getchannel("A").getextrema() == (255, 255)
            im.convert("RGB").save(p, dpi=(DPI, DPI))
        print("wrote", os.path.relpath(p, REPO))
    print(f"layout: no text overlaps or clipping; smallest text-to-edge margin {text_margin:.3f} in")
    print(f"crystal: {ref} (TRAIN index {idx}); {split}; oracle ok; figure {W_IN} x {H_IN} in at {DPI} dpi")
    print(f"panel 1 body-frame in-page angle {deg1} deg, clearance to symmetry symbols {clear1:.3f} in")
    print(f"TEST-B n = {n}; table1 picked1_mean_pct: " + ", ".join(f"{k} {v}" for k, v in base.items()))
    for lab, v, txt, _, key in rows:
        print(f"  {lab}: {key} = {v} -> '{txt}'")


if __name__ == "__main__":
    main()
