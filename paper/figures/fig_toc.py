"""
Purpose-built ACS Table-of-Contents (TOC) graphic for the JCIM manuscript
"Fully Symmetry-Conditioned Rigid-Body Flow Matching for Molecular-Crystal
Structure Prediction".

One conceptual idea, left -> right:
  (A) a rigid-body molecular crystal (unit cell, two symmetry-related molecules)
  (B) symmetry conditioning: orientation on the space-group coset,
      lattice on the crystal-family mask  ->  fully symmetry-conditioned flow
  (C) held-out exact match rises 0% -> 6.9% (parity with the symmetry-free
      MolCrystalFlow), 10.7% with orientation TTA.

ACS size spec: 3.25 in x 1.75 in. Output vector PDF (+ PNG proof).
Palette: Okabe-Ito (matches fig2/fig3).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Circle, FancyBboxPatch

# Okabe-Ito
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
VERM   = "#D55E00"
GREY   = "#555555"
LGREY  = "#BBBBBB"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.6,
    "mathtext.fontset": "dejavusans",
})

fig = plt.figure(figsize=(3.25, 1.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def molecule(cx, cy, angle_deg, color, s=1.0):
    """A small rigid 'molecule' = a triatomic glyph, drawn at an orientation."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    # local atom offsets (a bent triatomic)
    local = np.array([[0, 0], [4.6, 1.8], [-1.8, 4.6]]) * s
    pts = (local @ R.T) + [cx, cy]
    # bonds
    for i in (1, 2):
        ax.plot([pts[0, 0], pts[i, 0]], [pts[0, 1], pts[i, 1]],
                color=color, lw=1.1, zorder=3, solid_capstyle="round")
    ax.add_patch(Circle(pts[0], 2.3 * s, fc=color, ec="white", lw=0.5, zorder=4))
    ax.add_patch(Circle(pts[1], 1.6 * s, fc=color, ec="white", lw=0.5, zorder=4))
    ax.add_patch(Circle(pts[2], 1.6 * s, fc=color, ec="white", lw=0.5, zorder=4))


# ---------------- (A) unit cell with two symmetry-related molecules ----------
# monoclinic-ish cell (sheared parallelogram)
ox, oy, w, h, shear = 6, 26, 26, 34, 7
cell = np.array([[ox, oy], [ox + w, oy], [ox + w + shear, oy + h], [ox + shear, oy + h]])
ax.add_patch(Polygon(cell, closed=True, fill=True, fc="#F2F5F8",
                     ec=GREY, lw=1.0, zorder=1))
molecule(ox + 9, oy + 11, 20, BLUE)
molecule(ox + 20, oy + 24, 200, ORANGE)  # 2-fold-related copy
# 2-fold rotation arrow between the copies
ax.add_patch(FancyArrowPatch((ox + 11, oy + 15), (ox + 18, oy + 22),
             connectionstyle="arc3,rad=0.55", arrowstyle="-|>",
             mutation_scale=7, lw=1.0, color=VERM, zorder=5))
ax.text(ox + w / 2 + 3, oy - 6.5, "rigid-body\nmolecular crystal",
        ha="center", va="top", color=GREY, fontsize=6.2, linespacing=1.0)

# decomposition equation
ax.text(6, 92, r"$R_m=\mathrm{rot}(g_m)\,R_{\mathrm{asym}}$",
        ha="left", va="center", color="#222222", fontsize=7.0)
ax.text(6, 83.5, "learnable coset  $\\times$  free pose",
        ha="left", va="center", color=GREY, fontsize=5.8)

# ---------------- (B) conditioning arrow ------------------------------------
ax.add_patch(FancyArrowPatch((40, 50), (57, 50), arrowstyle="-|>",
             mutation_scale=12, lw=2.2, color=GREEN, zorder=2))
ax.text(48.5, 70, "fully\nsymmetry-conditioned\nflow", ha="center", va="center",
        color=GREEN, fontsize=6.2, fontweight="bold", linespacing=1.05)
# two conditioning chips
for (yy, txt, col) in [(41, "orientation $\\to$ coset", BLUE),
                       (32.5, "lattice $\\to$ family mask", ORANGE)]:
    ax.add_patch(FancyBboxPatch((35.0, yy - 3.0), 29, 6.0,
                 boxstyle="round,pad=0.3,rounding_size=1.5",
                 fc="white", ec=col, lw=0.9, zorder=3))
    ax.text(49.5, yy, txt, ha="center", va="center", color=col, fontsize=5.3, zorder=4)

# ---------------- (C) exact-match result bars -------------------------------
bx = 67.0
bars = [("raw\nflow", 0.0, LGREY), ("full\nstack", 6.9, GREEN), ("+TTA", 10.7, BLUE)]
bw, gap, base, scale = 7.0, 3.2, 22.0, 3.55
# MCF parity reference line + label centered above the bar group
mcf_y = base + 8.0 * scale
xr = bx + 3 * (bw + gap)
ax.plot([bx - 1.5, xr], [mcf_y, mcf_y], ls=(0, (3, 2)), color=VERM, lw=0.9, zorder=2)
ax.text(bx - 1.5, mcf_y + 1.2, "8%", ha="left", va="bottom", color=VERM, fontsize=5.2)
for i, (lab, v, col) in enumerate(bars):
    x = bx + i * (bw + gap)
    ax.add_patch(plt.Rectangle((x, base), bw, max(v, 0.15) * scale,
                 fc=col, ec="white", lw=0.4, zorder=3))
    ax.text(x + bw / 2, base + v * scale + 1.4, f"{v:.1f}%" if v else "0%",
            ha="center", va="bottom", color="#222222", fontsize=5.8, zorder=4)
    ax.text(x + bw / 2, base - 1.6, lab, ha="center", va="top",
            color=GREY, fontsize=5.2, linespacing=0.95)
ax.text((bx + xr) / 2 - 1, base - 9.7, "held-out exact match\n(vs MolCrystalFlow, dashed)",
        ha="center", va="top", color=GREY, fontsize=5.3, linespacing=1.0)

fig.savefig("fig_toc.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_toc.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote fig_toc.pdf / fig_toc.png")
