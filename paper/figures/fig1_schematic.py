"""Figure 1: rigid-body factorization and the orientation decomposition.

Panel (a): a molecular crystal is parametrized on lattice x T^3 x SO(3) --
a unit cell holding rigid molecules, each carried by a fractional centroid c_m
and a body-frame orientation R_m, reconstructed by cart = c_m L + R_m local.

Panel (b): the per-molecule orientation target factors as
R_m = rot(g_m) . R_asym into a space-group-determined relative rotation
(learnable) and the gauge-free asymmetric-unit pose (not learnable here),
with the paper's evidence attached to each branch.

Renders fig1_schematic.pdf / .png (vector + 300 dpi raster). Exact LaTeX math
via mathtext; Okabe-Ito colorblind-safe palette.
Run from repo root: python paper/figures/fig1_schematic.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, RegularPolygon, Polygon

# --- Okabe-Ito (colorblind-safe) + soft fills --------------------------------
OK = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "verm": "#D55E00", "grey": "#8A8A8A", "ink": "#1A1A1A",
    "lblue": "#DCEBF7", "lgreen": "#D5F0E5", "lred": "#FBE2D5",
    "lgrey": "#ECECEC",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "mathtext.fontset": "dejavuserif",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(7.4, 3.35), gridspec_kw={"width_ratios": [1.02, 1.18]})
for ax in (axA, axB):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def oriented_molecule(ax, cx, cy, r, angle_deg, face, edge, lw=1.4, z=5, alpha=1.0):
    """A rigid pentagon glyph with a body-frame 'north' tick marking R_m."""
    poly = RegularPolygon((cx, cy), 5, radius=r, orientation=np.radians(angle_deg),
                          facecolor=face, edgecolor=edge, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(poly)
    a = np.radians(angle_deg) + np.pi / 2          # pointer to top vertex
    ax.add_patch(FancyArrowPatch(
        (cx, cy), (cx + 1.45 * r * np.cos(a), cy + 1.45 * r * np.sin(a)),
        arrowstyle="-|>", mutation_scale=8, lw=1.3, color=edge, zorder=z + 1, alpha=alpha))
    ax.plot([cx], [cy], marker="o", ms=2.2, color=edge, zorder=z + 1, alpha=alpha)


# =====================================================================
# (a) rigid-body factorization
# =====================================================================
axA.set_title("(a) Rigid-body factorization", fontsize=9.5, loc="left", pad=6)

O = np.array([0.95, 2.25])
va, vb, vc = np.array([3.05, 0.0]), np.array([0.0, 3.55]), np.array([0.95, 0.9])
def P(fa, fb, fc): return O + fa * va + fb * vb + fc * vc

# unit cell as a parallelepiped (back face lighter, front face solid)
back = [P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)]
front = [P(0, 0, 0), P(1, 0, 0), P(1, 1, 0), P(0, 1, 0)]
axA.add_patch(Polygon(back, closed=True, fill=False, ec=OK["grey"],
                      lw=0.9, ls=(0, (3, 2)), zorder=1))
for f, b in zip(front, back):
    axA.plot([f[0], b[0]], [f[1], b[1]], color=OK["grey"], lw=0.9,
             ls=(0, (3, 2)), zorder=1)
axA.add_patch(Polygon(front, closed=True, fill=False, ec=OK["ink"],
                      lw=1.5, zorder=2))

# lattice vectors a, b, c from the origin
for v, name, col in [(va, "$\\mathbf{a}$", OK["ink"]),
                     (vb, "$\\mathbf{b}$", OK["ink"]),
                     (vc, "$\\mathbf{c}$", OK["ink"])]:
    tip = O + 0.42 * v
    axA.add_patch(FancyArrowPatch(O, tip, arrowstyle="-|>", mutation_scale=8,
                                  lw=1.2, color=col, zorder=3))
    axA.text(*(tip + 0.18 * v / np.linalg.norm(v) + np.array([0.0, 0.0])),
             name, fontsize=8.5, ha="center", va="center", zorder=4)

# two rigid molecules at fractional centroids, each with its own orientation
oriented_molecule(axA, *P(0.34, 0.60, 0.5), 0.42, 18, OK["lblue"], OK["blue"])
oriented_molecule(axA, *P(0.72, 0.28, 0.5), 0.42, -135, OK["lred"], OK["verm"])

axA.text(2.75, 7.65, "molecular crystal", ha="center", fontsize=8.6,
         color=OK["ink"])

# factorize arrow -> three manifold tokens
axA.add_patch(FancyArrowPatch((6.05, 6.45), (6.85, 6.45), arrowstyle="-|>",
                              mutation_scale=13, lw=1.4, color=OK["ink"]))
axA.text(6.45, 6.95, "factorize", ha="center", fontsize=7.3, style="italic",
         color=OK["grey"])
tokens = [("lattice  $L\\in\\mathbb{R}^{3\\times3}$", OK["lblue"], OK["blue"], 8.25),
          ("centroid  $c_m\\in\\mathbb{T}^3$", OK["lgreen"], OK["green"], 6.45),
          ("orientation  $R_m\\in\\mathrm{SO}(3)$", OK["lred"], OK["verm"], 4.65)]
for txt, fc, ec, y in tokens:
    axA.add_patch(FancyBboxPatch((7.0, y - 0.62), 2.85, 1.18,
                  boxstyle="round,pad=0.05,rounding_size=0.18",
                  fc=fc, ec=ec, lw=1.2, zorder=3))
    axA.text(8.42, y, txt, ha="center", va="center", fontsize=7.8, zorder=4)
axA.text(8.42, 2.95, "$\\mathrm{cart}_{m,a}=c_mL+R_m\\,\\mathrm{local}_{m,a}$",
         ha="center", va="center", fontsize=7.8, color=OK["ink"])

# =====================================================================
# (b) orientation target decomposes
# =====================================================================
axB.set_title("(b) Orientation target decomposes", fontsize=9.5, loc="left", pad=6)
axB.text(5.0, 8.9, "$R_m \\;=\\; \\mathrm{rot}(g_m)\\;\\cdot\\;R_{\\mathrm{asym}}$",
         ha="center", va="center", fontsize=13.5, color=OK["ink"])

# --- LEFT: rot(g_m) learnable -----------------------------------------------
axB.add_patch(FancyBboxPatch((0.35, 3.0), 4.4, 4.55,
              boxstyle="round,pad=0.08,rounding_size=0.2",
              fc=OK["lgreen"], ec=OK["green"], lw=1.6, zorder=2))
axB.text(2.55, 7.05, "$\\mathrm{rot}(g_m)$", ha="center", fontsize=11,
         color=OK["ink"], zorder=3)
axB.text(2.55, 6.35, "space-group operation\nbetween symmetry copies",
         ha="center", va="center", fontsize=7.4, color=OK["ink"], zorder=3)
# glyph: two copies related by a rotation
oriented_molecule(axB, 1.65, 4.95, 0.34, 20, "#FFFFFF", OK["green"], lw=1.2, z=4)
oriented_molecule(axB, 3.35, 4.95, 0.34, 200, "#FFFFFF", OK["green"], lw=1.2, z=4)
axB.add_patch(FancyArrowPatch((2.05, 5.35), (2.95, 5.35),
              connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
              mutation_scale=8, lw=1.2, color=OK["green"], zorder=5))
axB.text(2.55, 3.55, "LEARNABLE", ha="center", fontsize=8.3,
         color=OK["green"], weight="bold", zorder=3)

# --- RIGHT: R_asym free ------------------------------------------------------
axB.add_patch(FancyBboxPatch((5.25, 3.0), 4.4, 4.55,
              boxstyle="round,pad=0.08,rounding_size=0.2",
              fc=OK["lred"], ec=OK["verm"], lw=1.6, zorder=2))
axB.text(7.45, 7.05, "$R_{\\mathrm{asym}}$", ha="center", fontsize=11,
         color=OK["ink"], zorder=3)
axB.text(7.45, 6.35, "free asymmetric-unit pose\nHaar-uniform on $\\mathrm{SO}(3)$",
         ha="center", va="center", fontsize=7.4, color=OK["ink"], zorder=3)
# glyph: one copy with ambiguous (free) orientation -- ghosted alternates
for ang, alpha in [(95, 0.28), (-60, 0.28), (20, 1.0)]:
    oriented_molecule(axB, 7.45, 4.95, 0.34, ang, "#FFFFFF", OK["verm"],
                      lw=1.2, z=4, alpha=alpha)
axB.text(8.45, 5.5, "?", fontsize=12, color=OK["verm"], weight="bold",
         ha="center", va="center", zorder=6)
axB.text(7.45, 3.55, "NOT LEARNABLE (here)", ha="center", fontsize=8.3,
         color=OK["verm"], weight="bold", zorder=3)

# --- evidence strip ----------------------------------------------------------
axB.text(2.55, 2.1, "$+27\\%$ held-out loss\n$13.7\\%$ packings reconstructed",
         ha="center", va="center", fontsize=7.3, color=OK["green"])
axB.text(7.45, 2.1, "$0\\%$ beyond the\npredict-zero floor",
         ha="center", va="center", fontsize=7.3, color=OK["verm"])

fig.tight_layout(w_pad=2.0)
fig.savefig("paper/figures/fig1_schematic.pdf")
fig.savefig("paper/figures/fig1_schematic.png", dpi=300)
print("wrote fig1_schematic.pdf / .png")
