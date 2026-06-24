"""Figure 1: rigid-body factorization and the orientation decomposition.

A *visual* schematic (real molecule glyphs, a pseudo-3D unit cell, an SO(3)
orientation frame, and a symmetry-copy rotation), with every element placed on
an explicit non-overlapping grid.

Panel (a): a molecular crystal (unit cell + oriented rigid molecules) is
factorized onto lattice x T^3 x SO(3) -- a lattice L, a fractional centroid
c_m per molecule, and a body-frame orientation R_m, reconstructed by
cart = c_m L + R_m local.

Panel (b): the per-molecule orientation target factors as
R_m = rot(g_m) . R_asym into a space-group-determined relative rotation between
symmetry copies (learnable) and the gauge-free asymmetric-unit pose (not
learnable here), with the paper's evidence attached to each branch.

Renders fig1_schematic.pdf / .png (vector + 300 dpi raster). Exact LaTeX math
via mathtext; Okabe-Ito colorblind-safe palette.
Run from repo root: python paper/figures/fig1_schematic.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon)

# --- Okabe-Ito (colorblind-safe) + soft fills --------------------------------
OK = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "verm": "#D55E00", "grey": "#8A8A8A", "ink": "#1A1A1A",
    "lblue": "#DCEBF7", "lgreen": "#D5F0E5", "lred": "#FBE2D5",
    "lgrey": "#ECECEC", "cellface": "#F4F7FA",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "mathtext.fontset": "dejavuserif",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# atom palette (by role, so orientation is unambiguous)
ATOM = {"C": (OK["blue"], 0.16), "O": (OK["verm"], 0.185),
        "N": (OK["green"], 0.17), "H": (OK["grey"], 0.11)}
# an asymmetric rigid body template (local frame); asymmetry makes pose visible
BODY = [(0.00, 0.00, "C"), (0.95, 0.18, "O"),
        (0.22, 0.92, "N"), (-0.72, 0.36, "H")]
BONDS = [(0, 1), (0, 2), (0, 3)]


def _rot(pts, deg):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pts @ R.T


def molecule(ax, cx, cy, deg, s=0.5, z=5, alpha=1.0):
    """Draw the rigid body centered at (cx, cy), rotated by `deg`, scaled `s`."""
    xy = _rot(np.array([(x, y) for x, y, _ in BODY]) * s, deg) + (cx, cy)
    for i, j in BONDS:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color=OK["ink"], lw=1.6 * s / 0.5, solid_capstyle="round",
                zorder=z, alpha=alpha)
    for (x, y, el), (px, py) in zip(BODY, xy):
        col, r = ATOM[el]
        ax.add_patch(Circle((px, py), r * s / 0.5, fc=col, ec="white",
                            lw=0.8, zorder=z + 1, alpha=alpha))


def frame(ax, cx, cy, deg, L=0.95, z=8):
    """Two orthogonal body axes (an SO(3) orientation frame)."""
    for d, col in ((0, OK["verm"]), (90, OK["blue"])):
        v = _rot(np.array([[L, 0.0]]), deg + d)[0]
        ax.add_patch(FancyArrowPatch(
            (cx, cy), (cx + v[0], cy + v[1]), arrowstyle="-|>",
            mutation_scale=9, lw=1.5, color=col, zorder=z))


def cell(ax, x, y, w, h, dx=0.7, dy=0.55, lw=1.4, z=3, face=True):
    """Pseudo-3D unit-cell parallelepiped; front bottom-left at (x, y)."""
    F = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    B = F + (dx, dy)
    if face:
        ax.add_patch(Polygon(B, closed=True, fc=OK["cellface"], ec=OK["grey"],
                            lw=lw * 0.8, zorder=z - 1))
    for f, b in zip(F, B):                       # connecting edges
        ax.plot([f[0], b[0]], [f[1], b[1]], color=OK["grey"], lw=lw * 0.8,
                zorder=z)
    if face:
        ax.add_patch(Polygon(F, closed=True, fc=OK["cellface"], ec=OK["ink"],
                            lw=lw, zorder=z + 1, alpha=0.92))
    else:
        ax.add_patch(Polygon(F, closed=True, fill=False, ec=OK["ink"],
                            lw=lw, zorder=z + 1))


fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": [1.0, 1.08]})
for ax in (axA, axB):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def box(ax, x, y, w, h, fc, ec, lw=1.3, z=2):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.16",
        fc=fc, ec=ec, lw=lw, zorder=z))


# =====================================================================
# (a) rigid-body factorization
# =====================================================================
axA.set_title("(a) Rigid-body factorization", fontsize=9.5, loc="left", pad=4)

# molecular crystal: unit cell holding two oriented rigid molecules
cell(axA, 0.8, 5.3, 2.7, 2.5)
molecule(axA, 1.7, 6.25, 22, s=0.42, z=6)
molecule(axA, 2.75, 6.95, 158, s=0.42, z=6)
axA.text(2.45, 4.55, "molecular crystal", ha="center", va="center", fontsize=8.6)
axA.text(2.45, 4.05, "rigid molecules in a unit cell", ha="center",
         va="center", fontsize=6.8, style="italic", color=OK["grey"])

# factorize arrow
axA.add_patch(FancyArrowPatch((4.55, 6.4), (5.35, 6.4), arrowstyle="-|>",
                              mutation_scale=13, lw=1.4, color=OK["ink"], zorder=4))
axA.text(4.95, 6.92, "factorize", ha="center", fontsize=6.9, style="italic",
         color=OK["grey"])

# three manifold tokens (icon on the left, exact label on the right)
def token(y, fc, ec, label):
    box(axA, 7.72, y, 4.45, 1.55, fc, ec, lw=1.3)
    axA.text(6.5, y, label, ha="left", va="center", fontsize=8.2, color=OK["ink"])

token(8.15, OK["lblue"], OK["blue"], "lattice  $L\\in\\mathbb{R}^{3\\times3}$")
cell(axA, 5.72, 7.78, 0.62, 0.6, dx=0.26, dy=0.22, lw=1.0, z=6, face=False)

token(6.05, OK["lgreen"], OK["green"], "centroid  $c_m\\in\\mathbb{T}^3$")
axA.add_patch(Rectangle((5.66, 5.7), 0.78, 0.74, fill=False, ec=OK["green"],
                        lw=1.2, zorder=6))
axA.add_patch(Circle((6.18, 6.0), 0.085, fc=OK["green"], ec="none", zorder=7))
for (sx, sy, ex, ey) in [(5.66, 6.25, 5.46, 6.25), (6.05, 6.44, 6.05, 6.64)]:
    axA.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>",
                                  mutation_scale=6, lw=0.9, color=OK["green"],
                                  zorder=7))  # periodic wrap hints

token(3.95, OK["lred"], OK["verm"], "orientation  $R_m\\in\\mathrm{SO}(3)$")
molecule(axA, 6.02, 3.95, 30, s=0.26, z=6)
frame(axA, 6.02, 3.95, 30, L=0.62, z=8)

# reconstruction equation
axA.text(5.0, 1.7, "$\\mathrm{cart}_{m,a}=c_mL+R_m\\,\\mathrm{local}_{m,a}$",
         ha="center", va="center", fontsize=9.0, color=OK["ink"])

# =====================================================================
# (b) orientation target decomposes
# =====================================================================
axB.set_title("(b) Orientation target decomposes", fontsize=9.5, loc="left", pad=4)
axB.text(5.0, 9.15, "$R_m \\;=\\; \\mathrm{rot}(g_m)\\;\\cdot\\;R_{\\mathrm{asym}}$",
         ha="center", va="center", fontsize=13.5, color=OK["ink"])

# ---- LEFT: rot(g_m) learnable (two symmetry copies related by a rotation) ----
box(axB, 2.6, 5.15, 4.35, 4.3, OK["lgreen"], OK["green"], lw=1.6)
axB.text(2.6, 6.85, "$\\mathrm{rot}(g_m)$", ha="center", va="center",
         fontsize=11.5, color=OK["ink"])
molecule(axB, 1.78, 5.55, 18, s=0.30, z=6)
molecule(axB, 3.42, 5.55, 198, s=0.30, z=6)
axB.add_patch(FancyArrowPatch((2.18, 6.18), (3.02, 6.18), arrowstyle="-|>",
                              mutation_scale=9, lw=1.4, color=OK["green"],
                              connectionstyle="arc3,rad=-0.55", zorder=7))
axB.text(2.6, 6.5, "$C_2$", ha="center", va="center", fontsize=7.2,
         color=OK["green"])
axB.text(2.6, 4.35, "space-group operation\nrelating symmetry copies",
         ha="center", va="center", fontsize=7.4, color=OK["ink"])
axB.text(2.6, 3.5, "LEARNABLE", ha="center", va="center", fontsize=9,
         color=OK["green"], weight="bold")

# ---- RIGHT: R_asym free (one molecule, free orientation) --------------------
box(axB, 7.4, 5.15, 4.35, 4.3, OK["lred"], OK["verm"], lw=1.6)
axB.text(7.4, 6.85, "$R_{\\mathrm{asym}}$", ha="center", va="center",
         fontsize=11.5, color=OK["ink"])
molecule(axB, 7.4, 5.5, 35, s=0.30, z=6)
for a0, a1 in [(35, 150), (215, 330)]:            # free-rotation arcs
    th = np.linspace(np.deg2rad(a0), np.deg2rad(a1), 30)
    axB.plot(7.4 + 0.78 * np.cos(th), 5.5 + 0.78 * np.sin(th), color=OK["verm"],
             lw=1.1, ls=(0, (3, 2)), zorder=7)
axB.add_patch(FancyArrowPatch((7.4 + 0.78 * np.cos(np.deg2rad(150)),
                               5.5 + 0.78 * np.sin(np.deg2rad(150))),
                              (7.4 + 0.78 * np.cos(np.deg2rad(158)),
                               5.5 + 0.78 * np.sin(np.deg2rad(158))),
                              arrowstyle="-|>", mutation_scale=8, lw=1.1,
                              color=OK["verm"], zorder=7))
axB.text(8.28, 6.18, "?", ha="center", va="center", fontsize=11,
         color=OK["verm"], weight="bold")
axB.text(7.4, 4.35, "free asymmetric-unit pose\n(gauge-arbitrary)",
         ha="center", va="center", fontsize=7.4, color=OK["ink"])
axB.text(7.4, 3.5, "NOT LEARNABLE here", ha="center", va="center", fontsize=9,
         color=OK["verm"], weight="bold")

# evidence strip (below the two boxes)
axB.text(2.6, 2.15, "evidence:  $+27\\%$ held-out loss,\n"
         "$13.7\\%$ packings reconstructed", ha="center", va="center",
         fontsize=7.2, color=OK["green"])
axB.text(7.4, 2.15, "evidence:  $0\\%$ beyond\nthe predict-zero floor",
         ha="center", va="center", fontsize=7.2, color=OK["verm"])

fig.tight_layout(w_pad=2.0)
fig.savefig("paper/figures/fig1_schematic.pdf")
fig.savefig("paper/figures/fig1_schematic.png", dpi=300)
print("wrote fig1_schematic.pdf / .png")
