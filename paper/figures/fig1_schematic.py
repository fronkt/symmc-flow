"""Figure 1: rigid-body factorization and the orientation decomposition.

A clean box-and-text diagram (no illustrative glyphs).

Panel (a): a molecular crystal is factorized onto lattice x T^3 x SO(3) --
a lattice L, a fractional centroid c_m per molecule, and a body-frame
orientation R_m, reconstructed by cart = c_m L + R_m local.

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
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
    1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.0, 1.12]})
for ax in (axA, axB):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def box(ax, x, y, w, h, fc, ec, lw=1.3, z=2):
    """Rounded box anchored by its center (x, y)."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.16",
        fc=fc, ec=ec, lw=lw, zorder=z))


# =====================================================================
# (a) rigid-body factorization
# =====================================================================
axA.set_title("(a) Rigid-body factorization", fontsize=9.5, loc="left", pad=6)

# source: molecular crystal
box(axA, 1.95, 5.5, 3.0, 2.2, OK["lgrey"], OK["ink"], lw=1.4)
axA.text(1.95, 6.05, "molecular crystal", ha="center", va="center", fontsize=9)
axA.text(1.95, 5.1, "rigid molecules\nin a unit cell", ha="center", va="center",
         fontsize=7, style="italic", color=OK["grey"])

# factorize arrow
axA.add_patch(FancyArrowPatch((3.6, 5.5), (4.55, 5.5), arrowstyle="-|>",
                              mutation_scale=13, lw=1.4, color=OK["ink"], zorder=3))
axA.text(4.08, 6.0, "factorize", ha="center", fontsize=7.3, style="italic",
         color=OK["grey"])

# three manifold tokens
tokens = [("lattice  $L\\in\\mathbb{R}^{3\\times3}$", OK["lblue"], OK["blue"], 7.5),
          ("centroid  $c_m\\in\\mathbb{T}^3$", OK["lgreen"], OK["green"], 5.5),
          ("orientation  $R_m\\in\\mathrm{SO}(3)$", OK["lred"], OK["verm"], 3.5)]
for txt, fc, ec, y in tokens:
    box(axA, 7.35, y, 5.0, 1.45, fc, ec, lw=1.3)
    axA.text(7.35, y, txt, ha="center", va="center", fontsize=8.3, color=OK["ink"])

# reconstruction equation
axA.text(5.0, 1.55, "$\\mathrm{cart}_{m,a}=c_mL+R_m\\,\\mathrm{local}_{m,a}$",
         ha="center", va="center", fontsize=8.6, color=OK["ink"])

# =====================================================================
# (b) orientation target decomposes
# =====================================================================
axB.set_title("(b) Orientation target decomposes", fontsize=9.5, loc="left", pad=6)
axB.text(5.0, 9.0, "$R_m \\;=\\; \\mathrm{rot}(g_m)\\;\\cdot\\;R_{\\mathrm{asym}}$",
         ha="center", va="center", fontsize=13.5, color=OK["ink"])

# LEFT: rot(g_m) learnable
box(axB, 2.55, 5.45, 4.4, 4.0, OK["lgreen"], OK["green"], lw=1.6)
axB.text(2.55, 6.95, "$\\mathrm{rot}(g_m)$", ha="center", va="center",
         fontsize=11.5, color=OK["ink"])
axB.text(2.55, 5.55, "space-group operation\nrelating symmetry copies",
         ha="center", va="center", fontsize=7.7, color=OK["ink"])
axB.text(2.55, 4.15, "LEARNABLE", ha="center", va="center", fontsize=9,
         color=OK["green"], weight="bold")

# RIGHT: R_asym free
box(axB, 7.45, 5.45, 4.4, 4.0, OK["lred"], OK["verm"], lw=1.6)
axB.text(7.45, 6.95, "$R_{\\mathrm{asym}}$", ha="center", va="center",
         fontsize=11.5, color=OK["ink"])
axB.text(7.45, 5.55, "free asymmetric-unit pose\n(gauge-arbitrary)",
         ha="center", va="center", fontsize=7.7, color=OK["ink"])
axB.text(7.45, 4.15, "NOT LEARNABLE here", ha="center", va="center", fontsize=9,
         color=OK["verm"], weight="bold")

# evidence strip
axB.text(2.55, 2.45, "evidence:  $+27\\%$ held-out loss,\n"
         "$13.7\\%$ packings reconstructed", ha="center", va="center",
         fontsize=7.3, color=OK["green"])
axB.text(7.45, 2.45, "evidence:  $0\\%$ beyond\nthe predict-zero floor",
         ha="center", va="center", fontsize=7.3, color=OK["verm"])

fig.tight_layout(w_pad=2.2)
fig.savefig("paper/figures/fig1_schematic.pdf")
fig.savefig("paper/figures/fig1_schematic.png", dpi=300)
print("wrote fig1_schematic.pdf / .png")
