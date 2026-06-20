"""Figure 1: rigid-body factorization and the orientation decomposition.

A schematic: a molecular crystal -> rigid blocks parametrized on
lattice x T^3 x SO(3); and the per-molecule orientation target factored as
R_m = rot(g_m) . R_asym into a learnable symmetry-relative part and a
gauge-free asymmetric-unit pose.

Renders fig1_schematic.pdf / .png. This is a draft schematic; for final
typesetting it can be redrawn in TikZ (see fig1_schematic.tex).
Run: python paper/figures/fig1_schematic.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "grey": "#999999", "black": "#000000",
      "lblue": "#D6EAF8", "lgreen": "#D5F5E3", "lred": "#FADBD8"}

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                               gridspec_kw={"width_ratios": [1.0, 1.25]})
for ax in (axA, axB):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ---------- (a) factorization ----------
axA.set_title("(a) rigid-body factorization", fontsize=10, loc="left")
# unit cell
axA.add_patch(Rectangle((0.6, 1.2), 4.2, 4.2, fill=False, lw=1.4,
                        edgecolor=OK["black"]))
# two molecule glyphs (triangles = oriented rigid bodies)
for (cx, cy, ang, col) in [(2.0, 3.7, 0, OK["blue"]), (3.6, 2.4, 180, OK["orange"])]:
    axA.plot([cx], [cy], marker=(3, 0, ang), markersize=20, color=col)
axA.text(2.7, 5.7, "crystal", ha="center", fontsize=9)
# arrow to parametrization
axA.add_patch(FancyArrowPatch((5.0, 3.3), (6.2, 3.3),
              arrowstyle="-|>", mutation_scale=14, lw=1.3, color=OK["black"]))
# parametrization boxes
labels = [(r"lattice $L$", OK["lblue"], 8.4),
          (r"centroid $c_m\in\mathbb{T}^3$", OK["lgreen"], 6.2),
          (r"orient $R_m\in\mathrm{SO}(3)$", OK["lred"], 4.0)]
for txt, col, y in labels:
    axA.add_patch(FancyBboxPatch((6.4, y - 0.7), 3.4, 1.3,
                  boxstyle="round,pad=0.06", fc=col, ec=OK["black"], lw=1.0))
    axA.text(8.1, y - 0.05, txt, ha="center", va="center", fontsize=8.5)
axA.text(8.1, 2.4, r"$\mathrm{cart}=c_mL+R_m\,\mathrm{local}$",
         ha="center", va="center", fontsize=8)

# ---------- (b) orientation decomposition ----------
axB.set_title("(b) orientation target decomposes", fontsize=10, loc="left")
axB.text(5.0, 9.1, r"$R_m \;=\; \mathrm{rot}(g_m)\;\cdot\;R_{\mathrm{asym}}$",
         ha="center", va="center", fontsize=13)

# left box: rot(g_m) learnable
axB.add_patch(FancyBboxPatch((0.4, 4.6), 4.3, 3.2, boxstyle="round,pad=0.1",
              fc=OK["lgreen"], ec=OK["green"], lw=1.6))
axB.text(2.55, 7.3, r"$\mathrm{rot}(g_m)$", ha="center", fontsize=10)
axB.text(2.55, 6.5, "space-group operation\nbetween symmetry copies",
         ha="center", va="center", fontsize=7.8)
axB.text(2.55, 5.2, "LEARNABLE", ha="center", fontsize=8.5,
         color=OK["green"], weight="bold")

# right box: R_asym free
axB.add_patch(FancyBboxPatch((5.3, 4.6), 4.3, 3.2, boxstyle="round,pad=0.1",
              fc=OK["lred"], ec=OK["vermillion"], lw=1.6))
axB.text(7.45, 7.3, r"$R_{\mathrm{asym}}$", ha="center", fontsize=10)
axB.text(7.45, 6.5, "free asymmetric-unit pose\n(per-crystal, gauge-arbitrary)",
         ha="center", va="center", fontsize=7.8)
axB.text(7.45, 5.2, "UNLEARNABLE", ha="center", fontsize=8.5,
         color=OK["vermillion"], weight="bold")

# evidence strip
axB.text(5.0, 3.4, "evidence", ha="center", fontsize=8, style="italic",
         color=OK["grey"])
axB.text(2.55, 2.5, "+27% loss\n16.8% packings\nreconstructed", ha="center",
         va="center", fontsize=7.6, color=OK["green"])
axB.text(7.45, 2.5, "0% beyond\npredict-zero\nfloor", ha="center",
         va="center", fontsize=7.6, color=OK["vermillion"])

fig.tight_layout()
fig.savefig("paper/figures/fig1_schematic.pdf", bbox_inches="tight")
fig.savefig("paper/figures/fig1_schematic.png", bbox_inches="tight", dpi=200)
print("wrote fig1_schematic.pdf / .png")
