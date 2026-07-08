"""Generate the RSC Digital Discovery graphical-abstract/TOC entry.
Two panels: (a) the rotation decomposition R_m = rot(g_m).R_asym, (b) the
reconstruction-accuracy progression that motivates it. Sized to RSC's spec
(<=8cm x 4cm), exported at 600dpi TIFF plus a PNG preview.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

CM = 1 / 2.54
fig = plt.figure(figsize=(8 * CM, 4 * CM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.45,
                       left=0.03, right=0.98, top=0.86, bottom=0.16)

# --- Panel (a): rotation decomposition schematic ---
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(-1.3, 1.3); ax0.set_ylim(-1.3, 1.3)
ax0.set_aspect("equal")
ax0.axis("off")

circle = plt.Circle((0, 0), 1.0, fill=False, lw=0.9, color="0.5")
ax0.add_patch(circle)

def arrow(ax, angle_deg, color, lw=1.6):
    a = np.deg2rad(angle_deg)
    ax.add_patch(FancyArrowPatch((0, 0), (np.cos(a), np.sin(a)),
                                  arrowstyle="-|>", mutation_scale=8,
                                  lw=lw, color=color, zorder=3))

arrow(ax0, 25, "#d95f02")     # R_asym: free (gauge-arbitrary)
arrow(ax0, 100, "#1b9e77")    # rot(g_m): space-group relative rotation
ax0.text(0, -1.28, r"$R_m=\mathrm{rot}(g_m)\!\cdot\!R_{\mathrm{asym}}$",
          ha="center", va="top", fontsize=6.0)
ax0.text(1.05, -0.05, "free\n(gauge)", color="#d95f02", fontsize=4.6,
          ha="left", va="top")
ax0.text(-0.35, 1.08, "symmetry-\ndetermined", color="#1b9e77", fontsize=4.6,
          ha="right", va="bottom")

# --- Panel (b): reconstruction accuracy bars ---
ax1 = fig.add_subplot(gs[1])
labels = ["predict-\nfloor", "re-gauged", "+ coset"]
vals = [0.0, 13.7, 48.0]
colors = ["0.75", "#d95f02", "#1b9e77"]
bars = ax1.bar(range(3), vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:g}%",
              ha="center", va="bottom", fontsize=5.4)
ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, fontsize=5.0)
ax1.set_ylim(0, 58)
ax1.set_yticks([])
for spine in ("top", "right", "left"):
    ax1.spines[spine].set_visible(False)
ax1.set_title("exact packing\nreconstruction", fontsize=5.4, pad=2)

for out, dpi in [("toc_graphic.png", 300), ("toc_graphic.tiff", 600)]:
    kwargs = {"dpi": dpi}
    if out.endswith(".tiff"):
        kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(out, **kwargs)
plt.close(fig)
print("done")
