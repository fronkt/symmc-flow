"""Figure 3: the exact-match lever ladder (Phase F).

(a) Waterfall of held-out best-of-10 exact match rate (StructureMatcher,
    stol=1.0) as each lever is added, from the raw fully symmetry-conditioned
    flow (0%) to parity with MolCrystalFlow (6.9%, strict match@10). The
    orientation-TTA finishing pass (best-of-30, 3x finish candidates) reaches
    10.7%, shown hatched as it uses a larger finishing budget.
(b) The same best configuration across the StructureMatcher site-tolerance
    sweep, against MolCrystalFlow's reported match@10. Strict match@10 is at
    parity; orientation-TTA finishing clears it.

Numbers: gpu_results/FINDINGS_F_FINAL.md and phaseF3{a..f}/ logs (n=131).
Renders fig3_ladder.pdf (vector, for LaTeX) and fig3_ladder.png (preview).
Colorblind-safe Okabe-Ito palette. Run: python paper/figures/fig3_ladder.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette (matches fig2_results.py)
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "grey": "#999999", "black": "#000000"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.4))

# ---- (a) exact-match lever ladder (waterfall), strict match@10, stol=1.0 ----
levers = ["raw flow", "+rigid press", "+centroid fix", "+2.3$\\times$ scale",
          "+self-cond.", "+match@10/steps", "+recycling", "+orient-TTA"]
vals   = [0.0, 1.5, 3.1, 3.1, 3.8, 6.1, 6.9, 10.7]
# color the strict-ladder bars blue; the TTA bar (larger finish budget) hatched grey
bar_colors = [OK["blue"]] * 7 + [OK["grey"]]
x = np.arange(len(levers))
bars = axL.bar(x, vals, color=bar_colors, width=0.68)
bars[-1].set_hatch("////")
bars[-1].set_edgecolor(OK["black"])
bars[-1].set_linewidth(0.6)
# MolCrystalFlow reference band (~8% at stol=1.0)
axL.axhline(8.0, color=OK["vermillion"], ls="--", lw=1.0)
axL.text(0.05, 8.4, "MolCrystalFlow $\\approx$8% (stol 1.0)",
         color=OK["vermillion"], fontsize=7.5, va="bottom")
axL.set_xticks(x)
axL.set_xticklabels(levers, rotation=32, ha="right", fontsize=7.5)
axL.set_ylabel("best-of-10 exact match (%)")
axL.set_title("(a) 0% $\\to$ parity: the lever ladder")
axL.set_ylim(0, 12.5)
for b, v in zip(bars, vals):
    axL.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}",
             ha="center", va="bottom", fontsize=7.5)
axL.axhline(0, color=OK["black"], lw=0.6)
axL.text(x[-1], 11.4, "3x finish\nbudget", ha="center", va="bottom",
         fontsize=6.5, color=OK["grey"], style="italic")

# ---- (b) best config vs MolCrystalFlow across the stol sweep ----
stols = ["stol 0.8", "stol 1.0", "stol 1.2"]
mcf     = [6.8, 8.0, np.nan]          # MolCrystalFlow reported (— at 1.2)
strict  = [6.1, 6.9, 7.6]            # ours, strict match@10 (F3f recycle)
tta     = [7.6, 10.7, 13.7]          # ours, +orientation-TTA best-of-30
xb = np.arange(len(stols))
w = 0.26
b1 = axR.bar(xb - w, mcf,    w, color=OK["vermillion"], label="MolCrystalFlow")
b2 = axR.bar(xb,     strict, w, color=OK["blue"],  label="ours, strict match@10")
b3 = axR.bar(xb + w, tta,    w, color=OK["grey"],  label="ours, +orient-TTA",
             hatch="////", edgecolor=OK["black"], linewidth=0.6)
axR.set_xticks(xb)
axR.set_xticklabels(stols)
axR.set_ylabel("best-of-10 exact match (%)")
axR.set_title("(b) direct comparison, tolerance sweep")
axR.set_ylim(0, 15.5)
axR.legend(loc="upper left", frameon=False, fontsize=7)
for bars_ in (b1, b2, b3):
    for b in bars_:
        h = b.get_height()
        if not np.isnan(h):
            axR.text(b.get_x() + b.get_width() / 2, h + 0.2, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=7)
axR.axhline(0, color=OK["black"], lw=0.6)

fig.tight_layout()
fig.savefig("paper/figures/fig3_ladder.pdf", bbox_inches="tight")
fig.savefig("paper/figures/fig3_ladder.png", bbox_inches="tight", dpi=200)
print("wrote fig3_ladder.pdf / .png")
