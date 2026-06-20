"""Figure 2: the orientation results.

(a) Orientation-isolated StructureMatcher reconstruction rate by condition
    (true lattice+centroid+conformer; only SO(3) sampled; best-of-8; n=131).
(b) Held-out non-reference orientation loss drop under the strengthening levers,
    with the no-coset paired control.

Renders fig2_results.pdf (vector, for LaTeX) and fig2_results.png (preview).
Colorblind-safe Okabe-Ito palette. Run: python paper/figures/fig2_results.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "grey": "#999999", "black": "#000000"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# ---- (a) match rate by condition ----
conds = ["oracle", "identity\n(R=I)", "untrained\n(floor)", "trained"]
rates = [100.0, 1.5, 0.0, 16.8]
colors = [OK["grey"], OK["orange"], OK["vermillion"], OK["blue"]]
bars = axL.bar(conds, rates, color=colors, width=0.66)
axL.set_ylabel("exact match rate (%)")
axL.set_title("(a) orientation-isolated reconstruction")
axL.set_ylim(0, 105)
for b, r in zip(bars, rates):
    axL.text(b.get_x() + b.get_width() / 2, r + 1.5, f"{r:.1f}",
             ha="center", va="bottom", fontsize=8)
axL.axhline(0, color=OK["black"], lw=0.6)

# ---- (b) non-reference orientation loss drop under levers ----
levers = ["baseline", "+capacity\n& steps", "+coset\ncond.", "coset\ncontrol"]
drops = [27.1, 33.4, 50.6, 27.1]
lcolors = [OK["blue"], OK["green"], OK["vermillion"], OK["grey"]]
bars2 = axR.bar(levers, drops, color=lcolors, width=0.66)
axR.set_ylabel("non-ref. orientation loss drop (%)")
axR.set_title("(b) the ceiling is inference-limited")
axR.set_ylim(0, 60)
for b, d in zip(bars2, drops):
    axR.text(b.get_x() + b.get_width() / 2, d + 0.8, f"{d:.1f}",
             ha="center", va="bottom", fontsize=8)
axR.axhline(27.1, color=OK["black"], ls=":", lw=0.8, label="baseline level")
axR.legend(loc="upper left", frameon=False, fontsize=7)

fig.tight_layout()
fig.savefig("paper/figures/fig2_results.pdf", bbox_inches="tight")
fig.savefig("paper/figures/fig2_results.png", bbox_inches="tight", dpi=200)
print("wrote fig2_results.pdf / .png")
