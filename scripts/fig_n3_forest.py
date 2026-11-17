"""Figure (main text): forest plot of OURS minus each arm on TEST-B (percentage points).

Every number comes from `results/n3/n3_results.json` (committed in f1c713a before interpretation):
  sets.testB.report.table1[<arm>].delta_ours_minus_arm.{delta_pp, bca_pp, two_stage_pp}
  sets.testB.report.family_primary.{H1.components, H2.comparator, H3.comparator}  (which rows are H1-H3)
  sets.testB.report.{n, matcher}
Thick bar: paired BCa 95% CI over crystals. Thin bar: the two-stage interval that also resamples seeds.
The x axis is broken: iii-s and i-E (about -90 and -50 pp) sit in the left segment so that the
+6..+10 pp contrasts stay readable. Single ACS column (3.3 in).

    python scripts/fig_n3_forest.py
writes paper/submissions/ACSOmega/figures/fig_n3_forest.{pdf,png}
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "n3", "n3_results.json")
OUT = os.path.join(REPO, "paper", "submissions", "ACSOmega", "figures")
MINUS = "−"

# Okabe-Ito
BLUE, GREY, BLACK = "#0072B2", "#666666", "#000000"

# top -> bottom; the order of Table 1, then the descriptive rows
ARMS = ["floor", "iii-u", "iii-p", "iii-s", "iv-P", "MCF-R", "MCF-R+G", "ii-S", "iv-S", "i-E"]
DESCRIPTIVE = {"ii-S", "iv-S", "i-E"}
LABEL = {"floor": "random floor"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.default": "regular",
})


def signed(x, nd=2):
    s = f"{abs(x):.{nd}f}"
    return (MINUS if x < 0 else "+") + s


def load():
    d = json.load(open(RES, encoding="utf-8"))
    rep = d["sets"]["testB"]["report"]
    assert rep["set"] == "testB" and rep["matcher"] == "primary"
    fam = rep["family_primary"]
    assert fam["complete"]
    hyp = {c: "H1" for c in fam["H1"]["components"]}
    hyp[fam["H2"]["comparator"]] = "H2"
    hyp[fam["H3"]["comparator"]] = "H3"
    rows = []
    for arm in ARMS:
        x = rep["table1"][arm]["delta_ours_minus_arm"]
        assert x["n"] == rep["n"]
        rows.append({"arm": arm, "d": x["delta_pp"], "bca": x["bca_pp"], "two": x["two_stage_pp"],
                     "hyp": hyp.get(arm), "desc": arm in DESCRIPTIVE})
    # the H1 row values must be the family's own components (same numbers, two sources in the JSON)
    for r in rows:
        if r["hyp"] == "H1":
            c = fam["H1"]["components"][r["arm"]]
            assert c["delta_pp"] == r["d"] and c["bca_pp"] == r["bca"] and c["two_stage_pp"] == r["two"]
    return rows, rep["n"]


def main():
    rows, n = load()
    neg = [r for r in rows if r["two"][1] < -20]
    assert {r["arm"] for r in neg} == {"iii-s", "i-E"}, "axis break assumes only iii-s and i-E are far negative"
    lo_neg = min(r["two"][0] for r in neg)
    hi_neg = max(r["two"][1] for r in neg)
    hi_pos = max(r["two"][1] for r in rows if r not in neg)

    fig = plt.figure(figsize=(3.3, 2.75))
    # columns: [left segment | right segment]; labels left of the left segment, values right of the right
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.06,
                          left=0.215, right=0.80, top=0.865, bottom=0.165)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1], sharey=axL)
    gap = 0.55                                    # extra space before the descriptive rows
    y_of = {}
    yy = len(rows) - 1 + gap
    for r in rows:
        if r["desc"] and not any(q["desc"] for q in rows if q["arm"] in y_of):
            yy -= gap
        y_of[r["arm"]] = yy
        yy -= 1.0

    def style(r):
        if r["hyp"]:
            return dict(c=BLUE, marker="D", ms=3.6, mfc=BLUE)
        if r["desc"]:
            return dict(c=GREY, marker="o", ms=3.4, mfc="white")
        return dict(c=BLACK, marker="o", ms=3.4, mfc=BLACK)

    for ax in (axL, axR):
        # shade the descriptive block
        yd = [y_of[r["arm"]] for r in rows if r["desc"]]
        ax.axhspan(min(yd) - 0.5, max(yd) + 0.5, color="#F2F2F2", zorder=0, lw=0)
        for r in rows:
            s, y = style(r), y_of[r["arm"]]
            ax.plot(r["two"], [y, y], color=s["c"], lw=0.8, solid_capstyle="butt", zorder=2)
            ax.plot(r["bca"], [y, y], color=s["c"], lw=2.8, solid_capstyle="butt", zorder=3)
            ax.plot([r["d"]], [y], marker=s["marker"], ms=s["ms"], mfc=s["mfc"], mec=s["c"], mew=0.8,
                    color=s["c"], zorder=4, ls="none")
    axR.axvline(0, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)

    axL.set_xlim(lo_neg - 3.0, hi_neg + 4.0)
    axR.set_xlim(-1.5, hi_pos + 1.2)
    axL.set_xticks([-90, -70, -50])
    axR.set_xticks([0, 5, 10])
    axL.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{MINUS}{abs(v):.0f}"))
    axR.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ytop = max(y_of.values()) + 0.6
    ybot = min(y_of.values()) - 0.6
    axL.set_ylim(ybot, ytop)
    axL.set_yticks([y_of[r["arm"]] for r in rows])
    axL.set_yticklabels([LABEL.get(r["arm"], r["arm"]) for r in rows])
    for t, r in zip(axL.get_yticklabels(), rows):
        if r["desc"]:
            t.set_color(GREY)
            t.set_fontstyle("italic")
    plt.setp(axR.get_yticklabels(), visible=False)
    for ax in (axL, axR):
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.spines["right"].set_visible(False)
    # break marks on the bottom spine (matplotlib's broken-axis idiom: marker size in points)
    mk = dict(marker=[(-1, -1.8), (1, 1.8)], markersize=5, ls="none", color=BLACK, mec=BLACK, mew=0.6,
              clip_on=False)
    axL.plot([1], [0], transform=axL.transAxes, **mk)
    axR.plot([0], [0], transform=axR.transAxes, **mk)

    # right-hand columns: delta value and hypothesis tag
    trR = matplotlib.transforms.blended_transform_factory(axR.transAxes, axR.transData)
    for r in rows:
        y = y_of[r["arm"]]
        col = BLUE if r["hyp"] else GREY if r["desc"] else BLACK
        axR.text(1.03, y, signed(r["d"]), transform=trR, ha="left", va="center", fontsize=6.3, color=col)
        if r["hyp"]:
            axR.text(1.30, y, r["hyp"], transform=trR, ha="left", va="center", fontsize=6.5,
                     color=BLUE, fontweight="bold")
    axR.text(1.03, ytop, "Δ (pp)", transform=trR, ha="left", va="bottom", fontsize=6.3, color=BLACK)
    yd = [y_of[r["arm"]] for r in rows if r["desc"]]
    axR.text(1.30, sum(yd) / len(yd), "descriptive", transform=trR, ha="left", va="center", fontsize=6.0,
             color=GREY, fontstyle="italic", rotation=90)
    # direction cues
    axR.text(0.99, 1.005, "favours OURS →", transform=axR.transAxes, ha="right", va="bottom",
             fontsize=6.3, color=GREY)
    axL.text(0.02, 1.005, "← favours arm", transform=axL.transAxes, ha="left", va="bottom",
             fontsize=6.3, color=GREY)
    fig.text(0.5, 0.035,
             f"Δ = OURS {MINUS} arm, picked@1 (percentage points); TEST-B, n = {n:,}",
             ha="center", va="bottom", fontsize=6.8)
    # one-line legend above the panels
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=BLACK, lw=2.8, solid_capstyle="butt"),
         Line2D([0], [0], color=BLACK, lw=0.8, solid_capstyle="butt"),
         Line2D([0], [0], color=BLUE, lw=0, marker="D", ms=3.6),
         Line2D([0], [0], color=GREY, lw=0, marker="o", ms=3.4, mfc="white")]
    fig.legend(h, ["BCa 95% CI", "two-stage", "preregistered", "descriptive"], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=6.0, handlelength=1.4,
               handletextpad=0.35, columnspacing=0.9, borderaxespad=0.0)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig_n3_forest.{ext}")
        fig.savefig(p, dpi=300)
        print("wrote", os.path.relpath(p, REPO))


if __name__ == "__main__":
    main()
