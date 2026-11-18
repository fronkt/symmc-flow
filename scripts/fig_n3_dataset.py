"""SI figure: TRAIN vs TEST-B composition (aggregates only).

Inputs: the frozen N3 set items, read through `n3_common.load_set` (validated per-set caches
data/n3_sets/{train,testB}.pt; TRAIN = seed-0 perm[300:] of data/csd_mol/g2_asym.pt, TEST-B = the frozen
order of tasks/n3_sets/testB_refcodes.txt). Per crystal it reads only
  sg  -> crystal class with `n3_labels.sg_class` (gemmi: centrosymmetric / Sohncke / other)
  K   -> number of space-group operations in the CSD cell (= Z for Z' = 1)
  Z   -> heavy-atom count (Z > 1) and the set of elements in the molecule
  L   -> lattice lengths a, b, c (row norms, CSD setting)
and plots set-level distributions (percent of crystals), so no structure leaves the private caches.
The TEST-B class and K counts are checked against the committed results/n3/labels_summary_testB.json.

    python scripts/fig_n3_dataset.py
writes paper/submissions/ACSOmega/figures/fig_n3_dataset.{pdf,png}
"""
from __future__ import annotations

import collections
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = os.path.join(REPO, "paper", "submissions", "ACSOmega", "figures")
SUMMARY = os.path.join(REPO, "results", "n3", "labels_summary_testB.json")

SETS = [("train", "TRAIN", "#0072B2"), ("testB", "TEST-B", "#E69F00")]   # Okabe-Ito blue / orange
CLASSES = [("centro", "centro-\nsymmetric"), ("sohncke", "Sohncke"), ("other", "other")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
})


FAMS = [("tric", "tric."), ("mono", "mono."), ("orth", "orth."), ("tetr", "tetr."), ("hex", "trig./hex."),
        ("cub", "cubic")]


def crystal_family(sg):
    """Crystal family from the international space-group number (trigonal and hexagonal merged)."""
    for hi, fam in ((2, "tric"), (15, "mono"), (74, "orth"), (142, "tetr"), (194, "hex"), (230, "cub")):
        if sg <= hi:
            return fam
    raise ValueError(sg)


def aggregates(name):
    import n3_common as C
    from n3_labels import sg_class
    from pymatgen.core.periodic_table import Element
    items = C.load_set(name)
    assert len(items) == C.SET_SIZES[name]
    out = {"n": len(items), "class": collections.Counter(), "K": collections.Counter(), "heavy": [],
           "elem": collections.Counter(), "abc": [], "fam": collections.Counter()}
    for a in items:
        out["class"][sg_class(int(a["sg"]))] += 1
        out["fam"][crystal_family(int(a["sg"]))] += 1
        out["K"][int(a["K"])] += 1
        Z = a["Z"].numpy()
        out["heavy"].append(int((Z > 1).sum()))
        for s in {Element.from_Z(int(z)).symbol for z in Z}:
            out["elem"][s] += 1
        out["abc"].append(np.linalg.norm(a["L"].double().numpy(), axis=1))
    out["heavy"] = np.array(out["heavy"])
    out["abc"] = np.array(out["abc"])
    return out


def check_testB(agg):
    s = json.load(open(SUMMARY, encoding="utf-8"))["counts"]
    assert s["n"] == agg["n"]
    assert {c: agg["class"][c] for c in ("centro", "sohncke", "other")} == s["class"], "class counts differ"
    assert {str(k): v for k, v in sorted(agg["K"].items())} == s["K_values"], "K counts differ"


def bars(ax, cats, data, width=0.38):
    x = np.arange(len(cats))
    for s, (key, label, col) in enumerate(SETS):
        ax.bar(x + (s - 0.5) * width, data[key], width, color=col, label=label, lw=0)
    ax.set_xticks(x)


def main():
    agg = {k: aggregates(k) for k, _, _ in SETS}
    check_testB(agg["testB"])
    pct = lambda k, c: 100.0 * c / agg[k]["n"]

    fig = plt.figure(figsize=(3.3, 4.75))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.62, wspace=0.42,
                          left=0.13, right=0.985, top=0.905, bottom=0.075)

    # (a) crystal family (space-group class -- centrosymmetric / Sohncke / other -- is in main Table 5 and SI)
    ax = fig.add_subplot(gs[0, 0])
    fams = [f for f in FAMS if agg["train"]["fam"][f[0]] + agg["testB"]["fam"][f[0]] > 0]
    bars(ax, fams, {k: [pct(k, agg[k]["fam"][f]) for f, _ in fams] for k, _, _ in SETS})
    ax.set_xticklabels([lab for _, lab in fams], rotation=35, ha="right")
    ax.set_ylabel("crystals (%)")
    ax.set_title("(a) crystal family", loc="left")
    for k, _, _ in SETS:
        print(k, "families", dict(agg[k]["fam"]), "classes", dict(agg[k]["class"]))

    # (b) K = number of operations
    ax = fig.add_subplot(gs[0, 1])
    Ks = sorted(set(agg["train"]["K"]) | set(agg["testB"]["K"]))
    bars(ax, Ks, {k: [pct(k, agg[k]["K"][K]) for K in Ks] for k, _, _ in SETS})
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel("K (operations = Z)")
    ax.set_ylabel("crystals (%)")
    ax.set_title("(b) copies per cell", loc="left")

    # (c) heavy-atom count
    ax = fig.add_subplot(gs[1, 0])
    hmax = int(max(agg[k]["heavy"].max() for k, _, _ in SETS))
    edges = np.arange(0.5, hmax + 2.5, 2)
    for key, label, col in SETS:
        h = agg[key]["heavy"]
        w = np.full(len(h), 100.0 / len(h))
        ax.hist(h, bins=edges, weights=w, histtype="step", color=col, lw=1.1, label=label)
    ax.set_xlabel("heavy atoms per molecule")
    ax.set_ylabel("crystals (%)")
    ax.set_title("(c) molecule size", loc="left")

    # (d) lattice lengths (box: quartiles, line: median, whiskers: 5th-95th percentile)
    ax = fig.add_subplot(gs[1, 1])
    pos, data, cols = [], [], []
    for i in range(3):
        for s, (key, _, col) in enumerate(SETS):
            pos.append(i + (s - 0.5) * 0.36)
            data.append(agg[key]["abc"][:, i])
            cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.3, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.6),
                    capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
    for line, col in zip(bp["whiskers"], np.repeat(cols, 2)):
        line.set_color(col)
    for line, col in zip(bp["caps"], np.repeat(cols, 2)):
        line.set_color(col)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["$a$", "$b$", "$c$"])
    ax.set_ylabel("length (Å)")
    ax.set_title("(d) lattice lengths", loc="left")

    # (e) elements: percent of crystals whose molecule contains the element (C and H noted in the panel)
    ax = fig.add_subplot(gs[2, :])
    shown = [e for e, _ in sorted(((e, c) for e, c in (agg["train"]["elem"] + agg["testB"]["elem"]).items()
                                   if e not in ("C", "H")), key=lambda t: -t[1])]
    keep = [e for e in shown if max(pct(k, agg[k]["elem"][e]) for k, _, _ in SETS) >= 0.5]
    rest = [e for e in shown if e not in keep]
    data = {}
    for key, _, _ in SETS:
        vals = [pct(key, agg[key]["elem"][e]) for e in keep]
        if rest:   # crystals containing at least one of the rare elements
            vals.append(None)
        data[key] = vals
    if rest:
        import n3_common as C
        for key, _, _ in SETS:
            from pymatgen.core.periodic_table import Element
            n_rest = sum(1 for a in C.load_set(key)
                         if {Element.from_Z(int(z)).symbol for z in a["Z"].numpy()} & set(rest))
            data[key][-1] = pct(key, n_rest)
    cats = keep + (["other"] if rest else [])
    bars(ax, cats, data, width=0.4)
    ax.set_xticklabels(cats)
    ax.set_ylabel("crystals (%)")
    ax.set_title("(e) elements present (besides C and H)", loc="left")
    ch = ", ".join(f"{lab}: C {pct(k, agg[k]['elem']['C']):.0f}%, H {pct(k, agg[k]['elem']['H']):.1f}%"
                   for k, lab, _ in SETS)
    ax.text(0.99, 0.97, ch.replace(", TEST", "\nTEST"), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.0, color="#444444")
    if rest:
        ax.text(0.99, 0.62, "other = " + ", ".join(rest), transform=ax.transAxes, ha="right", va="top",
                fontsize=5.6, color="#444444")

    h, _ = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, [f"{lab} (n = {agg[k]['n']:,})" for k, lab, _ in SETS], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.2, columnspacing=1.5)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig_n3_dataset.{ext}")
        fig.savefig(p, dpi=300)
        print("wrote", os.path.relpath(p, REPO))


if __name__ == "__main__":
    main()
