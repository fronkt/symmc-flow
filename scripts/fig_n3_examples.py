"""SI figure: representative OURS successes and failures on TEST-B (seed 0, LJ pick, primary matcher).

Selection rule (fixed, no cherry-picking): walk TEST-B in its frozen order (tasks/n3_sets/testB_refcodes.txt)
and take the FIRST THREE crystals whose seed-0 OURS pick is a match under the primary matcher and the FIRST
THREE whose pick is not.

Frozen inputs (nothing is re-sampled or re-scored):
  results/n3/private/draws/ours_s0_testB.pt           OURS seed-0 TEST-B bundle: R [1000, 16, 3, 3] (R_asym
                                                      draws), valid [1000, 16], sel["lj"] [1000, 16]
  results/n3/private/match/ours_s0_testB__primary.jsonl  frozen scorer rows {i, j, ok} for every draw
  results/n3/private/match/ours_s0_testB__primary.meta.json  bundle SHA-256 the rows were scored against
  results/n3/private/step9/testB_checksums.tsv        step-9 checksum of the match rows
  results/n3/n3_results.json                          table1.OURS.picked1_per_seed_pct[0] (consistency check)
The pick is `n3_score.argmin_pick(valid, sel["lj"])`, the scorer's own LJ selector; its `ok` is read from the
frozen rows. The script asserts that the bundle and the rows are the frozen ones (hashes) and that the
seed-0 picked@1 recomputed from them equals the committed number.

Drawing: the K copies of the cell, projected along its shortest axis, built with OURS's `expand` from the true
R_asym (truth, thick grey) and from the picked R_asym (OURS, thin colour), heavy atoms only. Annotation:
for a match, the RMS of `StructureMatcher().get_rms_dist(truth, pick)` (the primary matcher's defaults;
normalised by (V/n)^(1/3)); the frozen rows store no RMS for the primary matcher, so it is recomputed for
display, and the script asserts that `fit` reproduces the frozen `ok` for every shown crystal. dtheta is the
geodesic angle between the picked and the true R_asym. Only rendered images are written.

    python scripts/fig_n3_examples.py
writes paper/submissions/ACSOmega/figures/fig_n3_examples.{pdf,png}
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = os.path.join(REPO, "paper", "submissions", "ACSOmega", "figures")
PRIV = os.path.join(REPO, "results", "n3", "private")
BUNDLE = os.path.join(PRIV, "draws", "ours_s0_testB.pt")
MATCH_META = os.path.join(PRIV, "match", "ours_s0_testB__primary.meta.json")
MATCH_ROWS = os.path.join(PRIV, "match", "ours_s0_testB__primary.jsonl")
CHECKSUMS = os.path.join(PRIV, "step9", "testB_checksums.tsv")
RESULTS = os.path.join(REPO, "results", "n3", "n3_results.json")
N_EACH = 3
TRUTH = "#AAAAAA"
HIT, MISS = "#0072B2", "#D55E00"         # Okabe-Ito blue / vermillion

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
})


def sg_label(n):
    """Space-group number -> short Hermann-Mauguin symbol with screw subscripts and rotoinversion bars."""
    import gemmi
    s = gemmi.find_spacegroup_by_number(int(n)).short_name()
    out, i = "", 0
    while i < len(s):
        ch = s[i]
        if ch == "-" and i + 1 < len(s):
            out += "$\\bar{" + s[i + 1] + "}$"
            i += 2
        elif ch.isdigit() and i + 1 < len(s) and s[i + 1].isdigit() and int(s[i + 1]) < int(ch):
            out += ch + "$_" + s[i + 1] + "$"
            i += 2
        else:
            out += ch
            i += 1
    return out


def frozen_inputs():
    """Load the bundle and match rows and prove they are the frozen step-9 files."""
    import n3_common as C
    import n3_score as S
    meta = json.load(open(MATCH_META, encoding="utf-8"))
    assert meta["matcher"] == "primary" and meta["step9"] is True
    assert C.sha256(BUNDLE) == meta["bundle_sha256"], "bundle is not the one the frozen rows scored"
    rel = os.path.relpath(MATCH_ROWS, REPO).replace(os.sep, "/")
    frozen = {line.split("\t")[1].strip(): line.split("\t")[0] for line in open(CHECKSUMS, encoding="utf-8")}
    assert frozen[rel] == C.sha256(MATCH_ROWS), "match rows differ from the step-9 checksum"
    b = C.load_bundle(BUNDLE)
    assert b["arm"] == "ours" and b["seed"] == 0 and b["set"] == "testB" and b["kind"] == "rasym"
    assert b["refcodes"] == C.frozen_refcodes("testB")
    rows = S.load_match(BUNDLE, "primary")
    return b, rows


def picks(b, rows):
    import n3_score as S
    out = []
    for i in range(len(b["refcodes"])):
        j = S.argmin_pick([bool(v) for v in b["valid"][i]], [float(v) for v in b["sel"]["lj"][i]])
        out.append((j, None if j is None else bool(rows[(i, j)]["ok"])))
    k = sum(1 for _, ok in out if ok)
    rep = json.load(open(RESULTS, encoding="utf-8"))["sets"]["testB"]["report"]["table1"]["OURS"]
    assert abs(100.0 * k / len(out) - rep["picked1_per_seed_pct"][0]) < 1e-9, "seed-0 picked@1 differs"
    return out, k


def geodesic_deg(Ra, Rb):
    c = (float(torch.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def cell_coords(a, R):
    from g2_asym_baselines import expand
    L, cent, orient, local, Z, am, mm = expand(a, R.double())
    L = L.double().numpy()
    X = (cent.double().numpy() @ L)[:, None, :] + np.einsum("kij,kaj->kai", orient.double().numpy(),
                                                            local.double().numpy())
    return L, X, Z[0].numpy()


def draw_panel(ax, a, R_pick, ok, rms, dth):
    from n3_chirality import graph
    L, Xt, Zs = cell_coords(a, a["R0"])
    _, Xp, _ = cell_coords(a, R_pick)
    heavy = Zs > 1
    remap = {int(i): n for n, i in enumerate(np.nonzero(heavy)[0])}
    bonds = [(remap[i], remap[j]) for i, j in graph(a["local"].double().numpy(), Zs).edges()
             if heavy[i] and heavy[j]]
    Xt, Xp = Xt[:, heavy], Xp[:, heavy]
    # view along the shortest lattice axis (s into the page); e1 along the next axis, made orthogonal to s
    n = np.linalg.norm(L, axis=1)
    i_s = int(np.argmin(n))
    others = [i for i in range(3) if i != i_s]
    sh = L[i_s] / n[i_s]
    e1 = L[others[0]] - (L[others[0]] @ sh) * sh
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, sh)
    proj = lambda P: np.stack([P @ e1, P @ e2], -1)
    u, v = L[others[0]], L[others[1]]
    C2 = proj(np.array([np.zeros(3), u, u + v, v]))
    ax.add_patch(Polygon(C2, closed=True, fill=False, edgecolor="#666666", lw=0.5, zorder=0.5))
    col = HIT if ok else MISS
    for X, c, lw, z in ((Xt, TRUTH, 2.2, 1), (Xp, col, 0.8, 3)):
        for k in range(X.shape[0]):
            P = proj(X[k])
            for i, j in bonds:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=c, lw=lw, solid_capstyle="round", zorder=z)
    allP = np.concatenate([proj(Xt.reshape(-1, 3)), proj(Xp.reshape(-1, 3)), C2])
    lo, hi = allP.min(0) - 0.6, allP.max(0) + 0.6
    span = max(hi - lo)
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax_names = "abc"
    ax.set_title(f"{a['refcode']}   {sg_label(a['sg'])}   $K$ = {int(a['K'])}", fontsize=6.6, pad=10)
    txt = (f"match, RMS {rms:.3f}" if ok else "no match") + f";  $\\Delta\\theta$ = {dth:.0f}°"
    ax.text(0.5, 1.01, txt, transform=ax.transAxes, fontsize=6.2, color=col, ha="center", va="bottom",
            fontweight="bold")
    ax.text(0.99, 0.01, f"along ${ax_names[i_s]}$", transform=ax.transAxes, fontsize=5.8, color="#666666",
            ha="right", va="bottom", zorder=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.1"))


def main():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    import n3_common as C
    from pymatgen.analysis.structure_matcher import StructureMatcher
    b, rows = frozen_inputs()
    pk, k = picks(b, rows)
    hits = [i for i, (j, ok) in enumerate(pk) if ok][:N_EACH]
    miss = [i for i, (j, ok) in enumerate(pk) if j is not None and not ok][:N_EACH]
    items = C.load_set("testB")
    sm = StructureMatcher()

    fig, axes = plt.subplots(N_EACH, 2, figsize=(3.3, 5.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.875, bottom=0.005, wspace=0.05, hspace=0.27)
    log = []
    for col, idxs in ((0, hits), (1, miss)):
        for r, i in enumerate(idxs):
            a = items[i]
            assert a["refcode"] == b["refcodes"][i]
            j, ok = pk[i]
            R = b["R"][i, j].double()
            truth, cand = C.truth_structure(a), C.rasym_structure(a, R)
            assert bool(sm.fit(truth, cand)) == ok, f"{a['refcode']}: matcher disagrees with the frozen row"
            rr = sm.get_rms_dist(truth, cand) if ok else None
            rms = None if rr is None else float(rr[0])
            dth = geodesic_deg(R, a["R0"].double())
            draw_panel(axes[r, col], a, R, ok, rms, dth)
            log.append(f"{'match' if ok else 'no match':8s} TEST-B #{i:4d} {a['refcode']:10s} sg {int(a['sg']):3d} "
                       f"K {int(a['K'])}  pick j={j}  rms {'-' if rms is None else f'{rms:.4f}'}  dtheta {dth:.1f}")
    fig.text(0.25, 0.950, "matches (first 3)", ha="center", va="center", fontsize=7.2, color=HIT,
             fontweight="bold")
    fig.text(0.75, 0.950, "no match (first 3)", ha="center", va="center", fontsize=7.2, color=MISS,
             fontweight="bold")
    h = [Line2D([0], [0], color=TRUTH, lw=2.2), Line2D([0], [0], color=HIT, lw=0.8),
         Line2D([0], [0], color=MISS, lw=0.8)]
    fig.legend(h, ["experimental (truth)", "OURS pick (match)", "OURS pick (no match)"], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.0, handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig_n3_examples.{ext}")
        fig.savefig(p, dpi=300)
        print("wrote", os.path.relpath(p, REPO))
    print(f"seed-0 picked@1 recomputed from the frozen rows: {k}/{len(pk)} (matches n3_results.json)")
    print("\n".join(log))


if __name__ == "__main__":
    main()
