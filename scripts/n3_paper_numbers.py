"""Every N3 number the ACS Omega manuscript quotes, generated from the committed results.

Reads `results/n3/n3_results.json` (committed in f1c713a before interpretation) and
`results/n3/table_r_testB.json`, and writes into `paper/submissions/ACSOmega/generated/`:

  n3_numbers.tex   \\newcommand macros for every number used in the text (no number is typed by hand)
  n3_table1.tex    Table 1 (headline, TEST-B, primary matcher; P section 6)
  n3_hyp.tex       the H1-H3 families (primary and N2), with the ii-S identity
  n3_table2.tex    Table 2 (descriptive strata; P section 6)
  n3_tableR.tex    Table R (representability; P section 5)

Usage: python scripts/n3_paper_numbers.py [--check]   (--check: regenerate in memory, fail if files differ)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "n3", "n3_results.json")
TABR = os.path.join(REPO, "results", "n3", "table_r_testB.json")
OUT = os.path.join(REPO, "paper", "submissions", "ACSOmega", "generated")
MINUS = "\u2212"


def rnd(x, nd):
    """Round half away from zero on the value as written in the JSON (8.645 -> 8.65, not float's 8.64)."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(abs(x))).quantize(q, rounding=ROUND_HALF_UP))


def f2(x, sign=False, nd=2):
    if x is None:
        return "---"
    s = rnd(x, nd)
    if x < 0 and float(s) != 0:
        return "$-$" + s
    return ("+" if sign and x > 0 else "") + s


def pct(x):
    """Percent with 2 decimals below 10 and 1 decimal above (99.4, 58.9, 8.47, 0.13)."""
    if x is None:
        return "---"
    return rnd(x, 1) if x >= 10 else rnd(x, 2)


def ci(lo_hi):
    lo, hi = lo_hi
    return f"[{f2(lo)}, {f2(hi)}]"


def pval(p):
    if p is None:
        return "---"
    e = math.floor(math.log10(p))
    m = p / 10 ** e
    m = round(m, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"${m:.1f}\\times10^{{{e}}}$"


def macro(name, value):
    assert name.isalpha(), name
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


ROWS = [  # (key, label, information code, representable); information codes follow Table 0 (P section 0)
    ("floor", "random floor", "$L$, $c_0$, ops", "1000/1000"),
    ("iii-u", "iii-u (Haar, LJ pick)", "$L$, $c_0$, ops", "1000/1000"),
    ("iii-p", "iii-p (steric press)", "$L$, $c_0$, ops", "1000/1000"),
    ("iii-s", "iii-s (FF$^\\ast$ search)", "$L$, $c_0$, ops", "1000/1000"),
    ("iv-P", "iv-P (previous architecture)", "$L$, all $c_k$, op labels", "---$^{b}$"),
    ("MCF-R", "MCF-R", "$L$, all $c_k$", "1000/1000"),
    ("MCF-R+G", "MCF-R+G", "$L$, all $c_k$", "1000/1000"),
    ("OURS", "\\textbf{OURS}", "$L$, $c_0$, ops", "1000/1000"),
    ("OURS-N2", "OURS-N2", "$L$, $c_0$, ops", "1000/1000"),
]


def build():
    d = json.load(open(RES, encoding="utf-8"))
    tr = json.load(open(TABR, encoding="utf-8"))
    rep = d["sets"]["testB"]["report"]
    assert rep["set"] == "testB" and rep["n"] == 1000 and d["h2_form"]["form"] == "EXACT"
    t1, fam, fn2, w = rep["table1"], rep["family_primary"], rep["family_n2"], rep["wording"]
    assert fam["complete"] and fn2["complete"]
    out = {}

    # ---------------------------------------------------------------- macros
    m = []
    o = t1["OURS"]
    m.append(macro("nN", f"{rep['n']:,}".replace(",", "{,}")))
    m.append(macro("nOursPick", pct(o["picked1_mean_pct"])))
    m.append(macro("nOursSd", f2(o["picked1_sd_pct"])))
    m.append(macro("nOursSeeds", ", ".join(f"{x:.1f}" for x in o["picked1_per_seed_pct"])))
    m.append(macro("nOursDraw", pct(o["draw1_mean_pct"])))
    m.append(macro("nOursAny", pct(o["any16_mean_pct"])))
    m.append(macro("nOursSec", f"{o['compute']['sec_per_crystal_per_seed_median']:.1f}"))
    m.append(macro("nOursNtwoPick", pct(t1["OURS-N2"]["picked1_mean_pct"])))
    m.append(macro("nOursNtwoSd", f2(t1["OURS-N2"]["picked1_sd_pct"])))
    for key, nm in [("iii-s", "Search"), ("i-E", "IE"), ("MCF-R", "McfR"), ("MCF-R+G", "McfRG"),
                    ("iv-P", "IvP"), ("ii-S", "IiS"), ("floor", "Floor"), ("iii-u", "Haar"), ("iii-p", "Press"),
                    ("iv-S", "IvS"), ("MCF-A", "McfA")]:
        m.append(macro(f"n{nm}Pick", pct(t1[key]["picked1_mean_pct"])))
    names = {"H1": "One", "H2": "Two", "H3": "Three"}
    for fam_d, tag in [(fam, ""), (fn2, "Ntwo")]:
        for h, hn in names.items():
            x = fam_d[h]
            m.append(macro(f"nH{hn}{tag}Delta", f2(x["delta_pp"], sign=True)))
            m.append(macro(f"nH{hn}{tag}CI", ci(x["bca_pp"])))
            m.append(macro(f"nH{hn}{tag}P", pval(x["p_holm"])))
    for comp, nm in [("MCF-R", "R"), ("MCF-R+G", "G")]:
        x = fam["H1"]["components"][comp]
        m.append(macro(f"nHOne{nm}Delta", f2(x["delta_pp"], sign=True)))
        m.append(macro(f"nHOne{nm}CI", ci(x["bca_pp"])))
        m.append(macro(f"nHOne{nm}Two", ci(x["two_stage_pp"])))
    for h, hn in names.items():
        x = fam[h] if h != "H1" else fam["H1"]["components"][fam["H1"]["delta_from"]]
        m.append(macro(f"nH{hn}Two", ci(x["two_stage_pp"])))
    dec = w["ii_S_decomposition"]
    m.append(macro("nLearnDelta", f2(dec["OURS - ii-S"]["delta_pp"], sign=True)))
    m.append(macro("nLearnCI", ci(dec["OURS - ii-S"]["bca_pp"])))
    m.append(macro("nExpDelta", f2(dec["ii-S - MCF-R+G"]["delta_pp"], sign=True)))
    m.append(macro("nExpCI", ci(dec["ii-S - MCF-R+G"]["bca_pp"])))
    m.append(macro("nZmcfMin", f2(w["zmcf_ge3_min_delta_pp"], sign=True)))
    fl = t1["floor"]["delta_ours_minus_arm"]
    m.append(macro("nFloorDelta", f2(fl["delta_pp"], sign=True)))
    m.append(macro("nFloorCI", ci(fl["bca_pp"])))
    ie = rep["i_e_vs_h2"]
    m.append(macro("nIEvsSearchDelta", f2(ie["delta_pp"], sign=True)))
    m.append(macro("nIEvsSearchCI", ci(ie["bca_pp"])))
    lab = w["H2_by_ff_label"]
    for k, nm in [("ANCHORED", "Anch"), ("DRIFTED", "Drift"), ("UNRESOLVED", "Unres")]:
        m.append(macro(f"nFF{nm}N", str(lab[k]["n"])))
    m.append(macro("nAnchDelta", f2(lab["ANCHORED"]["delta_pp"], sign=True)))
    m.append(macro("nAnchCI", ci(lab["ANCHORED"]["bca_pp"])))
    anc = d["sets"]["testB"]["ff_truth_anchoring"]
    m.append(macro("nAnchPct", f"{100 * anc['anchoring']:.1f}"))
    omc = rep["h2_omc25_nonoverlap"]
    m.append(macro("nOmcN", str(omc["n"])))
    m.append(macro("nOmcDelta", f2(omc["OURS - iii-s"]["delta_pp"], sign=True)))
    m.append(macro("nOmcCI", ci(omc["OURS - iii-s"]["bca_pp"])))
    gd = o["geodesic_deg"]["per_seed"]
    m.append(macro("nOursGeoMed", f"{sum(s['median'] for s in gd) / len(gd):.0f}"))
    m.append(macro("nOursGeoQten", f"{sum(s['q10'] for s in gd) / len(gd):.0f}"))  # seed-mean of per-seed q10
    m.append(macro("nOursEns", pct(o["ensemble_picked1_pct"])))
    wr = tr["wording"]
    assert wr["final"] and wr["n"] == 1000
    m.append(macro("nRLegacy", str(wr["N_legacy_unrep"])))
    m.append(macro("nRLegacyCentro", str(wr["M_legacy_unrep_centro"])))
    m.append(macro("nRMixed", str(wr["n_mixed"])))
    m.append(macro("nRMixedMiss", str(wr["P_mixed_not_matched_primary"])))
    m.append(macro("nRMixedApprox", str(wr["Q_mixed_matched_stol0.5"])))
    out["n3_numbers.tex"] = ("% GENERATED by scripts/n3_paper_numbers.py from results/n3/n3_results.json and "
                             "results/n3/table_r_testB.json -- do not edit\n" + "".join(m))

    # ---------------------------------------------------------------- Table 1
    def hyp_cell(fd):
        return "/".join("S" if fd[h]["supported"] else "N" for h in ("H1", "H2", "H3"))

    lines = []
    for key, label, info, reprn in ROWS:
        x = t1[key]
        sd = "" if x.get("picked1_sd_pct") is None else f" $\\pm$ {f2(x['picked1_sd_pct'])}"
        dl = x.get("delta_ours_minus_arm") or {}
        if key == "OURS":
            dcell, tcell, hcell = "---", "---", hyp_cell(fam)
        else:
            dcell = f"{f2(dl['delta_pp'], sign=True)} {ci(dl['bca_pp'])}"
            tcell = ci(dl["two_stage_pp"])
            hcell = hyp_cell(fn2) if key == "OURS-N2" else ""
        lines.append(f"{label} & {info} & {reprn} & {pct(x['picked1_mean_pct'])}{sd} & {dcell} & {tcell} & "
                     f"{pct(x['draw1_mean_pct'])} & {pct(x['any16_mean_pct'])} & {hcell} \\\\\n")
    out["n3_table1.tex"] = ("% GENERATED by scripts/n3_paper_numbers.py -- do not edit\n" + "".join(lines))

    # ---------------------------------------------------------------- hypotheses table
    hl = []
    comp = {"H1": "MCF-R, MCF-R+G", "H2": "iii-s", "H3": "iv-P"}
    for fam_d, fname in [(fam, "OURS"), (fn2, "OURS-N2")]:
        for h in ("H1", "H2", "H3"):
            x = fam_d[h]
            ts = x["two_stage_pp"] if "two_stage_pp" in x else x["components"][x["delta_from"]]["two_stage_pp"]
            verdict = {"OURS better": "supported", "iii-s better": "not supported$^{a}$"}.get(
                x["verdict"], x["verdict"])
            hl.append(f"{fname} & {h} & {comp[h]} & {f2(x['delta_pp'], sign=True)} & {ci(x['bca_pp'])} & "
                      f"{ci(ts)} & {pval(x['p_holm'])} & {verdict} \\\\\n")
        if fname == "OURS":
            hl.append("\\midrule\n")
    out["n3_hyp.tex"] = "% GENERATED by scripts/n3_paper_numbers.py -- do not edit\n" + "".join(hl)

    # ---------------------------------------------------------------- Table 2 (strata x arms)
    t2 = rep["table2"]
    arms = [k for k, *_ in ROWS]
    strata = [("K", "$K$", ["<=2", "3-4", ">=6"]), ("z_mcf", "$z_\\mathrm{MCF}$", ["1", "2", ">=3"]),
              ("class", "class", ["centro", "sohncke", "other"]), ("mixed", "mixed handedness", ["True", "False"]),
              ("formula_disjoint", "formula-disjoint", ["True", "False"]),
              ("omc25_nonoverlap", "OMC25 non-overlap", ["True", "False"]),
              ("ff_label", "FF$^\\ast$ truth label", ["ANCHORED", "DRIFTED", "UNRESOLVED"])]
    pretty = {"<=2": "$\\le2$", "3-4": "3--4", ">=6": "$\\ge6$", ">=3": "$\\ge3$", "True": "yes", "False": "no",
              "centro": "centrosymmetric", "sohncke": "Sohncke", "other": "other", "ANCHORED": "anchored",
              "DRIFTED": "drifted", "UNRESOLVED": "unresolved"}
    sl = []
    for skey, sname, levels in strata:
        for i, lv in enumerate(levels):
            cell = t2[skey][lv]
            n = cell["n"]
            vals = []
            for a in arms:
                if a not in cell["arms"]:  # the FF* truth label is reported for the H2 pair only (P section 6)
                    vals.append("")
                    continue
                k = cell["arms"][a]["k_per_seed"]
                vals.append(pct(100 * sum(k) / len(k) / n))
            flag = "$^{\\dagger}$" if cell.get("too_small") else ""
            sl.append(f"{sname if i == 0 else ''} & {pretty.get(lv, lv)}{flag} & {n} & " + " & ".join(vals) + " \\\\\n")
        sl.append("\\addlinespace\n")
    out["n3_table2.tex"] = "% GENERATED by scripts/n3_paper_numbers.py -- do not edit\n" + "".join(sl[:-1])
    # range of seed-mean picked@1 over the strata the text quotes (n >= 30, not too_small)
    for a, nm in [("OURS", "Ours"), ("iii-s", "Search")]:
        vals = []
        for skey, _, levels in strata:
            for lv in levels:
                cell = t2[skey][lv]
                if cell["n"] < 30 or cell.get("too_small") or a not in cell["arms"]:
                    continue
                k = cell["arms"][a]["k_per_seed"]
                vals.append(100 * sum(k) / len(k) / cell["n"])
        m.append(macro(f"n{nm}StratMin", pct(min(vals))))
        m.append(macro(f"n{nm}StratMax", pct(max(vals))))
    out["n3_numbers.tex"] = ("% GENERATED by scripts/n3_paper_numbers.py from results/n3/n3_results.json and "
                             "results/n3/table_r_testB.json -- do not edit\n" + "".join(m))

    # ---------------------------------------------------------------- Table R
    rows = tr["rows"]
    rl = []
    rname = {"all": "all", "centro": "centrosymmetric", "sohncke": "Sohncke", "other": "other",
             "mixed": "mixed handedness", "improper_achiral": "improper, achiral",
             "legacy_unrep": "legacy-unrepresentable"}
    for k in ["all", "centro", "sohncke", "other", "mixed", "improper_achiral", "legacy_unrep"]:
        r = rows[k]
        rl.append(f"{rname[k]} & {r['n']} & {r['a']['primary']} & {r['b']['primary']} & {r['c']['primary']} & "
                  f"{r['c']['stol0.5']} & {r['d']['flag_kept']} \\\\\n")
    out["n3_tableR.tex"] = "% GENERATED by scripts/n3_paper_numbers.py -- do not edit\n" + "".join(rl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = build()
    os.makedirs(OUT, exist_ok=True)
    bad = 0
    for name, text in out.items():
        p = os.path.join(OUT, name)
        if a.check:
            cur = open(p, encoding="utf-8").read() if os.path.exists(p) else None
            if cur != text:
                print("DIFFERS:", name)
                bad += 1
        else:
            open(p, "w", encoding="utf-8", newline="\n").write(text)
            print("wrote", os.path.relpath(p, REPO))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
