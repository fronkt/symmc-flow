"""Every number in the ACS Omega Supporting Information (si.tex), generated from the committed N3 records.

Reads only named evidence records (PAPER_ACCEPTANCE_CONTRACT assertion 1):
  results/n3/n3_results.json            sets.testB / sets.devtest: report (table1, table2, family_*, wording,
                                        steric_basin, i_e_vs_h2, h2_omc25_nonoverlap, h2_basin_reference),
                                        mcf_diagnostics, ff_truth_anchoring, provenance; top level: h2_form,
                                        gate_c, mcf_diagnostic_a
  results/n3/n3_selection.json          the frozen choices and the VALSEL selection grids (valsel_cell_scores)
  results/n3/table_r_{testB,devtest}.json
  results/n3/gateF.json, results/n3/gateC.json
  results/n3/omc25_outside_elements_{testB,devtest}.json
  tasks/n3_step4_record_2026-10-11.md   run-health counts that no results file holds (FF* OOM retries, MCF branch-cut
                                        draws, residual and LJ health, X-H normalisation). Each is found by a regex on
                                        the record and cited by file:line in the generated TeX; nothing is typed here.

Writes paper/submissions/ACSOmega/generated/si_*.tex: si_numbers.tex (\\newcommand macros, each with a source
comment) and table-row files read with \\inputrows (the primitive \\@@input, as in main.tex), plus two pgfplots figures.
Rounding and number formats follow scripts/n3_paper_numbers.py (half-up on the value as written in the JSON).
A field the records lack is printed as "not recorded"; no value is supplied for it.

Usage: python scripts/n3_si_tables.py [--check]   (--check: regenerate in memory, fail if files differ)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from n3_paper_numbers import ci, f2, pct, rnd  # noqa: E402  (shared number conventions with the main text)

P = lambda *a: os.path.join(REPO, *a)  # noqa: E731
RES, SELF = "results/n3/n3_results.json", "results/n3/n3_selection.json"
TABR = {"testB": "results/n3/table_r_testB.json", "devtest": "results/n3/table_r_devtest.json"}
GATEF, GATEC = "results/n3/gateF.json", "results/n3/gateC.json"
OMC = {"testB": "results/n3/omc25_outside_elements_testB.json",
       "devtest": "results/n3/omc25_outside_elements_devtest.json"}
REC = "tasks/n3_step4_record_2026-10-11.md"
OUT = P("paper", "submissions", "ACSOmega", "generated")
NR = "not recorded"
SETNAME = {"testB": "TEST-B", "devtest": "DEV-TEST"}

LAB = {"floor": "random floor", "iii-u": "iii-u", "iii-p": "iii-p", "iii-s": "iii-s", "iv-P": "iv-P",
       "MCF-R": "MCF-R", "MCF-R+G": "MCF-R+G", "OURS": "\\textbf{OURS}", "OURS-N2": "OURS-N2", "ii-S": "ii-S",
       "iv-S": "iv-S", "i-E": "i-E", "MCF-A": "MCF-A", "MCF-A-floor": "MCF-A prior floor"}
PLAIN = {k: v.replace("\\textbf{", "").replace("}", "") for k, v in LAB.items()}
T1 = ["floor", "iii-u", "iii-p", "iii-s", "iv-P", "MCF-R", "MCF-R+G", "OURS", "OURS-N2"]
ALL = T1 + ["ii-S", "iv-S", "i-E", "MCF-A", "MCF-A-floor"]
SELWORD = {"lj": "LJ", "random": "RANDOM", "resid": "RESID", "ff": "FF$^\\ast$ minimum", "lj_pre": "LJ (pre-press)"}
MATCHER = {"primary": "primary", "mcf05": "MCF rule, stol 0.5", "mcf08": "MCF rule, stol 0.8",
           "mcf10": "MCF rule, stol 1.0"}


# ----------------------------------------------------------------------------- formatting
def tex(s):
    """Escape a JSON string for TeX text (and spell out the section sign, as the manuscript does)."""
    s = str(s).replace("\u00a7", "Section ")
    out = []
    for ch in s:
        out.append({"\\": "\\textbackslash{}", "_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$",
                    "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}",
                    "\u2212": "$-$"}.get(ch, ch))
    return "".join(out)


def sel(s):
    if s is None:
        return "---"
    if s.startswith("mode@"):
        return "MODE@" + s[5:]
    return SELWORD.get(s, tex(s))


def num(n):
    """An integer count with a thin thousands separator, as \\nN in the main text."""
    return f"{int(n):,}".replace(",", "{,}")


def cnt(x):
    """A (possibly fractional, for RANDOM's exact expectation) count: integers plain, else up to 4 decimals."""
    if x is None:
        return "---"
    if float(x).is_integer():
        return num(x)
    q = Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP).normalize()
    return format(q, "f")


def pc(x):
    """A percentage (paper convention: 2 decimals below 10, 1 above), '---' when absent."""
    return "---" if x is None else pct(x)


def fr(x):
    """A fraction shown as a percentage."""
    return "---" if x is None else pct(100 * x)


def seeds(lst, f=pc):
    return "/".join(f(x) for x in lst)


def wil(lohi):
    lo, hi = lohi
    return f"[{fr(lo)}, {fr(hi)}]"


def g3(x):
    """Three significant digits, half-up; scientific notation outside [1e-3, 1e4)."""
    if x is None:
        return "---"
    if isinstance(x, float) and not math.isfinite(x):
        return "non-finite"
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    if e < -3 or e >= 4:
        return sci(x)
    q = Decimal(1).scaleb(e - 2)
    return format(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP), "f")


def sci(x, nd=1):
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = Decimal(repr(abs(x) / 10 ** e)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    if m >= 10:
        m, e = (m / 10).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP), e + 1
    sign = "-" if x < 0 else ""
    return f"${sign}{m}\\times10^{{{e}}}$"


def dec6(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "non-finite"
    return rnd(x, 6) if x >= 0 else "$-$" + rnd(x, 6)


def deg(x):
    return rnd(x, 1)


def dl(d):
    """Delta with its BCa CI."""
    return f"{f2(d['delta_pp'], sign=True)} {ci(d['bca_pp'])}"


def refs(lst):
    if lst is None:
        return NR
    return ", ".join(tex(r) for r in lst) if lst else "none"


def hdr(*src):
    return "% GENERATED by scripts/n3_si_tables.py -- do not edit\n" + "".join(f"% src: {s}\n" for s in src)


# ----------------------------------------------------------------------------- the step record
class Record:
    """Finds counts in the step record by regex (never by line number) and cites the line it found."""

    def __init__(self, rel):
        self.rel = rel
        self.text = open(P(rel), encoding="utf-8").read()

    def find(self, pattern):
        ms = list(re.finditer(pattern, self.text, flags=re.S))
        assert len(ms) == 1, f"{self.rel}: pattern {pattern!r} matched {len(ms)} times"
        m = ms[0]
        a = self.text.count("\n", 0, m.start()) + 1
        b = self.text.count("\n", 0, m.end()) + 1
        return m, f"{self.rel}:{a}" + (f"-{b}" if b != a else "")


# ----------------------------------------------------------------------------- build
def build():
    d = json.load(open(P(RES), encoding="utf-8"))
    S = json.load(open(P(SELF), encoding="utf-8"))
    TR = {k: json.load(open(P(v), encoding="utf-8")) for k, v in TABR.items()}
    GF = json.load(open(P(GATEF), encoding="utf-8"))
    GC = json.load(open(P(GATEC), encoding="utf-8"))
    OM = {k: json.load(open(P(v), encoding="utf-8")) for k, v in OMC.items()}
    R = Record(REC)
    sets = d["sets"]
    assert sets["testB"]["report"]["n"] == 1000 and d["h2_form"]["form"] == "EXACT"
    assert d["selection"]["sha256"] == sets["testB"]["provenance"]["selection_sha256"]
    out, M = {}, []

    def mac(name, value, src):
        assert name.isalpha() and name.startswith("sv"), name
        M.append(f"\\newcommand{{\\{name}}}{{{value}}}% src: {src}\n")

    def rows(name, src, lines):
        out[name] = hdr(*src) + "".join(lines)

    for s in ("testB", "devtest"):
        mac(f"svN{'' if s == 'testB' else 'Dev'}", num(sets[s]["report"]["n"]), f"{RES} sets.{s}.report.n")

    # ======================================================================== S1: OURS checkpoints
    L = []
    for arm, key in (("OURS", "OURS"), ("OURS-N2", "OURS-N2")):
        fz = S["frozen"][key]
        for sd, ck in sorted(fz["checkpoints"].items()):
            L.append(f"{LAB[arm]} & {sd} & \\texttt{{{tex(ck['ckpt'])}}} & \\texttt{{{ck['sha256'][:12]}}} \\\\\n")
    rows("si_ours_ckpt.tex", [f"{SELF} frozen.OURS, frozen.OURS-N2 (checkpoints)"], L)
    fo, fn = S["frozen"]["OURS"], S["frozen"]["OURS-N2"]
    assert (fo["selector"], fo["S"], fo["euler_steps"]) == (fn["selector"], fn["S"], fn["euler_steps"])
    mac("svOursSel", sel(fo["selector"]), f"{SELF} frozen.OURS.selector (= OURS-N2)")
    mac("svOursS", num(fo["S"]), f"{SELF} frozen.OURS.S (= OURS-N2)")
    mac("svOursEuler", num(fo["euler_steps"]), f"{SELF} frozen.OURS.euler_steps (= OURS-N2)")

    # ======================================================================== per-set tables
    for s in ("testB", "devtest"):
        st = sets[s]
        rep = st["report"]
        t1 = rep["table1"]
        n = rep["n"]
        tag = "" if s == "testB" else "Dev"
        src1 = f"{RES} sets.{s}.report.table1"

        # ---------------- headline analogue (DEV-TEST in S8; TEST-B is main-text Table 1, so only DEV-TEST here)
        if s == "devtest":
            L = []
            for a in ALL:
                x = t1[a]
                sd = "" if x.get("picked1_sd_pct") is None else f" $\\pm$ {f2(x['picked1_sd_pct'])}"
                de = x.get("delta_ours_minus_arm")
                dcell, tcell = ("---", "---") if de is None else (dl(de), ci(de["two_stage_pp"]))
                L.append(f"{LAB[a]} & {sel(x['selector'])} & {seeds(x['picked1_per_seed_pct'])} & "
                         f"{pc(x['picked1_mean_pct'])}{sd} & {dcell} & {tcell} & {pc(x['draw1_mean_pct'])} & "
                         f"{pc(x['any16_mean_pct'])} \\\\\n")
                if a == "OURS-N2":
                    L.append("\\midrule\n")
            rows("si_t1_devtest.tex", [f"{src1}.<arm>.(selector, picked1_per_seed_pct, picked1_mean_pct, picked1_sd_pct,"
                                       " delta_ours_minus_arm, draw1_mean_pct, any16_mean_pct)"], L)

        # ---------------- S2 secondary matchers
        L = []
        for a in ALL:
            x = t1[a]
            sc = x["secondary"]
            lab = LAB[a]
            cells = []
            for k in ("mcf05", "mcf08"):
                cells += [seeds(sc[k]["picked1_per_seed_pct"]), pc(sc[k]["ensemble_picked1_pct"])]
            L.append(f"{lab} & {sel(x['selector'])} & " + " & ".join(cells) + " \\\\\n")
            if a == "OURS-N2":
                L.append("\\midrule\n")
        rows(f"si_secondary_{s}.tex", [f"{src1}.<arm>.secondary.mcf05, .mcf08 (picked1_per_seed_pct, ensemble_picked1_pct)"], L)

        # ---------------- S3 ensembles and any@48
        L = []
        for a in ALL:
            x = t1[a]
            L.append(f"{LAB[a]} & {tex(x['ensemble_label'])} & {pc(x['picked1_mean_pct'])} & "
                     f"{pc(x['ensemble_picked1_pct'])} & {pc(x['any16_mean_pct'])} & {pc(x['any_all_pct'])} \\\\\n")
            if a == "OURS-N2":
                L.append("\\midrule\n")
        rows(f"si_ensemble_{s}.tex", [f"{src1}.<arm>.(ensemble_label, picked1_mean_pct, ensemble_picked1_pct,"
                                      " any16_mean_pct, any_all_pct)"], L)

        # ---------------- S3 continuous endpoint: RMS ECDF (seed mean of the per-seed ECDF)
        XS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        L, curves = [], {}
        for a in ALL:
            e = t1[a]["secondary"].get("mcf10")
            if not isinstance(e, dict) or "rms_ecdf" not in e:
                why = "missing$^{a}$" if e == "missing" else "no ECDF$^{b}$"
                L.append(f"{LAB[a]} & \\multicolumn{{{len(XS)}}}{{c}}{{{why}}} \\\\\n")
                continue
            ec = e["rms_ecdf"]
            mean = [sum(p[j] for p in ec["per_seed"]) / len(ec["per_seed"]) for j in range(len(ec["x"]))]
            curves[a] = list(zip(ec["x"], mean))
            idx = [min(range(len(ec["x"])), key=lambda j: abs(ec["x"][j] - xv)) for xv in XS]
            assert all(abs(ec["x"][j] - xv) < 1e-9 for j, xv in zip(idx, XS))
            L.append(f"{LAB[a]} & " + " & ".join(fr(mean[j]) for j in idx) + " \\\\\n")
            if a == "OURS-N2":
                L.append("\\midrule\n")
        rows(f"si_ecdf_{s}.tex", [f"{src1}.<arm>.secondary.mcf10.rms_ecdf (x, per_seed); value = mean over seeds of"
                                  " the per-seed ECDF, as a percentage"], L)
        if s == "testB":
            FIG = [("OURS", "black, very thick"), ("OURS-N2", "black, dashed, thick"), ("ii-S", "blue, thick"),
                   ("MCF-R+G", "red, thick"), ("MCF-R", "red, dashed, thick"), ("iv-P", "orange, thick"),
                   ("iii-u", "gray, thick"), ("i-E", "teal, thick"), ("iii-s", "violet, thick")]
            g = ["\\begin{tikzpicture}\n\\begin{axis}[width=0.78\\linewidth, height=7cm, xmin=0, xmax=1, ymin=0,"
                 " ymax=100, xlabel={normalised RMS displacement $x$ (\\texttt{get\\_rms\\_dist}, stol 1.0)},"
                 " ylabel={picked structures with RMS $\\le x$ (\\%)}, legend pos=outer north east,"
                 " legend cell align=left, tick label style={font=\\footnotesize}, label style={font=\\footnotesize},"
                 " legend style={font=\\footnotesize}]\n"]
            for a, style in FIG:
                pts = " ".join(f"({x:g},{100 * y:.4f})" for x, y in curves[a])
                g.append(f"\\addplot[{style}] coordinates {{{pts}}};\n\\addlegendentry{{{PLAIN[a]}}}\n")
            g.append("\\end{axis}\n\\end{tikzpicture}\n")
            out["si_ecdf_fig_testB.tex"] = hdr(f"{src1}.<arm>.secondary.mcf10.rms_ecdf (seed mean of per_seed)") + "".join(g)

        # ---------------- S3 geodesic error quantiles (R_asym arms)
        L = []
        for a in ["OURS", "OURS-N2", "ii-S", "iv-S", "iii-u", "iii-p", "i-E", "iii-s"]:
            gd = t1[a].get("geodesic_deg")
            if gd is None:
                continue
            ps = gd["per_seed"]
            for i, q in enumerate(ps):
                L.append(f"{LAB[a] if i == 0 else ''} & s{i} & " +
                         " & ".join(deg(q[k]) for k in ("q10", "q25", "median", "q75", "q90")) + " \\\\\n")
            if len(ps) > 1:
                q = gd["pooled"]
                L.append(" & pooled & " + " & ".join(deg(q[k]) for k in ("q10", "q25", "median", "q75", "q90")) +
                         " \\\\\n")
            L.append("\\addlinespace\n")
        rows(f"si_geodesic_{s}.tex", [f"{src1}.<arm>.geodesic_deg (per_seed, pooled; q10..q90, degrees)"], L[:-1])

        # (the forest plot of the TEST-B contrasts is the main-text figure scripts/fig_n3_forest.py; not repeated here)

        # ---------------- S4 ablations and descriptive arms
        L = []
        for a in ["ii-S", "iv-S", "i-E"]:
            x = t1[a]
            de = x["delta_ours_minus_arm"]
            L.append(f"{LAB[a]} & {sel(x['selector'])} & {seeds(x['picked1_per_seed_pct'])} & "
                     f"{pc(x['picked1_mean_pct'])} $\\pm$ {f2(x['picked1_sd_pct'])} & {pc(x['draw1_mean_pct'])} & "
                     f"{pc(x['any16_mean_pct'])} & {dl(de)} & {ci(de['two_stage_pp'])} \\\\\n")
        rows(f"si_ablation_{s}.tex", [f"{src1}.ii-S, .iv-S, .i-E"], L)

        # steric basin
        sb = rep["steric_basin"]
        L = []
        for a, lab in (("p1-OURS", "P1 of OURS"), ("p1-iii-p", "P1 of iii-p")):
            x = sb["arms"][a]
            L.append(f"{lab} & {sel(x['selector'])} & {seeds(x['picked1_per_seed_pct'])} & "
                     f"{pc(x['picked1_mean_pct'])} & & \\\\\n")
        c = sb["contrasts"]["p1-OURS - p1-iii-p"]
        L.append("\\midrule\n")
        L.append(f"P1 of OURS $-$ P1 of iii-p ($n={num(c['n'])}$) & & & & {dl(c)} & {ci(c['two_stage_pp'])} \\\\\n")
        rows(f"si_steric_{s}.tex", [f"{RES} sets.{s}.report.steric_basin (arms, contrasts)"], L)
        mac(f"svStericRefUnconv{tag}", refs(sb["reference"]["ref_unconverged_refcodes"]),
            f"{RES} sets.{s}.report.steric_basin.reference.ref_unconverged_refcodes")
        ctx = re.sub(r"\s*\(printed beside it, §3\.3\)$", "", sb["g2_devtest_context"])
        assert ctx != sb["g2_devtest_context"]
        mac(f"svGtwoContext{tag}", tex(ctx), f"{RES} sets.{s}.report.steric_basin.g2_devtest_context (its trailing"
            " cross-reference to the protocol dropped)")

        # ii-S decomposition (TEST-B; the DEV-TEST report has no wording block, by design)
        if "wording" in rep:
            dec = rep["wording"]["ii_S_decomposition"]
            tot = t1["MCF-R+G"]["delta_ours_minus_arm"]
            a_, b_ = dec["ii-S - MCF-R+G"], dec["OURS - ii-S"]
            assert abs(tot["delta_pp"] - (a_["delta_pp"] + b_["delta_pp"])) < 1e-9
            h1g = rep["family_primary"]["H1"]["components"]["MCF-R+G"]
            assert (h1g["delta_pp"], h1g["bca_pp"]) == (tot["delta_pp"], tot["bca_pp"])
            L = []
            for lab, x in (("OURS $-$ MCF-R+G (H1 component)", tot),
                           ("ii-S $-$ MCF-R+G (expansion of MCF's orientations)", a_),
                           ("OURS $-$ ii-S (learner; ablation estimate)", b_)):
                L.append(f"{lab} & {f2(x['delta_pp'], sign=True)} & {ci(x['bca_pp'])} & {ci(x['two_stage_pp'])} & "
                         f"{num(x['nonzero'])} \\\\\n")
            rows("si_decomp_testB.tex", [f"{src1}.MCF-R+G.delta_ours_minus_arm; {RES} sets.testB.report.wording."
                                         "ii_S_decomposition (identity checked to 1e-9 in the script)"], L)
            for x, nm, key in ((tot, "Tot", "table1.MCF-R+G.delta_ours_minus_arm"),
                               (a_, "Exp", "wording.ii_S_decomposition.'ii-S - MCF-R+G'"),
                               (b_, "Learn", "wording.ii_S_decomposition.'OURS - ii-S'")):
                mac(f"svDec{nm}", f2(x["delta_pp"], sign=True), f"{RES} sets.testB.report.{key}.delta_pp")
                mac(f"svDec{nm}CI", ci(x["bca_pp"]), f"{RES} sets.testB.report.{key}.bca_pp")
            mac("svDecompSum", f2(a_["delta_pp"] + b_["delta_pp"], sign=True),
                f"{RES} sets.testB.report.wording.ii_S_decomposition: (ii-S - MCF-R+G) + (OURS - ii-S), delta_pp")
            mac("svDecompLearner", tex(rep["wording"]["ii_S_decomposition"]["learner"]),
                f"{RES} sets.testB.report.wording.ii_S_decomposition.learner")
            mac("svZmcfMin", f2(rep["wording"]["zmcf_ge3_min_delta_pp"], sign=True),
                f"{RES} sets.testB.report.wording.zmcf_ge3_min_delta_pp")

        # H2 descriptive contrasts
        L = []
        ie = rep["i_e_vs_h2"]
        L.append(f"i-E $-$ iii-s & {tex(ie['form'])} & {num(ie['n'])} & {dl(ie)} & {ci(ie['two_stage_pp'])} & "
                 f"{num(ie['nonzero'])} \\\\\n")
        if "wording" in rep:
            for lv, x in rep["wording"]["H2_by_ff_label"].items():
                small = " (too small to interpret)" if x["n"] < 30 else ""
                L.append(f"OURS $-$ iii-s, FF$^\\ast$ truth {lv.lower()}{small} & EXACT & {num(x['n'])} & {dl(x)} & "
                         f"{ci(x['two_stage_pp'])} & {num(x['nonzero'])} \\\\\n")
        om = rep["h2_omc25_nonoverlap"]
        for k in ("OURS - iii-s", "OURS-N2 - iii-s"):
            x = om[k]
            L.append(f"{k.replace(' - ', ' $-$ ')}, OMC25 non-overlap & {tex(om['form'])} & {num(x['n'])} & {dl(x)} & "
                     f"{ci(x['two_stage_pp'])} & {num(x['nonzero'])} \\\\\n")
        hb = t1["iii-s"].get("h2_basin_delta")
        if hb:
            L.append(f"OURS $-$ iii-s, BASIN form & BASIN & {num(hb['n'])} & {dl(hb)} & {ci(hb['two_stage_pp'])} & "
                     f"{num(hb['nonzero'])} \\\\\n")
        rows(f"si_h2desc_{s}.tex", [f"{RES} sets.{s}.report.i_e_vs_h2; .wording.H2_by_ff_label (TEST-B only);"
                                    " .h2_omc25_nonoverlap; .table1.iii-s.h2_basin_delta"], L)
        L = []
        for a in ("OURS", "OURS-N2", "i-E", "iii-s"):
            x = t1[a]
            if "h2_basin_picked1_per_seed_pct" in x:
                L.append(f"{LAB[a]} & {seeds(x['h2_basin_picked1_per_seed_pct'])} & {pc(x['h2_basin_picked1_pct'])} & "
                         f"{pc(x['picked1_mean_pct'])} \\\\\n")
        rows(f"si_basin_{s}.tex", [f"{src1}.<arm>.(h2_basin_picked1_per_seed_pct, h2_basin_picked1_pct, picked1_mean_pct)"], L)
        hbr = rep["h2_basin_reference"]
        mac(f"svBasinUnconvN{tag}", num(len(hbr["ref_unconverged_refcodes"])),
            f"{RES} sets.{s}.report.h2_basin_reference.ref_unconverged_refcodes (count)")
        mac(f"svBasinUnconv{tag}", refs(hbr["ref_unconverged_refcodes"]),
            f"{RES} sets.{s}.report.h2_basin_reference.ref_unconverged_refcodes")
        mac(f"svBasinInvalid{tag}", refs(hbr["ref_invalid_refcodes"]),
            f"{RES} sets.{s}.report.h2_basin_reference.ref_invalid_refcodes")
        mac(f"svOmcN{tag}", num(om["n"]), f"{RES} sets.{s}.report.h2_omc25_nonoverlap.n")

        # ---------------- S5 MCF diagnostics (b)-(d) (approximate) and the MCF-A context table
        md = st["mcf_diagnostics"]
        assert md["approximate"] is True and md["reported"] is True
        ZB = [("z_MCF=1", "1"), ("z_MCF=2", "2"), ("z_MCF>=3", "$\\ge3$")]
        L = []
        for mk in ("primary", "mcf05", "mcf08"):
            for i, (zk, zl) in enumerate(ZB):
                x = md["b"]["summary"][mk][zk]
                L.append(f"{MATCHER[mk] if i == 0 else ''} & {zl} & {num(x['n'])} & {fr(x['mean_c_draw1_ceiling'])} & "
                         f"{fr(x['mean_any16_ceiling'])} & {tex(x['label'])} \\\\\n")
            L.append("\\addlinespace\n")
        rows(f"si_diagb_{s}.tex", [f"{RES} sets.{s}.mcf_diagnostics.b.summary.<matcher>.<z_MCF> (n, mean_c_draw1_ceiling,"
                                   " mean_any16_ceiling, label); fractions shown as percentages"], L[:-1])
        mac(f"svDiagBNote{tag}", tex(md["b"]["note"]), f"{RES} sets.{s}.mcf_diagnostics.b.note")
        mac(f"svDiagBGrid{tag}", str(md["b"]["grid_deg"]), f"{RES} sets.{s}.mcf_diagnostics.b.grid_deg")
        mac(f"svDiagBWalk{tag}", str(md["b"]["walk_deg"]), f"{RES} sets.{s}.mcf_diagnostics.b.walk_deg")
        L = []
        for mk in ("primary", "mcf05", "mcf08"):
            blk = md["c"]["matchers"][mk]
            for i, (zk, zl) in enumerate((("z_MCF = 2", "2"), ("z_MCF >= 3", "$\\ge3$"))):
                x = blk[zk]
                L.append(f"{MATCHER[mk] if i == 0 else ''} & {zl} & {num(x['n'])} & {seeds(x['draw1_per_seed_pct'])} & "
                         f"{pc(x['draw1_mean_pct'])} & {pc(x['any16_mean_pct'])} & {seeds(x['picked1_per_seed_pct'])} & "
                         f"{pc(x['picked1_mean_pct'])} \\\\\n")
            L.append("\\addlinespace\n")
        rows(f"si_diagc_{s}.tex", [f"{RES} sets.{s}.mcf_diagnostics.c.matchers.<matcher>.<z_MCF> (selector "
                                   f"{md['c']['selector']})"], L[:-1])
        z1 = md["c"]["z_MCF = 1"]
        mac(f"svDiagCZoneN{tag}", num(z1["n"]), f"{RES} sets.{s}.mcf_diagnostics.c.'z_MCF = 1'.n")
        L = []
        for mk in ("primary", "mcf05", "mcf08"):
            blk = md["d"]["matchers"][mk]
            for i, (zk, zl) in enumerate((("z_MCF=2", "2"), ("z_MCF>=3", "$\\ge3$"))):
                x = blk["strata"][zk]
                L.append(f"{MATCHER[mk] if i == 0 else ''} & {zl} & {num(x['n'])} & {fr(x['draw1'])} & {wil(x['ci95'])} & "
                         f"{fr(x['ceiling_mean_c'])} & {'yes' if x['exceeds_ceiling'] else 'no'} \\\\\n")
            L.append("\\addlinespace\n")
        rows(f"si_diagd_{s}.tex", [f"{RES} sets.{s}.mcf_diagnostics.d.matchers.<matcher>.strata (n, draw1, ci95,"
                                   " ceiling_mean_c, exceeds_ceiling)"], L[:-1])
        mac(f"svDiagDConsistent{tag}", "yes" if md["d_consistent_all"] else "no",
            f"{RES} sets.{s}.mcf_diagnostics.d_consistent_all")
        L = []
        for a in ("MCF-A", "MCF-A-floor"):
            x = t1[a]
            for i, mk in enumerate(("primary", "mcf05", "mcf08", "mcf10")):
                if mk == "primary":
                    ps, en, d1, a16 = (x["picked1_per_seed_pct"], x["ensemble_picked1_pct"], pc(x["draw1_mean_pct"]),
                                       pc(x["any16_mean_pct"]))
                else:
                    sc = x["secondary"][mk]
                    ps, en, d1, a16 = sc["picked1_per_seed_pct"], sc["ensemble_picked1_pct"], NR, NR
                L.append(f"{LAB[a] if i == 0 else ''} & {sel(x['selector']) if i == 0 else ''} & {MATCHER[mk]} & "
                         f"{seeds(ps)} & {pc(en)} & {d1} & {a16} \\\\\n")
            L.append("\\addlinespace\n")
        rows(f"si_mcfa_{s}.tex", [f"{src1}.MCF-A, .MCF-A-floor (picked1_per_seed_pct, ensemble_picked1_pct, draw1_mean_pct,"
                                  " any16_mean_pct, secondary.mcf05/mcf08/mcf10)"], L[:-1])

        # ---------------- S6 / S8 Table 2 detail (strata)
        t2 = rep["table2"]
        FAM = [("K", "$K$"), ("z_mcf", "$z_\\mathrm{MCF}$"), ("class", "space-group class"), ("chiral", "chiral conformer"),
               ("mixed", "mixed handedness"), ("improper_achiral", "improper, achiral"),
               ("legacy_unrep", "legacy-unrepresentable"), ("formula_disjoint", "formula-disjoint"),
               ("omc25_nonoverlap", "OMC25 non-overlap"), ("ff_label", "FF$^\\ast$ truth label")]
        SENS = [(f"sensitivity.{dfn}.{p}", f"{lab}, {nm}") for dfn, nm in
                (("parser>0.3", "parser-capped definition (max\\_iso 200), 0.3~\\AA"),
                 ("refined>0.1", "refined definition, 0.1~\\AA"), ("refined>0.5", "refined definition, 0.5~\\AA"))
                for p, lab in (("chiral", "chiral conformer"), ("mixed", "mixed handedness"),
                               ("improper_achiral", "improper, achiral"))]
        assert set(t2) == {k for k, _ in FAM + SENS}, set(t2) ^ {k for k, _ in FAM + SENS}
        ORDER = ["<=2", "3-4", ">=6", "1", "2", ">=3", "centro", "sohncke", "other", "True", "False",
                 "ANCHORED", "DRIFTED", "UNRESOLVED"]
        PRET = {"<=2": "$\\le2$", "3-4": "3--4", ">=6": "$\\ge6$", ">=3": "$\\ge3$", "True": "yes", "False": "no",
                "centro": "centrosymmetric", "sohncke": "Sohncke", "other": "other", "ANCHORED": "anchored",
                "DRIFTED": "drifted", "UNRESOLVED": "unresolved"}
        ZMAP = {"1": "z_MCF=1", "2": "z_MCF=2", ">=3": "z_MCF>=3"}
        ceil = md["b"]["summary"]["primary"]

        def strata_detail(fams):
            Ld = []
            for key, name in fams:
                for lv in sorted(t2[key], key=ORDER.index):
                    cell = t2[key][lv]
                    note = "; too small to interpret" if cell["too_small"] or cell["n"] < 30 else ""
                    if key == "z_mcf" and ZMAP[lv] in ceil:
                        cz = ceil[ZMAP[lv]]
                        lab = ("Haar floor" if lv == "1" else "draw@1 ceiling, oracle, approximate")
                        note += f"; MCF diagnostic (b), primary matcher: {lab} {fr(cz['mean_c_draw1_ceiling'])}\\%"
                    Ld.append(f"\\multicolumn{{7}}{{@{{}}p{{0.92\\linewidth}}}}{{\\textit{{{name}: {PRET.get(lv, tex(lv))}}}, "
                              f"$n={num(cell['n'])}${note}}} \\\\\n")
                    for a in ALL:
                        if a not in cell["arms"]:
                            continue
                        x = cell["arms"][a]
                        ws = [wil(w) for w in x["wilson_per_seed"]] + [""] * (3 - len(x["wilson_per_seed"]))
                        Ld.append(f"\\quad {LAB[a]} & {seeds(x['k_per_seed'], cnt)} & {ws[0]} & {ws[1]} & {ws[2]} & "
                                  f"{cnt(x['k_ens'])} & {wil(x['wilson_ens'])} \\\\\n")
                    Ld.append("\\addlinespace\n")
            return Ld[:-1]

        if s == "testB":
            src2 = (f"{RES} sets.testB.report.table2.<stratum>.<level> (n, too_small, arms.<arm>.k_per_seed, "
                    "wilson_per_seed, k_ens, wilson_ens; Wilson fractions shown as percentages); "
                    f"{RES} sets.testB.mcf_diagnostics.b.summary.primary (z_MCF ceilings)")
            rows("si_strata_testB.tex", [src2], strata_detail(FAM))
            rows("si_strata_sens_testB.tex", [src2], strata_detail(SENS))
        # seed-mean summary (DEV-TEST continuity table, S8)
        if s == "devtest":
            Ls = []
            for key, name in FAM + SENS:
                for i, lv in enumerate(sorted(t2[key], key=ORDER.index)):
                    cell = t2[key][lv]
                    vals = []
                    for a in T1:
                        if a not in cell["arms"]:
                            vals.append("")
                            continue
                        k = cell["arms"][a]["k_per_seed"]
                        vals.append(pc(100 * sum(k) / len(k) / cell["n"]))
                    flag = "$^{*}$" if cell["too_small"] or cell["n"] < 30 else ""
                    Ls.append(f"{name if i == 0 else ''} & {PRET.get(lv, tex(lv))}{flag} & {num(cell['n'])} & " +
                              " & ".join(vals) + " \\\\\n")
                Ls.append("\\addlinespace\n")
            rows("si_strata_devtest.tex", [f"{RES} sets.devtest.report.table2.<stratum>.<level>.arms.<arm>.k_per_seed;"
                                           " value = mean_seeds(k)/n, as a percentage"], Ls[:-1])

        # ---------------- S10 run health per set
        L = []
        for a in ALL:
            x = t1[a]
            miss = sorted({r for lst in x["arm_miss_refcodes_per_seed"] for r in lst} | set(x["arm_miss_refcodes"]))
            en = x["energy_det_fail_refcodes_per_seed"]
            if all(e is None for e in en):
                ecell = NR
            else:
                ecell = " / ".join(refs(e) for e in en)
            L.append(f"{LAB[a]} & {refs(miss)} & {num(x['det_fail_fits'])} & {refs(x['det_fail_list'])} & {ecell} & "
                     f"{num(x['arm_fail_draws'])} \\\\\n")
        rows(f"si_fail_{s}.tex", [f"{src1}.<arm>.(arm_miss_refcodes, arm_miss_refcodes_per_seed, det_fail_fits, "
                                  "det_fail_list, energy_det_fail_refcodes_per_seed, arm_fail_draws)"], L)
        mac(f"svIiisArmFail{tag}", num(t1["iii-s"]["arm_fail_draws"]), f"{src1}.iii-s.arm_fail_draws")
        o = OM[s]
        assert o["n"] == n
        L = [f"{tex(c['refcode'])} & {', '.join(c['elements_outside'])} \\\\\n" for c in o["crystals"]]
        rows(f"si_omc25_{s}.tex", [f"{OMC[s]} crystals (refcode, elements_outside)"], L)
        mac(f"svOmcOutside{tag}", num(o["n_outside"]), f"{OMC[s]} n_outside")
        if s == "testB":
            mac("svOmcElements", ", ".join(o["omc25_element_set"]), f"{OMC[s]} omc25_element_set")

        # ---------------- S11 compute
        L = []
        for a in ALL:
            c = t1[a]["compute"]
            sec = c["sec_per_crystal_per_seed_median"]
            L.append(f"{LAB[a]} & {NR if sec is None else rnd(sec, 2)} & "
                     f"{NR if c['lj_evals_per_crystal'] is None else num(c['lj_evals_per_crystal'])} & "
                     f"{NR if c['ff_evals_per_crystal'] is None else num(c['ff_evals_per_crystal'])} & {NR} \\\\\n")
        rows(f"si_compute_{s}.tex", [f"{src1}.<arm>.compute (sec_per_crystal_per_seed_median, lj_evals_per_crystal, "
                                     "ff_evals_per_crystal); training cost: no key in any named record -> 'not recorded'"], L)
    oc = sets["testB"]["report"]["table1"]["OURS"]["compute"]
    mac("svOursSec", rnd(oc["sec_per_crystal_per_seed_median"], 2),
        f"{RES} sets.testB.report.table1.OURS.compute.sec_per_crystal_per_seed_median")
    mac("svOursLj", num(oc["lj_evals_per_crystal"]), f"{RES} sets.testB.report.table1.OURS.compute.lj_evals_per_crystal")

    # ======================================================================== S5 diagnostic (a)
    da = d["mcf_diagnostic_a"]
    assert da["verdict"] == "PASS" and d["mcf_diagnostic_a_passed"] is True
    CK = {"mcfR_s0": "MCF-R s0", "mcfR_s1": "MCF-R s1", "mcfR_s2": "MCF-R s2", "mcfRG_s0": "MCF-R+G s0",
          "mcfA_s0": "MCF-A s0", "mcfA_s1": "MCF-A s1", "mcfA_s2": "MCF-A s2", "thurlemann": "Thurlemann (provenance)"}
    tols = {tuple(sorted(v["tolerances"].items())) for v in da["per_checkpoint"].values()}
    assert len(tols) == 1
    tol = da["per_checkpoint"]["mcfR_s0"]["tolerances"]
    L1, L2 = [], []
    for k, lab in CK.items():
        x = da["per_checkpoint"][k]
        kn = x["knobs"]
        knob = f"{cnt(kn['s_uR'])}" + (f", {cnt(kn['s_uF'])}" if k.startswith("mcfA") or k == "thurlemann" else "")
        stt = x["statistics"]
        assert all(x["requirements_met"].values()) == (x["verdict"] == "PASS")
        L1.append(f"{lab} & {knob} & {x['verdict']} & " + " & ".join(sci(stt[q]) for q in ("q", "lat", "tr", "R")) +
                  " & " + " & ".join(g3(x["controls_min"][q]) for q in ("q", "lat", "tr", "R")) + " \\\\\n")
        sd = x["sampler_deviation"]
        cb = x["corrupt_batch_state"]
        bc = cb["branch_cut_edges"]
        L2.append(f"{lab} & {g3(sd['float32_native']['R'])} & {g3(sd['float64_replica']['R'])} & "
                  f"{'no' if x['sampler_deviation_threshold_1e-3']['float32_native'] else 'yes'} & "
                  f"{tex(cb['verdict'])} & {bc['on_branch_cut']}/{bc['edges']} \\\\\n")
    rows("si_diaga_stats.tex", [f"{RES} mcf_diagnostic_a.per_checkpoint.<ckpt>.(knobs, verdict, statistics, controls_min)"], L1)
    rows("si_diaga_sampler.tex", [f"{RES} mcf_diagnostic_a.per_checkpoint.<ckpt>.(sampler_deviation, "
                                  "sampler_deviation_threshold_1e-3 [true = within 1e-3], corrupt_batch_state)"], L2)
    for q, nm in (("q", "Q"), ("lat", "Lat"), ("tr", "Tr"), ("R", "R")):
        mac(f"svTol{nm}", sci(tol[q]), f"{RES} mcf_diagnostic_a.per_checkpoint.mcfR_s0.tolerances.{q} (identical for all)")
    mac("svDiagAVerdict", da["verdict"], f"{RES} mcf_diagnostic_a.verdict")
    mac("svDiagAApproxN", num(len(da["approximate_checkpoints"])), f"{RES} mcf_diagnostic_a.approximate_checkpoints (count)")
    mac("svDiagAApprox", ", ".join(CK[c] for c in da["approximate_checkpoints"]),
        f"{RES} mcf_diagnostic_a.approximate_checkpoints")
    mac("svThreads", num(da["per_checkpoint"]["mcfR_s0"]["torch_threads"]), f"{RES} mcf_diagnostic_a.per_checkpoint.*.torch_threads")

    # MCF fork: P1 maps patched on every exported set (Gate F)
    pm = {k: all(v["p1_maps_patched"].values()) for k, v in GF["sets"].items()}
    assert all(pm.values())
    mac("svForkSets", ", ".join(SETNAME.get(k, k.upper()) for k in GF["sets"]), f"{GATEF} sets.<set>.p1_maps_patched")
    mac("svMcfBase", tex(GF["sets"]["testB"]["mcf_base"]), f"{GATEF} sets.testB.mcf_base")

    # ======================================================================== S7 Table R detail
    RN = {"all": "all", "centro": "centrosymmetric", "sohncke": "Sohncke", "other": "other",
          "mixed": "mixed handedness", "improper_achiral": "improper, achiral", "legacy_unrep": "legacy-unrepresentable"}
    La, Lc = [], []
    for s in ("testB", "devtest"):
        tr = TR[s]
        assert tr["wording"]["final"] and tr["complete"]["all"] and not tr["partial"]
        head = SETNAME[s] + ("" if s == "testB" else " (used during development, not confirmatory)")
        La.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\\n")
        Lc.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\\n")
        for k, nm in RN.items():
            r = tr["rows"][k]
            m3 = lambda b: " & ".join(num(b[m]) for m in ("primary", "stol0.5", "stol0.8"))  # noqa: E731
            La.append(f"\\quad {nm} & {num(r['n'])} & {m3(r['a'])} & {m3(r['b'])} & {num(r['d']['flag_kept'])} & "
                      f"{m3(r['d']['oracle'])} \\\\\n")
            Lc.append(f"\\quad {nm} & {num(r['n'])} & {m3(r['c'])} & {m3(r['c_frozen16'])} & "
                      f"{num(r['c']['from_a_no_improper_op'])} & {num(r['c'].get('multiple_optima', 0))} \\\\\n")
        La.append("\\addlinespace\n")
        Lc.append("\\addlinespace\n")
        tag = "" if s == "testB" else "Dev"
        w = tr["wording"]
        for key, nm in (("N_legacy_unrep", "N"), ("M_legacy_unrep_centro", "M"), ("N_flag_or_oracle_miss", "NOracle"),
                        ("M_flag_or_oracle_miss", "MOracle"), ("n_mixed", "Mixed"), ("P_mixed_not_matched_primary", "P"),
                        ("Q_mixed_matched_stol0.5", "Q"), ("P_frozen16", "PSixteen"), ("Q_frozen16", "QSixteen"),
                        ("mixed_c_unresolved", "Unres"), ("oracle_unresolved_flag_kept", "OracleUnres")):
            mac(f"svTR{nm}{tag}", num(w[key]), f"{TABR[s]} wording.{key}")
        mac(f"svTRText{tag}", tex(w["text"]), f"{TABR[s]} wording.text")
        mac(f"svTRTextSixteen{tag}", tex(w["text_frozen16"]), f"{TABR[s]} wording.text_frozen16")
        cd = tr["c_diagnostics"]
        mac(f"svTRImproperOk{tag}", num(cd["n_improper_ok"]), f"{TABR[s]} c_diagnostics.n_improper_ok")
        mac(f"svTRAutoImproved{tag}", num(cd["improved_by_automorphism_start_1e-6"]),
            f"{TABR[s]} c_diagnostics.improved_by_automorphism_start_1e-6")
        mac(f"svTRChangedHalf{tag}", num(cd["outcome_changed_vs_frozen16"]["stol0.5"]),
            f"{TABR[s]} c_diagnostics.outcome_changed_vs_frozen16.stol0.5")
        mac(f"svTRChangedPrimary{tag}", num(cd["outcome_changed_vs_frozen16"]["primary"]),
            f"{TABR[s]} c_diagnostics.outcome_changed_vs_frozen16.primary")
        mac(f"svTRMaxGain{tag}", g3(cd["max_rmsd_gain_A"]), f"{TABR[s]} c_diagnostics.max_rmsd_gain_A")
        mac(f"svTRRmsdMedMixed{tag}", rnd(cd["homochiral_rmsd"]["median_mixed"], 2),
            f"{TABR[s]} c_diagnostics.homochiral_rmsd.median_mixed")
        bad = sum(tr["rows"][k][col].get(f, 0) for k in RN for col in ("a", "b", "c", "c_frozen16")
                  for f in ("timeout", "deterministic", "error", "missing"))
        bad += sum(tr["rows"][k]["d"][f] for k in RN for f in ("oracle_timeout", "oracle_deterministic", "oracle_error",
                                                               "oracle_missing", "oracle_fresh_registry_rejected"))
        mac(f"svTRBad{tag}", num(bad), f"{TABR[s]} rows.*.{{a,b,c,c_frozen16}}.{{timeout,deterministic,error,missing}} "
                                       "+ rows.*.d.oracle_* (sum)")
        mac(f"svTRByConstruction{tag}", "yes" if tr["a_by_construction"] else "no (computed)",
            f"{TABR[s]} a_by_construction")
        mac(f"svTRTimeout{tag}", rnd(tr["timing_s"]["timeout_per_job"], 0), f"{TABR[s]} timing_s.timeout_per_job")
    rows("si_tabler_abd.tex", [f"{TABR['testB']}, {TABR['devtest']} rows.<row>.(n, a, b, d.flag_kept, d.oracle)"], La[:-1])
    rows("si_tabler_c.tex", [f"{TABR['testB']}, {TABR['devtest']} rows.<row>.(n, c, c_frozen16, "
                             "c.from_a_no_improper_op, c.multiple_optima)"], Lc[:-1])
    bc = TR["testB"]["corpus_context"]["pool"]["build_counts"]
    assert bc == TR["testB"]["build_step2"] | {k: v for k, v in bc.items() if k not in TR["testB"]["build_step2"]}
    L = [f"pool CIFs & {num(bc['pool_cifs'])} \\\\\n", f"kept by the O(3) parser & {num(bc['o3_kept'])} \\\\\n",
         f"after the G1 oracle & {num(bc['after_g1_oracle'])} \\\\\n"]
    for k, v in bc["step2_drops"].items():
        L.append(f"\\quad build step 2 removes: {tex(k)} & $-${num(v)} \\\\\n")
    L += [f"after build step 2 & {num(bc['after_step2'])} \\\\\n",
          f"\\quad build step 3 removes: family duplicates & $-${num(bc['step3_family_drops'])} \\\\\n",
          f"\\quad build step 3 removes: WL-species duplicates & $-${num(bc['step3_wl_drops'])} \\\\\n",
          f"after build step 3 & {num(bc['after_step3'])} \\\\\n",
          f"after family de-duplication & {num(bc['P_after_family_dedup'])} \\\\\n",
          f"SEL / TEST-B drawn & {num(bc['SEL'])} / {num(bc['TEST_B'])} \\\\\n"]
    rows("si_build.tex", [f"{TABR['testB']} corpus_context.pool.build_counts (= build_step2 for the step-2 entries)"], L)
    dv = TR["testB"]["corpus_context"]["dev"]
    for key, nm in (("cifs", "Cifs"), ("kept_det1", "DetOne"), ("kept_o3", "OThree"), ("eligible_g2_asym", "Eligible"),
                    ("o3_kept_not_eligible", "NotEligible")):
        mac(f"svDev{nm}", num(dv[key]), f"{TABR['testB']} corpus_context.dev.{key}")
    mac("svPoolDetOne", num(TR["testB"]["corpus_context"]["pool"]["kept_det1"]), f"{TABR['testB']} corpus_context.pool.kept_det1")

    # ======================================================================== S9 VALSEL grids
    V = S["valsel_cell_scores"]
    TIEW = {"hits": "seed-summed hits", "selector": "selector preference", "draw1": "draw@1", "any16": "any@16",
            "prefer_cell": "preferred cell", "picked": "VALSEL picked@1", "val_loss": "fixed-draw VAL loss",
            "any10@mcf10": "any-of-10 at stol 1.0", "any10@mcf08": "any-of-10 at stol 0.8",
            "picked@mcf10": "picked@1 at stol 1.0"}

    def tiew(lst, last=None):
        w = [TIEW[t] for t in lst]
        if last:
            w[-1] = last
        return " $\\rightarrow$ ".join(w)

    def cellname(c):
        m = re.fullmatch(r"u(\d+)", c)
        if m:
            return f"$s_{{uR}}={m.group(1)}$"
        m = re.fullmatch(r"f(\d+)_u(\d+)", c)
        if m:
            return f"$(s_{{uF}},s_{{uR}})=({m.group(1)},{m.group(2)})$"
        return tex(c)

    GA = [("iii-u", "iii-u: selector"), ("iii-p", "iii-p: selector"), ("p1-ours", "steric basin P1 of OURS: selector"),
          ("p1-iii-p", "steric basin P1 of iii-p: selector"), ("mcfR_A", "MCF-R Stage A: $s_{uR}$ and selector"),
          ("mcfRG_A", "MCF-R+G Stage A: $s_{uR}$ and selector"), ("iiS", "ii-S: selector"),
          ("mcfA_2", "MCF-A Stage 2: $(s_{uF},s_{uR})$"), ("mcfA_3", "MCF-A Stage 3: selector"),
          ("ivP", "iv-P Stage A: selector"), ("ivS", "iv-S: selector")]
    LA, LS = [], []
    for key, title in GA:
        g = V[key]["stage_a"]
        crit = g["tie"][0]
        ch = g["chosen"]
        LA.append(f"\\multicolumn{{7}}{{@{{}}p{{0.92\\linewidth}}}}{{\\textit{{{title}}}; score = {TIEW[crit]}; ties: {tiew(g['tie'])}; "
                  f"decided by {TIEW[g['decided_by']]}}} \\\\\n")
        for r in g["table"]:
            score = r.get(crit)
            per = r.get(f"{crit}_per_seed") or r.get("hits_per_seed")
            mark = "chosen" if (r["cell"] == ch["cell"] and r["selector"] == ch["selector"]) else ""
            pref = " (pref.)" if r.get("preferred") else ""
            cname = cellname(r["cell"]) if len({q["cell"] for q in g["table"]}) > 1 else "---"
            LA.append(f"\\quad {cname}{pref} & {sel(r['selector'])} & {cnt(r.get('draw1'))} & "
                      f"{cnt(r.get('any16'))} & {cnt(score)} & {seeds(per, cnt) if per else '---'} & {mark} \\\\\n")
        LA.append("\\addlinespace\n")
        multi = len({r["cell"] for r in g["table"]}) > 1
        chosen = ", ".join(([cellname(ch["cell"])] if multi else []) + ([sel(ch["selector"])] if ch["selector"] else []))
        LS.append(f"{title} & {chosen} & {TIEW[g['decided_by']]} \\\\\n")
    rows("si_sel_stagea.tex", [f"{SELF} valsel_cell_scores.<grid>.stage_a (tie, table, chosen, decided_by); counts are "
                               "summed over the 3 seeds on the 400 VALSEL crystals"], LA[:-1])
    nv_rows = {r["n"] for key, _ in GA for r in V[key]["stage_a"]["table"]}
    assert len(nv_rows) == 1
    mac("svValselN", num(nv_rows.pop()), f"{SELF} valsel_cell_scores.<grid>.stage_a.table[*].n")
    GB = [("mcfR_B", "MCF-R Stage B: checkpoint", ("picked", "draw1", "any16"), "fixed-draw VAL loss"),
          ("mcfRG_B", "MCF-R+G Stage B: checkpoint", ("picked", "draw1", "any16"), "fixed-draw VAL loss"),
          ("mcfA_1", "MCF-A Stage 1: checkpoint", ("any10@mcf10", "any10@mcf08"), "MCF's logged validation loss"),
          ("ivP", "iv-P Stage B: checkpoint", ("picked", "draw1", "any16"), "fixed-draw VAL loss")]
    LB = []
    for key, title, crits, vl in GB:
        g = V[key]["stage_b"]
        assert list(g["tie"]) == list(crits) + ["val_loss"], (key, g["tie"])
        LB.append(f"\\multicolumn{{6}}{{@{{}}p{{0.92\\linewidth}}}}{{\\textit{{{title}}}; ties: {tiew(g['tie'], vl)}; "
                  f"c1--c3 = {', '.join(TIEW[c] for c in crits)}}} \\\\\n")
        dby = []
        for sd, y in sorted(g["seeds"].items()):
            for r in y["table"]:
                cs = [cnt(r.get(c)) for c in crits] + [""] * (3 - len(crits))
                mark = "chosen" if r["label"] == y["chosen"] else ""
                LB.append(f"\\quad s{sd}: {tex(r['label'])} & " + " & ".join(cs) + f" & {dec6(r['val_loss'])} & {mark} \\\\\n")
            word = vl if y["decided_by"] == "val_loss" else TIEW[y["decided_by"]]
            dby.append(f"s{sd} {tex(y['chosen'])} ({word})")
        LB.append("\\addlinespace\n")
        LS.append(f"{title} & {'; '.join(dby)} & per seed \\\\\n")
    rows("si_sel_stageb.tex", [f"{SELF} valsel_cell_scores.<grid>.stage_b.seeds.<s> (table, chosen, decided_by)"], LB[:-1])
    # iv-P learning-rate window
    w = V["ivP_window"]
    L = []
    for lr, v in w["lr_min_val_in_window"].items():
        L.append(f"{tex(lr.replace('lr', ''))} & {dec6(v)} & {'yes' if w['improving_seed0'][lr] else 'no'} & "
                 f"{'chosen' if float(lr[2:]) == w['lr'] else ''} \\\\\n")
    rows("si_sel_ivp_lr.tex", [f"{SELF} valsel_cell_scores.ivP_window (lr_min_val_in_window, improving_seed0, lr)"], L)
    mac("svIvPWindow", num(w["window"]), f"{SELF} valsel_cell_scores.ivP_window.window")
    mac("svIvPLate", "no" if not w["late_extension"] else "yes", f"{SELF} valsel_cell_scores.ivP_window.late_extension")
    fz = S["frozen"]
    LS.insert(0, f"OURS, OURS-N2 & {sel(fz['OURS']['selector'])}, $S={fz['OURS']['S']}$, "
                 f"{fz['OURS']['euler_steps']} Euler steps & {tex(fz['OURS']['frozen'])} \\\\\n")
    LS.insert(1, f"iii-s & {sel('ff')} & {tex(fz['iii-s']['selector'])} \\\\\n")
    LS.append(f"iv-P learning rate & {tex(fz['iv-P']['lr'])} & lowest seed-0 fixed-draw VAL loss in epochs "
              f"1--{num(w['window'])} \\\\\n")
    rows("si_sel_summary.tex", [f"{SELF} frozen.<arm>; valsel_cell_scores.<grid>.stage_a.chosen/decided_by; "
                                "stage_b.seeds.<s>.chosen/decided_by; ivP_window"], LS)

    # ======================================================================== S10 run health
    m, c1 = R.find(r"\*\*FF\* \u00a73\.3 retries:\*\* ([\d,]+) in total, only in iii-s: 7S (\d+), 7Sd (\d+)\. All were CUDA"
                   r" OOM in (\d+)-structure batches,\s+and all were `retry_ok`\. (\d+) deterministic FF\* failures in any run")
    tot, t7s, t7sd, bsz, ndet = m.groups()
    assert int(tot.replace(",", "")) == int(t7s) + int(t7sd)
    mac("svFFRetryTotal", num(int(tot.replace(",", ""))), c1)
    mac("svFFRetryT", num(int(t7s)), c1 + " (7S = iii-s/testB)")
    mac("svFFRetryD", num(int(t7sd)), c1 + " (7Sd = iii-s/devtest)")
    mac("svFFRetryBatch", bsz, c1)
    mac("svFFDet", ndet, c1)
    m, c2 = R.find(r"The (\d+) MCF inference calls \(MCF-R (\d+), MCF-A (\d+), plus the (\d+) extra \+G checkpoint infers"
                   r" at s0\) all have\s+`check_pass: true`, (\d+) non-finite draws and finite residuals\.")
    mac("svMcfCalls", m.group(1), c2)
    mac("svMcfNonfiniteDraws", m.group(5), c2)
    m, c3 = R.find(r"`branch_cut_draws` is 0 in (\d+) of them and (\d+) in MCF-R/testB/s0 \(`(\w+)`, (\d+) of ([\d,]+)"
                   r" draws\)")
    mac("svBranchClean", m.group(1), c3)
    mac("svBranchDraws", m.group(4), c3)
    mac("svBranchOf", num(int(m.group(5).replace(",", ""))), c3)
    mac("svBranchCkpt", tex(m.group(3)), c3)
    m, c4 = R.find(r"All (\d+) LJ bundles \(MCF-R/A/floor\): `valid` = n x 16, `det_fail` (\d+), `nonfinite` (\d+)\.")
    mac("svLJBundles", m.group(1), c4)
    mac("svLJDetFail", m.group(2), c4)
    mac("svLJNonfinite", m.group(3), c4)
    m, c5 = R.find(r"It wrote `draws/iii-s_s-1_val\.pt`: (\d+) MISS crystals, (\d+) deterministic FF\* failures, (\d+)"
                   r" retries; X-H bonds\s+normalised C-H ([\d,]+), N-H (\d+), O-H (\d+) \((\d+) shortened\)\.")
    mac("svXHCH", num(int(m.group(4).replace(",", ""))), c5)
    mac("svXHNH", m.group(5), c5)
    mac("svXHOH", m.group(6), c5)
    mac("svXHShort", m.group(7), c5)
    m, c6 = R.find(r"`ie` on (\d+) VAL crystals \(0 MISS; C-H normalised (\d+)\)")
    mac("svXHIeN", m.group(1), c6)
    mac("svXHIeCH", m.group(2), c6)

    # FF* truth anchoring (VALSEL from the H2-form block; TEST-B and DEV-TEST from each set)
    L = []
    hf = d["h2_form"]
    nv = sum(hf["labels"].values())
    L.append(f"VALSEL (fixes the H2 form) & {num(nv)} & {fr(hf['anchoring'])} & " +
             " & ".join(num(hf["labels"][k]) for k in ("ANCHORED", "DRIFTED", "UNRESOLVED")) +
             f" & {num(hf['no_minimum'])} & {rnd(hf['median_drift_deg'], 2)} \\\\\n")
    for s in ("testB", "devtest"):
        a = sets[s]["ff_truth_anchoring"]
        assert sum(a["labels"].values()) == a["n"] and not a["partial"]
        L.append(f"{SETNAME[s]}{'' if s == 'testB' else ' (not confirmatory)'} & {num(a['n'])} & {fr(a['anchoring'])} & " +
                 " & ".join(num(a["labels"][k]) for k in ("ANCHORED", "DRIFTED", "UNRESOLVED")) +
                 f" & {num(a['no_minimum'])} & {rnd(a['median_drift_deg'], 2)} \\\\\n")
    rows("si_anchor.tex", [f"{RES} h2_form (anchoring, labels, no_minimum, median_drift_deg; n = sum of labels); "
                           f"sets.<set>.ff_truth_anchoring"], L)
    mac("svAnchValOnly", fr(hf["anchoring_val_only"]), f"{RES} h2_form.anchoring_val_only")
    mac("svFairchem", tex(hf["fairchem_core"]), f"{RES} h2_form.fairchem_core")
    mac("svFFSettings", tex(f"{hf['ffident']['settings']}, {hf['ffident']['dtype']}, {hf['ffident']['device']}"),
        f"{RES} h2_form.ffident")

    # gates F and C
    L = []
    for k, g in GF["sets"].items():
        f1, f2_, f3, f4, f5 = g["F1"], g["F2"], g["F3"], g["F4"], g["F5"]
        f5c = (f"{num(f5['le_1e-4'])}/{num(g['n'])}" + ("" if f5["gated"] else
               f", not gated ({num(f5['fail_explained_by_energy_image_shell'])} explained by the energy image shell)"))
        L.append(f"{SETNAME.get(k, k.upper())} & {num(g['n'])} & {num(f1['n_exported'])} & {num(f2_['match_primary'])} & "
                 f"{num(f3['chi_parity_agree_up_to_relabel'])} & {refs(f4['determine_cap_hit_refcodes'])} & {f5c} & "
                 f"{'pass' if g['gate_pass'] else 'FAIL'} \\\\\n")
    rows("si_gatef.tex", [f"{GATEF} sets.<set>.(n, F1.n_exported, F2.match_primary, F3.chi_parity_agree_up_to_relabel, "
                          "F4.determine_cap_hit_refcodes, F5.le_1e-4/gated/fail_explained_by_energy_image_shell, gate_pass)"], L)
    assert GF["all_pass"] and GC["all_pass"] and d["gate_c"]["C-a"]["pass"]
    L = []
    for lk, nm in (("test", "MCF TEST split"), ("val", "MCF VAL split")):
        g = GC["C-a"]["labels"][lk]
        for mk, stl in (("mcf09", "0.9"), ("mcf10", "1.0")):
            L.append(f"{nm} ($n={num(g['n'])}$) & {stl} & {pc(g['mcf_scorer_median_any_pct'][mk])} & "
                     f"{pc(g['n3_mcf_reading_median_any_pct'][mk])} & {pc(g['n3_own_reading_median_any_pct'][mk])} & "
                     f"$\\approx${rnd(g['fig3a_ref_pct'][mk], 0)} & {'yes' if g['ii_within_5pp'][mk] else 'no'} \\\\\n")
    rows("si_gateca.tex", [f"{GATEC} C-a.labels.<split>.(n, mcf_scorer_median_any_pct, n3_mcf_reading_median_any_pct, "
                           "n3_own_reading_median_any_pct, fig3a_ref_pct [reference values read from MCF's Fig. 3a], ii_within_5pp)"], L)
    ag = GC["C-a"]["labels"]["test"]["agreement"]
    mac("svGcaAgree", f"{num(ag['agree'])}/{num(ag['outcomes'])}", f"{GATEC} C-a.labels.test.agreement")
    agv = GC["C-a"]["labels"]["val"]["agreement"]
    mac("svGcaAgreeVal", f"{num(agv['agree'])}/{num(agv['outcomes'])}", f"{GATEC} C-a.labels.val.agreement")
    cb = GC["C-b"]
    mac("svGcbCpu", f"{num(cb['cpu']['bitwise_equal'])}/{num(cb['cpu']['tensors'])}", f"{GATEC} C-b.cpu.(bitwise_equal, tensors)")
    mac("svGcbRuns", num(cb["cuda"]["runs"]), f"{GATEC} C-b.cuda.runs")
    mac("svGcbStat", g3(cb["cuda"]["stat_patched"]), f"{GATEC} C-b.cuda.stat_patched")
    mac("svGcbMaxUn", g3(max(cb["cuda"]["stat_unpatched"])), f"{GATEC} C-b.cuda.stat_unpatched (max)")
    mac("svGcbBound", rnd(cb["cuda"]["false_fail_bound"], 2), f"{GATEC} C-b.cuda.false_fail_bound")
    cc = GC["C-c"]
    mac("svGcc", f"{num(cc['lattice_bitwise'])}/{num(cc['S'])} lattices and {num(cc['trans_bitwise'])}/{num(cc['S'])} "
                 f"translations bitwise identical on {num(cc['n'])} crystals", f"{GATEC} C-c.(lattice_bitwise, trans_bitwise, S, n)")

    # step-9 provenance
    L = []
    for s in ("testB", "devtest"):
        p = sets[s]["provenance"]
        lib = p["environment"]["libraries"]
        L.append(f"{SETNAME[s]} & {', '.join(p['slurm_jobs'])} & \\texttt{{{tex(p['host'].split('.')[0])}}} & {tex(p['first'])} & "
                 f"{tex(p['last'])} & {num(p['events']['done'])}/{num(p['events']['start'])} & {num(p['output_files'])} & "
                 f"{'none' if not p['stops'] else tex(p['stops'])} \\\\\n")
    rows("si_provenance.tex", [f"{RES} sets.<set>.provenance (slurm_jobs, host, first, last, events.done/start, "
                               "output_files, stops)"], L)
    env = sets["testB"]["provenance"]["environment"]
    assert env == sets["devtest"]["provenance"]["environment"] and d["one_scoring_environment"]
    mac("svScoreEnv", tex(f"Python {env['python']}, PyTorch {env['libraries']['torch']}, pymatgen "
                          f"{env['libraries']['pymatgen']}, SciPy {env['libraries']['scipy']}, NumPy {env['libraries']['numpy']}"),
        f"{RES} sets.<set>.provenance.environment (identical for both sets)")

    # ======================================================================== figures (S12) and misc
    mac("svNTrain", num(GF["sets"]["train"]["n"]), f"{GATEF} sets.train.n")
    mac("svSelSha", S_short(d["selection"]["sha256"]), f"{RES} selection.sha256")
    mac("svSelCommit", tex(d["selection"]["commit"]), f"{RES} selection.commit")

    out["si_numbers.tex"] = ("% GENERATED by scripts/n3_si_tables.py from the named N3 evidence records -- do not edit\n"
                             + "".join(M))
    return out


def S_short(h):
    return h[:12] + "\\ldots"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = build()
    os.makedirs(OUT, exist_ok=True)
    bad = 0
    for name, text in sorted(out.items()):
        assert name.startswith("si_"), name
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
