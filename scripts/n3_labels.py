"""N3: frozen per-crystal stratum labels (protocol §2.6, §1.1 input-side list, §3.3 OMC25 overlap).

For every crystal of a frozen set, in frozen order (input-side only; no arm output is read):
  K, K_stratum        number of spglib ops in the CSD cell; <=2 / 3-4 / >=6
  mult, z_mcf         mult = 1 + #ops with W = I and t != 0 (pure centring, min-image > 1e-4); z_MCF = K / mult
                      (the copy count of the exported MCF cell, §3.2 export step 2); z stratum 1 / 2 / >=3
  class               gemmi: centro / sohncke / other (same rule as g1_o3_reparse and n3_build_testB)
  chiral, mixed,      scripts/n3_chirality.label_items: CHIRAL = refined mirror RMSD > 0.3 A; MIXED = an
  improper_achiral    improper op (det W = -1) and CHIRAL; IMPROPER-ACHIRAL = an improper op and not CHIRAL.
                      Sensitivity rows: refined > 0.1 A, refined > 0.5 A, parser-capped (max_iso 200) > 0.3 A
  legacy_unrep        old_ok False (tuple field 1) in the G1 part files <cache>.parts/part_*.pt of the run that
                      built the set: data/csd_mol for val/devtest, data/csd_testB for sel/testB (mapped by refcode)
  formula_disjoint    the asym molecular formula does not occur among the TRAIN (1,687) molecular formulas
  omc25_*             family: refcode[:6] among csd_refcode[:6] of facebook/OMC25 omc25-starting-crystals.csv;
                      nonoverlap = not family; formula: formula equals some OMC25 mol.composition (descriptive);
                      elements outside the OMC25 element set (the union over mol.composition) are flagged
Writes results/n3/private/labels_<set>.json (per crystal; CSD-derived, gitignored), the committed-safe
results/n3/labels_summary_<set>.json (counts only + SHA-256 of the private file, of the inputs and of the code,
library versions) and results/n3/omc25_outside_elements_<set>.json (§3.3: the crystals with elements outside
OMC25's element set, listed by refcode with those elements; refcode lists are committed, §7 hygiene).
DEV-TEST and VAL counts are checked against the protocol / review reference values (exit 1 on a mismatch).

    python scripts/n3_labels.py --set devtest val
    python scripts/n3_labels.py --set sel testB valsel --workers 2
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

import torch

OMC25_CSV = os.path.join(C.REPO, "external", "omc25", "omc25-starting-crystals.csv")
PART_SRC = {"val": "csd_mol", "devtest": "csd_mol", "sel": "csd_testB", "testB": "csd_testB"}
COMPONENTS = {"valsel": ("val", "sel")}
LABEL_SETS = ("val", "sel", "valsel", "testB", "devtest")
SENS = (("refined>0.1", "refined", 0.1), ("refined>0.5", "refined", 0.5), ("parser>0.3", "parser", 0.3))
OMC25_ELEMENT_SET_DEF = "union of the elements of mol.composition over all rows (all splits) of omc25-starting-crystals.csv"
SCRIPTS = ("scripts/n3_labels.py", "scripts/n3_chirality.py", "scripts/n3_common.py", "scripts/n3_tpool.py")

# protocol §2.6 / §2.2 / §3.2 and review2 oldflag.py / zmcf_count.py reference counts
REFERENCE = {
    "devtest": {"n": 200, "chiral": 178, "mixed": 137, "improper_achiral": 19,
                "K_stratum": {"<=2": 55, "3-4": 119, ">=6": 26}, "centred": 12, "legacy_kept": 57,
                "K_z": {"2,2": 55, "4,2": 1, "4,4": 118, "8,4": 9, "8,8": 15, "16,4": 2}},
    "val": {"n": 100, "centred": 6, "legacy_kept": 30},
}


def k_stratum(K):
    return "<=2" if K <= 2 else "3-4" if K <= 4 else ">=6" if K >= 6 else "K=5"


def z_stratum(z):
    return "1" if z == 1 else "2" if z == 2 else ">=3"


def centring(a):
    """(mult, z_MCF) from the crystal's own ops: pure translations W = I, t != 0 (A/B/C/F/I/R)."""
    W, t = a["W"], a["t"]
    eye = torch.eye(3, dtype=W.dtype)
    n_c = sum(1 for k in range(W.shape[0]) if torch.allclose(W[k], eye, atol=1e-6)
              and float((t[k] - torch.round(t[k])).abs().max()) > 1e-4)
    mult = 1 + n_c
    K = int(a["K"])
    assert K % mult == 0, f"{a['refcode']}: K={K} not divisible by centring multiplicity {mult}"
    return mult, K // mult


_CLS = {}


def sg_class(n):
    if n not in _CLS:
        import gemmi
        g = gemmi.find_spacegroup_by_number(n)
        _CLS[n] = "centro" if g.is_centrosymmetric() else "sohncke" if g.is_sohncke() else "other"
    return _CLS[n]


def formula_of(Z):
    """Asym molecular formula -> (canonical key, Hill string). Key = sorted (symbol, count) pairs."""
    from pymatgen.core.periodic_table import Element
    cnt = collections.Counter(Element.from_Z(int(z)).symbol for z in Z)
    return tuple(sorted(cnt.items())), hill(cnt)


def hill(cnt):
    order = (["C", "H"] if "C" in cnt else []) + sorted(s for s in cnt if not ("C" in cnt and s in ("C", "H")))
    return " ".join(s + (str(cnt[s]) if cnt[s] != 1 else "") for s in order)


def parse_composition(s):
    """OMC25 mol.composition ('C22 H19 N O') -> canonical key."""
    cnt = collections.Counter()
    for tok in s.split():
        m = re.fullmatch(r"([A-Z][a-z]?)(\d*)", tok)
        if m is None:
            raise ValueError(f"unparsed OMC25 composition token {tok!r}")
        cnt[m.group(1)] += int(m.group(2) or 1)
    return tuple(sorted(cnt.items()))


def load_omc25(path):
    fams, comps, elems = set(), set(), set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            fams.add(r["csd_refcode"][:6])
            k = parse_composition(r["mol.composition"])
            comps.add(k)
            elems.update(s for s, _ in k)
    return fams, comps, elems


def legacy_flags(src):
    """refcode -> old_ok from the G1 part files of data/<src>/ds_o3.pt.parts (tuple field 1)."""
    parts = sorted(glob.glob(os.path.join(C.REPO, "data", src, "ds_o3.pt.parts", "part_*.pt")))
    if not parts:
        raise SystemExit(f"no G1 part files under data/{src}/ds_o3.pt.parts")
    flags, n_rows = {}, 0
    for p in parts:
        for row in torch.load(p, weights_only=False):
            n_rows += 1
            ref, old_ok = row[0], bool(row[1])
            if ref in flags and flags[ref] != old_ok:
                raise SystemExit(f"data/{src}: refcode {ref} has conflicting old_ok flags in the part files")
            flags[ref] = old_ok
    info = {"n_part_files": len(parts), "rows": n_rows, "unique_refcodes": len(flags),
            "old_ok": sum(flags.values()), "sha256": {os.path.basename(p): C.sha256(p) for p in parts}}
    return flags, info


def _chiral_one(slim):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from n3_chirality import label_items
    a = dict(slim, W=torch.from_numpy(slim["W"]), local=torch.from_numpy(slim["local"]),
             Z=torch.from_numpy(slim["Z"]))
    return label_items([a])[0]


def chirality_rows(items, workers):
    # numpy payloads: plain pickling to spawn workers (no torch shared-memory handles)
    slims = [{"refcode": a["refcode"], "sg": int(a["sg"]), "K": int(a["K"]), "W": a["W"].numpy(),
              "local": a["local"].numpy(), "Z": a["Z"].numpy()} for a in items]
    if workers <= 1:
        return [_chiral_one(s) for s in slims]
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(_chiral_one, slims, chunksize=4)


def git_head():
    try:
        return subprocess.run(["git", "-C", C.REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def label_component(name, workers, train_formulas, omc, flag_cache):
    items = C.load_set(name)
    src = PART_SRC[name]
    if src not in flag_cache:
        flag_cache[src] = legacy_flags(src)
    flags, _ = flag_cache[src]
    missing = [a["refcode"] for a in items if a["refcode"] not in flags]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} crystals not in the data/{src} G1 part files")
    fams, comps, elems = omc
    t0 = time.time()
    chir = chirality_rows(items, workers)
    t_chir = time.time() - t0
    rows = []
    for a, ch in zip(items, chir):
        assert ch["refcode"] == a["refcode"]
        K, sg = int(a["K"]), int(a["sg"])
        mult, z = centring(a)
        fkey, fstr = formula_of(a["Z"].tolist())
        el = [s for s, _ in fkey]
        sens = {}
        for lab, field, tol in SENS:
            c = ch[field] > tol
            sens[lab] = {"chiral": c, "mixed": ch["improper"] and c, "improper_achiral": ch["improper"] and not c}
        rows.append({
            "refcode": a["refcode"], "source": src, "sg": sg, "class": sg_class(sg),
            "K": K, "K_stratum": k_stratum(K), "mult": mult, "centred": mult > 1, "z_mcf": z,
            "z_stratum": z_stratum(z),
            "improper": ch["improper"], "parser": ch["parser"], "refined": ch["refined"], "n_iso": ch["n_iso"],
            "iso_cap_hit": ch["n_iso"] >= 200, "connected": ch["connected"], "wl_hash": ch["hash"],
            "chiral": ch["chiral"], "mixed": ch["mixed"], "improper_achiral": ch["improper_achiral"],
            "sensitivity": sens,
            "legacy_ok": flags[a["refcode"]], "legacy_unrep": not flags[a["refcode"]],
            "formula": fstr, "formula_disjoint": fkey not in train_formulas,
            "elements": el, "omc25_family": a["refcode"][:6] in fams,
            "omc25_nonoverlap": a["refcode"][:6] not in fams, "omc25_formula": fkey in comps,
            "elements_outside_omc25": sorted(set(el) - elems),
        })
    return rows, t_chir


def summarize(rows):
    n = len(rows)
    cnt = lambda f: sum(1 for r in rows if f(r))
    by = lambda key: dict(sorted(collections.Counter(str(r[key]) for r in rows).items()))
    s = {
        "n": n,
        "K_stratum": {k: cnt(lambda r, k=k: r["K_stratum"] == k) for k in ("<=2", "3-4", ">=6")},
        "K_values": dict(sorted(collections.Counter(r["K"] for r in rows).items())),
        "z_mcf_stratum": {k: cnt(lambda r, k=k: r["z_stratum"] == k) for k in ("1", "2", ">=3")},
        "K_z": {f"{k[0]},{k[1]}": v for k, v in sorted(collections.Counter((r["K"], r["z_mcf"]) for r in rows).items())},
        "centred": cnt(lambda r: r["centred"]),
        "mult": by("mult"),
        "class": {c: cnt(lambda r, c=c: r["class"] == c) for c in ("centro", "sohncke", "other")},
        "improper": cnt(lambda r: r["improper"]),
        "chiral": cnt(lambda r: r["chiral"]),
        "mixed": cnt(lambda r: r["mixed"]),
        "improper_achiral": cnt(lambda r: r["improper_achiral"]),
        "sensitivity": {lab: {k: cnt(lambda r, lab=lab, k=k: r["sensitivity"][lab][k])
                              for k in ("chiral", "mixed", "improper_achiral")} for lab, _, _ in SENS},
        "iso_cap_hit": cnt(lambda r: r["iso_cap_hit"]),
        "disconnected_graph": cnt(lambda r: not r["connected"]),
        "legacy_kept": cnt(lambda r: r["legacy_ok"]),
        "legacy_unrep": cnt(lambda r: r["legacy_unrep"]),
        "legacy_unrep_by_class": {c: cnt(lambda r, c=c: r["legacy_unrep"] and r["class"] == c)
                                  for c in ("centro", "sohncke", "other")},
        "formula_disjoint": cnt(lambda r: r["formula_disjoint"]),
        "omc25_family_overlap": cnt(lambda r: r["omc25_family"]),
        "omc25_nonoverlap": cnt(lambda r: r["omc25_nonoverlap"]),
        "omc25_formula_overlap": cnt(lambda r: r["omc25_formula"]),
        "elements_outside_omc25": cnt(lambda r: bool(r["elements_outside_omc25"])),
        "class_x_handedness": {c: {h: cnt(lambda r, c=c, h=h: r["class"] == c and (
            r["mixed"] if h == "mixed" else r["improper_achiral"] if h == "improper_achiral" else not r["improper"]))
            for h in ("mixed", "improper_achiral", "no_improper")} for c in ("centro", "sohncke", "other")},
    }
    return s


def reference_check(name, s):
    ref = REFERENCE.get(name)
    if ref is None:
        return None
    got = {"n": s["n"], "chiral": s["chiral"], "mixed": s["mixed"], "improper_achiral": s["improper_achiral"],
           "K_stratum": s["K_stratum"], "centred": s["centred"], "legacy_kept": s["legacy_kept"], "K_z": s["K_z"]}
    bad = {k: {"expected": v, "got": got[k]} for k, v in ref.items() if got[k] != v}
    return {"expected": ref, "mismatch": bad, "pass": not bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", required=True, choices=LABEL_SETS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--omc25", default=OMC25_CSV)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from n3_tpool import provenance

    train_formulas = {formula_of(a["Z"].tolist())[0] for a in C.load_set("train")}
    omc = load_omc25(args.omc25)
    omc_sha = C.sha256(args.omc25)
    flag_cache, comp_rows, failed = {}, {}, []
    for name in args.set:
        t0 = time.time()
        rows, t_chir = [], 0.0
        for comp in COMPONENTS.get(name, (name,)):
            if comp not in comp_rows:                      # deterministic, so a component is labelled once
                comp_rows[comp] = label_component(comp, args.workers, train_formulas, omc, flag_cache)
                t_chir += comp_rows[comp][1]
            rows += comp_rows[comp][0]
        expect = [a["refcode"] for a in C.load_set(name)]
        assert [r["refcode"] for r in rows] == expect, f"{name}: label order differs from the frozen order"
        for i, r in enumerate(rows):
            r["i"] = i
        os.makedirs(C.PRIVATE, exist_ok=True)
        path = os.path.join(C.PRIVATE, f"labels_{name}.json")
        with open(path + ".tmp", "w", newline="\n") as fh:
            json.dump(rows, fh, indent=0)
        os.replace(path + ".tmp", path)
        s = summarize(rows)
        chk = reference_check(name, s)
        srcs = sorted({PART_SRC[c] for c in COMPONENTS.get(name, (name,))})
        out = {"set": name, "n": len(rows), "labels_file": os.path.relpath(path, C.REPO).replace("\\", "/"),
               "labels_sha256": C.sha256(path), "counts": s, "reference_check": chk,
               "definitions": {"chiral": "n3_chirality refined mirror RMSD > 0.3 A",
                               "centring": "ops with W = I and min-image |t| > 1e-4",
                               "legacy": "old_ok (tuple field 1) of data/<src>/ds_o3.pt.parts",
                               "formula_disjoint": "asym formula not among the 1687 TRAIN formulas",
                               "omc25_family": "refcode[:6] in OMC25 csd_refcode[:6]",
                               "omc25_element_set": OMC25_ELEMENT_SET_DEF},
               "inputs": {"omc25_csv_sha256": omc_sha, "omc25_n_families": len(omc[0]),
                          "omc25_n_formulas": len(omc[1]), "omc25_elements": sorted(omc[2]),
                          "train_n_formulas": len(train_formulas),
                          "g1_parts": {src: {k: v for k, v in flag_cache[src][1].items()} for src in srcs},
                          "code_head": git_head()},
               "timing_s": {"chirality": round(t_chir, 1), "total": round(time.time() - t0, 1)},
               "workers": args.workers}
        opath = os.path.join(C.REPO, "results", "n3", f"omc25_outside_elements_{name}.json")
        outside = [{"refcode": r["refcode"], "elements_outside": r["elements_outside_omc25"]}
                   for r in rows if r["elements_outside_omc25"]]
        with open(opath, "w", newline="\n") as fh:
            json.dump({"set": name, "n": len(rows), "n_outside": len(outside),
                       "omc25_element_set": sorted(omc[2]), "omc25_csv_sha256": omc_sha,
                       "definition": OMC25_ELEMENT_SET_DEF, "crystals": outside}, fh, indent=1)
        out["omc25_outside_elements_file"] = os.path.relpath(opath, C.REPO).replace("\\", "/")
        out["provenance"] = provenance(SCRIPTS)
        spath = os.path.join(C.REPO, "results", "n3", f"labels_summary_{name}.json")
        with open(spath, "w", newline="\n") as fh:
            json.dump(out, fh, indent=1)
        print(f"{name}: n={s['n']} K {s['K_stratum']} z_MCF {s['z_mcf_stratum']} centred {s['centred']} "
              f"class {s['class']} chiral {s['chiral']} mixed {s['mixed']} improper-achiral {s['improper_achiral']} "
              f"legacy-unrep {s['legacy_unrep']} formula-disjoint {s['formula_disjoint']} "
              f"omc25-family {s['omc25_family_overlap']} omc25-formula {s['omc25_formula_overlap']} "
              f"outside-omc25-elements {s['elements_outside_omc25']} ({time.time() - t0:.0f}s)")
        if chk is not None:
            print(f"  reference check {name}: {'PASS' if chk['pass'] else 'MISMATCH ' + json.dumps(chk['mismatch'])}")
            if not chk["pass"]:
                failed.append(name)
    if failed:
        raise SystemExit(f"reference mismatch on {failed}")


if __name__ == "__main__":
    main()
