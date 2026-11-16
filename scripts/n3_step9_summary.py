"""N3 §3.2 MCF diagnostic (c) summary at step 9 (tasks/n3_step9_clarification_2026-11-15.md): the frozen list makes the
oracle-aligned MCF-R bundles (n3_mcf_diag.py c) and the scorer's rows for them (primary, mcf05, mcf08; every draw),
but no frozen command aggregates them. P §3.2 (c): "Report draw@1, any@16 and picked@1 per z_MCF and matcher. It is
not computed for z_MCF = 1, where it is exact by construction (counted)." This script does exactly that and nothing
else: no match is made, the rows are the scorer's own (refused unless current for the bundle, the scoring identity and
the freeze, as `report` refuses them), picked@1 uses MCF-R's frozen selector from report_configs[set] (LJ, recomputed
after alignment by (c)), seeds are the three MCF-R training seeds. Oracle item, labelled as such; not a result.

    python scripts/n3_step9_summary.py c --set testB --step9 --selection-sha256 <freeze>   # box, after the list

Run by scripts/n3_step9.py as the last unit of a set (post/<set>/diag-c-summary), never from the freeze.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

import n3_common as C  # noqa: E402
import n3_score as S  # noqa: E402

MATCHERS = ("primary", "mcf05", "mcf08")


def oal_bundles(sel, set_name):
    """The (c) bundles of runs.step9, in seed order (their listed outputs)."""
    runs = [r for r in sel["runs"]["step9"] if r["set"] == set_name and r["id"].startswith("diag-c/")]
    paths = [o for r in sorted(runs, key=lambda r: r["id"]) for o in r["outputs"]]
    if len(paths) != 3 or not all(os.path.basename(p).startswith("mcfRoal_") for p in paths):
        raise SystemExit(f"runs.step9 does not list three diagnostic (c) bundles for {set_name}")
    return paths


def pct(x):
    return round(100.0 * float(x), 4)


def summarize(set_name, selection, selection_sha256, step9, out=None):
    sel = json.load(open(selection, encoding="utf-8"))
    sel_rec = S.guard(set_name, step9, selection, selection_sha256)
    selector = sel["report_configs"][set_name]["arms"]["MCF-R"]["selector"]
    paths = [S.rel(p) for p in oal_bundles(sel, set_name)]
    bundles = [C.load_bundle(p) for p in paths]
    S.validate_bundles(bundles, seed_ids=[0, 1, 2])
    items = C.load_set(set_name)
    refcodes = [a["refcode"] for a in items]
    if any(b["set"] != set_name or b["refcodes"] != refcodes for b in bundles):
        raise SystemExit("a (c) bundle's set or crystal order differs from the frozen set")
    z = [list(b["meta"]["z_MCF"]) for b in bundles]
    if any(zz != z[0] for zz in z) or len(z[0]) != len(refcodes):
        raise SystemExit("the (c) bundles disagree on z_MCF")
    z = np.array(z[0])
    strata = {"z_MCF = 2": z == 2, "z_MCF >= 3": z >= 3}
    res = {"diagnostic": "c (oracle-aligned score, ORACLE ITEM, not a result)", "set": set_name,
           "protocol": "P §3.2 (c): draw@1, any@16 and picked@1 per z_MCF and matcher; z_MCF = 1 counted",
           "bundles": paths, "bundle_sha256": [C.sha256(p) for p in paths], "selector": selector,
           "seeds": [b["seed"] for b in bundles], "selection": sel_rec,
           "z_MCF = 1": {"n": int((z < 2).sum()), "note": "exact by construction (not computed)"},
           "matchers": {}}
    for m in MATCHERS:
        results = []
        for p, b in zip(paths, bundles):
            if S.rows_current(p, m, selection=sel_rec, bundle=b) is None:
                raise SystemExit(f"no current {m} rows for {p}")
            results.append(S.load_match(p, m))
        o = S.arm_outcomes(bundles, results, items, selector, None, ensemble=False)
        block = {}
        for name, mask in strata.items():
            n = int(mask.sum())
            row = {"n": n}
            for key, lab in (("draw1", "draw1"), ("any", "any16"), ("picked", "picked1")):
                if n == 0:
                    row[f"{lab}_per_seed_pct"], row[f"{lab}_mean_pct"] = [], None
                    continue
                per = o[key][mask].mean(0)
                row[f"{lab}_per_seed_pct"] = [pct(x) for x in per]
                row[f"{lab}_mean_pct"] = pct(per.mean())
            block[name] = row
        res["matchers"][m] = block
    res["identity"] = S.scoring_identity(set_name)
    out = out or os.path.join(REPO, "results", "n3", "private", "mcf", f"diag_c_{set_name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {os.path.relpath(out, REPO)}")
    return res


def main():
    top = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = top.add_subparsers(dest="cmd", required=True)
    ap = sub.add_parser("c", help="diagnostic (c) per z_MCF stratum and matcher")
    ap.add_argument("--set", required=True, choices=["testB", "devtest"])
    ap.add_argument("--step9", action="store_true")
    ap.add_argument("--selection", default=os.path.join(REPO, "results", "n3", "n3_selection.json"))
    ap.add_argument("--selection-sha256", default=None)
    ap.add_argument("--out", default=None)
    a = top.parse_args()
    summarize(a.set, a.selection, a.selection_sha256, a.step9, a.out)


if __name__ == "__main__":
    main()
