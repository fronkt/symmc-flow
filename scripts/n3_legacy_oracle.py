"""N3 Table R (d): oracle rebuild of the legacy det+1 representation (protocol §5 (d), §2.6).

The part-file keep flag (old_ok, tuple field 1 of the G1 part files; `legacy_ok` in labels_<set>.json) says
which crystals the legacy det+1 parser kept. G1 never saved those items, so for every flag-kept crystal this
script re-runs `symmc_flow.molcrystal._parse_structure(allow_mirror=False)` on the crystal's CIF with a FRESH
conformer registry (G1 defaults: max_mols 16, max_atoms 64, symprec 0.1, conf_tol 0.3; CIF read with
`read_cif`, as G1 did), rebuilds it with `rigid_to_structure`, and matches the rebuild to the CIF structure
under every §2.1 matcher:
  primary     StructureMatcher().fit(cif, rebuilt)
  secondary   StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=s).get_rms_dist(cif, rebuilt)
              is not None, s in {0.5, 0.8}
CIFs: data/csd_mol/cif for val/devtest, data/csd_testB/cif for sel/testB. Per-crystal wall-clock limit 600 s,
reported as its own count (§5), not re-run. An error or crash is re-run once alone in a fresh process (§1.3);
the same failure again is DETERMINISTIC and counts as a non-match. Input-side only: no arm output is read.

Resumable: per-crystal rows go to results/n3/private/legacy_oracle_<set>.jsonl (the last row per refcode
wins; attempt 2 = the §1.3 re-run); counts (flag-kept, parsed with a fresh registry, matched per matcher,
timeouts, deterministic failures, errors) plus provenance go to results/n3/legacy_oracle_<set>.json.

    python scripts/n3_labels.py --set devtest          # labels_<set>.json must exist first
    python scripts/n3_legacy_oracle.py --set devtest val --workers 2
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

G1_COMMON = (16, 64, 0.1, 0.3)                          # max_mols, max_atoms, symprec, conf_tol (G1 defaults)
CIF_DIR = {"csd_mol": os.path.join(C.REPO, "data", "csd_mol", "cif"),
           "csd_testB": os.path.join(C.REPO, "data", "csd_testB", "cif")}
STOLS = (0.5, 0.8)


def load_labels(name):
    path = os.path.join(C.PRIVATE, f"labels_{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run scripts/n3_labels.py --set {name} first")
    rows = json.load(open(path))
    assert [r["refcode"] for r in rows] == [a["refcode"] for a in C.load_set(name)], f"{name}: stale labels"
    return rows


def matchers():
    from pymatgen.analysis.structure_matcher import StructureMatcher
    return {"primary": StructureMatcher(),
            **{f"stol{s}": StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=s) for s in STOLS}}


def match_all(sms, truth, cand):
    """Every §2.1 matcher, truth first."""
    out = {"primary": bool(sms["primary"].fit(truth, cand))}
    for s in STOLS:
        out[f"stol{s}"] = sms[f"stol{s}"].get_rms_dist(truth, cand) is not None
    return out


def _init():
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    return matchers()


def _oracle(sms, task):
    """task = (refcode, cif_path) -> dict(parsed, why, n_mol, primary, stol0.5, stol0.8)."""
    warnings.filterwarnings("ignore")
    from symmc_flow.molcrystal import _ConformerRegistry, _parse_structure, read_cif, rigid_to_structure
    ref, path = task
    st = read_cif(path)
    item, why = _parse_structure(st, _ConformerRegistry(), *G1_COMMON, allow_mirror=False)
    if item is None:
        return {"parsed": False, "why": why}
    rebuilt = rigid_to_structure(item["lattice"], item["Z"], item["local"], item["centroid"], item["orient"],
                                 item["atom_mask"], item["mol_mask"])
    return {"parsed": True, "why": None, "n_mol": int(item["mol_mask"].sum()), **match_all(sms, st, rebuilt)}


SCRIPTS = ("scripts/n3_legacy_oracle.py", "scripts/n3_tpool.py", "scripts/n3_common.py",
           "symmc_flow/molcrystal.py")


def oracle_matched(x, m):
    """A final oracle row matched under matcher m (deterministic failures and rejections are non-matches)."""
    return bool(x and x["status"] == "ok" and x.get("parsed") and x[m])


def summarize(name, labels, done):
    kept = [r for r in labels if r["legacy_ok"]]
    res = [done.get(r["refcode"]) for r in kept]
    ok = [x for x in res if x is not None and x["status"] == "ok"]
    parsed = [x for x in ok if x["parsed"]]
    st = lambda *v: len([x for x in res if x is not None and x["status"] in v])
    s = {"set": name, "n": len(labels), "flag_kept": len(kept), "flag_unrepresentable": len(labels) - len(kept),
         "evaluated": len([x for x in res if x is not None]),
         "missing": len([x for x in res if x is None]),
         "fresh_registry_parsed": len(parsed),
         "fresh_registry_rejected": len(ok) - len(parsed),
         "oracle_match": {m: sum(bool(x[m]) for x in parsed) for m in ("primary", *[f"stol{s}" for s in STOLS])},
         "timeout": st("timeout"),
         "deterministic_failure": st("deterministic"),
         "error": st("error", "crash"),
         "error_awaiting_rerun": len([x for x in res if x is not None and x["status"] in ("error", "crash")
                                      and x.get("attempt", 1) == 1]),
         "reruns": len([x for x in res if x is not None and x.get("attempt", 1) == 2]),
         "complete": all(x is not None and (x["status"] in ("ok", "timeout", "deterministic")
                                            or x.get("attempt", 1) == 2) for x in res),
         "wall_s_sum": round(sum(x.get("dt", 0.0) for x in res if x is not None), 1),
         "config": {"G1_common(max_mols,max_atoms,symprec,conf_tol)": list(G1_COMMON), "allow_mirror": False,
                    "registry": "fresh per crystal", "timeout_s": None, "truth": "the CIF structure (read_cif)",
                    "fit_order": "fit(cif, rebuilt)"}}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", required=True, choices=("val", "sel", "valsel", "testB", "devtest"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from n3_tpool import FAILED, provenance, read_jsonl, rerun_once, resolve_rerun, run

    for name in args.set:
        t0 = time.time()
        labels = load_labels(name)
        jl = os.path.join(C.PRIVATE, f"legacy_oracle_{name}.jsonl")
        done = {x["refcode"]: x for x in read_jsonl(jl)}           # last row per refcode wins
        cif = {r["refcode"]: os.path.join(CIF_DIR[r["source"]], r["refcode"] + ".cif") for r in labels}
        tasks = [(r["refcode"], cif[r["refcode"]]) for r in labels if r["legacy_ok"] and r["refcode"] not in done]
        for _, p in tasks:
            if not os.path.exists(p):
                raise SystemExit(f"missing CIF {p}")
        retry = [ref for ref, x in done.items() if x["status"] in FAILED and x.get("attempt", 1) == 1]
        print(f"{name}: {sum(r['legacy_ok'] for r in labels)} flag-kept, {len(tasks)} to run, "
              f"{len(retry)} to re-run", flush=True)
        with open(jl, "a", newline="\n") as fh:
            def write(row):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[row["refcode"]] = row

            for (ref, _), status, res, dt in run(_oracle, tasks, init=_init, workers=args.workers,
                                                 timeout=args.timeout):
                row = {"refcode": ref, "status": status, "dt": round(dt, 2), "attempt": 1}
                if status == "ok":
                    row.update(res)
                elif status == "error":
                    row["error"] = res
                write(row)
                if status in FAILED:
                    retry.append(ref)
            for ref in retry:                                       # §1.3: once, alone, in a fresh process
                first = done[ref]
                status, res, dt = rerun_once(_oracle, (ref, cif[ref]), init=_init, timeout=args.timeout)
                row = {"refcode": ref, "status": resolve_rerun(first, status, res), "dt": round(dt, 2),
                       "attempt": 2, "first_status": first["status"], "first_error": first.get("error")}
                if status == "ok":
                    row.update(res)
                elif status == "error":
                    row["error"] = res
                write(row)
                print(f"  re-run {ref}: {first['status']} -> {row['status']}", flush=True)
        s = summarize(name, labels, done)
        s["config"]["timeout_s"] = args.timeout
        s["this_run_wall_s"] = round(time.time() - t0, 1)
        s["private_jsonl_sha256"] = C.sha256(jl)
        s["labels_sha256"] = C.sha256(os.path.join(C.PRIVATE, f"labels_{name}.json"))
        s["provenance"] = provenance(SCRIPTS)
        with open(os.path.join(C.REPO, "results", "n3", f"legacy_oracle_{name}.json"), "w", newline="\n") as fh:
            json.dump(s, fh, indent=1)
        print(f"{name}: flag-kept {s['flag_kept']}/{s['n']}; fresh-registry parsed {s['fresh_registry_parsed']}; "
              f"oracle match {s['oracle_match']}; timeouts {s['timeout']}; deterministic "
              f"{s['deterministic_failure']}; errors {s['error']}; missing {s['missing']} "
              f"({s['this_run_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
