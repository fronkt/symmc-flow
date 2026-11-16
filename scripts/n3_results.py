"""N3 §7 step 9: assemble results/n3/n3_results.json (the committed result) from the step-9 outputs pulled back from
the box. Pure JSON assembly: nothing is matched, re-scored, re-selected or re-tested here; every number is copied from
the report / diagnostic files the frozen commands wrote on the box, in the scoring environment of the freeze.

    python scripts/n3_results.py            # laptop, after the step-9 pullback is unpacked and verified

Contents per set (TEST-B confirmatory; DEV-TEST labelled as P §1.4 requires):
  report           results/n3/report_<set>.out.json, verbatim (Table 1 / Table 2 inputs, the H1-H3 families, the
                   wording and abstract inputs), with its SHA-256;
  mcf_diagnostics  (b) summary, (c) per z_MCF and matcher, (d) strata + consistent, each labelled ORACLE where P says so;
                   reported only if diagnostic (a) passed (P §3.2: if (a) failed, (b)-(d) are not reported);
  haar_floor       counts only (its per-crystal values are a (b) input);
  provenance       the set's first-start record (environment, host, driver), an events summary and the checksum list.
Top level: the freeze identity, the H2 FORM, Gate C, diagnostic (a), degrade records from SEL.
Hygiene: the output holds numbers, labels and refcodes only; keys or text that look like structure data stop it.
"""
import argparse
import hashlib
import json
import os
import re
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION = "results/n3/n3_selection.json"
FREEZE_SHA256 = "553f1dc9efc0ec2c6d2d92b6d61bd5c76bbdc46c1d8a8ef9c814843222e1f491"
FREEZE_COMMIT = "772a36e"
OUT = "results/n3/n3_results.json"
P = "results/n3/private"
SETS = ("testB", "devtest")
DEV_LABEL = "DEV-TEST: used during development, not confirmatory (P §1.4)"
MATCHERS_BD = ("primary", "mcf05", "mcf08")
# a key holding an array is structure data if it is one of these (exact) or contains one of the tokens (any case)
STRUCTURE_KEYS = {"R", "R0", "mol", "matrix", "cell", "cells", "local"}
STRUCTURE_TOKENS = ("cart", "frac", "coord", "lattice", "rotmat", "position", "species", "atom_type", "xyz", "cif",
                    "pred_r", "trans_1", "gt_")
XYZ_TEXT = re.compile(r"^\s*\S*\s*-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*$", re.M)  # xyz / POSCAR rows
CIF_TEXT = re.compile(r"_cell_length_|_atom_site_|^data_\S", re.M)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(rel, tsv=None):
    """JSON + SHA-256; with tsv (the box's checksum list) the file must be listed there with that hash."""
    p = os.path.join(REPO, rel)
    if not os.path.isfile(p):
        raise SystemExit(f"missing {rel}: unpack and verify the step-9 pullback first")
    h = sha256(p)
    if tsv is not None and tsv.get(rel) != h:
        raise SystemExit(f"{rel} is not the file step 9 wrote on the box (absent from, or another hash in, the "
                         f"set's checksum list)")
    return json.load(open(p, encoding="utf-8")), h


def hygiene(obj, path="$"):
    """Problems: structure-data keys, CIF text, or long numeric arrays (coordinates) anywhere in obj."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if isinstance(v, list) and (str(k) in STRUCTURE_KEYS or any(t in kl for t in STRUCTURE_TOKENS)):
                bad.append(f"{path}.{k}: structure-data key holding an array")   # scalars named R are statistics
            bad += hygiene(v, f"{path}.{k}")
    elif isinstance(obj, list):
        rows = [x for x in obj if isinstance(x, list)]
        if len(obj) >= 2 and len(rows) == len(obj) and all(
                len(x) in (3, 9) and all(isinstance(y, (int, float)) and not isinstance(y, bool) for y in x)
                for x in rows) and any(isinstance(y, float) and not float(y).is_integer() for x in rows for y in x):
            bad.append(f"{path}: a list of numeric 3- / 9-vectors (coordinates or cells?)")
        for i, v in enumerate(obj):
            bad += hygiene(v, f"{path}[{i}]")
    elif isinstance(obj, str) and (CIF_TEXT.search(obj) or len(XYZ_TEXT.findall(obj)) >= 3):
        bad.append(f"{path}: CIF / XYZ / POSCAR text")
    return bad


def events_summary(st):
    p = os.path.join(REPO, P, "step9", f"events_{st}.jsonl")
    if not os.path.isfile(p):
        raise SystemExit(f"missing {P}/step9/events_{st}.jsonl")
    ev = [json.loads(x) for x in open(p) if x.strip()]
    count = {}
    for e in ev:
        count[e["event"]] = count.get(e["event"], 0) + 1
    if not ev or ev[-1]["event"] != "set_done":
        raise SystemExit(f"{st}: the step-9 events do not end in set_done")
    # failures, and every restart / resume (a timeout or kill leaves no failure event, only the later restart)
    stops = [{"event": e["event"], "unit": e["unit"], "command": e["command"], "time": e["time"],
              "slurm_job": e.get("slurm_job"), "detail": e.get("detail")} for e in ev
             if e["event"] in ("command_failed", "outputs_missing", "restart", "resume", "deleted_partial")]
    return {"events": count, "slurm_jobs": sorted({str(e.get("slurm_job")) for e in ev}), "stops": stops,
            "first": ev[0]["time"], "last": ev[-1]["time"]}


def mcf_block(st, diag_a, tsv):
    if diag_a.get("verdict") != "PASS":
        return {"reported": False, "reason": "MCF diagnostic (a) did not pass: (b)-(d) are not reported (P §3.2)"}
    b, bsha = load(f"{P}/mcf/diag_b_{st}.json", tsv)
    c, csha = load(f"{P}/mcf/diag_c_{st}.json", tsv)
    d = {}
    for m in MATCHERS_BD:
        dd, dsha = load(f"{P}/mcf/diag_d_{st}_{m}.json", tsv)
        d[m] = {"strata": dd["strata"], "consistent": dd["consistent"], "sha256": dsha}
    approx = diag_a.get("approximate") is True
    tag = " (approximate: A2 item 3)" if approx else ""
    return {"reported": True, "label": "MCF diagnostics: oracle items labelled as such; not results (P §3.2)" + tag,
            "approximate": approx, "approximate_rule": diag_a.get("approximate_rule"),
            "b": {"label": b["diagnostic"] + tag, "summary": b["summary"], "note": b.get("note"),
                  "grid_deg": b.get("grid_deg"), "walk_deg": b.get("walk_deg"), "sha256": bsha},
            "c": {"label": c["diagnostic"] + tag, "z_MCF = 1": c["z_MCF = 1"], "matchers": c["matchers"],
                  "selector": c["selector"], "sha256": csha},
            "d": {"label": "d (consistency)" + tag, "matchers": d},
            "d_consistent_all": all(v["consistent"] for v in d.values())}


def checksums(st):
    """{path: sha256} of the box's <set>_checksums.tsv (every file the set's step 9 wrote)."""
    p = os.path.join(REPO, P, "step9", f"{st}_checksums.tsv")
    if not os.path.isfile(p):
        raise SystemExit(f"missing {P}/step9/{st}_checksums.tsv")
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            h, path = line.rstrip("\n").split("\t")
            out[path] = h
    return out, sha256(p)


def assemble():
    sel, sel_sha = load(SELECTION)
    if sel_sha != FREEZE_SHA256:
        raise SystemExit(f"{SELECTION} is not the published freeze")
    diag_a = sel.get("mcf_diagnostic_a") or {}
    a_passed = diag_a.get("verdict") == "PASS"         # the freeze's rule: the Thurlemann repeat is not gated
    res = {"schema": "n3-results/1", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "protocol_sha256": sel["protocol_sha256"], "selection": {"path": SELECTION, "sha256": sel_sha,
                                                                     "commit": FREEZE_COMMIT},
           "h2_form": sel["h2_form"], "gate_c": sel.get("gate_c"), "mcf_diagnostic_a": diag_a,
           "mcf_diagnostic_a_passed": a_passed, "degrade": sel.get("degrade"),
           "clarification": "tasks/n3_step9_clarification_2026-11-15.md", "sets": {}}
    for st in SETS:
        tsv, tsv_sha = checksums(st)
        rep, rsha = load(f"results/n3/report_{st}.out.json", tsv)
        if rep.get("set") != st or rep.get("dry_run"):
            raise SystemExit(f"report_{st}.out.json is not a confirmatory {st} report")
        if (rep.get("selection") or {}).get("sha256") != sel_sha:
            raise SystemExit(f"report_{st}.out.json was made under another freeze")
        if rep.get("missing_arms"):
            raise SystemExit(f"report_{st}.out.json has missing arms {sorted(rep['missing_arms'])}: stop and inspect "
                             "(clarification section 5); an incomplete report is not published")
        for fam in ("family_primary", "family_n2"):
            if (rep.get(fam) or {}).get("complete") is not True:
                raise SystemExit(f"report_{st}.out.json: {fam} is not complete: stop and inspect")
        started, _ = load(f"{P}/step9/started_{st}.json")
        floor, fsha = load(f"{P}/mcf/haar_floor_{st}.json", tsv)
        anchor, ash = load(f"{P}/classical/anchor_{st}.json", tsv)
        a_sum = anchor.get("summary") or {}
        res["sets"][st] = {
            "role": "confirmatory (Table 1 / H1-H3)" if st == "testB" else DEV_LABEL,
            "report": rep, "report_sha256": rsha,
            "mcf_diagnostics": mcf_block(st, diag_a, tsv),
            "haar_floor": {"crystals": len(floor) if isinstance(floor, dict) else None, "sha256": fsha},
            "ff_truth_anchoring": {**{k: a_sum.get(k) for k in ("n", "anchoring", "labels", "no_minimum",
                                                                 "match_det_fail", "median_drift_deg", "partial",
                                                                 "match_step9")},
                                   "note": "FF*-minimised truths of this set (P §3.3); the H2 FORM itself was fixed "
                                           "on VALSEL", "sha256": ash},
            "provenance": {"environment": started.get("environment"), "host": started.get("host"),
                           "first_start": started.get("first_start"), "driver_sha256": started.get("driver_sha256"),
                           "selection_sha256": started.get("selection_sha256"), **events_summary(st),
                           "checksums_tsv_sha256": tsv_sha, "output_files": len(tsv)}}
        if started.get("selection_sha256") != sel_sha:
            raise SystemExit(f"{st}: step 9 ran under another freeze")
    envs = {json.dumps(res["sets"][st]["provenance"]["environment"], sort_keys=True) for st in SETS}
    res["one_scoring_environment"] = len(envs) == 1
    bad = hygiene(res)
    if bad:
        raise SystemExit(f"hygiene: {len(bad)} problem(s), e.g. {bad[:5]}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    res = assemble()
    out = os.path.join(REPO, a.out)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, sort_keys=False, default=float)
        fh.write("\n")
    print(f"wrote {a.out} ({os.path.getsize(out)} bytes, sha256 {sha256(out)[:16]}...)")


if __name__ == "__main__":
    main()
