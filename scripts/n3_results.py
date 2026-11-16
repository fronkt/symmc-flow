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
STRUCTURE_KEYS = {"cart", "frac", "frac_coords", "coords", "positions", "lattice", "lattices", "cell", "cells",
                  "species", "atom_types", "rotmats", "rotmats_1", "R", "pred_R", "mol", "cif", "xyz", "matrix"}
CIF_TEXT = re.compile(r"_cell_length_|_atom_site_|^data_\S", re.M)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(rel):
    p = os.path.join(REPO, rel)
    if not os.path.isfile(p):
        raise SystemExit(f"missing {rel}: unpack and verify the step-9 pullback first")
    return json.load(open(p, encoding="utf-8")), sha256(p)


def hygiene(obj, path="$"):
    """Problems: structure-data keys, CIF text, or long numeric arrays (coordinates) anywhere in obj."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in STRUCTURE_KEYS and isinstance(v, list):      # scalars named R / lattice are statistics
                bad.append(f"{path}.{k}: structure-data key holding an array")
            bad += hygiene(v, f"{path}.{k}")
    elif isinstance(obj, list):
        if len(obj) > 3 and all(isinstance(x, list) and len(x) == 3 and all(isinstance(y, float) for y in x)
                                for x in obj):
            bad.append(f"{path}: a list of 3-vectors (coordinates?)")
        for i, v in enumerate(obj):
            bad += hygiene(v, f"{path}[{i}]")
    elif isinstance(obj, str) and CIF_TEXT.search(obj):
        bad.append(f"{path}: CIF text")
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
    stops = [{"unit": e["unit"], "command": e["command"], "time": e["time"], "slurm_job": e.get("slurm_job"),
              "detail": e.get("detail")} for e in ev if e["event"] in ("command_failed", "outputs_missing")]
    return {"events": count, "slurm_jobs": sorted({str(e.get("slurm_job")) for e in ev}), "stops": stops,
            "first": ev[0]["time"], "last": ev[-1]["time"]}


def mcf_block(st, diag_a_passed):
    if not diag_a_passed:
        return {"reported": False, "reason": "MCF diagnostic (a) did not pass: (b)-(d) are not reported (P §3.2)"}
    b, bsha = load(f"{P}/mcf/diag_b_{st}.json")
    c, csha = load(f"{P}/mcf/diag_c_{st}.json")
    d = {}
    for m in MATCHERS_BD:
        dd, dsha = load(f"{P}/mcf/diag_d_{st}_{m}.json")
        d[m] = {"strata": dd["strata"], "consistent": dd["consistent"], "sha256": dsha}
    return {"reported": True, "label": "MCF diagnostics: oracle items labelled as such; not results (P §3.2)",
            "b": {"label": b["diagnostic"], "summary": b["summary"], "note": b.get("note"), "grid_deg": b.get("grid_deg"),
                  "walk_deg": b.get("walk_deg"), "sha256": bsha},
            "c": {"label": c["diagnostic"], "z_MCF = 1": c["z_MCF = 1"], "matchers": c["matchers"],
                  "selector": c["selector"], "sha256": csha},
            "d": d, "d_consistent_all": all(v["consistent"] for v in d.values())}


def assemble():
    sel, sel_sha = load(SELECTION)
    if sel_sha != FREEZE_SHA256:
        raise SystemExit(f"{SELECTION} is not the published freeze")
    diag_a = sel.get("mcf_diagnostic_a") or {}
    verdicts = [v.get("verdict") for v in (diag_a.get("per_checkpoint") or {}).values()]
    a_passed = bool(verdicts) and all(v == "PASS" for v in verdicts)
    res = {"schema": "n3-results/1", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "protocol_sha256": sel["protocol_sha256"], "selection": {"path": SELECTION, "sha256": sel_sha,
                                                                     "commit": FREEZE_COMMIT},
           "h2_form": sel["h2_form"], "gate_c": sel.get("gate_c"), "mcf_diagnostic_a": diag_a,
           "mcf_diagnostic_a_passed": a_passed, "degrade": sel.get("degrade"),
           "clarification": "tasks/n3_step9_clarification_2026-11-15.md", "sets": {}}
    for st in SETS:
        rep, rsha = load(f"results/n3/report_{st}.out.json")
        if rep.get("set") != st or rep.get("dry_run"):
            raise SystemExit(f"report_{st}.out.json is not a confirmatory {st} report")
        started, _ = load(f"{P}/step9/started_{st}.json")
        floor, fsha = load(f"{P}/mcf/haar_floor_{st}.json")
        cks = os.path.join(REPO, P, "step9", f"{st}_checksums.tsv")
        if not os.path.isfile(cks):
            raise SystemExit(f"missing {P}/step9/{st}_checksums.tsv")
        res["sets"][st] = {
            "role": "confirmatory (Table 1 / H1-H3)" if st == "testB" else DEV_LABEL,
            "report": rep, "report_sha256": rsha,
            "mcf_diagnostics": mcf_block(st, a_passed),
            "haar_floor": {"crystals": len(floor) if isinstance(floor, dict) else None, "sha256": fsha},
            "provenance": {"environment": started.get("environment"), "host": started.get("host"),
                           "first_start": started.get("first_start"), "driver_sha256": started.get("driver_sha256"),
                           "selection_sha256": started.get("selection_sha256"), **events_summary(st),
                           "checksums_tsv_sha256": sha256(cks),
                           "output_files": sum(1 for x in open(cks) if x.strip())}}
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
