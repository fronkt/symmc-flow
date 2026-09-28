"""N3 Table R: deterministic representability counts (protocol §5; computed before any sampling).

Rows    all / centro / sohncke / other / MIXED-HANDEDNESS / IMPROPER-ACHIRAL / LEGACY-UNREPRESENTABLE
        (from results/n3/private/labels_<set>.json, scripts/n3_labels.py).
Columns counts of crystals whose TRUE structure (to_structure(orig)) each representation can emit, under every
        §2.1 matcher: primary StructureMatcher().fit(truth, cand); secondary StructureMatcher(ltol=0.3,
        angle_tol=10, scale=False, stol=s).get_rms_dist(truth, cand) is not None, s in {0.5, 0.8}.
  (a) OURS = CLASSICAL: expand(a, R0), plus build step 2 (i) orbit check. Holds by construction on SEL/TEST-B
      (build step 2; the caption prints the step-2 drops from results/n3/n3_build_testB.json); computed here
      on every set as a check.
  (b) MCF per-copy representation (Gate F2), under every matcher: the export pickle's own pose
      (scripts/n3_mcf_export.py load_export / entry_cells, read-only) matched against the truth here. A
      crystal counts iff it was exported, passes F2's non-matcher checks (assemble_err <= 1e-3 A, species;
      results/n3/private/gateF_<set>_detail.jsonl) and matches. The primary count is cross-checked against
      results/n3/gateF_<set>.json F2.fail_refcodes. Needs the export of the set; otherwise (b) is absent.
  (c) Best homochiral approximant (the information of MCF's released CSP driver, packing_gen.py:376-388):
      true lattice and centroids; every improper copy (det W_k = -1) is replaced by the RMSD-optimal proper
      rotation onto its mirror image. Element-blocked Hungarian ICP (at most 30 iterations, stop at max
      displacement < 1e-8 A, Kabsch det+1) of the conformer onto its mirror.
        c           (proposed A2) the 16 frozen starts PLUS the §2.6 (i) best automorphism rotation; the
                    approximant is the lowest-RMSD end point; geometrically distinct end points (> 1e-3 A
                    apart) within 1e-6 A of the best RMSD are each built and matched, and the crystal counts
                    as matched if any matches
        c_frozen16  the protocol text alone: the 4 sign-diagonal rotations and 12
                    scipy Rotation.random(random_state=int(rng.integers(1e9))), rng = np.random.default_rng(0)
                    per molecule; the first strict minimum over the 16 end points
      Crystals with no improper op are the expansion itself, so their (c) entries are their (a) result.
  (d) OLD det+1: the part-file keep flag (labels) and the scripts/n3_legacy_oracle.py rebuild counts.
Per-crystal wall-clock limit 600 s per column job, reported as its own count (timeouts are not re-run). An
error or crash is re-run once alone in a fresh process (§1.3); the same failure again is DETERMINISTIC and
counts as a non-match. Corpus context row: G1/G2 counts from the part files (3,500 dev CIFs; 4,500 pool CIFs)
and g2_asym.pt. All constructions are truth-derived and input-side (§1.1); no arm output is read.

Resumable: per-(crystal, column) rows in results/n3/private/table_r_<set>.jsonl (c rows of another method and
b rows of another export pickle are ignored); counts only in results/n3/table_r_<set>.json, whose `complete`
block says which columns are final. The frozen wording is filled only when every column is complete.

    python scripts/n3_labels.py --set devtest && python scripts/n3_legacy_oracle.py --set devtest
    python scripts/n3_table_r.py --set devtest --workers 2 [--limit 40] [--cols a b c]
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

import numpy as np

STOLS = (0.5, 0.8)
MATCHERS = ("primary",) + tuple(f"stol{s}" for s in STOLS)
MIR = np.array([1.0, 1.0, -1.0])
SIGN_DIAG = ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1])
# SHA-256 of the 16 frozen start matrices (float64, rounded to 1e-8, C order); scipy 1.17.1 legacy
# Rotation.random(random_state=int) semantics. A scipy that draws differently fails here, loudly.
STARTS16_SHA256 = "d5f2c23ba01dd773d042d02e1d5ac6bcc1d1605d0ce2966a4c1c65bf0a0d0352"
C_METHOD = "16 frozen + 2.6(i) automorphism start; ties 1e-6 A; distinct > 1e-3 A; any-match"
TIE_RMSD, DISTINCT_A, MAX_ISO = 1e-6, 1e-3, 200
TOL_ASSEMBLE = 1e-3
ROWS = ("all", "centro", "sohncke", "other", "mixed", "improper_achiral", "legacy_unrep")
ROW_PRED = {"all": lambda r: True, "centro": lambda r: r["class"] == "centro",
            "sohncke": lambda r: r["class"] == "sohncke", "other": lambda r: r["class"] == "other",
            "mixed": lambda r: r["mixed"], "improper_achiral": lambda r: r["improper_achiral"],
            "legacy_unrep": lambda r: r["legacy_unrep"]}
SCRIPTS = ("scripts/n3_table_r.py", "scripts/n3_legacy_oracle.py", "scripts/n3_tpool.py", "scripts/n3_common.py",
           "scripts/n3_chirality.py", "scripts/g2_asym_baselines.py", "scripts/n3_mcf_export.py",
           "external/mcf/data-preprocess/thurlemann23/common.py")
FINAL = ("ok", "timeout", "deterministic")
_STARTS = []


# ---------------------------------------------------------------------------------------------------
# (c) homochiral approximant
# ---------------------------------------------------------------------------------------------------

def frozen_starts():
    """The 16 frozen ICP starts of §5 (c), identical for every molecule; fingerprint-checked."""
    if not _STARTS:
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(0)
        S = [np.diag(s).astype(float) for s in SIGN_DIAG] + \
            [Rotation.random(random_state=int(rng.integers(1e9))).as_matrix() for _ in range(12)]
        fp = hashlib.sha256(np.ascontiguousarray(np.round(np.stack(S), 8) + 0.0, dtype="<f8").tobytes()).hexdigest()
        if fp != STARTS16_SHA256:
            raise RuntimeError(f"n3_table_r: the 16 frozen ICP starts changed (SHA-256 {fp}); scipy "
                               "Rotation.random(random_state=int) no longer draws as scipy 1.17.1 did")
        _STARTS.extend(S)
    return list(_STARTS)


def _icp(Y, Ym, blocks, R, iters=30):
    """Element-blocked Hungarian ICP of Y onto the mirror Ym from start R -> (rmsd, Yh = Y[p] R^T)."""
    from scipy.optimize import linear_sum_assignment
    from n3_chirality import kabsch
    P = Y @ R.T
    for _ in range(iters):
        p = np.empty(len(Y), dtype=int)
        for idx in blocks:
            D = ((Ym[idx][:, None] - P[idx][None]) ** 2).sum(-1)
            ri, ci = linear_sum_assignment(D)
            p[idx[ri]] = idx[ci]
        r, R = kabsch(Ym, Y[p])
        Pn = Y @ R.T
        if np.abs(Pn - P).max() < 1e-8:
            break
        P = Pn
    return r, Y[p] @ R.T


def automorphism_start(Y, Z):
    """§2.6 (i), as n3_chirality.mirror_rmsd: det+1 Kabsch of the mirror onto the conformer over the first 200
    element-matched bond-graph automorphisms -> (parser rmsd, start rotation). With X ~ Ymir[perm] R^T the
    homochiral form is Ymir ~ X[perm^-1] R, i.e. start rotation R^T (same RMSD)."""
    from networkx.algorithms.isomorphism import GraphMatcher
    from n3_chirality import graph, kabsch
    g = graph(Y, Z)
    Ymir = Y * MIR
    gm = GraphMatcher(g, g, node_match=lambda u, v: u["element"] == v["element"])
    parser, bestR = np.inf, None
    for n, mp in enumerate(gm.isomorphisms_iter()):
        if n >= MAX_ISO:
            break
        perm = np.array([mp[k] for k in range(len(Z))])
        r, R = kabsch(Y, Ymir[perm])
        if r < parser:
            parser, bestR = r, R
    return float(parser), (bestR.T if bestR is not None else None)


def set_dist(A, B, Z):
    """Distance between two placements of the same atoms, as element-blocked point sets (max nearest-neighbour)."""
    d = 0.0
    for z in np.unique(Z):
        m = Z == z
        D = np.sqrt(((A[m][:, None] - B[m][None]) ** 2).sum(-1))
        d = max(d, float(D.min(1).max()), float(D.min(0).max()))
    return d


def homochiral_approximants(Y, Z, iters=30):
    """-> dict(rmsd, optima [Yh, best first], n_tied, best_start (16 = automorphism start), parser,
    rmsd16, Yh16). Yh[i] approximates the mirror Ym[i] = Y[i] diag(1,1,-1) with a proper rotation of Y."""
    Y, Z = np.asarray(Y, float), np.asarray(Z)
    Ym = Y * MIR
    blocks = [np.nonzero(Z == z)[0] for z in np.unique(Z)]
    parser, Ra = automorphism_start(Y, Z)
    starts = frozen_starts() + ([Ra] if Ra is not None else [])
    ends = [_icp(Y, Ym, blocks, R, iters) for R in starts]
    k16 = min(range(16), key=lambda k: ends[k][0])                   # first strict minimum (protocol text)
    kb = min(range(len(ends)), key=lambda k: ends[k][0])
    best = ends[kb][0]
    tied = [kb] + [k for k in range(len(ends)) if k != kb and ends[k][0] - best < TIE_RMSD]
    optima = []
    for k in tied:
        if all(set_dist(ends[k][1], o, Z) > DISTINCT_A for o in optima):
            optima.append(ends[k][1])
    return {"rmsd": float(best), "optima": optima, "n_tied": len(tied), "best_start": kb, "parser": parser,
            "rmsd16": float(ends[k16][0]), "Yh16": ends[k16][1]}


def approximant_structure(a, Yh):
    """True lattice/centroids/orientations; each improper copy k -> proper orient_k diag(1,1,-1) with the
    homochiral conformer Yh in place of the mirrored one."""
    import torch
    from g2_asym_baselines import expand, to_structure
    L, cent, orient, local, Zc, am, mm = expand(a, a["R0"])
    orient, local = orient.clone(), local.clone()
    Dm = torch.diag(torch.tensor(MIR, dtype=orient.dtype))
    imp = torch.nonzero(torch.linalg.det(orient) < 0).flatten().tolist()
    for k in imp:
        orient[k] = orient[k] @ Dm
        local[k] = torch.as_tensor(Yh, dtype=local.dtype)
    assert bool((torch.linalg.det(orient) > 0).all())
    return to_structure(L, cent, orient, local, Zc, am, mm), len(imp)


def orbit_check(a):
    """Build step 2 (i): max residual of W_k c0 + t_k to the stored copy centroids and min orbit separation."""
    import torch
    o = a["orig"]
    real = [m for m in range(o["mol_mask"].shape[0]) if bool(o["mol_mask"][m])]
    cents = o["centroid"][real].double()
    orb = torch.einsum("kij,j->ki", a["W"], a["c0"]) + a["t"]
    wrap = lambda x: x - torch.round(x)
    d_oc = wrap(orb[:, None, :] - cents[None, :, :]).abs().amax(-1)
    d_oo = wrap(orb[:, None, :] - orb[None, :, :]).abs().amax(-1)
    d_oo.fill_diagonal_(1.0)
    res, sep = float(d_oc.min(1).values.max()), float(d_oo.min())
    return res <= 1e-3 and sep >= 1e-3, res, sep


# ---------------------------------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------------------------------

def _init(set_name, want_b):
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from n3_legacy_oracle import matchers
    state = {"items": C.load_set(set_name), "sms": matchers()}
    if want_b:
        from n3_mcf_export import load_export
        entries, side = load_export(set_name)
        state["export"] = dict(zip(side["refcodes"], entries))
        state["export_sha"] = side["sha256"]
    return state


def _job(state, task):
    """task = (i, col) -> matcher outcomes for column a, b or c of crystal i."""
    warnings.filterwarnings("ignore")
    from n3_legacy_oracle import match_all
    i, col = task
    a = state["items"][i]
    truth = C.truth_structure(a)
    sms = state["sms"]
    if col == "a":
        ok, res, sep = orbit_check(a)
        return {"orbit_ok": ok, "orbit_res": res, "orbit_sep": sep,
                **match_all(sms, truth, C.rasym_structure(a, a["R0"]))}
    if col == "b":
        from n3_mcf_export import entry_cells
        cand = C.cell_structure(entry_cells(state["export"][a["refcode"]]))
        return {"pickle_sha256": state["export_sha"], **match_all(sms, truth, cand)}
    if col == "c":
        t0 = time.time()
        Z = a["Z"].numpy()
        h = homochiral_approximants(a["local"].numpy(), Z)
        t_icp = time.time() - t0
        outs = []
        for Yh in h["optima"]:
            cand, n_imp = approximant_structure(a, Yh)
            outs.append(match_all(sms, truth, cand))
        k16 = next((k for k, Yh in enumerate(h["optima"]) if set_dist(Yh, h["Yh16"], Z) <= DISTINCT_A), None)
        f16 = outs[k16] if k16 is not None else match_all(sms, truth, approximant_structure(a, h["Yh16"])[0])
        return {"method": C_METHOD, "homochiral_rmsd": h["rmsd"], "rmsd_frozen16": h["rmsd16"],
                "best_start": h["best_start"], "auto_parser_rmsd": h["parser"], "n_tied": h["n_tied"],
                "n_optima": len(outs), "optima": outs, "frozen16": {**f16, "optimum_index": k16},
                "n_improper_copies": n_imp, "t_icp": round(t_icp, 3),
                **{m: any(o[m] for o in outs) for m in MATCHERS}}
    raise ValueError(col)


# ---------------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------------

def is_final(x):
    return x is not None and (x["status"] in FINAL or x.get("attempt", 1) == 2)


def outcome(x, sub=None):
    """Row -> (status, {matcher: bool} or None); status ok / timeout / deterministic / error / missing."""
    if x is None:
        return "missing", None
    if x["status"] == "ok":
        src = x[sub] if sub else x
        return "ok", {m: bool(src[m]) for m in MATCHERS}
    return ("error" if x["status"] in ("error", "crash") else x["status"]), None


def cell(outs, extra=()):
    """outs: list of (status, matches) -> counts per matcher (non-ok = non-match) + one count per status."""
    c = {m: 0 for m in MATCHERS}
    c.update({k: 0 for k in ("timeout", "deterministic", "error", "missing", *extra)})
    for st, mt in outs:
        if st == "ok":
            for m in MATCHERS:
                c[m] += mt[m]
        else:
            c[st] += 1
    return c


def b_source(name, labels):
    """Export + Gate F state of the set -> (info, None or dict(exported, nm_ok, f2_fail, f2_match, sha))."""
    rel = lambda p: os.path.relpath(p, C.REPO).replace("\\", "/")
    gpath = os.path.join(C.REPO, "results", "n3", f"gateF_{name}.json")
    if not os.path.exists(gpath):
        return {"status": "absent", "gate": rel(gpath)}, None
    g = json.load(open(gpath))
    try:
        from n3_mcf_export import paths
        p = paths(name)
    except Exception as ex:
        return {"status": f"n3_mcf_export unavailable: {type(ex).__name__}"}, None
    if not (os.path.exists(p["pickle"]) and os.path.exists(p["sidecar"])):
        return {"status": "no export pickle", "gate_sha256": C.sha256(gpath),
                "export_failures": len(g.get("export_failure_refcodes") or [])}, None
    side = json.load(open(p["sidecar"]))
    if g.get("pickle_sha256") not in (None, side["sha256"]):
        return {"status": "gateF and export pickle differ (stale)", "gate_sha256": C.sha256(gpath)}, None
    refs = {r["refcode"] for r in labels}
    exported = list(side["refcodes"])
    fails = set(g.get("export_failure_refcodes") or [])
    assert set(exported) <= refs and not (set(exported) & fails), f"{name}: sidecar refcodes not in the set"
    f2 = g.get("F2") or {}
    f2_fail = set(f2.get("fail_refcodes") or [])
    dpath = os.path.join(C.PRIVATE, f"gateF_{name}_detail.jsonl")
    if os.path.exists(dpath):
        from n3_tpool import read_jsonl
        det = {x["refcode"]: x for x in read_jsonl(dpath, trim=False)}
        nm_ok = {r: bool(det.get(r) and det[r].get("ok") and det[r].get("species_ok")
                         and det[r].get("assemble_err", np.inf) <= TOL_ASSEMBLE) for r in exported}
        f2_match = {r: bool(det.get(r) and det[r].get("match")) for r in exported}
        nm_src = "gateF detail (assemble_err <= 1e-3 A and species)"
    else:                                                  # conservative: any F2 failure fails every matcher
        nm_ok = {r: r not in f2_fail for r in exported}
        f2_match = None
        nm_src = "F2.fail_refcodes (conservative: detail file absent)"
    info = {"status": "computed", "gate": rel(gpath), "gate_sha256": C.sha256(gpath),
            "pickle": rel(p["pickle"]), "pickle_sha256": side["sha256"], "exported": len(exported),
            "export_failures": len(set(r["refcode"] for r in labels) - set(exported)),
            "f2_fail_refcodes": len(f2_fail), "nonmatch_checks_from": nm_src,
            "detail_sha256": C.sha256(dpath) if os.path.exists(dpath) else None}
    return info, {"exported": set(exported), "nm_ok": nm_ok, "f2_fail": f2_fail, "f2_match": f2_match,
                  "sha": side["sha256"]}


def corpus_context():
    import torch
    out = {}
    for src, n_label in (("csd_mol", "dev"), ("csd_testB", "pool")):
        parts = sorted(glob.glob(os.path.join(C.REPO, "data", src, "ds_o3.pt.parts", "part_*.pt")))
        if not parts:
            continue
        rows = [r for p in parts for r in torch.load(p, weights_only=False)]
        out[n_label] = {"cifs": len(rows), "kept_det1": sum(bool(r[1]) for r in rows),
                        "kept_o3": sum(r[2] is not None for r in rows)}
    if "dev" in out:
        out["dev"]["eligible_g2_asym"] = len(torch.load(C.DEV_CACHE, weights_only=False))
        out["dev"]["o3_kept_not_eligible"] = out["dev"]["kept_o3"] - out["dev"]["eligible_g2_asym"]
    build = os.path.join(C.REPO, "results", "n3", "n3_build_testB.json")
    if "pool" in out and os.path.exists(build):
        out["pool"]["build_counts"] = json.load(open(build)).get("counts")
    out["note"] = ("MCF's per-copy representation also covers the Z' > 1 and special-position crystals "
                   "our design excludes (o3_kept_not_eligible).")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("val", "sel", "valsel", "testB", "devtest"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--cols", nargs="+", default=["a", "b", "c"], choices=["a", "b", "c"])
    ap.add_argument("--limit", type=int, default=0, help="first N crystals only (validation; output marked partial)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from n3_legacy_oracle import load_labels, oracle_matched
    from n3_tpool import FAILED, provenance, read_jsonl, rerun_once, resolve_rerun, run

    t_start = time.time()
    name = args.set
    labels = load_labels(name)
    n = len(labels)
    lim = min(args.limit or n, n)
    frozen_starts()                                          # fingerprint check before any work
    b_info, bsrc = b_source(name, labels)
    want_b = "b" in args.cols and bsrc is not None
    if "b" in args.cols and bsrc is None:
        print(f"{name}: column (b) not computable: {b_info['status']}", flush=True)

    jl = os.path.join(C.PRIVATE, f"table_r_{name}.jsonl")
    done, stale = {}, 0
    for x in read_jsonl(jl):
        assert labels[x["i"]]["refcode"] == x["refcode"], f"{jl}: stale row for i={x['i']}"
        if (x["col"] == "c" and x.get("method") != C_METHOD) or \
                (x["col"] == "b" and (bsrc is None or x.get("pickle_sha256") != bsrc["sha"])):
            stale += 1                                       # another (c) method or another export pickle
            continue
        done[(x["i"], x["col"])] = x
    need = {"a": [i for i in range(lim)], "c": [i for i in range(lim) if labels[i]["improper"]],
            "b": [i for i in range(lim) if bsrc and labels[i]["refcode"] in bsrc["exported"]]}
    cols = [c for c in ("a", "b", "c") if c in args.cols and (c != "b" or want_b)]
    tasks = [(i, col) for col in cols for i in need[col] if (i, col) not in done]
    retry = [k for k, x in done.items() if k[1] in cols and k[0] < lim and x["status"] in FAILED
             and x.get("attempt", 1) == 1]
    print(f"{name}: n={n}, {len(tasks)} column jobs to run, {len(retry)} to re-run (limit {lim}, workers "
          f"{args.workers}; {stale} superseded rows ignored)", flush=True)
    t_run = time.time()
    init_args = (name, want_b)
    with open(jl, "a", newline="\n") as fh:
        def write(row):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done[(row["i"], row["col"])] = row

        for k, ((i, col), status, res, dt) in enumerate(run(_job, tasks, init=_init, init_args=init_args,
                                                             workers=args.workers, timeout=args.timeout)):
            row = {"i": i, "refcode": labels[i]["refcode"], "col": col, "status": status, "dt": round(dt, 2),
                   "attempt": 1}
            if status == "ok":
                row.update(res)
            elif status == "error":
                row["error"] = res
            if col == "c":
                row.setdefault("method", C_METHOD)
            if col == "b":
                row.setdefault("pickle_sha256", bsrc["sha"])
            write(row)
            if status in FAILED:
                retry.append((i, col))
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(tasks)} ({time.time() - t_run:.0f}s)", flush=True)
        for (i, col) in retry:                               # §1.3: once, alone, in a fresh process
            first = done[(i, col)]
            status, res, dt = rerun_once(_job, (i, col), init=_init, init_args=init_args, timeout=args.timeout)
            row = {"i": i, "refcode": labels[i]["refcode"], "col": col, "status": resolve_rerun(first, status, res),
                   "dt": round(dt, 2), "attempt": 2, "first_status": first["status"],
                   "first_error": first.get("error")}
            if status == "ok":
                row.update(res)
            elif status == "error":
                row["error"] = res
            if col == "c":
                row.setdefault("method", C_METHOD)
            if col == "b":
                row.setdefault("pickle_sha256", bsrc["sha"])
            write(row)
            print(f"  re-run ({i}, {col}): {first['status']} -> {row['status']}", flush=True)
    t_run = time.time() - t_run

    # per-crystal outcomes; (c) for crystals with no improper op is the expansion itself -> (a)
    def c_out(i, sub=None):
        if labels[i]["improper"]:
            return outcome(done.get((i, "c")), sub)
        return outcome(done.get((i, "a")))

    def b_out(i):
        r = labels[i]["refcode"]
        if r not in bsrc["exported"]:
            return "export_failure", None
        if not bsrc["nm_ok"][r]:
            return "f2_nonmatch_fail", None
        return outcome(done.get((i, "b")))

    lo_path = os.path.join(C.REPO, "results", "n3", f"legacy_oracle_{name}.json")
    lo_jl = os.path.join(C.PRIVATE, f"legacy_oracle_{name}.jsonl")
    lo = {x["refcode"]: x for x in read_jsonl(lo_jl, trim=False)}

    table = {}
    for rname in ROWS:
        idx = [i for i in range(lim) if ROW_PRED[rname](labels[i])]
        cell_a = cell([outcome(done.get((i, "a"))) for i in idx])
        cell_a["orbit_ok"] = sum(1 for i in idx if (done.get((i, "a")) or {}).get("orbit_ok"))
        cell_c = cell([c_out(i) for i in idx])
        cell_c16 = cell([c_out(i, "frozen16") for i in idx])
        cell_c["from_a_no_improper_op"] = cell_c16["from_a_no_improper_op"] = \
            sum(1 for i in idx if not labels[i]["improper"])
        cell_c["multiple_optima"] = sum(1 for i in idx if labels[i]["improper"]
                                        and (done.get((i, "c")) or {}).get("n_optima", 0) > 1)
        kept = [i for i in idx if labels[i]["legacy_ok"]]
        orc = [lo.get(labels[i]["refcode"]) for i in kept]
        cell_d = {"flag_kept": len(kept),
                  "oracle": {m: sum(1 for x in orc if oracle_matched(x, m)) for m in MATCHERS},
                  "oracle_fresh_registry_rejected": sum(1 for x in orc if x and x["status"] == "ok"
                                                        and not x.get("parsed")),
                  "oracle_timeout": sum(1 for x in orc if x and x["status"] == "timeout"),
                  "oracle_deterministic": sum(1 for x in orc if x and x["status"] == "deterministic"),
                  "oracle_error": sum(1 for x in orc if x and x["status"] in ("error", "crash")),
                  "oracle_missing": sum(1 for x in orc if x is None)}
        cell_b = cell([b_out(i) for i in idx], extra=("export_failure", "f2_nonmatch_fail")) if bsrc else None
        table[rname] = {"n": len(idx), "a": cell_a, "b": cell_b, "c": cell_c, "c_frozen16": cell_c16,
                        "d": cell_d}

    # completeness: every column final on every crystal of the set
    miss = {"a": sum(1 for i in range(n) if not is_final(done.get((i, "a")))),
            "c": sum(1 for i in range(n) if labels[i]["improper"] and not is_final(done.get((i, "c")))),
            "b": (sum(1 for i in range(n) if labels[i]["refcode"] in bsrc["exported"]
                      and not is_final(done.get((i, "b")))) if bsrc else None),
            "d_oracle": sum(1 for r in labels if r["legacy_ok"] and not is_final(lo.get(r["refcode"])))}
    complete = {"a": miss["a"] == 0, "b": bsrc is not None and miss["b"] == 0, "c": miss["c"] == 0,
                "d": miss["d_oracle"] == 0}
    complete["all"] = all(complete.values()) and lim == n

    # (b) primary cross-check against Gate F2
    b_check = None
    if bsrc:
        mine = {labels[i]["refcode"]: b_out(i) for i in range(lim)}
        f2_pass = {r for r in mine if r in bsrc["exported"] and r not in bsrc["f2_fail"]}
        mine_pass = {r for r, (st, mt) in mine.items() if st == "ok" and mt["primary"]}
        judged = {r for r, (st, _) in mine.items() if st in ("ok", "export_failure", "f2_nonmatch_fail")}
        b_check = {"judged": len(judged), "disagree_with_F2_fail_refcodes": len((f2_pass ^ mine_pass) & judged)}
        if bsrc["f2_match"] is not None:
            b_check["disagree_with_detail_match"] = sum(
                1 for i in range(lim) if (i, "b") in done and done[(i, "b")]["status"] == "ok"
                and bool(done[(i, "b")]["primary"]) != bsrc["f2_match"][labels[i]["refcode"]])
        if b_check["disagree_with_F2_fail_refcodes"] or b_check.get("disagree_with_detail_match"):
            print(f"  WARNING (b) primary disagrees with Gate F2: {b_check}", flush=True)

    # (c) diagnostics
    imp = [i for i in range(lim) if labels[i]["improper"] and (done.get((i, "c")) or {}).get("status") == "ok"]
    cr = {i: done[(i, "c")] for i in imp}
    rms = [cr[i]["homochiral_rmsd"] for i in imp]
    c_diag = {
        "n_improper_ok": len(imp),
        "homochiral_rmsd": {"median": statistics.median(rms) if rms else None, "min": min(rms, default=None),
                            "max": max(rms, default=None),
                            "median_mixed": statistics.median(mx) if (mx := [cr[i]["homochiral_rmsd"] for i in imp
                                                                             if labels[i]["mixed"]]) else None},
        "above_refined_mirror_rmsd_1e-6": sum(cr[i]["homochiral_rmsd"] > labels[i]["refined"] + 1e-6 for i in imp),
        "frozen16_above_refined_mirror_rmsd_1e-6": sum(cr[i]["rmsd_frozen16"] > labels[i]["refined"] + 1e-6
                                                       for i in imp),
        "improved_by_automorphism_start_1e-6": sum(cr[i]["rmsd_frozen16"] - cr[i]["homochiral_rmsd"] > 1e-6
                                                   for i in imp),
        "max_rmsd_gain_A": max((cr[i]["rmsd_frozen16"] - cr[i]["homochiral_rmsd"] for i in imp), default=None),
        "multiple_distinct_optima": sum(cr[i]["n_optima"] > 1 for i in imp),
        "outcome_changed_vs_frozen16": {m: sum(bool(cr[i][m]) != bool(cr[i]["frozen16"][m]) for i in imp)
                                        for m in MATCHERS},
        "automorphism_parser_vs_labels_max_abs": max((abs(cr[i]["auto_parser_rmsd"] - labels[i]["parser"])
                                                      for i in imp), default=None)}

    mixed = table["mixed"]
    unres = lambda c: c["timeout"] + c["error"] + c["missing"]
    kept_all = [i for i in range(lim) if labels[i]["legacy_ok"]]
    o_miss = [i for i in kept_all if is_final(lo.get(labels[i]["refcode"]))
              and not oracle_matched(lo.get(labels[i]["refcode"]), "primary")
              and lo[labels[i]["refcode"]]["status"] != "timeout"]
    N = table["legacy_unrep"]["n"]
    M = sum(1 for i in range(lim) if labels[i]["legacy_unrep"] and labels[i]["class"] == "centro")
    wording = {"final": complete["all"], "n": table["all"]["n"], "N_legacy_unrep": N, "M_legacy_unrep_centro": M,
               "N_flag_or_oracle_miss": N + len(o_miss),
               "M_flag_or_oracle_miss": M + sum(1 for i in o_miss if labels[i]["class"] == "centro"),
               "oracle_unresolved_flag_kept": sum(1 for i in kept_all if not is_final(lo.get(labels[i]["refcode"]))
                                                  or lo[labels[i]["refcode"]]["status"] == "timeout"),
               "n_mixed": mixed["n"],
               "P_mixed_not_matched_primary": mixed["n"] - mixed["c"]["primary"] - unres(mixed["c"]),
               "Q_mixed_matched_stol0.5": mixed["c"]["stol0.5"],
               "P_frozen16": mixed["n"] - mixed["c_frozen16"]["primary"] - unres(mixed["c_frozen16"]),
               "Q_frozen16": mixed["c_frozen16"]["stol0.5"],
               "mixed_c_unresolved": unres(mixed["c"]),
               "rules": {"N, M": "part-file keep flag (§5 d); *_flag_or_oracle_miss adds flag-kept crystals whose "
                                 "det+1 rebuild misses the primary matcher",
                         "P, Q": "c (16 frozen starts + automorphism start, tied optima any-match; proposed A2)",
                         "P_frozen16, Q_frozen16": "the 16 frozen starts alone (protocol text)",
                         "P": "deterministic matcher failures count as non-matches; timeouts, errors and missing "
                              "rows are excluded and given as mixed_c_unresolved"},
               "template": ("The legacy det+1 pipeline cannot represent N/n test crystals (M of the centrosymmetric "
                            "ones). The released MolCrystalFlow CSP driver cannot produce P/n_mixed mixed-handedness "
                            "cells under the default StructureMatcher; a homochiral approximant nevertheless matches "
                            "Q/n_mixed at stol 0.5.")}
    if complete["all"]:
        fill = lambda P, Q: (f"The legacy det+1 pipeline cannot represent {N}/{wording['n']} test crystals ({M} of the "
                             f"centrosymmetric ones). The released MolCrystalFlow CSP driver cannot produce "
                             f"{P}/{mixed['n']} mixed-handedness cells under the default StructureMatcher; a "
                             f"homochiral approximant nevertheless matches {Q}/{mixed['n']} at stol 0.5.")
        wording["text"] = fill(wording["P_mixed_not_matched_primary"], wording["Q_mixed_matched_stol0.5"])
        wording["text_frozen16"] = fill(wording["P_frozen16"], wording["Q_frozen16"])
    else:
        wording["text"] = None
        wording["not_final_because"] = [k for k, v in complete.items() if k != "all" and not v] + \
            (["--limit"] if lim < n else [])

    build_drops = None
    bpath = os.path.join(C.REPO, "results", "n3", "n3_build_testB.json")
    if name in ("sel", "testB", "valsel") and os.path.exists(bpath):
        bc = json.load(open(bpath)).get("counts", {})
        build_drops = {"after_g1_oracle": bc.get("after_g1_oracle"), "step2_drops": bc.get("step2_drops"),
                       "after_step2": bc.get("after_step2")}
    dts = {col: [x["dt"] for (i, c), x in done.items() if c == col and i < lim] for col in ("a", "b", "c")}
    out = {"set": name, "n": n, "evaluated_crystals": lim, "partial": not complete["all"],
           "complete": complete, "missing_column_jobs": miss,
           "rows": table,
           "a_by_construction": name in ("sel", "testB"), "build_step2": build_drops,
           "b_source": b_info, "b_check_vs_gateF2": b_check,
           "c_method": C_METHOD, "c_frozen16_method": "16 frozen starts, first strict minimum (protocol text)",
           "c_diagnostics": c_diag,
           "d_source": {"legacy_oracle_json": os.path.relpath(lo_path, C.REPO).replace("\\", "/")
                        if os.path.exists(lo_path) else None,
                        "legacy_oracle_jsonl_sha256": C.sha256(lo_jl) if os.path.exists(lo_jl) else None},
           "wording": wording,
           "corpus_context": corpus_context(),
           "matchers": {"primary": "StructureMatcher() fit(truth, cand)",
                        **{f"stol{s}": f"StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol={s})"
                                       ".get_rms_dist(truth, cand) is not None" for s in STOLS}},
           "timing_s": {"this_run_wall": round(t_run, 1), "this_run_jobs": len(tasks),
                        "this_run_reruns": len(retry), "workers": args.workers, "timeout_per_job": args.timeout,
                        **{f"{col}_job_sum": round(sum(v), 1) for col, v in dts.items()},
                        **{f"{col}_job_max": max(v) if v else None for col, v in dts.items()}},
           "labels_sha256": C.sha256(os.path.join(C.PRIVATE, f"labels_{name}.json")),
           "private_jsonl_sha256": C.sha256(jl),
           "starts16_sha256": STARTS16_SHA256,
           "provenance": provenance(SCRIPTS)}
    with open(os.path.join(C.REPO, "results", "n3", f"table_r_{name}.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    state = "complete" if complete["all"] else f"PARTIAL: missing {miss}, limit {lim}/{n}"
    print(f"{name} Table R ({state}; {time.time() - t_start:.0f}s):")
    fmt = lambda c: "/".join(str(c[m]) for m in MATCHERS) + "".join(
        f" {k} {c[k]}" for k in ("timeout", "deterministic", "error", "missing", "export_failure",
                                 "f2_nonmatch_fail") if c.get(k))
    for rname in ROWS:
        r = table[rname]
        d = r["d"]
        d_bad = {k: d[f"oracle_{k}"] for k in ("fresh_registry_rejected", "timeout", "deterministic", "error",
                                               "missing") if d[f"oracle_{k}"]}
        print(f"  {rname:17s} n={r['n']:4d}  a {fmt(r['a'])}  b {fmt(r['b']) if r['b'] else '-'}  "
              f"c {fmt(r['c'])}  c16 {fmt(r['c_frozen16'])}  d flag {d['flag_kept']} oracle "
              f"{'/'.join(str(d['oracle'][m]) for m in MATCHERS)}" + (f" {d_bad}" if d_bad else ""))
    print(f"  (c) diagnostics: {json.dumps(c_diag)}")
    if b_check:
        print(f"  (b) vs Gate F2: {json.dumps(b_check)}")
    print(f"  wording{'' if wording['final'] else ' (PROVISIONAL, not final)'}: "
          f"{json.dumps({k: v for k, v in wording.items() if k not in ('template', 'rules', 'text', 'text_frozen16')})}")


if __name__ == "__main__":
    main()
