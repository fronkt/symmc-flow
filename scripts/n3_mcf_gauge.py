"""N3 / G3: MCF-R+G gauge scan (tasks/n3_protocol.md §3.2 "(ii-R+G)"; deployable, no truth used).

MCF's network sees the lattice only through L L^T and orientations only through R_i^T R_j, and its symmetric
prior conjugates copy i by g_(i mod 4) in {I, C2z, C2y, C2x} of its standardized Cartesian frame, so the sampled
orientation distribution is invariant under the commutant H of the g_i drawn. H by z_MCF, in the EXPORTED
standardized frame (the pickle's lattice_1 frame; z = the normal to the two shortest cell vectors):
    z_MCF = 2   {Rz(t)} u {Rz(t) C2x}: both cosets on a 2 degree grid (360 members, identity first)
    z_MCF >= 3  D2 = {I, C2x, C2y, C2z}
    z_MCF = 1   no scan: MCF-R+G = MCF-R (its draws and LJ are reused unchanged)
For every draw of an MCF-R run, each h in H is applied as a common left rotation h R_k of all copies with the
lattice and centroids fixed (X'_k = (X_k - T_k) h^T + T_k, T_k = pred_trans_k @ lattice, MCF's own assembly),
its unrelaxed steric LJ is evaluated through the Gate-F5 adapter (energy() unchanged) and the lowest-LJ member
is kept (non-finite last; exact ties -> lowest member index, i.e. the identity first). The +G candidate carries
its source draw's RESID (exact whenever diagnostic (a) passes at that state: the RESID pass records, per draw,
the copy-graph edges on MCF's azimuth branch cut, where it is not guaranteed) and validity.

Outputs: a +G draw bundle (kind 'cells', sel 'lj' = the kept member's LJ, 'resid'), per-draw rows
<run>/gauge_<S>_step<step>.jsonl (resumable; member energies; refused if its recorded inputs -- grid step,
predictions / pickle SHA-256, validity -- differ)
and results/n3/private/mcf/gauge_counts/<arm>_s<seed>_<set>.json: LJ evaluations per crystal (disclosed for
every arm: 5,760 at z_MCF = 2, 64 at z_MCF >= 3, 16 at z_MCF = 1). Draws whose scan fails deterministically
under the §1.3 harness are re-scanned member by member in fresh processes; a failing member counts as +inf.

    python scripts/n3_mcf_gauge.py --run_dir n3_mcf/infer/mcfR/valsel/s0_last_u3 --workers 8
    python scripts/n3_mcf_gauge.py --run_dir <smoke run> --workers 2                 # laptop smoke (VAL)
"""
import argparse
import json
import os
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import torch

import n3_common as nc
import n3_mcf_lib as ML


def scan_draw(task):
    """All members of H for one draw -> energies (list, member order)."""
    p, T, M, z, step = task
    out = []
    for _, h in ML.commutant_members(z, step):
        q = dict(p)
        q["cart"] = ML.apply_common_rotation(p["cart"], p["mol"], T, h)
        out.append(ML.lj_of_payload(q, M)[0])
    return out


def member_task(task):
    p, T, M, z, step, m = task
    h = ML.commutant_members(z, step)[m][1]
    q = dict(p)
    q["cart"] = ML.apply_common_rotation(p["cart"], p["mol"], T, h)
    return ML.lj_of_payload(q, M)[0]


def resid_cut_draws(rp, zs):
    """Scanned (z_MCF >= 2) draws whose returned state has copy-graph edges on MCF's azimuth branch cut (from the
    RESID pass): there the carried RESID is not guaranteed equal to the RESID of the rotated state."""
    if not os.path.exists(rp["resid"]):
        return None
    cut = torch.load(rp["resid"], map_location="cpu", weights_only=False).get("branch_cut_edges")
    if cut is None:
        return None
    z2 = torch.tensor([z >= 2 for z in zs])
    return int(((cut > 0) & z2.unsqueeze(0)).sum())


def _key(v):
    v = float(v)
    return (0, v) if np.isfinite(v) else (1, 0.0)


def gauge(run_dir, src_bundle=None, workers=2, step=2.0, out=None):
    man, rp = ML.load_manifest(run_dir)
    assert man["arm"] == "R", "+G is MCF-R's gauge scan"
    S = man["num_samples"]
    src_path = src_bundle or man.get("bundle")
    if src_path is None:
        import n3_mcf_run as RUN
        sub, set_name = RUN.smoke_subset(man)
        arm = RUN.bundle_arm(man)
        src_path = (nc.bundle_path(arm, man["train_seed"], set_name) if sub is None
                    else os.path.join(ML.SMOKE_DIR, "draws", f"{arm}_s{man['train_seed']}_{set_name}.pt"))
    src = nc.load_bundle(src_path)
    assert src["meta"]["check"]["predictions_sha256"] == ML.sha256(rp["pred"]), "bundle is not this run's"
    entries, side = ML.export_set(man["set"])
    pred = ML.load_predictions(rp["pred"])
    n = len(entries)
    zs = [it["z_MCF"] for it in side["items"]]
    Ms = [it["M"] for it in side["items"]]
    members = {z: ML.commutant_members(z, step) for z in set(zs)}
    T = {}
    for s in range(S):
        tr = pred["pred_trans"][s].double().numpy()
        lat = pred["lattices"][s].double().numpy()
        nb = [int(v) for v in pred["num_bbs"][s]]
        off = np.concatenate([[0], np.cumsum(nb)])
        for i in range(n):
            T[(i, s)] = tr[off[i]:off[i + 1]] @ lat[i]
    tasks = [((i, s), (ML.cell_payload(src["cells"][i][s]), T[(i, s)], Ms[i], zs[i], step))
             for i in range(n) for s in range(S) if zs[i] >= 2 and bool(src["valid"][i, s])]
    jl = os.path.join(run_dir, f"gauge_{S}_step{step:g}.jsonl")
    params = {"step_deg": step, "predictions_sha256": ML.sha256(rp["pred"]), "pickle_sha256": side["sha256"],
              "valid_sha256": ML.tensors_sha256(src["valid"]), "members": {str(z): len(m) for z, m in members.items()}}
    t0 = time.time()
    rows = ML.run_jobs(scan_draw, tasks, jl, workers=workers, row=lambda e: {"e": e},
                       log_path=jl[:-6] + ".events.jsonl", label="+G scan", params=params)
    # deterministic draw failures: member by member in fresh processes (a failing member = +inf)
    bad = [k for k, r in rows.items() if r["status"] != "ok"]
    mem = {}
    if bad:
        tt = dict(tasks)
        mtasks = [((i, s, m), (*tt[(i, s)], m)) for (i, s) in bad for m in range(len(members[zs[i]]))]
        mem = ML.run_jobs(member_task, mtasks, jl[:-6] + "_members.jsonl", workers=workers,
                          row=lambda e: {"e": e}, label="+G members", params=params)
    cells = [[None] * S for _ in range(n)]
    lj = torch.full((n, S), float("nan"), dtype=torch.float64)
    picked, counts, fails, id_mismatch = {}, [0] * n, 0, 0
    for i in range(n):
        for s in range(S):
            if zs[i] < 2 or not bool(src["valid"][i, s]):
                cells[i][s] = src["cells"][i][s]
                lj[i, s] = src["sel"]["lj"][i, s]
                counts[i] += 1 if zs[i] < 2 and bool(src["valid"][i, s]) else 0
                continue
            r = rows[(i, s)]
            if r["status"] == "ok":
                e = r["e"]
            else:
                e = [(mem[(i, s, m)]["e"] if mem[(i, s, m)]["status"] == "ok" else float("inf"))
                     for m in range(len(members[zs[i]]))]
                fails += sum(mem[(i, s, m)]["status"] != "ok" for m in range(len(members[zs[i]])))
            e = [float("inf") if v is None else v for v in e]
            assert len(e) == len(members[zs[i]]), f"crystal {i} draw {s}: {len(e)} energies for {len(members[zs[i]])} members"
            src_e = float(src["sel"]["lj"][i, s])
            id_mismatch += not (e[0] == src_e or (np.isnan(e[0]) and np.isnan(src_e)))
            m = min(range(len(e)), key=lambda q: (_key(e[q]), q))
            (name, ang), h = members[zs[i]][m]
            c = dict(src["cells"][i][s])
            c["cart"] = torch.from_numpy(np.ascontiguousarray(
                ML.apply_common_rotation(c["cart"].double().numpy(), c["mol"].numpy(), T[(i, s)], h)))
            cells[i][s] = c
            lj[i, s] = e[m]
            counts[i] += len(e)
            picked[(i, s)] = {"member": m, "coset": name, "angle_deg": ang, "lj": e[m], "lj_identity": e[0]}
    valid = src["valid"].clone()
    import n3_mcf_run as RUN
    sub, set_name = RUN.smoke_subset(man)
    arm = src["arm"].replace("mcfR_", "mcfRG_", 1)
    seed = src["seed"]
    meta = {"source": "n3_mcf_gauge.py", "source_bundle": os.path.relpath(src_path, REPO).replace("\\", "/"),
            "source_bundle_sha256": ML.sha256(src_path), "run": src["meta"]["run"], "check": src["meta"]["check"],
            "grid_step_deg": step, "H": {"1": "none (MCF-R unchanged)", "2": "Rz(t) u Rz(t)C2x, 2 deg grid",
                                         ">=3": "D2 {I, C2x, C2y, C2z}"},
            "frame": "exported standardized frame (pickle lattice_1)", "resid": "source draw's RESID",
            "lj_evaluations_per_crystal": dict(zip(side["refcodes"], counts)),
            "member_failures": fails, "identity_lj_differs_from_source": id_mismatch, "picked": {f"{i},{s}": v for (i, s), v in picked.items()},
            "resid_branch_cut_draws": resid_cut_draws(rp, zs),
            "z_MCF": zs, "M": Ms, "sec": round(time.time() - t0, 1)}
    if sub is not None:
        meta["smoke_subset"] = sub
    out = out or (nc.bundle_path(arm, seed, set_name) if sub is None
                  else os.path.join(ML.SMOKE_DIR, "draws", f"{arm}_s{seed}_{set_name}.pt"))
    ML.save_bundle(out, arm=arm, seed=seed, set_name=set_name, refcodes=src["refcodes"], kind="cells", valid=valid,
                   sel={"lj": lj, "resid": src["sel"]["resid"].clone()}, meta=meta, cells=cells,
                   subset=sub is not None)
    cp = os.path.join(ML.SMOKE_DIR if sub is not None else ML.PRIV, "gauge_counts", f"{arm}_s{seed}_{set_name}.json")
    by_z = {}
    for z, c in zip(zs, counts):
        by_z.setdefault(str(z), []).append(c)
    ML.atomic_json({"bundle": os.path.relpath(out, REPO).replace("\\", "/"), "per_crystal": dict(zip(side["refcodes"], counts)),
                    "by_z_MCF": {z: {"n": len(v), "min": min(v), "max": max(v)} for z, v in sorted(by_z.items())},
                    "expected": {"z_MCF=1": S, "z_MCF=2": S * 360, "z_MCF>=3": S * 4}}, cp, indent=1)
    return out, meta, by_z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir", required=True, help="an MCF-R sampling run (after to_bundle)")
    ap.add_argument("--bundle", default=None, help="its MCF-R bundle (default: the path to_bundle wrote)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--step", type=float, default=2.0, help="z_MCF = 2 grid step in degrees (protocol: 2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.step != 2.0:
        print("WARNING: --step differs from the protocol's 2 degrees (not a reportable +G run)")
    out, meta, by_z = gauge(args.run_dir, args.bundle, args.workers, args.step, args.out)
    print(json.dumps({"bundle": os.path.relpath(out, REPO).replace("\\", "/"), "sec": meta["sec"],
                      "member_failures": meta["member_failures"],
                      "identity_lj_differs_from_source": meta["identity_lj_differs_from_source"],
                      "resid_branch_cut_draws": meta["resid_branch_cut_draws"],
                      "lj_evaluations_by_z_MCF": {z: sorted(set(v)) for z, v in by_z.items()},
                      "picked_identity": sum(v["member"] == 0 for v in meta["picked"].values()),
                      "picked_total": len(meta["picked"])}))


if __name__ == "__main__":
    main()
