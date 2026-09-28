"""N3 / G3: ii-S, MCF-R + post-hoc space-group expansion (tasks/n3_protocol.md §3.2 "(ii-S)"; ABLATION,
descriptive, no test; "MCF orientations inside our exact SG expansion"). CPU, from the MCF-R+G draws.

Per draw (one candidate per draw):
  1. map to our frame: the MCF cell is first expanded to the stored supercell through M (the Gate-F5 adapter:
     L_c = M L_MCF, each copy at its mult translations n L_MCF, n M^-1 in [0,1)^3), then X = X_MCF Q^T and
     L = L_c Q^T (Q from the sidecar; det Q = -1 allowed: the map is a mirror then, never assumed proper);
  2. assign each copy to op k by centroid match to W_k c0 + t_k (fractional in the true L, min-image, Hungarian;
     every match within 1e-3 and a bijection over the K copies, else the draw is an arm failure);
  3. Y_k = Rc_k^-1 (X_k - cent_k), Rc_k = L^T W_k L^-T (g2_asym_baselines.cart_ops);
  4. R^(k) = det+1 Kabsch of the asym local onto Y_k (atom order of the export = order of `orig`);
  5. expand each R^(k) exactly (g2_asym_baselines.expand) and keep the lowest steric-LJ one (energy() of the
     expanded cell; non-finite last; exact ties -> lowest k).
Output: a 'rasym' draw bundle (R [n, S, 3, 3]; sel 'lj' of the kept candidate, 'resid' of the source draw),
rows <out>.jsonl (resumable; kept k, its LJ, the max pairwise angle between the K recovered R^(k), the worst
centroid match) and K LJ evaluations per draw.

Rows <out>.jsonl are refused if the source bundle's content (cells, validity) or the pickle differs from the
inputs recorded beside them.

--oracle (input-side): the export pickle's own poses in place of draws; reports the angle between every
recovered R^(k) and R0 (protocol: recovers R0 in 200/200 DEV-TEST, max 0.014 deg). §1.1 lists the operations
allowed on TEST-B / SEL before sampling and this is not one: sel, testB and valsel are refused unless
--after_sampling (that set's MCF runs exist: VALSEL after §7 step 5 sampling, TEST-B at step 9).

    python scripts/n3_mcf_sg.py --bundle results/n3/private/draws/mcfRG_last_u3_s0_valsel.pt --workers 8
    python scripts/n3_mcf_sg.py --oracle --set val --workers 2
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

TOL_CENT = 1e-3      # fractional, max-abs component after min-image (build step 2's orbit tolerance)


def kabsch_proper(P, Y):
    """Proper R minimising sum_a |R p_a - y_a|^2 (P, Y rows; both centred here)."""
    P = P - P.mean(0)
    Y = Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Y)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def angle(Ra, Rb):
    """Degrees (ML.geodesic: accurate at every angle)."""
    return float(np.degrees(float(ML.geodesic(torch.from_numpy(np.asarray(Ra)), torch.from_numpy(np.asarray(Rb))))))


def asym_payload(a):
    return {k: a[k].double().numpy() if torch.is_tensor(a[k]) and a[k].is_floating_point() else
            (a[k].numpy() if torch.is_tensor(a[k]) else a[k]) for k in ("L", "W", "t", "c0", "local", "Z", "K", "R0")}


def recover(p, Q, M, a):
    """Steps 1-4 for one MCF 'cells' payload -> (list of K proper R^(k) in op order, worst centroid match)."""
    import n3_mcf_export as X
    from scipy.optimize import linear_sum_assignment
    Lc, cent, _, local, _, am, _ = X.mcf_cells_to_supercell(p["lattice"], p["cart"], p["species"], p["mol"], M)
    Xm = (cent @ Lc).unsqueeze(1) + local                             # whole molecules, MCF frame, [K, A, 3]
    Q = np.asarray(Q, dtype=np.float64)
    Xo = Xm.numpy() @ Q.T                                               # our frame
    L, W, t, c0, K = a["L"], a["W"], a["t"], a["c0"], int(a["K"])
    assert Xo.shape[0] == K, f"{Xo.shape[0]} copies after the supercell, K = {K}"
    ams = am.numpy()
    cents = np.stack([Xo[k][ams[k]].mean(0) for k in range(K)])
    f = cents @ np.linalg.inv(L)
    orb = (np.einsum("kij,j->ki", W, c0) + t) % 1.0
    d = f[:, None, :] - orb[None, :, :]
    d = np.abs(d - np.round(d)).max(-1)                                 # [copy, op]
    rows, cols = linear_sum_assignment(d)
    worst = float(d[rows, cols].max())
    if worst > TOL_CENT:
        raise ValueError(f"centroid match {worst:.2e} > {TOL_CENT} (no op bijection)")
    Lt = L.T
    Rc = Lt @ W @ np.linalg.inv(Lt)                                    # [K, 3, 3], det +-1
    loc = a["local"]
    out = [None] * K
    for c, k in zip(rows, cols):
        Xk = Xo[c][ams[c]]
        Y = (Xk - Xk.mean(0)) @ np.linalg.inv(Rc[k]).T                  # rows: Rc_k^-1 (X - cent)
        assert Y.shape == loc.shape, "copy atom count differs from the asym local"
        out[k] = kabsch_proper(loc, Y)
    return out, worst


def draw_task(task):
    """Steps 1-5 for one draw -> dict(kept k, R (list), lj, lj per k, spread, worst match)."""
    from g2_asym_baselines import energy, expand
    p, Q, M, a = task
    Rs, worst = recover(p, Q, M, a)
    at = {k: torch.from_numpy(np.asarray(v)) if isinstance(v, np.ndarray) else v for k, v in a.items()}
    at["Z"] = at["Z"].long()
    es = [float(energy(*expand(at, torch.from_numpy(R)))) for R in Rs]
    k = min(range(len(es)), key=lambda q: ((0, es[q]) if np.isfinite(es[q]) else (1, 0.0), q))
    spread = max((angle(Rs[x], Rs[y]) for x in range(len(Rs)) for y in range(x + 1, len(Rs))), default=0.0)
    return {"k": k, "R": Rs[k].tolist(), "lj": es[k], "lj_all": es, "spread_deg": spread, "worst_match": worst}


def oracle_task(task):
    p, Q, M, a = task
    Rs, worst = recover(p, Q, M, a)
    return {"max_deg": max(angle(R, a["R0"]) for R in Rs), "worst_match": worst}


def run_oracle(set_name, workers, limit=None):
    """Input-side: the pickle's own poses -> every R^(k) vs R0."""
    import n3_mcf_export as X
    entries, side = X.load_export(set_name)
    items = nc.load_set(set_name)
    n = len(entries) if limit is None else min(limit, len(entries))
    tasks = [((i,), (ML.cell_payload(X.entry_cells(entries[i])), side["items"][i]["Q"], side["items"][i]["M"],
                     asym_payload(items[i]))) for i in range(n)]
    jl = os.path.join(ML.PRIV, "sg_oracle", f"{set_name}.jsonl")
    if os.path.exists(jl):
        os.remove(jl)
    rows = ML.run_jobs(oracle_task, tasks, jl, workers=workers, row=lambda r: r, label="ii-S oracle")
    ok = [r for r in rows.values() if r["status"] == "ok"]
    mx = max(r["max_deg"] for r in ok) if ok else None
    res = {"set": set_name, "n": n, "recovered": len(ok), "max_deg": mx,
           "within_0.1_deg": sum(r["max_deg"] <= 0.1 for r in ok),
           "worst_centroid_match": max(r["worst_match"] for r in ok) if ok else None,
           "failed": [side["refcodes"][k[0]] for k, r in rows.items() if r["status"] != "ok"]}
    ML.atomic_json(res, os.path.join(ML.PRIV, "sg_oracle", f"{set_name}.json"), indent=1)
    return res


def run_sg(bundle_path, workers, out=None):
    src = nc.load_bundle(bundle_path)
    assert src["kind"] == "cells" and src["arm"].startswith("mcfRG"), "ii-S runs from the MCF-R+G draws"
    run = src["meta"]["run"]
    entries, side = ML.export_set(run["set"])
    assert side["sha256"] == src["meta"]["run"]["pickle_sha256"]
    items = ML.smoke_items(src, src["set"])
    n, S = src["valid"].shape
    assert len(items) == n and [a["refcode"] for a in items] == list(src["refcodes"]) == side["refcodes"]
    ap = [asym_payload(a) for a in items]
    tasks = [((i, s), (ML.cell_payload(src["cells"][i][s]), side["items"][i]["Q"], side["items"][i]["M"], ap[i]))
             for i in range(n) for s in range(S) if bool(src["valid"][i, s])]
    arm = src["arm"].replace("mcfRG_", "mcfRS_", 1)
    sub = src["meta"].get("smoke_subset")
    out = out or (nc.bundle_path(arm, src["seed"], src["set"]) if sub is None
                  else os.path.join(ML.SMOKE_DIR, "draws", f"{arm}_s{src['seed']}_{src['set']}.pt"))
    jl = out[:-3] + ".jsonl"
    t0 = time.time()
    params = {"source_content_sha256": ML.tensors_sha256(src["valid"], src["cells"]), "pickle_sha256": side["sha256"],
              "tol_centroid": TOL_CENT}
    rows = ML.run_jobs(draw_task, tasks, jl, workers=workers, row=lambda r: r, log_path=jl[:-6] + ".events.jsonl",
                       label="ii-S", params=params)
    R = torch.tile(torch.eye(3, dtype=torch.float64), (n, S, 1, 1))
    valid = torch.zeros(n, S, dtype=torch.bool)
    lj = torch.full((n, S), float("nan"), dtype=torch.float64)
    fails = []
    for (i, s), _ in tasks:
        r = rows[(i, s)]
        if r["status"] == "ok":
            R[i, s] = torch.tensor(r["R"], dtype=torch.float64)
            valid[i, s] = True
            lj[i, s] = r["lj"]
        else:
            fails.append({"refcode": src["refcodes"][i], "draw": s, "fails": r.get("fails")})
    meta = {"source": "n3_mcf_sg.py", "source_bundle": os.path.relpath(bundle_path, REPO).replace("\\", "/"),
            "source_bundle_sha256": ML.sha256(bundle_path), "run": run,
            "label": "MCF orientations inside our exact SG expansion (ablation; descriptive)",
            "lj_evaluations_per_draw": "K (one per recovered R^(k))", "resid": "source draw's RESID",
            "arm_failures": fails, "max_spread_deg": max((rows[k]["spread_deg"] for k, _ in tasks
                                                          if rows[k]["status"] == "ok"), default=None),
            "sec": round(time.time() - t0, 1)}
    if sub is not None:
        meta["smoke_subset"] = sub
    ML.save_bundle(out, arm=arm, seed=src["seed"], set_name=src["set"], refcodes=src["refcodes"], kind="rasym",
                   valid=valid, sel={"lj": lj, "resid": src["sel"]["resid"].clone()}, meta=meta, R=R,
                   subset=sub is not None)
    return out, meta, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default=None, help="an MCF-R+G bundle (n3_mcf_gauge.py)")
    ap.add_argument("--oracle", action="store_true", help="input-side: the pickle's own poses")
    ap.add_argument("--after_sampling", action="store_true",
                    help="allow --oracle on sel / testB / valsel (only after that set was sampled, §1.1)")
    ap.add_argument("--set", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.oracle:
        if args.set in ML.BLIND_SETS and not args.after_sampling:
            raise SystemExit(f"--oracle on {args.set} before its sampling is not a §1.1 input-side operation "
                             "(--after_sampling once that set's MCF runs exist)")
        print(json.dumps(run_oracle(args.set, args.workers, args.limit)))
        return
    out, meta, valid = run_sg(args.bundle, args.workers, args.out)
    print(json.dumps({"bundle": os.path.relpath(out, REPO).replace("\\", "/"), "valid": int(valid.sum()),
                      "arm_failures": len(meta["arm_failures"]), "max_spread_deg": meta["max_spread_deg"],
                      "sec": meta["sec"]}))


if __name__ == "__main__":
    main()
