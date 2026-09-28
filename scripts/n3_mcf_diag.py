"""N3 / G3: MolCrystalFlow diagnostics (tasks/n3_protocol.md §3.2 "MCF diagnostics"; oracle items labelled as such;
not results).

  a  EQUIVARIANCE, on a selected MCF-R / MCF-A checkpoint (box, §7 step 5, un-shimmed, CPU, float64): model and
     inputs in float64, the first --n VALSEL crystals (VAL part), --G random G in SO(3) per crystal applied to
     every copy (R_k -> G R_k). Gate state "generic": lattice, centroids and t of one corrupt_batch draw (seed 0;
     MCF-R: true lattice/centroids) with an independent Haar orientation per copy. Also reported: the same
     statistics at the corrupt_batch state itself, and per state the edges on the azimuth branch cut of MCF's
     gen_edges (atan2 of the relative-rotation axis, used raw in rot_unit_dots): copies that are exact C2
     conjugates (the symmetric prior at t = 0; states keeping a true mirror relation) put that axis exactly on
     the cut, where G flips the feature by 1 through a rounding zero. Statistics (maxima over crystals and G):
       q    max|q_vec(GR) - q_vec(R)| / max(1, max|q_vec(R)|)   (forward hook on FlowModel.rot_update)   <= 1e-6
       lat  max|pred_lattice(GR) - pred_lattice(R)|                                                        <= 1e-6
       tr   max|pred_b_trans(GR) - pred_b_trans(R)|                                                        <= 1e-6
       R    max|pred_R(GR) - G pred_R(R)|   (openfold casts pred_R to float32; floor ~6-7e-7)              <= 1e-5
     Controls: independent per-copy rotations G_k in place of G; each control statistic must exceed 100 x the
     corresponding G statistic (else "check the harness" before (a) is declared). Verdict PASS / FAIL /
     HARNESS_CHECK. The 50-step sampler deviation (reported, not gated): the same prior draw with R_0 and G R_0
     (lattice / centroid priors shared; P3 for MCF-R), final max|R_B - G R_A|, |d lattice|, |d trans mod 1|, for
     MCF's own FlowModule.forward in float32 and for a float64 replica of its Euler loop (MCF's sampler cannot run
     in float64: pred_R is float32 and geodesic_t's einsum rejects mixed dtypes). --pickle/--provenance: the
     repeat on the released Thurlemann checkpoint and its own test pickle.
  b  PERFECT-MODEL CEILING from truth (input-side; not in §1.1's list of operations allowed on TEST-B / SEL before
     sampling, so sel / testB / valsel need --after_sampling: VALSEL after §7 step 5 sampling, TEST-B at step 9;
     val / devtest / train any time): c_i = Haar fraction of H whose common rotation of the
     true frames (pickle frame) still matches the truth. z_MCF = 2: both cosets on a 5 degree grid, walking at 1
     degree from every hit to the window edges, c_i = (w_rot + w_flip)/720; z_MCF >= 3: #matching D2 / 4;
     z_MCF = 1: the crystal's Haar floor (iii-u draw@1, --haar_floor), labelled as such. Summary per z_MCF and
     matcher: mean c_i and mean[1 - (1 - c_i)^16]. Resumable per (crystal, matcher); the rows are refused if the
     grid, walk or pickle differ from the ones recorded beside them.
  c  ORACLE-ALIGNED draws (MCF-R only; writes a bundle, matched only by the scorer): per draw the h in H
     minimising sum_k |h R_k - R_k^true|_F^2, R^true = rotmats_1 of the same pickle (z_MCF = 2: closed form in
     theta per coset, better coset kept; z_MCF >= 3: the 4 D2 elements); LJ recomputed after alignment (F5
     adapter), RESID of the source draw. z_MCF = 1 is not computed (exact by construction): those crystals keep
     the source draws unchanged with their source validity (never arm failures), are listed in meta
     'excluded_z1_refcodes' / 'c_computed', and (c) is reported per z_MCF stratum only (z_MCF = 2, >= 3), the
     z_MCF = 1 row printed as "exact by construction (n)"; never aggregated over all crystals.
  d  CONSISTENCY (scorer-side check, from the scorer's per-draw match rows): per z_MCF stratum, MCF-R's unaligned
     draw@1 (seed-mean) with a crystal-bootstrap 95% CI (10,000, seed 0) against the (b) ceiling mean; an
     exceedance (CI entirely above the ceiling) means (b) is wrong.

    python scripts/n3_mcf_diag.py a --ckpt <selected ckpt> --set valsel --s_uR 3 --label mcfR_s0
    python scripts/n3_mcf_diag.py a --ckpt <thurlemann23/best.ckpt> --pickle <.../normalized/test_molcrystal_normalized.pkl.gz> --provenance --label thurlemann
    python scripts/n3_mcf_diag.py b --set testB --workers 8 --after_sampling          # step 9
    python scripts/n3_mcf_diag.py c --run_dir n3_mcf/infer/mcfR/testB/s0_last_u3 --workers 8
    python scripts/n3_mcf_diag.py d --ceiling results/n3/private/mcf/diag_b_testB.json --bundles <MCF-R bundles> --matcher primary
"""
import argparse
import copy
import json
import math
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

TOL = {"q": 1e-6, "lat": 1e-6, "tr": 1e-6, "R": 1e-5}
MATCHERS_B = ("primary", "mcf05", "mcf08")


# ============================================================================================ (a)
def rand_rot(n, gen):
    q = torch.randn(n, 4, dtype=torch.float64, generator=gen)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(n, 3, 3)


def _to64(b):
    b = copy.deepcopy(b)
    for k in b.keys():
        v = b[k]
        if torch.is_tensor(v) and v.is_floating_point():
            b[k] = v.double()
    return b


def diag_cfg(ckpt, arm, s_uR, s_uF, steps):
    from omegaconf import OmegaConf
    cfg = ML.ckpt_config(ckpt)
    OmegaConf.set_struct(cfg, False)
    cfg.inference = {"num_samples": 1, "gt_lattice_trans": arm == "R", "save_trajectories": False}
    cfg.interpolant.sampling.num_timesteps = steps
    cfg.interpolant.rots.sample_schedule = "exp"
    cfg.interpolant.rots.exp_rate = float(s_uR)
    cfg.interpolant.trans.scaling = 9.0 if arm == "R" else float(s_uF)
    return cfg


def replica_sampler64(mod, batch, steps, p3):
    """FlowModule.forward's Euler loop in float64 (P3 as patched), with pred_R cast to float64 before the step."""
    ip = mod.interpolant
    B = batch["lattice_1"].shape[0]
    nbb = batch["num_bbs"]
    ts = torch.linspace(float(ip._cfg.min_t), 1.0, steps, dtype=torch.float64)
    tr, R, L = batch["trans_0"], batch["rotmats_0"], batch["lattice_0"]
    if p3:
        tr, L = batch["trans_1"].clone(), batch["lattice_1"].clone()
    t1 = ts[0]
    d_t = None
    for t2 in ts[1:]:
        t = torch.ones((B, 1), dtype=torch.float64) * t1
        batch["trans_t"], batch["rotmats_t"], batch["lattice_t"] = tr, R, L
        batch["so3_t"] = batch["r3_t"] = t.repeat_interleave(nbb, dim=0)
        batch["l_t"] = t
        d_t = t2 - t1
        with torch.no_grad():
            o = mod.model(batch)
        tr2 = ip._b_trans_euler_step(d_t, t1, o["pred_b_trans"], tr)
        R2 = ip._rots_euler_step(d_t, t1, o["pred_rotmats"].double(), R)
        L2 = ip._trans_euler_step(d_t, t1, o["pred_lattice"], L, scaling=0)
        if p3:
            tr2, L2 = batch["trans_1"].clone(), batch["lattice_1"].clone()
        tr, R, L = tr2, R2, L2
        t1 = t2
    t = torch.ones((B, 1), dtype=torch.float64) * ts[-1]
    batch["trans_t"], batch["rotmats_t"], batch["lattice_t"] = tr, R, L
    batch["so3_t"] = batch["r3_t"] = t.repeat_interleave(nbb, dim=0)
    batch["l_t"] = t
    with torch.no_grad():
        o = mod.model(batch)
    ptr = batch["trans_1"].clone() if p3 else ip._b_trans_euler_step(d_t, ts[-1], o["pred_b_trans"], tr)
    pl_ = batch["lattice_1"].clone() if p3 else o["pred_lattice"]
    return {"pred_rotmats": o["pred_rotmats"].double(), "pred_trans": ptr, "lattices": pl_}


def _dev(rA, rB, Gm):
    dR = float((rB["pred_rotmats"].double() - Gm @ rA["pred_rotmats"].double()).abs().max())
    dT = float((((rB["pred_trans"].double() - rA["pred_trans"].double()) + 0.5) % 1.0 - 0.5).abs().max())
    dL = float((rB["lattices"].double() - rA["lattices"].double()).abs().max())
    return {"R": dR, "trans_mod1": dT, "lattice": dL}


def branch_cut_edges(mod, b64):
    """Edges of MCF's copy graph whose relative-rotation axis lies on gen_edges' azimuth branch cut
    (ML.branch_cut_counts), summed over the crystals of the state."""
    e, on = ML.branch_cut_counts(b64["rotmats_t"], b64["num_bbs"])
    return {"edges": int(e.sum()), "on_branch_cut": int(on.sum())}


def equivariance_stats(mod, b64, bv, n_cryst, n_G, gen):
    """The four (a) statistics over n_G common rotations G (one per crystal) and the per-copy controls."""
    cap = {}
    h = mod.model.rot_update.register_forward_hook(lambda m, i, o: cap.__setitem__("q", o.detach().clone()))

    def run(R):
        b = copy.deepcopy(b64)
        b["rotmats_t"] = R
        with torch.no_grad():
            o = mod.model(b)
        return cap["q"], o["pred_lattice"], o["pred_b_trans"], o["pred_rotmats"].double()
    try:
        q0, l0, t0_, R0 = run(b64["rotmats_t"])
        qs = max(1.0, float(q0.abs().max()))

        def stats(Gm, out):
            q, l, t, R = out
            return {"q": float((q - q0).abs().max()) / qs, "lat": float((l - l0).abs().max()),
                    "tr": float((t - t0_).abs().max()), "R": float((R - Gm @ R0).abs().max())}
        trials, ctrl = [], []
        for _ in range(n_G):
            Gm = rand_rot(n_cryst, gen)[bv]
            trials.append(stats(Gm, run(Gm @ b64["rotmats_t"])))
        for _ in range(n_G):
            Gk = rand_rot(int(bv.shape[0]), gen)
            ctrl.append(stats(Gk, run(Gk @ b64["rotmats_t"])))
    finally:
        h.remove()
    stat = {k: max(t[k] for t in trials) for k in TOL}
    cstat = {k: min(t[k] for t in ctrl) for k in TOL}
    req = {k: stat[k] <= TOL[k] for k in TOL}
    ctrl_ok = {k: cstat[k] > 100 * stat[k] for k in TOL}
    verdict = "PASS" if all(req.values()) and all(ctrl_ok.values()) else (
        "FAIL" if not all(req.values()) and all(ctrl_ok.values()) else "HARNESS_CHECK")
    return {"verdict": verdict, "statistics": stat, "requirements_met": req, "per_trial": trials,
            "controls_min": cstat, "controls_exceed_100x_statistic": ctrl_ok,
            "controls_exceed_100x_tolerance": {k: cstat[k] > 100 * TOL[k] for k in TOL}, "controls_per_trial": ctrl,
            "branch_cut_edges": branch_cut_edges(mod, b64)}


def sampler_pair(m, cast, batch, rot0, trans0, lat0, Gm, steps, p3):
    """The same prior draw with R_0 and G R_0 through the sampler (MCF's forward in float32, or the float64
    replica); deviation of the final state from exact equivariance."""
    outs = []
    for R_init in (rot0.double(), Gm @ rot0.double()):
        b = copy.deepcopy(batch)
        if cast == torch.float64:
            b = _to64(b)
        b["rotmats_0"] = R_init.to(cast)
        b["trans_0"] = trans0.to(cast)
        b["lattice_0"] = lat0.to(cast)
        t_s = time.time()
        if cast == torch.float32:
            with torch.no_grad():
                r = m.forward(b)
            r = {"pred_rotmats": r["pred_rotmats"][0], "pred_trans": r["pred_trans"][0], "lattices": r["lattices"][0]}
        else:
            r = replica_sampler64(m, b, steps, p3)
        r["sec"] = time.time() - t_s
        outs.append(r)
    return {**_dev(outs[0], outs[1], Gm), "moved_R": float((outs[1]["pred_rotmats"].double()
                                                            - outs[0]["pred_rotmats"].double()).abs().max()),
            "sec": round(outs[0]["sec"] + outs[1]["sec"], 1)}


def cmd_a(args):
    shims = ML.setup_mcf(args.shims)
    from molcrystalflow.data.dataset import MCDataset
    from torch_geometric.loader import DataLoader
    from molcrystalflow.data.interpolant import _symmetric_so3
    from molcrystalflow.data.utils import lattice6_to_mat33
    info = ML.ckpt_info(args.ckpt)
    arm = info["arm"] if not args.provenance else "A"          # released Thurlemann = joint model, P3 off
    if arm == "A" and args.s_uF is None:
        args.s_uF = 9.0
    cfg = diag_cfg(args.ckpt, arm, args.s_uR, args.s_uF, args.steps)
    if args.pickle:
        cache, pk = os.path.dirname(os.path.abspath(args.pickle)), os.path.abspath(args.pickle)
        stem = os.path.basename(pk).replace("_molcrystal_normalized.pkl.gz", "")
    else:
        cache, stem = ML.cache_dir(args.set), "test"
        ML.export_set(args.set)                                      # SHA-256 re-check
    ds = MCDataset(cache_path=os.path.join(cache, f"{stem}.pt"), dataset_cfg=cfg.data, is_training=False)
    idx = list(range(min(args.n, len(ds))))
    batch = next(iter(DataLoader([ds[i] for i in idx], batch_size=len(idx), shuffle=False)))
    torch.manual_seed(0)
    np.random.seed(0)
    mod32 = ML.load_flow_module(args.ckpt, cfg, "cpu")
    mod32.interpolant.set_device(torch.device("cpu"))
    noisy = mod32.interpolant.corrupt_batch(batch)                   # one corrupt_batch draw (float32)
    mod = copy.deepcopy(mod32).double()
    bv = noisy["batch"]
    gen = torch.Generator().manual_seed(args.g_seed)
    states = {"generic": _to64(noisy), "corrupt_batch": _to64(noisy)}
    states["generic"]["rotmats_t"] = rand_rot(int(bv.shape[0]), gen)  # independent Haar orientation per copy
    per_state = {s_: equivariance_stats(mod, b64, bv, len(idx), args.G, gen) for s_, b64 in states.items()}
    g = per_state["generic"]
    stat, cstat, req = g["statistics"], g["controls_min"], g["requirements_met"]
    verdict = g["verdict"]
    # 50-step sampler deviation (reported): same priors, R_0 vs G R_0
    torch.manual_seed(1)
    np.random.seed(1)
    M = int(bv.shape[0])
    rot0 = _symmetric_so3(M, batch.num_bbs, "cpu")
    prior_state = copy.deepcopy(states["generic"])
    prior_state["rotmats_t"] = rot0.double()
    prior_cut = branch_cut_edges(mod, prior_state)                   # the prior's copies are exact C2 conjugates
    trans0 = torch.rand(M, 3)
    ip = mod32.interpolant                                          # lattice prior draw (P3 replaces it for R)
    lat0 = lattice6_to_mat33(torch.cat([ip._lognormal.sample((len(idx),)), ip._uniform.sample((len(idx), 3))], -1))
    Gs = rand_rot(len(idx), gen)
    Gm = Gs[bv]
    samp = {}
    for name, m, cast in (("float32_native", mod32, torch.float32), ("float64_replica", mod, torch.float64)):
        try:
            samp[name] = sampler_pair(m, cast, batch, rot0, trans0, lat0, Gm, args.steps, arm == "R")
        except Exception as ex:  # noqa: BLE001 -- reported, never gated (e.g. a degenerate joint sample)
            samp[name] = {"error": f"{type(ex).__name__}: {ex}"[:300]}
    res = {"diagnostic": "a (equivariance)", "label": args.label, "arm": arm, "ckpt": info["ckpt"],
           "ckpt_sha256": info["ckpt_sha256"], "train_seed": info["seed"], "run_name": info["run_name"],
           "provenance_repeat": bool(args.provenance), "set": args.set if not args.pickle else os.path.basename(pk),
           "n_crystals": len(idx), "copies": M, "G_trials": args.G, "dtype": "float64", "device": "cpu",
           "torch_threads": torch.get_num_threads(),
           "knobs": {"s_uR": args.s_uR, "s_uF": 9.0 if arm == "R" else args.s_uF, "steps": args.steps},
           "gate_state": "generic: the first n crystals of the set in one unshuffled batch; torch.manual_seed(0) and "
                         "np.random.seed(0) before the module is loaded; one corrupt_batch draw of that batch gives "
                         "t and the lattice / centroids (MCF-R: trans/lattice corrupt=False, i.e. the truth); "
                         "rotmats_t replaced by an independent Haar orientation per copy from "
                         f"torch.Generator().manual_seed({args.g_seed}), which then draws the G and the controls",
           "sampler_deviation_threshold_1e-3": {k: (v.get("R", float("inf")) <= 1e-3 if "error" not in v else None)
                                                for k, v in samp.items()},
           "statistics": stat, "tolerances": TOL, "requirements_met": req, "states": per_state,
           "verdict": verdict, "sampler_deviation": samp, "branch_cut_edges_prior_draw": prior_cut, "shims": shims,
           "reportable": not shims, "versions": ML.lib_versions(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = args.out or os.path.join(ML.SMOKE_DIR if shims else ML.PRIV, f"diag_a_{args.label}.json")
    ML.atomic_json(res, out, indent=1)
    print(json.dumps({"verdict": verdict, "statistics": stat, "controls_min": cstat,
                      "corrupt_batch_state": {k: per_state["corrupt_batch"][k] for k in
                                              ("verdict", "statistics", "branch_cut_edges")},
                      "branch_cut_edges_generic": per_state["generic"]["branch_cut_edges"],
                      "branch_cut_edges_prior": prior_cut, "sampler_deviation": samp,
                      "reportable": res["reportable"], "out": os.path.relpath(out, REPO).replace("\\", "/")}))


# ============================================================================================ (b)
def _fit(truth, cell, T, h, matcher):
    import n3_score as SC
    q = dict(cell)
    q["cart"] = ML.apply_common_rotation(cell["cart"], cell["mol"], T, h)
    return bool(SC._compare(truth, SC.structure_from_payload(None, "cells", q), matcher)["ok"])


def ceiling_task(task):
    """c_i for one crystal and matcher (+ the matching angles / D2 elements)."""
    truth, cell, T, z, matcher, grid, walk = task
    if z >= 3:
        hits = [n for n, g in zip(ML.D2_NAMES, ML.D2) if _fit(truth, cell, T, g, matcher)]
        return {"c": len(hits) / 4.0, "hits": hits, "fits": 4}
    fits = 0
    w = {}
    for coset in ("rot", "flip"):
        def match(deg):
            h = ML.Rz(math.radians(deg))
            return _fit(truth, cell, T, h if coset == "rot" else h @ ML.C2X, matcher)
        cache = {}

        def m(deg):
            nonlocal fits
            deg %= 360
            if deg not in cache:
                cache[deg] = match(deg)
                fits += 1
            return cache[deg]
        on = set()
        for g in range(0, 360, grid):
            if m(g):
                on.add(g)
        for g in sorted(on):
            for sgn in (1, -1):
                d = g + sgn * walk
                while abs(d - g) < 360 and m(d):
                    d += sgn * walk
        w[coset] = sorted(k for k, v in cache.items() if v and k % walk == 0)
    c = (len(w["rot"]) + len(w["flip"])) * walk / 720.0
    return {"c": c, "w_rot_deg": len(w["rot"]) * walk, "w_flip_deg": len(w["flip"]) * walk,
            "rot_deg": w["rot"], "flip_deg": w["flip"], "fits": fits}


def _winit_b():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)


def cmd_b(args):
    import n3_mcf_export as X
    import n3_harness as H
    if args.set in ML.BLIND_SETS and not args.after_sampling:
        raise SystemExit(f"(b) on {args.set} before its sampling is not a §1.1 input-side operation "
                         "(--after_sampling once that set's MCF runs exist)")
    entries, side = X.load_export(args.set) if args.set in X.TARGET else ML.export_set(args.set)
    set_name = args.set if args.set in X.TARGET else side.get("n3_set")
    items = nc.load_set(set_name) if args.set in X.TARGET else [nc.load_set(set_name)[i] for i in side["val_indices"]]
    assert [a["refcode"] for a in items] == side["refcodes"]
    n = len(entries) if args.limit is None else min(args.limit, len(entries))
    sel = range(n) if args.crystals is None else [int(x) for x in args.crystals.split(",")]
    tag = args.label or os.path.basename(os.path.normpath(args.set))
    odir = ML.PRIV if args.set in X.TARGET else ML.SMOKE_DIR        # a smoke subset never lands beside real runs
    jl = os.path.join(odir, f"diag_b_{tag}.jsonl")
    ML.check_params(jl, {"grid_deg": args.grid, "walk_deg": args.walk, "pickle_sha256": side["sha256"]})
    H.repair_jsonl(jl)
    done = {(r["i"], r["matcher"]) for r in H.read_jsonl(jl)}
    haar = json.load(open(args.haar_floor)) if args.haar_floor else {}
    tasks = []
    for i in sel:
        z = side["items"][i]["z_MCF"]
        if z < 2:
            continue
        e = entries[i]
        cell = ML.cell_payload(X.entry_cells(e))
        T = e["trans_1"].double().numpy() @ e["lattice_1"].double().numpy()
        truth = nc.truth_structure(items[i])
        for m in args.matchers:
            if (i, m) not in done:
                tasks.append(((i, m), (truth, cell, T, z, m, args.grid, args.walk)))
    print(f"diag b {tag}: {len(tasks)} (crystal, matcher) tasks, {args.workers} workers", flush=True)
    t0 = time.time()
    with open(jl, "a") as fh:
        for (i, m), st, res, info in H.run(ceiling_task, tasks, workers=args.workers, slow=None, init=_winit_b):
            row = {"i": i, "refcode": side["refcodes"][i], "z_MCF": side["items"][i]["z_MCF"], "matcher": m,
                   "status": st, **(res if st == "ok" else {"c": 0.0, "fails": info.get("fails")})}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    rows = {(r["i"], r["matcher"]): r for r in H.read_jsonl(jl)}
    per, summ = {}, {}
    for i in sel:
        z = side["items"][i]["z_MCF"]
        rc = side["refcodes"][i]
        for m in args.matchers:
            if z < 2:
                c = haar.get(rc, {}).get(m) if isinstance(haar.get(rc), dict) else haar.get(rc)
                per.setdefault(rc, {"z_MCF": z})[m] = {"c": c, "source": "haar_floor (iii-u draw@1)"}
            else:
                r = rows.get((i, m))
                per.setdefault(rc, {"z_MCF": z})[m] = {"c": None if r is None else r["c"],
                                                         "status": None if r is None else r["status"]}
    for m in args.matchers:
        for zkey in ("1", "2", ">=3"):
            cs = [v[m]["c"] for v in per.values() if (str(v["z_MCF"]) == zkey or (zkey == ">=3" and v["z_MCF"] >= 3))
                  and v[m]["c"] is not None]
            nz = sum(1 for v in per.values() if str(v["z_MCF"]) == zkey or (zkey == ">=3" and v["z_MCF"] >= 3))
            summ.setdefault(m, {})["z_MCF" + ("" if zkey.startswith(">") else "=") + zkey] = {
                "n": nz, "n_with_c": len(cs),
                "mean_c_draw1_ceiling": float(np.mean(cs)) if cs else None,
                "mean_any16_ceiling": float(np.mean([1 - (1 - c) ** 16 for c in cs])) if cs else None,
                "label": "Haar floor (iii-u draw@1)" if zkey == "1" else "perfect-model ceiling (oracle)"}
    out = {"diagnostic": "b (perfect-model ceiling, oracle)", "set": set_name, "tag": tag, "matchers": args.matchers,
           "grid_deg": args.grid, "walk_deg": args.walk, "summary": summ, "per_crystal": per,
           "note": "c_i does NOT bound LJ-picked@1", "sec": round(time.time() - t0, 1)}
    path = os.path.join(odir, f"diag_b_{tag}.json")
    ML.atomic_json(out, path, indent=1)
    print(json.dumps(summ), "->", os.path.relpath(path, REPO).replace("\\", "/"))


# ============================================================================================ (c)
def align_z2(Rs, Rt):
    """argmin over h in {Rz(t)} u {Rz(t)C2x} of sum_k |h R_k - Rt_k|_F^2, closed form per coset."""
    best = None
    for coset, pre in (("rot", np.eye(3)), ("flip", ML.C2X)):
        A = np.einsum("ij,kjl->kil", pre, Rs)
        S = np.einsum("kij,klj->il", A, Rt)                  # sum_k A_k Rt_k^T
        th = math.atan2(S[0, 1] - S[1, 0], S[0, 0] + S[1, 1])
        val = math.hypot(S[0, 0] + S[1, 1], S[0, 1] - S[1, 0]) + S[2, 2]
        if best is None or val > best[0] + 1e-12:
            best = (val, coset, math.degrees(th), ML.Rz(th) @ pre)
    return best[1], best[2], best[3]


def align_d2(Rs, Rt):
    vals = [float(((np.einsum("ij,kjl->kil", g, Rs) - Rt) ** 2).sum()) for g in ML.D2]
    k = int(np.argmin(vals))
    return ML.D2_NAMES[k], 0.0, ML.D2[k]


def cmd_c(args):
    import n3_mcf_run as RUN
    man, rp = ML.load_manifest(args.run_dir)
    assert man["arm"] == "R", "(c) is MCF-R only"
    sub, set_name = RUN.smoke_subset(man)
    arm = RUN.bundle_arm(man)
    src_path = args.bundle or (nc.bundle_path(arm, man["train_seed"], set_name) if sub is None
                               else os.path.join(ML.SMOKE_DIR, "draws", f"{arm}_s{man['train_seed']}_{set_name}.pt"))
    src = nc.load_bundle(src_path)
    assert src["meta"]["check"]["predictions_sha256"] == ML.sha256(rp["pred"])
    entries, side = ML.export_set(man["set"])
    pred = ML.load_predictions(rp["pred"])
    n, S = src["valid"].shape
    zs = [it["z_MCF"] for it in side["items"]]
    Ms = [it["M"] for it in side["items"]]
    cells = [[None] * S for _ in range(n)]
    valid = src["valid"].clone()
    aligned = {}
    tasks = []
    for s in range(S):
        tr = pred["pred_trans"][s].double().numpy()
        lat = pred["lattices"][s].double().numpy()
        Rp = pred["pred_rotmats"][s].double().numpy()
        nb = [int(v) for v in pred["num_bbs"][s]]
        off = np.concatenate([[0], np.cumsum(nb)])
        for i in range(n):
            if zs[i] < 2:                       # not computed: exact by construction; source draw, source validity
                cells[i][s] = src["cells"][i][s]
                continue
            Rs, Rt = Rp[off[i]:off[i + 1]], entries[i]["rotmats_1"].double().numpy()
            name, ang, h = align_z2(Rs, Rt) if zs[i] == 2 else align_d2(Rs, Rt)
            c = dict(src["cells"][i][s])
            c["cart"] = torch.from_numpy(np.ascontiguousarray(ML.apply_common_rotation(
                c["cart"].double().numpy(), c["mol"].numpy(), tr[off[i]:off[i + 1]] @ lat[i], h)))
            cells[i][s] = c
            aligned[f"{i},{s}"] = {"h": name, "angle_deg": ang}
            if bool(valid[i, s]):
                tasks.append(((i, s), (ML.cell_payload(c), Ms[i], False)))
    jl = os.path.join(args.run_dir, f"oracle_aligned_{S}.jsonl")
    rows = ML.run_jobs(ML.lj_task, tasks, jl, workers=args.workers, row=lambda r: {"lj": r[0]}, label="(c) LJ",
                       params={"predictions_sha256": ML.sha256(rp["pred"]), "pickle_sha256": side["sha256"]})
    lj = src["sel"]["lj"].clone()                # z_MCF = 1 (and invalid draws): the source values
    for (i, s), _ in tasks:
        r = rows[(i, s)]
        lj[i, s] = r["lj"] if r["status"] == "ok" else float("inf")
    arm_c = src["arm"].replace("mcfR_", "mcfRoal_", 1)
    meta = {"source": "n3_mcf_diag.py c", "oracle": "aligned to the pickle's rotmats_1 (ORACLE ITEM, not a result)",
            "source_bundle": os.path.relpath(src_path, REPO).replace("\\", "/"), "run": src["meta"]["run"],
            "check": src["meta"]["check"],
            "excluded_z1_refcodes": [side["refcodes"][i] for i in range(n) if zs[i] < 2],
            "c_computed": [zs[i] >= 2 for i in range(n)],
            "reporting_rule": "per z_MCF stratum only (2, >= 3); z_MCF = 1 printed as 'exact by construction (n)' "
                              "(its draws here are the unaligned source draws); never aggregated over all crystals",
            "aligned": aligned, "resid": "source draw's RESID", "z_MCF": zs, "M": Ms}
    if sub is not None:
        meta["smoke_subset"] = sub
    out = args.out or (nc.bundle_path(arm_c, src["seed"], set_name) if sub is None
                       else os.path.join(ML.SMOKE_DIR, "draws", f"{arm_c}_s{src['seed']}_{set_name}.pt"))
    ML.save_bundle(out, arm=arm_c, seed=src["seed"], set_name=set_name, refcodes=src["refcodes"], kind="cells",
                   valid=valid, sel={"lj": lj, "resid": src["sel"]["resid"].clone()}, meta=meta, cells=cells,
                   subset=sub is not None)
    print(json.dumps({"bundle": os.path.relpath(out, REPO).replace("\\", "/"), "aligned_draws": len(aligned),
                      "excluded_z1_crystals": len(meta["excluded_z1_refcodes"])}))


# ============================================================================================ (d)
def cmd_d(args):
    import n3_score as SC
    ceil = json.load(open(args.ceiling))
    bundles = [nc.load_bundle(p) for p in args.bundles]
    refs = bundles[0]["refcodes"]
    zl = {rc: v["z_MCF"] for rc, v in ceil["per_crystal"].items()}
    hits = np.zeros(len(refs))
    for p, b in zip(args.bundles, bundles):
        assert b["refcodes"] == refs
        rows = SC.load_match(p, args.matcher)
        S = int(b["valid"].shape[1])
        for i in range(len(refs)):
            ok = [bool(rows[(i, j)]["ok"]) if (i, j) in rows else None for j in range(S)]
            if any(o is None for o in ok):
                raise SystemExit(f"{p}: match rows missing for crystal {i} (run n3_score.py match --draws all)")
            hits[i] += np.mean(ok) / len(bundles)
    rng = np.random.default_rng(0)
    out = {}
    for zkey in ("2", ">=3"):
        idx = [i for i, rc in enumerate(refs) if (zl[rc] == 2 if zkey == "2" else zl[rc] >= 3)]
        if not idx:
            continue
        h = hits[idx]
        boot = np.array([h[rng.integers(0, len(h), len(h))].mean() for _ in range(args.n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sk = "z_MCF" + ("" if zkey.startswith(">") else "=") + zkey
        cm = ceil["summary"][args.matcher][sk]["mean_c_draw1_ceiling"]
        out[sk] = {"n": len(idx), "draw1": float(h.mean()), "ci95": [float(lo), float(hi)],
                                "ceiling_mean_c": cm, "exceeds_ceiling": bool(cm is not None and lo > cm)}
    res = {"diagnostic": "d (consistency)", "matcher": args.matcher, "bundles": args.bundles, "strata": out,
           "consistent": not any(v["exceeds_ceiling"] for v in out.values())}
    path = args.out or os.path.join(ML.PRIV, f"diag_d_{ceil['tag']}_{args.matcher}.json")
    ML.atomic_json(res, path, indent=1)
    print(json.dumps(res["strata"]), "consistent" if res["consistent"] else "EXCEEDS: (b) must be fixed", "->", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("a")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--set", default="valsel")
    a.add_argument("--pickle", default=None, help="provenance: the released test pickle")
    a.add_argument("--provenance", action="store_true", help="released Thurlemann checkpoint (joint, P3 off)")
    a.add_argument("--n", type=int, default=10)
    a.add_argument("--G", type=int, default=3)
    a.add_argument("--g_seed", type=int, default=0)
    a.add_argument("--s_uR", type=float, default=3.0)
    a.add_argument("--s_uF", type=float, default=None)
    a.add_argument("--steps", type=int, default=50)
    a.add_argument("--label", required=True)
    a.add_argument("--out", default=None)
    a.add_argument("--shims", action="store_true", help="laptop smoke only (the result is not reportable)")
    b = sub.add_parser("b")
    b.add_argument("--set", required=True, help="valsel | testB | devtest | val (input-side) or a smoke subset dir")
    b.add_argument("--matchers", nargs="+", default=list(MATCHERS_B))
    b.add_argument("--grid", type=int, default=5)
    b.add_argument("--walk", type=int, default=1)
    b.add_argument("--haar_floor", default=None, help="JSON {refcode: draw@1 or {matcher: draw@1}} for z_MCF = 1")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--crystals", default=None, help="comma-separated positions (debug)")
    b.add_argument("--label", default=None)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--after_sampling", action="store_true",
                   help="allow sel / testB / valsel (only after that set was sampled, §1.1)")
    c = sub.add_parser("c")
    c.add_argument("--run_dir", required=True)
    c.add_argument("--bundle", default=None)
    c.add_argument("--out", default=None)
    c.add_argument("--workers", type=int, default=2)
    d = sub.add_parser("d")
    d.add_argument("--ceiling", required=True)
    d.add_argument("--bundles", nargs="+", required=True)
    d.add_argument("--matcher", default="primary")
    d.add_argument("--n_boot", type=int, default=10000)
    d.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    {"a": cmd_a, "b": cmd_b, "c": cmd_c, "d": cmd_d}[args.cmd](args)


if __name__ == "__main__":
    main()
