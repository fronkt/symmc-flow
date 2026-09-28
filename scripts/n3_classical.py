"""N3 / G3: the CLASSICAL arms and FF* (protocol tasks/n3_protocol.md §3.3; seeds §2.2; FF* budget §7).

Every classical arm uses the true lattice, the true asym centroid c0, the crystal's own spglib ops and the
rigid conformer. Only R_asym moves, and every copy comes from exact expansion (g2_asym_baselines.expand).
Outputs are n3_common draw bundles (kind "rasym", results/n3/private/draws/<arm>_s<seed>_<set>.pt), built
from resumable per-crystal JSONL shards (results/n3/private/classical/). Nothing here matches anything
(§1.3): `anchor` reads n3_score.py's primary-matcher rows of the FF*-minimised TRUTHS (the fftruth bundle).

  haar     iii-u  16 Haar R_asym per crystal and seed base b, torch.Generator().manual_seed(600000 +
                  10000 b + 1000 split + i) (i = position in VALSEL for val/sel); sel "lj" = unrelaxed LJ
  press    iii-p  the G2 press (80 Adam steps, lr 0.02, contact 0.90, cutoff 6.0) of every draw of a bundle
                  (default: the iii-u bundle of the same base); sel "lj" = POST-press LJ, "lj_pre" = pre-press
                  LJ. With --draws <OURS bundle> --arm p1-OURS it gives the steric-BASIN P1 draws of OURS
  p1basin         steric-BASIN reference press(true R0), one per crystal (arm "p1ref", input-side)
  grid            iii-s candidates: Yershova incremental Hopf grid level 2 (HEALPix Nside 4 x 24 psi = 4,608),
                  left-multiplied by one Haar rotation (torch.manual_seed(20260927);
                  random_so3((1,), float64)); checks count, coverage and uniformity
  lbgate          gate for the batched screen evaluator: 20 VAL crystals x 100 grid orientations, 1e-4 rel.
  screen          all 4,608 grid orientations by unrelaxed steric LJ -> the 64 lowest (never fewer)
  fdcheck         force-derived dE/d(delta) vs central finite differences (1e-4 rad) on 5 VAL crystals x
                  {truth, one Haar pose}; pass iff every relative error <= 1e-4 (meaningful in float64)
  fftime          §7 timing gate: FF* calls single and batched (--batch) on 20 VAL crystals, K = 2 / 4 / >= 8
  ffmin           FF* minimisation from the true pose and/or Haar starts (timing / debugging only)
  ffconfig        fixes the FF* configuration ONCE after the timing gate: settings, dtype, max_atoms, device,
                  plus the fairchem-core version, the weight hashes and the resolved inference state of a
                  probe call (toy cell) -> results/n3/private/classical/ffstar_config.json
  search   iii-s  screen -> 64 FF* minimisations -> merge <= 2 deg modulo the proper point group -> the 16
                  lowest distinct minima in energy order (seed -1; sel "ff")
  truths          FF* minimisation of the TRUE pose (<= 1,000 evaluations; arm "fftruth", seed -1): BASIN
                  reference, anchoring, ANCHORED/DRIFTED/UNRESOLVED (meta["converged"])
  selftest        synthetic checks (no CSD data, no FF*): force->torque formula with a det -1 op, L-BFGS,
                  evaluation cap, failure rules (i), point-group merge, Haar generator
  anchor          reads `n3_score.py match --matcher primary` rows of the fftruth bundle (the scorer makes the
                  match under the §1.3 harness; TEST-B / DEV-TEST rows only from a --step9 match) -> anchoring
                  rate, H2 FORM (VALSEL), and the FF* truth labels ff_labels_<set>.json for `n3_score.py report`
  ie       i-E    FF* minimisation of every draw of a bundle (sel "ff"; input selectors kept as "in_<name>")
  basin           FF* minimisation of each crystal's selector pick of a bundle (OURS-N2 LJ picks), or, with
                  --from-ie, the i-E pose of that pick (never minimised twice: a direct `basin` on an arm
                  whose i-E bundle exists is refused)

FF* (UMA s-1p1, OMC task) runs in its own venv through scripts/n3_ffstar.py (one server subprocess per worker).
One FF* configuration serves every job (§3.3 "with the iii-s settings" / "with the same settings"): the
--ff-settings / --ff-dtype / --ff-max-atoms / --device defaults come from ffstar_config.json (`ffconfig`),
full runs of truths / search / ie / basin refuse to start without it or with any other value, and every
server reply must show the resolved inference state recorded there (n3_ffstar.FFConfigError otherwise).
search / ie / basin also need the full-VALSEL anchor file made under the same configuration and from the
current VALSEL fftruth bundle, and TEST-B / DEV-TEST truths need the full VALSEL fftruth bundle made under it.
--ff-settings is fairchem's inference_settings name; with default/turbo the server is restarted per crystal
because fairchem merges the MOLE experts for one composition; --ff-dtype float64 for the FD check;
--ff-max-atoms splits a round into sub-batches (0 = off). Every FF* bundle's meta lists the stop reason of
every minimisation and the §3.3 deterministic-failure summary (runs ended by ff_fail_start / ff_fail, refcodes).
Every FF* evaluation uses a copy
of the conformer with X-H bonds set to C-H 1.089 / N-H 1.015 / O-H 0.993 A along the bond (parent = nearest
bonded non-H atom, n3_chirality.graph rule); the returned R_asym is applied to the ORIGINAL conformer.
Minimiser: L-BFGS on SO(3) (history 10, strong Wolfe c1 1e-4 c2 0.9, torch.optim.LBFGS's line search) with
the update R -> exp([delta]x) R and gradient dE/d(delta) = -sum_k sum_a (R l_a) x (Rc_k^T F_ka); converged
iff ||dE/d(delta)|| / K < 0.01 eV/rad at the returned pose; <= 100 energy+gradient evaluations (truths
1,000), all live starts of one crystal in one predict call per round.

    python scripts/n3_classical.py selftest
    python scripts/n3_classical.py grid
    python scripts/n3_classical.py haar  --set valsel --base 0 --workers 8
    python scripts/n3_classical.py press --set valsel --base 0 --workers 8
    python scripts/n3_classical.py lbgate --workers 2
    python scripts/n3_classical.py screen --set testB --workers 8
    python scripts/n3_classical.py ffconfig --device cuda --ff-settings batch --ff-dtype float32
    python scripts/n3_classical.py truths --set valsel --workers 4
    python scripts/n3_score.py match --bundles results/n3/private/draws/fftruth_s-1_valsel.pt --matcher primary
    python scripts/n3_classical.py anchor --set valsel
    python scripts/n3_classical.py search --set testB --workers 4
    python scripts/n3_classical.py ie    --draws results/n3/private/draws/ours_s0_testB.pt
    python scripts/n3_classical.py basin --draws results/n3/private/draws/i-E-ours_s0_testB.pt --from-ie
    python scripts/n3_classical.py basin --draws results/n3/private/draws/ours_n2_s0_testB.pt
Smoke runs: --limit N (first N crystals) or --only i,j write "<shard>.smoke.jsonl" and "<bundle>.partial.pt";
`anchor` needs a --limit (prefix) bundle, because the scorer indexes the truth by bundle row.
"""
import argparse
import json
import math
import os
import platform
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

import numpy as np
import torch

OUT = os.path.join(C.PRIVATE, "classical")
S_DRAWS = 16                       # draws per crystal and seed (§2.2)
SPLIT = dict(C.SPLIT_CODE, train=3)  # train = smoke only, outside every protocol seed range
GRID_SEED = 20260927
N_KEEP = 64                        # screen survivors; never reduced (§3.3)
XH_LEN = {6: 1.089, 7: 1.015, 8: 0.993}
FF_TOL = 0.01                      # eV/rad per molecule
MERGE_DEG = 2.0
BLINDED = ("testB", "devtest")     # sets on which no match of any kind is made here


# ------------------------------------------------------------------------------ generic helpers
def _set_index_offset(set_name):
    """i of §2.2: the position in the frozen list; val and sel are both indexed within VALSEL."""
    return len(C.load_set("val")) if set_name == "sel" else 0


def _read_jsonl(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[r["refcode"]] = r
    return out


def _pick_indices(n, args):
    if getattr(args, "only", None):
        return [int(x) for x in args.only.split(",")]
    if getattr(args, "limit", None):
        return list(range(min(n, args.limit)))
    return list(range(n))


def _partial(args):
    return bool(getattr(args, "only", None) or getattr(args, "limit", None))


def _smoke(args):
    """A crystal subset, or a job on a smoke (partial) input bundle."""
    return _partial(args) or bool(getattr(args, "src_partial", False))


def _save(path, partial, **kw):
    """Full sets go through n3_common.save_bundle (frozen-order check); smoke subsets to <path>.partial.pt."""
    if not partial:
        C.save_bundle(path, **kw)
        return path
    p = path[:-3] + ".partial.pt"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = {"arm": kw["arm"], "seed": kw["seed"], "set": kw["set_name"], "refcodes": list(kw["refcodes"]),
         "kind": kw["kind"], "R": kw.get("R"), "cells": kw.get("cells"), "valid": kw["valid"],
         "sel": kw["sel"], "meta": dict(kw["meta"], partial=True)}
    torch.save(d, p + ".tmp")
    os.replace(p + ".tmp", p)
    return p


def _meta(args, **extra):
    import subprocess
    try:
        head = subprocess.run(["git", "-C", C.REPO, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        head = "?"
    return dict({"argv": sys.argv, "git_head": head, "torch": torch.__version__,
                 "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, **extra)


def _run(tasks, fn, shard, workers, init=None, initargs=(), cfg=None, smoke=False):
    """Run fn over tasks (dicts with 'refcode'), append each result to the JSONL shard as it finishes.
    Smoke runs use their own shard; every record carries `cfg`, and a resume under another cfg is refused."""
    if smoke:
        shard = shard[:-len(".jsonl")] + ".smoke.jsonl"
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    done = _read_jsonl(shard)
    other = [rc for rc, r in done.items() if r.get("cfg") != cfg]
    if other:
        raise SystemExit(f"{shard}: {len(other)} records were made under another configuration "
                         f"({done[other[0]].get('cfg')} vs {cfg}); use a new shard")
    todo = [t for t in tasks if t["refcode"] not in done]
    print(f"{os.path.basename(shard)}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return done
    t0 = time.time()
    if workers <= 1:
        if init:
            init(*initargs)
        it = map(fn, todo)
        pool = None
    else:
        from multiprocessing import Pool
        torch.multiprocessing.set_sharing_strategy("file_system")
        pool = Pool(workers, initializer=init, initargs=initargs)
        it = pool.imap_unordered(fn, todo)
    with open(shard, "a") as fh:
        for n_, r in enumerate(it):
            r["cfg"] = cfg
            done[r["refcode"]] = r
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (n_ + 1) % 10 == 0 or n_ + 1 == len(todo):
                print(f"  {n_ + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    if pool is not None:
        pool.close(); pool.join()
    if workers <= 1 and _STATE.get("ff") is not None:
        _STATE["ff"].close(); _STATE["ff"] = None
    return done


def _R(x):
    return torch.tensor(x, dtype=torch.float64).reshape(-1, 3, 3)


def _Rlist(R):
    return [float(v) for v in R.reshape(-1).tolist()]


def _geo_deg(A, B):
    """Geodesic angle (deg) between rotation stacks, float64, via the trace (clamped)."""
    tr = torch.einsum("...ji,...ji->...", A, B)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


# ------------------------------------------------------------------------------ Haar (iii-u)
def haar(S, gen):
    """symmc_flow.manifolds.random_so3 with an explicit generator (same QR construction), float64."""
    A = torch.randn(S, 3, 3, generator=gen, dtype=torch.float64)
    Q, Rm = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rm, dim1=-2, dim2=-1)).unsqueeze(-2)
    det = torch.linalg.det(Q)
    Q[..., :, 0] = Q[..., :, 0] * det.unsqueeze(-1)
    return Q


def haar_seed(base, set_name, i):
    return 600000 + 10000 * base + 1000 * SPLIT[set_name] + i


def _w_haar(t):
    warnings.filterwarnings("ignore"); torch.set_num_threads(1)
    from g2_asym_baselines import energy, expand
    a = t["a"]
    t0 = time.time()
    R = haar(S_DRAWS, torch.Generator().manual_seed(t["seed"]))
    lj = [energy(*expand(a, R[s])) for s in range(S_DRAWS)]
    return {"refcode": a["refcode"], "i": t["i"], "seed": t["seed"], "R": [_Rlist(r) for r in R],
            "lj": lj, "sec": round(time.time() - t0, 2)}


def cmd_haar(args):
    items = C.load_set(args.set)
    idx = _pick_indices(len(items), args)
    off = _set_index_offset(args.set)
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "i": off + j,
              "seed": haar_seed(args.base, args.set, off + j)} for j in idx]
    shard = os.path.join(OUT, f"iii-u_s{args.base}_{args.set}.jsonl")
    done = _run(tasks, _w_haar, shard, args.workers, smoke=_partial(args))
    rows = [done[items[j]["refcode"]] for j in idx]
    R = torch.stack([_R(r["R"]) for r in rows])
    p = _save(C.bundle_path("iii-u", args.base, args.set), _partial(args), arm="iii-u", seed=args.base,
              set_name=args.set, refcodes=[r["refcode"] for r in rows], kind="rasym", R=R,
              valid=torch.ones(len(rows), S_DRAWS, dtype=torch.bool),
              sel={"lj": torch.tensor([r["lj"] for r in rows], dtype=torch.float64)},
              meta=_meta(args, seeds=[r["seed"] for r in rows], sec=[r["sec"] for r in rows]))
    print("wrote", p)


# ------------------------------------------------------------------------------ press (iii-p, P1)
def _pressed_R(a, R, steps):
    """R_asym after the G2 press (identity op first, so copy 0's orientation is the asym pose)."""
    from g2_asym_baselines import cart_ops, energy, press
    cell = press(a, R, steps)
    Rn = torch.linalg.solve(cart_ops(a)[0], cell[2][0].double())
    return Rn, energy(*cell)


def _w_press(t):
    warnings.filterwarnings("ignore"); torch.set_num_threads(1)
    from g2_asym_baselines import energy, expand
    a = t["a"]
    t0 = time.time()
    out_R, lj, pre = [], [], []
    for s, R in enumerate(t["R"]):
        R = torch.tensor(R, dtype=torch.float64).reshape(3, 3)
        if not t["valid"][s]:
            out_R.append(_Rlist(R)); lj.append(float("inf")); pre.append(float("inf"))
            continue
        pre.append(energy(*expand(a, R)))
        Rn, e = _pressed_R(a, R, t["steps"])
        out_R.append(_Rlist(Rn)); lj.append(e)
    return {"refcode": a["refcode"], "R": out_R, "lj": lj, "lj_pre": pre, "sec": round(time.time() - t0, 2)}


def cmd_press(args):
    src = args.draws or C.bundle_path("iii-u", args.base, args.set)
    b = C.load_bundle(src if os.path.exists(src) else src[:-3] + ".partial.pt")
    arm = args.arm or ("iii-p" if b["arm"] == "iii-u" else f"p1-{b['arm']}")
    items = {a["refcode"]: a for a in C.load_set(b["set"])}
    idx = _pick_indices(len(b["refcodes"]), args)
    tasks = [{"refcode": b["refcodes"][j], "a": items[b["refcodes"][j]], "R": b["R"][j].reshape(-1, 9).tolist(),
              "valid": b["valid"][j].tolist(), "steps": args.steps} for j in idx]
    shard = os.path.join(OUT, f"{arm}_s{b['seed']}_{b['set']}.jsonl")
    done = _run(tasks, _w_press, shard, args.workers, cfg={"steps": args.steps, "source": os.path.basename(src)},
                smoke=_partial(args) or b["meta"].get("partial", False))
    rows = [done[b["refcodes"][j]] for j in idx]
    p = _save(C.bundle_path(arm, b["seed"], b["set"]), _partial(args) or b["meta"].get("partial", False),
              arm=arm, seed=b["seed"], set_name=b["set"], refcodes=[r["refcode"] for r in rows], kind="rasym",
              R=torch.stack([_R(r["R"]) for r in rows]), valid=b["valid"][idx].clone(),
              sel={"lj": torch.tensor([r["lj"] for r in rows], dtype=torch.float64),
                   "lj_pre": torch.tensor([r["lj_pre"] for r in rows], dtype=torch.float64)},
              meta=_meta(args, source=src, steps=args.steps, lr=0.02, contact=0.90, cutoff=6.0,
                         sec=[r["sec"] for r in rows]))
    print("wrote", p)


def _w_p1ref(t):
    warnings.filterwarnings("ignore"); torch.set_num_threads(1)
    a = t["a"]
    t0 = time.time()
    Rn, e = _pressed_R(a, a["R0"].double(), t["steps"])
    return {"refcode": a["refcode"], "R": _Rlist(Rn), "lj": e, "sec": round(time.time() - t0, 2)}


def cmd_p1basin(args):
    items = C.load_set(args.set)
    idx = _pick_indices(len(items), args)
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "steps": args.steps} for j in idx]
    done = _run(tasks, _w_p1ref, os.path.join(OUT, f"p1ref_{args.set}.jsonl"), args.workers,
                cfg={"steps": args.steps}, smoke=_partial(args))
    rows = [done[items[j]["refcode"]] for j in idx]
    p = _save(C.bundle_path("p1ref", -1, args.set), _partial(args), arm="p1ref", seed=-1, set_name=args.set,
              refcodes=[r["refcode"] for r in rows], kind="rasym",
              R=torch.stack([_R(r["R"]) for r in rows]), valid=torch.ones(len(rows), 1, dtype=torch.bool),
              sel={"lj": torch.tensor([[r["lj"]] for r in rows], dtype=torch.float64)},
              meta=_meta(args, steps=args.steps, note="steric-BASIN reference press(true R0); input-side"))
    print("wrote", p)


# ------------------------------------------------------------------------------ Hopf grid
def _healpix_centres(nside):
    """(theta, phi) of the 12 nside^2 HEALPix pixel centres (ring scheme)."""
    th, ph = [], []
    N = nside
    for i in range(1, 4 * N):
        if i < N:
            z = 1 - i * i / (3 * N * N)
            phis = [math.pi / (2 * i) * (j - 0.5) for j in range(1, 4 * i + 1)]
        elif i <= 3 * N:
            z = 4 / 3 - 2 * i / (3 * N)
            s = (i - N + 1) % 2
            phis = [math.pi / (2 * N) * (j - s / 2) for j in range(1, 4 * N + 1)]
        else:
            ii = 4 * N - i
            z = -(1 - ii * ii / (3 * N * N))
            phis = [math.pi / (2 * ii) * (j - 0.5) for j in range(1, 4 * ii + 1)]
        for p in phis:
            th.append(math.acos(max(-1.0, min(1.0, z)))); ph.append(p)
    return np.array(th), np.array(ph)


def _quat_to_R(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def hopf_grid_raw(level=2):
    """Yershova et al. (2010) incremental Hopf grid: HEALPix(Nside = 2^level) x (6 2^level) psi.
    Hopf coordinates q = (cos(t/2)cos(p/2), cos(t/2)sin(p/2), sin(t/2)cos(f+p/2), sin(t/2)sin(f+p/2)),
    psi_k = (k + 1/2) 2 pi / n_psi. Order: HEALPix pixel (ring order) major, psi minor."""
    th, ph = _healpix_centres(2 ** level)
    npsi = 6 * 2 ** level
    psi = (np.arange(npsi) + 0.5) * 2 * math.pi / npsi
    T, P, S = np.repeat(th, npsi), np.repeat(ph, npsi), np.tile(psi, len(th))
    q = np.stack([np.cos(T / 2) * np.cos(S / 2), np.cos(T / 2) * np.sin(S / 2),
                  np.sin(T / 2) * np.cos(P + S / 2), np.sin(T / 2) * np.sin(P + S / 2)], -1)
    return _quat_to_R(torch.tensor(q, dtype=torch.float64))


_GRID = {}


def hopf_grid(level=2):
    """The iii-s candidate set: the raw grid left-multiplied by one fixed Haar rotation Q
    (torch.manual_seed(20260927); random_so3((1,), float64); the caller's RNG state is left untouched)."""
    if level not in _GRID:
        from symmc_flow import manifolds as M
        with torch.random.fork_rng():
            torch.manual_seed(GRID_SEED)
            Q = M.random_so3((1,), dtype=torch.float64)[0]
        _GRID[level] = (torch.einsum("ij,njk->nik", Q, hopf_grid_raw(level)), Q)
    G, Q = _GRID[level]
    return G.clone(), Q.clone()


def _nn_deg(G, Q=None, chunk=512):
    """Min geodesic angle (deg) from each of Q (default G itself, excluding self) to the set G."""
    same = Q is None
    Q = G if same else Q
    out = []
    for s in range(0, Q.shape[0], chunk):
        tr = torch.einsum("nji,mji->nm", Q[s:s + chunk], G)
        if same:
            tr[torch.arange(tr.shape[0]), torch.arange(s, s + tr.shape[0])] = -1.0
        out.append(torch.rad2deg(torch.arccos(((tr.max(1).values - 1) / 2).clamp(-1, 1))))
    return torch.cat(out)


def cmd_grid(args):
    G, Q = hopf_grid(2)
    rep = {"n": int(G.shape[0])}
    assert rep["n"] == 4608, rep
    rep["orthonormal_max_err"] = float((G @ G.transpose(-1, -2) - torch.eye(3, dtype=G.dtype)).abs().max())
    rep["det_min"] = float(torch.linalg.det(G).min())
    rep["prerot_Q"] = Q.reshape(-1).tolist()
    nn = _nn_deg(G)
    rep["nn_deg"] = {"min": float(nn.min()), "median": float(nn.median()), "max": float(nn.max())}
    rep["distinct"] = bool(nn.min() > 1e-3)
    # covering radius and cell occupancy from Haar probes (fixed seed; not part of any arm)
    probes = haar(args.probes, torch.Generator().manual_seed(12345))
    cov = []
    owner = []
    for s in range(0, probes.shape[0], 512):
        tr = torch.einsum("nji,mji->nm", probes[s:s + 512], G)
        m = tr.max(1)
        cov.append(torch.rad2deg(torch.arccos(((m.values - 1) / 2).clamp(-1, 1)))); owner.append(m.indices)
    cov, owner = torch.cat(cov), torch.cat(owner)
    rep["cover_deg"] = {"median": float(cov.median()), "p99": float(cov.quantile(0.99)), "max": float(cov.max())}
    cnt = torch.bincount(owner, minlength=G.shape[0]).double()
    lam = probes.shape[0] / G.shape[0]
    rep["occupancy"] = {"probes": args.probes, "expected": lam, "cv": float(cnt.std() / cnt.mean()),
                        "poisson_cv": 1 / math.sqrt(lam), "min": int(cnt.min()), "max": int(cnt.max()),
                        "chi2_over_dof": float(((cnt - lam) ** 2 / lam).sum() / (G.shape[0] - 1))}
    # uniformity moments: E[R] = 0 and E[R_ij^2] = 1/3 for Haar; grid means
    rep["mean_R_absmax"] = float(G.mean(0).abs().max())
    rep["mean_R2_dev"] = float(((G ** 2).mean(0) - 1 / 3).abs().max())
    # level-2 identity: the pre-rotation must be a rotation; raw grid is invariant to nothing in particular
    rep["Q_orth_err"] = float((Q @ Q.T - torch.eye(3, dtype=Q.dtype)).abs().max())
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, "grid_check.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------------------ steric screen
class BatchedLJ:
    """g2_asym_baselines.energy for many R_asym of one crystal at once, with energy()'s exact pair set:
    image shifts from rigid_press._image_shifts(L, cutoff + 1.5) (the neighbour-list shell energy() uses),
    intramolecular zero-shift pairs excluded, pairs counted iff dist < cutoff, float32 like energy().
    Candidate pairs are pruned by |C_i - C_j - s| - |l_a| - |l_b| < cutoff (exact bound, R-independent)."""

    def __init__(self, a, contact=0.90, cutoff=6.0, margin=1.5, d_floor=0.7, max_elems=2.5e7):
        from g2_asym_baselines import cart_ops
        from symmc_flow.rigid_press import _image_shifts, _radii
        self.cutoff, self.d_floor, self.max_elems = cutoff, d_floor, max_elems
        self.Lf = a["L"].float()
        self.Rc = cart_ops(a)                                       # float64, as expand()
        cent = torch.einsum("kij,j->ki", a["W"], a["c0"]) + a["t"]
        cent = cent - torch.floor(cent)
        self.base = (cent.float() @ self.Lf)                        # (K,3) as energy()
        self.loc = a["local"].float()
        K, A = a["K"], self.loc.shape[0]
        self.K, self.A = K, A
        r = _radii(a["Z"].repeat(K))
        shifts = _image_shifts(self.Lf, cutoff + margin)             # (nS,3) float32 integer vectors
        Scart = shifts @ self.Lf
        rho = self.loc.norm(dim=-1)                                  # |R l_a| = |l_a| for every copy
        I, J, Sx = [], [], []
        for s in range(shifts.shape[0]):
            zero = bool((shifts[s].abs().sum() == 0).item())
            Dc = (self.base[:, None, :] - (self.base[None, :, :] + Scart[s])).norm(dim=-1)   # (K,K)
            for k in range(K):
                for k2 in range(K):
                    if zero and k == k2:
                        continue
                    if float(Dc[k, k2]) - 2 * float(rho.max()) >= cutoff + 1e-3:
                        continue
                    ok = (float(Dc[k, k2]) - rho[:, None] - rho[None, :]) < cutoff + 1e-3
                    aa, bb = torch.nonzero(ok, as_tuple=True)
                    I.append(k * A + aa); J.append(k2 * A + bb)
                    Sx.append(shifts[s].expand(aa.shape[0], 3))
        if I:
            self.I, self.J = torch.cat(I), torch.cat(J)
            Sfrac = torch.cat(Sx).to(self.Lf)
            self.Scart = Sfrac @ self.Lf                             # per pair, as packing_energy_nbr
            self.Rij = contact * (r[self.I] + r[self.J])
        else:
            self.I = None
        self.n_pairs = 0 if self.I is None else int(self.I.shape[0])

    def __call__(self, Rs):
        if self.I is None:
            return torch.zeros(Rs.shape[0], dtype=torch.float64)
        orient = torch.matmul(self.Rc[None], Rs[:, None].double())   # (B,K,3,3) float64, as expand()
        chunk = max(1, int(self.max_elems // max(1, self.n_pairs)))
        out = []
        for s in range(0, Rs.shape[0], chunk):
            X = self.base[None, :, None, :] + torch.einsum("bmij,aj->bmai", orient[s:s + chunk].float(), self.loc)
            X = X.reshape(X.shape[0], -1, 3)
            dist = (X[:, self.I] - (X[:, self.J] + self.Scart)).norm(dim=-1)
            within = dist < self.cutoff
            r6 = (self.Rij / dist.clamp_min(self.d_floor)) ** 6
            phi = 1.0 * (r6 * r6 - 2.0 * r6)
            out.append((0.5 * (phi * within).sum(-1)).double())
        return torch.cat(out)


def _energy_many(a, Rs):
    from g2_asym_baselines import energy, expand
    return torch.tensor([energy(*expand(a, Rs[i])) for i in range(Rs.shape[0])], dtype=torch.float64)


def _gate_path():
    return os.path.join(OUT, "gate_screen.json")


def _gate_ok():
    p = _gate_path()
    if not os.path.exists(p):
        return False, "no gate file (run `lbgate`)"
    g = json.load(open(p))
    if not g.get("pass"):
        return False, "gate failed"
    if g.get("host") != platform.node() or g.get("torch") != torch.__version__:
        return False, f"gate was run on {g.get('host')} / torch {g.get('torch')}; re-run `lbgate` here"
    return True, "gate passed"


def _w_gate(t):
    warnings.filterwarnings("ignore"); torch.set_num_threads(1)
    a, Rs = t["a"], torch.tensor(t["Rs"], dtype=torch.float64).reshape(-1, 3, 3)
    t0 = time.time(); ref = _energy_many(a, Rs); t_ref = time.time() - t0
    ev = BatchedLJ(a)
    t0 = time.time(); b = ev(Rs); t_b = time.time() - t0
    rel = ((b - ref).abs() / ref.abs().clamp_min(1e-300)).tolist()
    # diagnostic beyond the gate: the low end of the full grid, where the screen decides
    G, _ = hopf_grid(2)
    Eg = ev(G)
    low = torch.argsort(Eg)[:t["n_low"]]
    ref_low = _energy_many(a, G[low])
    rel_low = ((Eg[low] - ref_low).abs() / ref_low.abs().clamp_min(1e-300)).tolist()
    return {"refcode": a["refcode"], "K": a["K"], "A": int(a["local"].shape[0]), "rel": rel,
            "abs": (b - ref).abs().tolist(), "e_ref_absmin": float(ref.abs().min()),
            "rel_low": rel_low, "e_low": ref_low.tolist(),
            "sec_energy": round(t_ref, 2), "sec_batched": round(t_b, 3)}


def cmd_lbgate(args):
    items = C.load_set("val")[:args.n_crystals]
    G, _ = hopf_grid(2)
    sub = torch.linspace(0, G.shape[0] - 1, args.n_orient).round().long()
    tasks = [{"refcode": a["refcode"], "a": a, "Rs": G[sub].reshape(-1, 9).tolist(), "n_low": args.n_low}
             for a in items]
    shard = os.path.join(OUT, f"gate_screen_{platform.node()}.jsonl")
    done = _run(tasks, _w_gate, shard, args.workers)
    rows = [done[a["refcode"]] for a in items]
    rel = [x for r in rows for x in r["rel"]]
    rep = {"pass": bool(max(rel) <= 1e-4 and len(rel) == args.n_crystals * args.n_orient),
           "n": len(rel), "max_rel": max(rel), "n_over_1e-4": sum(x > 1e-4 for x in rel),
           "median_rel": float(np.median(rel)), "min_abs_energy": min(r["e_ref_absmin"] for r in rows),
           "sec_per_energy_call": sum(r["sec_energy"] for r in rows) / len(rel),
           "sec_per_batched_orientation": sum(r["sec_batched"] for r in rows) / len(rel),
           "per_crystal_K": [r["K"] for r in rows], "host": platform.node(), "torch": torch.__version__,
           "orientations": "grid indices round(linspace(0, 4607, %d))" % args.n_orient,
           "diag_low_end": {"n": sum(len(r["rel_low"]) for r in rows),
                            "max_rel": max(x for r in rows for x in r["rel_low"]),
                            "min_abs_energy": min(abs(x) for r in rows for x in r["e_low"]),
                            "note": "not part of the gate: the %d batched-lowest grid points per crystal"
                                    % args.n_low}}
    json.dump(rep, open(_gate_path(), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_crystal_K"}, indent=1))


def screen_crystal(a, evaluator):
    G, _ = hopf_grid(2)
    t0 = time.time()
    E = BatchedLJ(a)(G) if evaluator == "batched" else _energy_many(a, G)
    sec = time.time() - t0
    key = torch.where(torch.isfinite(E), E, torch.full_like(E, float("inf")))
    order = sorted(range(G.shape[0]), key=lambda i: (float(key[i]), i))[:N_KEEP]
    return {"refcode": a["refcode"], "keep": order, "lj": [float(E[i]) for i in order],
            "evaluator": evaluator, "n_eval": int(G.shape[0]), "sec": round(sec, 2)}


def _w_screen(t):
    warnings.filterwarnings("ignore"); torch.set_num_threads(t["threads"])
    return screen_crystal(t["a"], t["evaluator"])


def _evaluator(choice):
    if choice == "energy":
        return "energy"
    ok, why = _gate_ok()
    if choice == "batched" and not ok:
        raise SystemExit(f"batched screen not allowed: {why}")
    print(f"screen evaluator: {'batched' if ok else 'energy'} ({why})", flush=True)
    return "batched" if ok else "energy"


def cmd_screen(args):
    items = C.load_set(args.set)
    idx = _pick_indices(len(items), args)
    ev = _evaluator(args.evaluator)
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "evaluator": ev, "threads": args.threads}
             for j in idx]
    _run(tasks, _w_screen, os.path.join(OUT, f"screen_{args.set}.jsonl"), args.workers, smoke=_partial(args))


# ------------------------------------------------------------------------------ FF* geometry
def normalise_xh(local, Z):
    """FF*-only conformer: each H bonded (n3_chirality.graph rule) to C/N/O is moved along the bond to the
    CSD neutron length; parent = nearest bonded non-H atom; heavy atoms and the frame origin unchanged."""
    from n3_chirality import graph
    X = local.double().clone()
    g = graph(local.double().numpy(), Z.numpy())
    cnt = {"C-H": 0, "N-H": 0, "O-H": 0, "shortened": 0}
    for h in (Z == 1).nonzero().flatten().tolist():
        nb = [j for j in g.neighbors(h) if int(Z[j]) != 1]
        if not nb:
            continue
        j = min(nb, key=lambda j: float((local[h] - local[j]).norm()))
        z = int(Z[j])
        if z not in XH_LEN:
            continue
        v = local[h].double() - local[j].double()
        d = float(v.norm())
        X[h] = local[j].double() + v / d * XH_LEN[z]
        cnt[{6: "C-H", 7: "N-H", 8: "O-H"}[z]] += 1
        cnt["shortened"] += int(d > XH_LEN[z])
    return X, cnt


def ff_prep(a):
    from g2_asym_baselines import cart_ops
    loc, cnt = normalise_xh(a["local"], a["Z"])
    cent = torch.einsum("kij,j->ki", a["W"], a["c0"]) + a["t"]
    cent = cent - torch.floor(cent)
    L = a["L"].double()
    return {"loc": loc, "Rc": cart_ops(a), "base": cent @ L, "L": L, "K": a["K"],
            "numbers": a["Z"].repeat(a["K"]).numpy().astype(np.int64), "xh": cnt}


def ff_struct(p, R):
    X = p["base"][:, None, :] + torch.einsum("kij,aj->kai", p["Rc"] @ R, p["loc"])
    return (p["numbers"], X.reshape(-1, 3).numpy(), p["L"].numpy())


def ff_grad(p, R, F):
    """dE/d(delta) for R -> exp([delta]x) R: -sum_k sum_a (R l_a) x (Rc_k^T F_ka)."""
    K, A = p["K"], p["loc"].shape[0]
    F = torch.as_tensor(F, dtype=torch.float64).reshape(K, A, 3)
    v = (R @ p["loc"].T).T.expand(K, A, 3)
    Fl = torch.einsum("kji,kaj->kai", p["Rc"], F)
    return -torch.linalg.cross(v, Fl, dim=-1).sum((0, 1))


def _so3_exp(w):
    from symmc_flow import manifolds as M
    return M.so3_exp(w)


# ------------------------------------------------------------------------------ L-BFGS on SO(3)
def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # torch.optim.lbfgs._cubic_interpolate
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 ** 2 - g1 * g2
    if d2_square >= 0:
        d2 = math.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    return (xmin_bound + xmax_bound) / 2.0


class _Stop(Exception):
    def __init__(self, reason):
        self.reason = reason


def lbfgs_so3(R0, K, max_evals, tol=FF_TOL, history=10, c1=1e-4, c2=0.9, max_ls=25, tol_change=1e-9):
    """Generator: yields a rotation to evaluate, is sent (E, grad) or None (deterministic FF* failure);
    returns a result dict. torch.optim.LBFGS (lr 1, history 10, strong Wolfe) transcribed to SO(3):
    x_{k+1} = exp([t d]x) x_k, s = t d, y = g_{k+1} - g_k (right-trivialised; identity transport), so the
    line-search derivative d/dt E(exp(t d) x) = g . d is exact. Stops: gradient test (converged), the
    evaluation cap, a line-search failure (strong Wolfe not met in 25 evaluations, bracket collapse, no
    descent direction or no progress), or a deterministic FF* failure (-> last finite iterate)."""
    st = {"n": 0}
    trace = []

    def ev(R):
        if st["n"] >= max_evals:
            raise _Stop("cap")
        res = yield R
        st["n"] += 1
        if res is None:
            raise _Stop("ff_fail")
        trace.append(res[0])
        return float(res[0]), res[1].double()

    def done(R, f, g, reason):
        gn = float(g.norm()) / K
        return {"R": R, "E": f, "gnorm_per_mol": gn, "converged": bool(gn < tol), "reason": reason,
                "n_eval": st["n"], "E_start": trace[0] if trace else float("nan")}

    try:
        f, g = yield from ev(R0)
    except _Stop:
        return {"R": None, "E": float("nan"), "gnorm_per_mol": float("nan"), "converged": False,
                "reason": "ff_fail_start", "n_eval": st["n"], "E_start": float("nan")}
    x = R0
    if float(g.norm()) / K < tol:
        return done(x, f, g, "converged")
    old_dirs, old_stps, ro = [], [], []
    H_diag = 1.0
    d = t = None
    n_iter = 0
    while True:
        n_iter += 1
        if n_iter == 1:
            d = -g
        else:
            y = g - g_prev
            s = d * t
            ys = float(y @ s)
            if ys > 1e-10:
                if len(old_dirs) == history:
                    old_dirs.pop(0); old_stps.pop(0); ro.pop(0)
                old_dirs.append(y); old_stps.append(s); ro.append(1.0 / ys)
                H_diag = ys / float(y @ y)
            q = -g
            al = [0.0] * len(old_dirs)
            for i in range(len(old_dirs) - 1, -1, -1):
                al[i] = float(old_stps[i] @ q) * ro[i]
                q = q - al[i] * old_dirs[i]
            r = q * H_diag
            for i in range(len(old_dirs)):
                be_i = float(old_dirs[i] @ r) * ro[i]
                r = r + (al[i] - be_i) * old_stps[i]
            d = r
        g_prev, f_prev = g.clone(), f
        t = min(1.0, 1.0 / float(g.abs().sum())) if n_iter == 1 else 1.0
        gtd = float(g @ d)
        if gtd > -tol_change:
            return done(x, f, g, "ls_fail:no_descent")
        # ---- strong-Wolfe line search along exp(t d) x (torch.optim.lbfgs._strong_wolfe) ----
        seen = []                                   # (f, t, g) of every trial point, for cut-short stops

        def evt(tt):
            Rt = _so3_exp(tt * d) @ x
            ff, gg = yield from ev(Rt)
            seen.append((ff, tt, gg))
            return ff, gg, float(gg @ d)

        try:
            d_norm = float(d.abs().max())
            f_new, g_new, gtd_new = yield from evt(t)
            t_prev, f_prev_ls, g_prev_ls, gtd_prev = 0.0, f, g, gtd
            ok = False
            ls_iter = 0
            bracket = bracket_f = bracket_g = bracket_gtd = None
            while ls_iter < max_ls:
                if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev_ls):
                    bracket, bracket_f = [t_prev, t], [f_prev_ls, f_new]
                    bracket_g, bracket_gtd = [g_prev_ls, g_new], [gtd_prev, gtd_new]
                    break
                if abs(gtd_new) <= -c2 * gtd:
                    bracket, bracket_f, bracket_g = [t], [f_new], [g_new]
                    ok = True
                    break
                if gtd_new >= 0:
                    bracket, bracket_f = [t_prev, t], [f_prev_ls, f_new]
                    bracket_g, bracket_gtd = [g_prev_ls, g_new], [gtd_prev, gtd_new]
                    break
                min_step = t + 0.01 * (t - t_prev)
                max_step = t * 10
                tmp = t
                t = _cubic_interpolate(t_prev, f_prev_ls, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step))
                t_prev, f_prev_ls, g_prev_ls, gtd_prev = tmp, f_new, g_new, gtd_new
                f_new, g_new, gtd_new = yield from evt(t)
                ls_iter += 1
            if ls_iter == max_ls:
                bracket, bracket_f, bracket_g = [0.0, t], [f, f_new], [g, g_new]
            insuf_progress = False
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
            while not ok and ls_iter < max_ls:
                if abs(bracket[1] - bracket[0]) * d_norm < tol_change:
                    break
                t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                                       bracket[1], bracket_f[1], bracket_gtd[1])
                eps = 0.1 * (max(bracket) - min(bracket))
                if min(max(bracket) - t, t - min(bracket)) < eps:
                    if insuf_progress or t >= max(bracket) or t <= min(bracket):
                        t = max(bracket) - eps if abs(t - max(bracket)) < abs(t - min(bracket)) else min(bracket) + eps
                        insuf_progress = False
                    else:
                        insuf_progress = True
                else:
                    insuf_progress = False
                f_new, g_new, gtd_new = yield from evt(t)
                ls_iter += 1
                if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
                    bracket[high_pos], bracket_f[high_pos] = t, f_new
                    bracket_g[high_pos], bracket_gtd[high_pos] = g_new, gtd_new
                    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
                else:
                    if abs(gtd_new) <= -c2 * gtd:
                        ok = True
                    elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                        bracket[high_pos], bracket_f[high_pos] = bracket[low_pos], bracket_f[low_pos]
                        bracket_g[high_pos], bracket_gtd[high_pos] = bracket_g[low_pos], bracket_gtd[low_pos]
                    bracket[low_pos], bracket_f[low_pos] = t, f_new
                    bracket_g[low_pos], bracket_gtd[low_pos] = g_new, gtd_new
            t = bracket[low_pos] if len(bracket) > 1 else bracket[0]
            f_new = bracket_f[low_pos] if len(bracket) > 1 else bracket_f[0]
            g_new = bracket_g[low_pos] if len(bracket) > 1 else bracket_g[0]
        except _Stop as e:
            # a deterministic FF* failure ends the run at the last finite iterate (§3.3 rule i); the cap keeps
            # the lowest point of the interrupted line search if it is below the iterate
            best = min(seen, key=lambda z: z[0]) if seen else None
            if e.reason == "cap" and best is not None and best[0] < f:
                return done(_so3_exp(best[1] * d) @ x, best[0], best[2], e.reason)
            return done(x, f, g, e.reason)
        if t == 0.0 or f_new > f:
            return done(x, f, g, "ls_fail:no_progress")
        x = _so3_exp(t * d) @ x
        f, g = f_new, g_new
        if float(g.norm()) / K < tol:
            return done(x, f, g, "converged")
        if not ok:
            return done(x, f, g, "ls_fail:wolfe")
        if st["n"] >= max_evals:
            return done(x, f, g, "cap")
        if abs(d.abs().max() * t) <= tol_change or abs(f - f_prev) < tol_change:
            return done(x, f, g, "ls_fail:no_progress")


def ff_minimise(ff, p, starts, max_evals, tag=""):
    """Minimise every start of one crystal; all live starts share one batched FF* call per round.
    Call times exclude FF* server start-up (model load)."""
    gens = [lbfgs_so3(R.double(), p["K"], max_evals) for R in starts]
    out = [None] * len(gens)
    pend = {}
    for i, gen in enumerate(gens):
        pend[i] = next(gen)
    rounds, calls_sec = 0, []
    while pend:
        ids = list(pend)
        t0, sp0 = time.time(), ff.stats["sec_spawn"]
        res = ff.evaluate([ff_struct(p, pend[i]) for i in ids], tag=f"{tag} round {rounds}")
        calls_sec.append((len(ids), time.time() - t0 - (ff.stats["sec_spawn"] - sp0)))
        rounds += 1
        nxt = {}
        for i, r in zip(ids, res):
            send = None if r is None else (r[0], ff_grad(p, pend[i], r[1]))
            try:
                nxt[i] = gens[i].send(send)
            except StopIteration as e:
                out[i] = e.value
        pend = nxt
    return out, {"rounds": rounds, "calls": calls_sec}


# ------------------------------------------------------------------------------ FF* client per worker
_STATE = {}


def _ff_init(cfg):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    _STATE["cfg"] = cfg
    _STATE["ff"] = None


def _ff():
    """(FFStar, stats snapshot) for the current crystal."""
    if _STATE.get("ff") is None:
        from n3_ffstar import FFStar
        c = _STATE["cfg"]
        _STATE["ff"] = FFStar(python=c["python"], ckpt=c["ckpt"], refs=c["refs"], device=c["device"],
                              threads=c["threads"], settings=c["settings"], dtype=c["dtype"],
                              max_atoms=c["max_atoms"], log_dir=c["log_dir"], expect=c["expect"])
    return _STATE["ff"], dict(_STATE["ff"].stats)


def _ff_row(ff, s0):
    """Per-crystal FF* record: the resolved inference state every reply showed, the server's versions, and the
    structure evaluations retried / deterministic (§3.3) during this crystal."""
    return {"ff_state": ff.state,
            "ff_server": {k: (ff.meta or {}).get(k) for k in ("fairchem_core", "torch", "device", "settings", "dtype")},
            "ff_retries": ff.stats["retries"] - s0["retries"], "ff_det": ff.stats["deterministic"] - s0["deterministic"]}


def _ff_done():
    """After each crystal. With MOLE-merging settings (default/turbo) a fairchem 2.23 server merges the experts
    for the first composition it sees and falls back for good on the next, so a fresh server per crystal
    keeps every crystal on the same code path whatever the processing order."""
    c = _STATE.get("cfg") or {}
    if c.get("settings") in ("default", "turbo") and _STATE.get("ff") is not None:
        _STATE["ff"].close()
        _STATE["ff"] = None


FF_IDENT = ("settings", "dtype", "max_atoms", "device")
FF_FALLBACK = {"settings": "default", "dtype": "float32", "max_atoms": None, "device": "cpu"}
FF_PRODUCTION = ("truths", "search", "ie", "basin")


def _ffconfig_path(args):
    return getattr(args, "ff_config", None) or os.path.join(OUT, "ffstar_config.json")


def _ffident(args):
    return {"settings": args.ff_settings, "dtype": args.ff_dtype, "max_atoms": args.ff_max_atoms,
            "device": args.device}


def _ff_resolve(args):
    """FF* flags left unset take the fixed configuration (ffstar_config.json; without one the fairchem
    defaults, for gates and smoke runs). Full runs of truths / search / ie / basin must equal the fixed
    configuration; whenever the job's flags equal it, every server reply must show its resolved state."""
    p = _ffconfig_path(args)
    cfg = json.load(open(p)) if os.path.exists(p) else None
    for k, attr in (("settings", "ff_settings"), ("dtype", "ff_dtype"), ("max_atoms", "ff_max_atoms"),
                    ("device", "device")):
        v = getattr(args, attr)
        if v is None:
            v = cfg[k] if cfg else FF_FALLBACK[k]
        if k == "max_atoms" and v == 0:
            v = None
        setattr(args, attr, v)
    same = cfg is not None and all(_ffident(args)[k] == cfg[k] for k in FF_IDENT)
    args.ff_expect = {"fairchem_core": cfg["fairchem_core"], "state": cfg["state"]} if same else None
    if args.cmd in FF_PRODUCTION and not _smoke(args):
        if cfg is None:
            raise SystemExit(f"{p} missing: fix the FF* configuration first (`ffconfig`, after the §7 step-4 gate)")
        if not same:
            raise SystemExit(f"FF* {_ffident(args)} differs from the fixed configuration "
                             f"{ {k: cfg[k] for k in FF_IDENT} } ({p}); drop the --ff-* / --device flags")
    return cfg


def _ff_cfg(args, job):
    from n3_ffstar import default_python, UMA_CKPT, UMA_REFS
    return {"python": args.ff_python or default_python(), "ckpt": args.ckpt or UMA_CKPT,
            "refs": args.refs or UMA_REFS, "device": args.device, "threads": args.ff_threads,
            "settings": args.ff_settings, "dtype": args.ff_dtype, "max_atoms": args.ff_max_atoms,
            "expect": args.ff_expect, "log_dir": os.path.join(OUT, "ff_logs", job)}


def _fftag(args):
    """Shard-name part for gate / diagnostic jobs run under several FF* configurations."""
    return f"{args.device}_{args.ff_settings}_{args.ff_dtype}" + (f"_ma{args.ff_max_atoms}" if args.ff_max_atoms else "")


def _ffkey(args, **extra):
    """What must not change within one shard: the FF* configuration (with the pinned resolved state) and the
    budget."""
    return dict(_ffident(args), max_evals=args.max_evals, expect=args.ff_expect, **extra)


def _ff_meta(args, job, rows):
    """Bundle meta of an FF* job: configuration, the ONE resolved state its crystals showed, retry counts."""
    states = {json.dumps(r["ff_state"], sort_keys=True) for r in rows if r.get("ff_state") is not None}
    if len(states) > 1:
        raise SystemExit(f"{job}: crystals ran under {len(states)} different resolved FF* states: {states}")
    vers = {r["ff_server"]["fairchem_core"] for r in rows if (r.get("ff_server") or {}).get("fairchem_core")}
    if len(vers) > 1:
        raise SystemExit(f"{job}: crystals ran under fairchem-core {sorted(vers)}")
    return {"ff": _ff_cfg(args, job), "ffident": _ffident(args),
            "ff_state": json.loads(states.pop()) if states else None, "fairchem_core": vers.pop() if vers else None,
            "ff_retries": sum(r.get("ff_retries", 0) for r in rows),
            "ff_det_structures": sum(r.get("ff_det", 0) for r in rows)}


def _det_summary(refcodes, res_lists):
    """§3.3 deterministic FF* failures of one arm and set: minimisations ended by one (ff_fail_start = no
    minimum; ff_fail = stopped at the last finite iterate), with their refcodes."""
    per = []
    for rc, rl in zip(refcodes, res_lists):
        s = sum(1 for x in rl if x and x["why"] == "ff_fail_start")
        m = sum(1 for x in rl if x and x["why"] == "ff_fail")
        if s or m:
            per.append({"refcode": rc, "ff_fail_start": s, "ff_fail": m})
    return {"n_ff_fail_start": sum(x["ff_fail_start"] for x in per), "n_ff_fail": sum(x["ff_fail"] for x in per),
            "n_crystals": len(per), "list": per}


def _det_txt(meta):
    d = meta["det_fail"]
    head = ", ".join(x["refcode"] for x in d["list"][:5]) + (" ..." if d["n_crystals"] > 5 else "")
    return (f"deterministic FF* failures: {d['n_ff_fail_start']} runs with no minimum, {d['n_ff_fail']} stopped "
            f"early, {d['n_crystals']} crystals [{head}]; structure evaluations retried {meta['ff_retries']}, "
            f"deterministic {meta['ff_det_structures']}")


def _res_json(r):
    """Minimisation result -> JSON (R as 9 floats; None if no minimum)."""
    return {"R": None if r["R"] is None else _Rlist(r["R"]), "E": r["E"], "E_start": r["E_start"],
            "g": r["gnorm_per_mol"], "conv": r["converged"], "why": r["reason"], "nev": r["n_eval"]}


def _xh_total(rows):
    """Normalised X-H bonds over a set (reported per set, protocol §3.3)."""
    tot = {}
    for r in rows:
        for k, v in r["xh"].items():
            tot[k] = tot.get(k, 0) + v
    return tot


def _same_ff(args, ident, state):
    return ident == _ffident(args) and state == (args.ff_expect or {}).get("state")


def _form_guard(args):
    """§3.3: the H2 FORM is fixed on the full VALSEL truths before any iii-s, i-E or BASIN minimisation runs,
    and those runs use the configuration the truths used ("with the iii-s settings"; "the same settings").
    --before-form (smoke subsets only) skips this; the bundle meta records it."""
    if getattr(args, "before_form", False):
        if not _smoke(args):
            raise SystemExit("--before-form is for smoke runs (--limit / --only, or a .partial.pt input) only")
        print("WARNING: --before-form: arm candidates are FF*-minimised before the H2 FORM exists (disclose)",
              flush=True)
        return
    p = os.path.join(OUT, "anchor_valsel.json")
    if not os.path.exists(p):
        raise SystemExit("protocol §3.3: fix the H2 FORM first (truths, n3_score match and anchor on the full "
                         "VALSEL); --before-form overrides for smoke subsets")
    s = json.load(open(p))["summary"]
    if s.get("partial"):
        raise SystemExit(f"{p} comes from a partial VALSEL run")
    if not _same_ff(args, s.get("ffident"), s.get("ff_state")):
        raise SystemExit(f"FF* {_ffident(args)} / {(args.ff_expect or {}).get('state')} is not the configuration "
                         f"of the VALSEL truths ({s.get('ffident')} / {s.get('ff_state')})")
    b = C.bundle_path("fftruth", -1, "valsel")
    if not os.path.exists(b) or C.sha256(b) != s.get("fftruth_sha256"):
        raise SystemExit(f"{b} is missing or changed since {p} was written: re-run match + anchor")


def _truths_guard(args):
    """TEST-B / DEV-TEST truths (BASIN reference, labels, TEST-B anchoring) run under the configuration of the
    full VALSEL truths."""
    if args.set not in BLINDED or _partial(args):
        return
    b = C.bundle_path("fftruth", -1, "valsel")
    if not os.path.exists(b):
        raise SystemExit(f"{b} missing: minimise the full VALSEL truths first")
    m = C.load_bundle(b)["meta"]
    if not _same_ff(args, m.get("ffident"), m.get("ff_state")):
        raise SystemExit(f"FF* {_ffident(args)} is not the configuration of the VALSEL truths "
                         f"({m.get('ffident')} / {m.get('ff_state')})")


def _call_stats(info, n_atoms):
    return {"rounds": info["rounds"], "struct_evals": sum(n for n, _ in info["calls"]),
            "sec": round(sum(s for _, s in info["calls"]), 2), "atoms": n_atoms,
            "sec_per_struct_eval": round(sum(s for _, s in info["calls"]) / max(1, sum(n for n, _ in info["calls"])), 4)}


# ------------------------------------------------------------------------------ fdcheck / ffmin
def _w_fd(t):
    a = t["a"]
    (ff, s0), p = _ff(), ff_prep(a)
    h = t["h"]
    rows = []
    for name, R in (("truth", a["R0"].double()), ("haar", haar(1, torch.Generator().manual_seed(t["seed"]))[0])):
        pts = [R] + [_so3_exp(sgn * h * torch.eye(3, dtype=torch.float64)[j]) @ R for j in range(3) for sgn in (1, -1)]
        t0 = time.time()
        res = ff.evaluate([ff_struct(p, Q) for Q in pts], tag=f"fd {a['refcode']}")
        sec = time.time() - t0
        if any(r is None for r in res):
            rows.append({"pose": name, "fail": True}); continue
        g = ff_grad(p, R, res[0][1])
        E = [r[0] for r in res]
        g_fd = torch.tensor([(E[1 + 2 * j] - E[2 + 2 * j]) / (2 * h) for j in range(3)], dtype=torch.float64)
        rows.append({"pose": name, "g": g.tolist(), "g_fd": g_fd.tolist(),
                     "rel_err": float((g - g_fd).norm() / g.norm().clamp_min(1e-12)),
                     "abs_err": float((g - g_fd).norm()), "E": E[0], "sec_batch7": round(sec, 2)})
    row = _ff_row(ff, s0)
    _ff_done()
    return {"refcode": a["refcode"], "K": a["K"], "atoms": int(a["K"] * a["local"].shape[0]),
            "xh": p["xh"], "rows": rows, **row}


def cmd_fdcheck(args):
    """§3.3 FD check. Pass rule (proposed amendment): every relative error <= 1e-4 over 5 VAL crystals x
    {truth, one Haar pose}; it tests the gradient code, so it is meaningful in float64 (float32 energy noise
    ~1e-5 eV / 1e-4 rad ~ 0.1 eV/rad); float32 figures are reported beside it."""
    _ff_resolve(args)
    items = C.load_set(args.set)
    idx = _pick_indices(len(items), args) if _partial(args) else list(range(5))
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "h": args.h, "seed": 31337 + j} for j in idx]
    job = f"fdcheck_{args.set}_{_fftag(args)}"
    done = _run(tasks, _w_fd, os.path.join(OUT, job + ".jsonl"), args.workers,
                init=_ff_init, initargs=(_ff_cfg(args, job),), cfg=_ffkey(args, h=args.h), smoke=_partial(args))
    rows = [done[items[j]["refcode"]] for j in idx]
    for r in rows:
        print(r["K"], r["atoms"], [(x["pose"], round(x.get("rel_err", float("nan")), 7)) for x in r["rows"]])
    rel = [x.get("rel_err", float("inf")) for r in rows for x in r["rows"]]
    rep = {"set": args.set, "ff": _ffident(args), "h": args.h, "n_crystals": len(rows), "n": len(rel),
           "max_rel_err": max(rel), "pass": bool(len(rel) == 2 * len(rows) and max(rel) <= 1e-4),
           "rule": "every relative error <= 1e-4 (5 VAL crystals x {truth, Haar}); meaningful in float64",
           "partial": _partial(args), "ff_state": [r.get("ff_state") for r in rows][0]}
    out = os.path.join(OUT, job + (".smoke" if _partial(args) else "") + ".summary.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n", "max_rel_err", "pass", "partial")}), "->", out)


def _w_ffmin(t):
    a = t["a"]
    (ff, s0), p = _ff(), ff_prep(a)
    starts = []
    if t["truth"]:
        starts.append(a["R0"].double())
    if t["n_haar"]:
        starts += list(haar(t["n_haar"], torch.Generator().manual_seed(t["seed"])))
    t0 = time.time()
    res, info = ff_minimise(ff, p, starts, t["max_evals"], tag=a["refcode"])
    row = _ff_row(ff, s0)
    _ff_done()
    return {"refcode": a["refcode"], "K": a["K"], "atoms": int(len(p["numbers"])), "xh": p["xh"],
            "res": [_res_json(r) for r in res], "stats": _call_stats(info, int(len(p["numbers"]))),
            "sec": round(time.time() - t0, 2), **row}


def cmd_ffmin(args):
    _ff_resolve(args)
    items = C.load_set(args.set)
    idx = _pick_indices(len(items), args)
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "truth": args.truth, "n_haar": args.n_haar,
              "seed": 777000 + j, "max_evals": args.max_evals} for j in idx]
    job = f"ffmin_{args.set}_t{int(args.truth)}_h{args.n_haar}_e{args.max_evals}_{_fftag(args)}"
    done = _run(tasks, _w_ffmin, os.path.join(OUT, job + ".jsonl"), args.workers,
                init=_ff_init, initargs=(_ff_cfg(args, job),), cfg=_ffkey(args), smoke=_partial(args))
    for j in idx:
        r = done[items[j]["refcode"]]
        print(r["K"], r["atoms"], r["stats"], [(x["conv"], x["why"], x["nev"], round(x["g"], 4)) for x in r["res"]])


def _w_time(t):
    a = t["a"]
    (ff, s0), p = _ff(), ff_prep(a)
    Rs = haar(t["batch"], torch.Generator().manual_seed(t["seed"]))
    t0 = time.time()
    ff.evaluate([ff_struct(p, a["R0"].double())], tag="warm-up")          # server start (+ compile) + first call
    warm = time.time() - t0
    s0p = ff.stats["sec_predict"]
    t0 = time.time()
    for r in range(t["reps"]):
        ff.evaluate([ff_struct(p, Rs[r % t["batch"]])], tag="time single")
    single, single_pred = (time.time() - t0) / t["reps"], (ff.stats["sec_predict"] - s0p) / t["reps"]
    s0p = ff.stats["sec_predict"]
    t0 = time.time()
    ff.evaluate([ff_struct(p, R) for R in Rs], tag="time batched")
    batched, batched_pred = (time.time() - t0) / t["batch"], (ff.stats["sec_predict"] - s0p) / t["batch"]
    meta = dict(ff.meta or {})
    row = _ff_row(ff, s0)
    _ff_done()
    return {"refcode": a["refcode"], "K": a["K"], "atoms": int(len(p["numbers"])), "warm_sec": round(warm, 2),
            "sec_single": round(single, 4), "sec_single_predict": round(single_pred, 4), "batch": t["batch"],
            "sec_per_struct_batched": round(batched, 4), "sec_per_struct_batched_predict": round(batched_pred, 4),
            "server": meta, "host": platform.node(), **row}


def cmd_fftime(args):
    """§7 step-4 timing: FF* energy+gradient calls, single and batched, on VAL crystals (K = 2, 4, >= 8)."""
    _ff_resolve(args)
    items = C.load_set(args.set)
    if _partial(args):
        idx = _pick_indices(len(items), args)
    else:
        by = {"2": [j for j, a in enumerate(items) if a["K"] == 2][:7],
              "4": [j for j, a in enumerate(items) if a["K"] == 4][:7],
              "8+": [j for j, a in enumerate(items) if a["K"] >= 8][:6]}
        idx = by["2"] + by["4"] + by["8+"]
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "batch": args.batch, "reps": args.reps,
              "seed": 888000 + j} for j in idx]
    job = f"fftime_{args.set}_{platform.node()}_{_fftag(args)}_b{args.batch}"
    done = _run(tasks, _w_time, os.path.join(OUT, job + ".jsonl"), args.workers, init=_ff_init,
                initargs=(_ff_cfg(args, job),), cfg=_ffkey(args, reps=args.reps), smoke=_partial(args))
    for j in idx:
        r = done[items[j]["refcode"]]
        print({k: r[k] for k in ("K", "atoms", "warm_sec", "sec_single", "sec_single_predict",
                                 "sec_per_struct_batched", "sec_per_struct_batched_predict")})


UMA_SHA256 = {"ckpt": "07068e9c76702ca173d13155095f2117c1b327ec228557e64cd2709c777b824a",     # protocol A1
              "refs": "8d6b57a33c6a139f80cd706f253a030bd9d93cbfb1bbe33fba066e23b7e8c29e"}


def cmd_ffconfig(args):
    """Fix the FF* configuration once, after the §7 step-4 timing gate: settings / dtype / max_atoms / device,
    checked weights, the fairchem-core version and the resolved inference state of one probe call on a synthetic
    cell. truths / search / ie / basin run only under it (§3.3 "with the iii-s settings", "the same settings")."""
    from n3_ffstar import FFStar, toy_structs, sha256, UMA_CKPT, UMA_REFS
    p = _ffconfig_path(args)
    ident = {"settings": args.ff_settings or FF_FALLBACK["settings"], "dtype": args.ff_dtype or FF_FALLBACK["dtype"],
             "max_atoms": args.ff_max_atoms or None, "device": args.device or FF_FALLBACK["device"]}
    ckpt, refs = args.ckpt or UMA_CKPT, args.refs or UMA_REFS
    sha = {"ckpt": sha256(ckpt), "refs": sha256(refs)}
    if sha != UMA_SHA256:
        raise SystemExit(f"UMA files differ from protocol A1: {sha}")
    ff = FFStar(python=args.ff_python, ckpt=ckpt, refs=refs, device=ident["device"], threads=args.ff_threads,
                settings=ident["settings"], dtype=ident["dtype"], max_atoms=ident["max_atoms"],
                log_dir=os.path.join(OUT, "ff_logs", "ffconfig"))
    try:
        t0 = time.time()
        r = ff.evaluate([toy_structs()], tag="ffconfig probe")
        sec = time.time() - t0
    finally:
        ff.close()
    if r[0] is None or ff.state is None:
        raise SystemExit("the probe call failed (see ff_logs/ffconfig): not a usable FF* configuration")
    rec = dict(ident, fairchem_core=ff.meta["fairchem_core"], torch_ff=ff.meta["torch"], state=ff.state,
               ckpt_sha256=sha["ckpt"], refs_sha256=sha["refs"], task="omc", host=platform.node(),
               time=time.strftime("%Y-%m-%d %H:%M:%S"), probe_sec_incl_start=round(sec, 1),
               note="fixed after the §7 step-4 timing gate; identical for truths, iii-s, i-E and BASIN on every set")
    if os.path.exists(p):
        old = json.load(open(p))
        keys = FF_IDENT + ("fairchem_core", "state", "ckpt_sha256", "refs_sha256")
        if any(old.get(k) != rec[k] for k in keys) and not args.force:
            raise SystemExit(f"{p} already fixes {({k: old.get(k) for k in keys})}; --force replaces it, and every "
                             "FF* job made under the old one must then be re-run")
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    json.dump(rec, open(p, "w"), indent=1)
    print(json.dumps(rec, indent=1), "\nwrote", p)


# ------------------------------------------------------------------------------ iii-s search
def proper_point_group(a, tol=0.3):
    """Proper rotations of the molecule's point group (PointGroupAnalyzer, tolerance 0.3 A), float64."""
    from pymatgen.core import Molecule
    from pymatgen.core.periodic_table import Element
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    mol = Molecule([Element.from_Z(int(z)).symbol for z in a["Z"].tolist()], a["local"].double().numpy())
    try:
        ops = PointGroupAnalyzer(mol, tolerance=tol).get_symmetry_operations()
        Ps = [torch.tensor(op.rotation_matrix, dtype=torch.float64) for op in ops]
        Ps = [P for P in Ps if float(torch.linalg.det(P)) > 0]
    except Exception:
        Ps = []
    return torch.stack(Ps) if Ps else torch.eye(3, dtype=torch.float64)[None]


def merge_minima(Rs, Es, Ps, deg=MERGE_DEG):
    """Energy order (ties: start index); a minimum within `deg` of a kept one (min over R P) is merged."""
    order = sorted([i for i in range(len(Rs)) if Rs[i] is not None and math.isfinite(Es[i])],
                   key=lambda i: (Es[i], i))
    kept = []
    for i in order:
        if kept:
            Rset = torch.einsum("rij,pjk->rpik", torch.stack([Rs[k] for k in kept]), Ps).reshape(-1, 3, 3)
            if float(_geo_deg(Rs[i][None], Rset).min()) <= deg:
                continue
        kept.append(i)
    return kept


def _w_search(t):
    a = t["a"]
    (ff, s0), p = _ff(), ff_prep(a)
    scr = t["screen"] or screen_crystal(a, t["evaluator"])
    G, _ = hopf_grid(2)
    assert len(scr["keep"]) == N_KEEP
    starts = [G[i] for i in scr["keep"][:t["n_starts"]]]
    t0 = time.time()
    res, info = ff_minimise(ff, p, starts, t["max_evals"], tag=a["refcode"])
    row = _ff_row(ff, s0)
    _ff_done()
    Ps = proper_point_group(a)
    kept = merge_minima([r["R"] for r in res], [r["E"] for r in res], Ps)
    return {"refcode": a["refcode"], "screen": scr, "starts": scr["keep"][:t["n_starts"]],
            "res": [_res_json(r) for r in res],
            "kept": kept, "pg_order": int(Ps.shape[0]), "xh": p["xh"],
            "stats": _call_stats(info, int(len(p["numbers"]))), "sec": round(time.time() - t0, 2), **row}


def _ranked_bundle(rows, S):
    """Rows with 'res' and 'kept' -> R [n,S,3,3], E [n,S], valid [n,S] (the S lowest distinct minima)."""
    n = len(rows)
    R = torch.eye(3, dtype=torch.float64).expand(n, S, 3, 3).clone()
    E = torch.full((n, S), float("inf"), dtype=torch.float64)
    valid = torch.zeros(n, S, dtype=torch.bool)
    for j, r in enumerate(rows):
        for s, k in enumerate(r["kept"][:S]):
            R[j, s] = _R(r["res"][k]["R"])[0]
            E[j, s] = r["res"][k]["E"]
            valid[j, s] = True
    return R, E, valid


def cmd_search(args):
    if args.set in ("sel", "valsel"):
        raise SystemExit("iii-s has no VALSEL knob and does not run on SEL (protocol §3.3)")
    _ff_resolve(args)
    _form_guard(args)
    items = C.load_set(args.set)
    idx = _pick_indices(len(items), args)
    scr = _read_jsonl(os.path.join(OUT, f"screen_{args.set}.jsonl"))
    ev = None if all(items[j]["refcode"] in scr for j in idx) else _evaluator(args.evaluator)
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "screen": scr.get(items[j]["refcode"]),
              "evaluator": ev, "max_evals": args.max_evals, "n_starts": args.n_starts} for j in idx]
    # the protocol budget is 64 starts x <= 100 evaluations; anything else is a tagged smoke run
    tag = "" if (args.max_evals, args.n_starts) == (100, N_KEEP) else f"_n{args.n_starts}_e{args.max_evals}"
    if tag and not _partial(args):
        raise SystemExit("iii-s budget is 64 starts x <= 100 evaluations; other values only with --limit/--only")
    job = f"iii-s{tag}_{args.set}"
    done = _run(tasks, _w_search, os.path.join(OUT, job + ".jsonl"), args.workers,
                init=_ff_init, initargs=(_ff_cfg(args, job),), cfg=_ffkey(args, n_starts=args.n_starts),
                smoke=_partial(args))
    rows = [done[items[j]["refcode"]] for j in idx]
    R, E, valid = _ranked_bundle(rows, S_DRAWS)
    top = [(r["kept"][:S_DRAWS] + [None] * S_DRAWS)[:S_DRAWS] for r in rows]
    conv = torch.tensor([[k is not None and r["res"][k]["conv"] for k in t] for r, t in zip(rows, top)]) & valid
    meta = _meta(args, max_evals=args.max_evals, n_keep=N_KEEP, n_starts=args.n_starts, merge_deg=MERGE_DEG,
                 **_ff_meta(args, job, rows), before_form=bool(getattr(args, "before_form", False)),
                 converged=conv, reason=[[None if k is None else r["res"][k]["why"] for k in t]
                                         for r, t in zip(rows, top)],
                 reason_all_starts=[[x["why"] for x in r["res"]] for r in rows],
                 det_fail=_det_summary([r["refcode"] for r in rows], [r["res"] for r in rows]),
                 n_distinct=[len(r["kept"]) for r in rows],
                 n_unconverged=[sum(not x["conv"] for x in r["res"]) for r in rows],
                 n_no_minimum=[sum(x["R"] is None for x in r["res"]) for r in rows],
                 miss=[r["refcode"] for r, v in zip(rows, valid) if not bool(v.any())],
                 xh=[r["xh"] for r in rows], xh_total=_xh_total(rows), stats=[r["stats"] for r in rows],
                 screen_evaluator=[r["screen"]["evaluator"] for r in rows])
    arm = "iii-s" + tag
    p = _save(C.bundle_path(arm, -1, args.set), _partial(args) or bool(tag), arm=arm, seed=-1,
              set_name=args.set, refcodes=[r["refcode"] for r in rows], kind="rasym", R=R, valid=valid,
              sel={"ff": E}, meta=meta)
    print("wrote", p, "| crystals with no finite minimum (MISS):", len(meta["miss"]),
          "| normalised X-H bonds:", _xh_total(rows))
    print(_det_txt(meta))


# ------------------------------------------------------------------------------ truths / anchor
def _w_truth(t):
    a = t["a"]
    (ff, s0), p = _ff(), ff_prep(a)
    t0 = time.time()
    res, info = ff_minimise(ff, p, [a["R0"].double()], t["max_evals"], tag=a["refcode"])
    row = _ff_row(ff, s0)
    _ff_done()
    return {"refcode": a["refcode"], "res": _res_json(res[0]), "xh": p["xh"], "K": a["K"],
            "stats": _call_stats(info, int(len(p["numbers"]))), "sec": round(time.time() - t0, 2), **row}


def cmd_truths(args):
    _ff_resolve(args)
    _truths_guard(args)
    items = C.load_set(args.set)
    idx = _pick_indices(len(items), args)
    tasks = [{"refcode": items[j]["refcode"], "a": items[j], "max_evals": args.max_evals} for j in idx]
    job = f"fftruth_{args.set}"
    done = _run(tasks, _w_truth, os.path.join(OUT, job + ".jsonl"), args.workers,
                init=_ff_init, initargs=(_ff_cfg(args, job),), cfg=_ffkey(args), smoke=_partial(args))
    rows = [done[items[j]["refcode"]] for j in idx]
    valid = torch.tensor([[r["res"]["R"] is not None] for r in rows])
    R = torch.stack([_R(r["res"]["R"]) if r["res"]["R"] is not None else torch.eye(3, dtype=torch.float64)[None]
                     for r in rows])
    E = torch.tensor([[r["res"]["E"] if r["res"]["R"] is not None else float("inf")] for r in rows],
                     dtype=torch.float64)
    meta = _meta(args, max_evals=args.max_evals, **_ff_meta(args, job, rows),
                 converged=torch.tensor([[bool(r["res"]["conv"])] for r in rows]) & valid,
                 n_eval=[r["res"]["nev"] for r in rows], reason=[r["res"]["why"] for r in rows],
                 det_fail=_det_summary([r["refcode"] for r in rows], [[r["res"]] for r in rows]),
                 E_start=[r["res"]["E_start"] for r in rows], xh=[r["xh"] for r in rows],
                 xh_total=_xh_total(rows),
                 stats=[r["stats"] for r in rows],
                 note="FF*-minimised TRUE pose. `anchor` (from n3_score match rows): ANCHORED = converged & "
                      "fit(truth, expand(R)); DRIFTED = converged & no fit; UNRESOLVED = not converged or no "
                      "minimum (valid False; §3.3 rule iii: a BASIN miss for every arm, not anchored)")
    p = _save(C.bundle_path("fftruth", -1, args.set), _partial(args), arm="fftruth", seed=-1, set_name=args.set,
              refcodes=[r["refcode"] for r in rows], kind="rasym", R=R, valid=valid, sel={"ff": E}, meta=meta)
    print("wrote", p, "| converged", int(meta["converged"].sum()), "/", len(rows),
          "| normalised X-H bonds:", _xh_total(rows))
    print(_det_txt(meta))
    print("next: python scripts/n3_score.py match --bundles", os.path.relpath(p, C.REPO).replace("\\", "/"),
          "--matcher primary" + (" --step9" if args.set in BLINDED else ""), "&& python scripts/n3_classical.py "
          "anchor --set", args.set)


def cmd_anchor(args):
    """Anchoring, H2 FORM and FF* truth labels from n3_score.py's primary-matcher rows of the fftruth bundle
    (every match is the scorer's, under the §1.3 harness: a deterministic matcher failure is a non-match).
    TEST-B / DEV-TEST rows are accepted only from a --step9 match. Nothing is matched here."""
    import n3_score as S
    src = C.bundle_path("fftruth", -1, args.set)
    if not os.path.exists(src):
        src = src[:-3] + ".partial.pt"
    if not os.path.exists(src):
        raise SystemExit(f"no fftruth bundle for {args.set}: run `truths --set {args.set}` first")
    b = C.load_bundle(src)
    n = len(b["refcodes"])
    rsrc = os.path.relpath(src, C.REPO).replace("\\", "/")
    cmd = f"python scripts/n3_score.py match --bundles {rsrc} --matcher primary" + \
          (" --step9" if args.set in BLINDED else "")
    mpath = S.match_path(src, "primary")
    mmeta = mpath[:-len(".jsonl")] + ".meta.json"
    if not os.path.exists(mmeta):
        raise SystemExit(f"no match rows for {rsrc}: run `{cmd}`")
    mm = json.load(open(mmeta))
    sha = C.sha256(src)
    if mm["bundle_sha256"] != sha:
        raise SystemExit(f"{rsrc} changed since it was matched: delete {mpath} (+ .meta/.events) and re-run `{cmd}`")
    if args.set in BLINDED and not mm.get("step9"):
        raise SystemExit(f"{args.set}: only rows of a --step9 match are used (§1.3)")
    items = C.load_set(args.set)
    if b["refcodes"] != [a["refcode"] for a in items[:n]]:
        raise SystemExit(f"{src}: rows are not the frozen {args.set} order or a prefix of it (the scorer indexes "
                         f"the truth by row: smoke with --limit, not --only)")
    res = S.load_match(src, "primary")
    miss = [j for j in range(n) if (j, 0) not in res]
    if miss:
        raise SystemExit(f"{len(miss)} of {n} truths have no match row yet: run `{cmd}`")
    n_val = len(C.load_set("val"))
    rows = []
    for j, rc in enumerate(b["refcodes"]):
        valid = bool(b["valid"][j, 0])
        conv = valid and bool(b["meta"]["converged"][j, 0])
        m = res[(j, 0)]
        fit = valid and bool(m["ok"])
        drift = float(_geo_deg(b["R"][j, 0][None], items[j]["R0"].double()[None])[0]) if valid else None
        lab = "UNRESOLVED" if not conv else ("ANCHORED" if fit else "DRIFTED")
        rows.append({"refcode": rc, "label": lab, "fit": fit, "conv": conv, "no_minimum": not valid,
                     "match_status": m.get("status"), "drift_deg": drift,
                     "in_val": args.set == "val" or (args.set == "valsel" and j < n_val)})
    anch = sum(r["label"] == "ANCHORED" for r in rows) / n
    vr = [r for r in rows if r["in_val"]]
    partial = bool(b["meta"].get("partial", False))
    rep = {"set": args.set, "n": n, "anchoring": anch,
           "form": ("EXACT" if anch >= 0.5 else "BASIN") if args.set == "valsel" else None,
           "form_rule": "anchoring on the 400 VALSEL truths >= 0.5 -> EXACT, else BASIN (§3.3)",
           "anchoring_val_only": (sum(r["label"] == "ANCHORED" for r in vr) / len(vr)) if vr else None,
           "labels": {k: sum(r["label"] == k for r in rows) for k in ("ANCHORED", "DRIFTED", "UNRESOLVED")},
           "no_minimum": sum(r["no_minimum"] for r in rows),
           "match_det_fail": sum(r["match_status"] == "det_fail" for r in rows), "partial": partial,
           "median_drift_deg": float(np.median([r["drift_deg"] for r in rows if r["drift_deg"] is not None]))
           if any(r["drift_deg"] is not None for r in rows) else None,
           "fftruth": rsrc, "fftruth_sha256": sha, "match": os.path.relpath(mpath, C.REPO).replace("\\", "/"),
           "matcher": "primary", "match_step9": bool(mm.get("step9")),
           "ffident": b["meta"].get("ffident"), "ff_state": b["meta"].get("ff_state"),
           "fairchem_core": b["meta"].get("fairchem_core"), "det_fail": b["meta"].get("det_fail")}
    sfx = ".smoke" if partial else ""
    out = os.path.join(OUT, f"anchor_{args.set}{sfx}.json")
    json.dump({"summary": rep, "rows": rows}, open(out, "w"), indent=1)
    lab_out = os.path.join(OUT, f"ff_labels_{args.set}{sfx}.json")
    json.dump([{"refcode": r["refcode"], "ff_label": r["label"]} for r in rows], open(lab_out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("det_fail", "ff_state")}, indent=1))
    print("wrote", out, "and", lab_out, "(n3_score report 'ff_labels')")


# ------------------------------------------------------------------------------ i-E / BASIN picks
def _w_ie(t):
    a = t["a"]
    (ff, s0), p = _ff(), ff_prep(a)
    live = [s for s in range(len(t["R"])) if t["valid"][s]]
    t0 = time.time()
    res, info = ff_minimise(ff, p, [torch.tensor(t["R"][s], dtype=torch.float64).reshape(3, 3) for s in live],
                            t["max_evals"], tag=a["refcode"])
    row = _ff_row(ff, s0)
    _ff_done()
    out = [None] * len(t["R"])
    for s, r in zip(live, res):
        out[s] = _res_json(r)
    return {"refcode": a["refcode"], "res": out, "xh": p["xh"],
            "stats": _call_stats(info, int(len(p["numbers"]))), "sec": round(time.time() - t0, 2), **row}


def _min_bundle(rows, b, S):
    n = len(rows)
    R = torch.eye(3, dtype=torch.float64).expand(n, S, 3, 3).clone()
    E = torch.full((n, S), float("inf"), dtype=torch.float64)
    valid = torch.zeros(n, S, dtype=torch.bool)
    conv = torch.zeros(n, S, dtype=torch.bool)
    for j, r in enumerate(rows):
        for s, x in enumerate(r["res"]):
            if x is not None and x["R"] is not None and math.isfinite(x["E"]):
                R[j, s] = _R(x["R"])[0]; E[j, s] = x["E"]; valid[j, s] = True; conv[j, s] = x["conv"]
    return R, E, valid, conv


def cmd_ie(args):
    b = C.load_bundle(args.draws)
    assert b["kind"] == "rasym"
    args.src_partial = bool(b["meta"].get("partial", False))
    _ff_resolve(args)
    _form_guard(args)
    idx = _pick_indices(len(b["refcodes"]), args)
    items = {a["refcode"]: a for a in C.load_set(b["set"])}
    tasks = [{"refcode": b["refcodes"][j], "a": items[b["refcodes"][j]], "R": b["R"][j].reshape(-1, 9).tolist(),
              "valid": b["valid"][j].tolist(), "max_evals": args.max_evals} for j in idx]
    arm = f"i-E-{b['arm']}"
    job = f"{arm}_s{b['seed']}_{b['set']}"
    done = _run(tasks, _w_ie, os.path.join(OUT, job + ".jsonl"), args.workers,
                init=_ff_init, initargs=(_ff_cfg(args, job),), cfg=_ffkey(args, source=os.path.basename(args.draws)),
                smoke=_partial(args) or b["meta"].get("partial", False))
    rows = [done[b["refcodes"][j]] for j in idx]
    R, E, valid, conv = _min_bundle(rows, b, b["R"].shape[1])
    sel = {"ff": E}
    sel.update({f"in_{k}": v[idx].clone() for k, v in b["sel"].items()})
    meta = _meta(args, source=args.draws, max_evals=args.max_evals, **_ff_meta(args, job, rows),
                 before_form=bool(getattr(args, "before_form", False)), converged=conv,
                 in_valid=b["valid"][idx].clone(),
                 reason=[[x["why"] if x else None for x in r["res"]] for r in rows],
                 det_fail=_det_summary([r["refcode"] for r in rows], [r["res"] for r in rows]),
                 miss=[r["refcode"] for r, v in zip(rows, valid) if not bool(v.any())],
                 n_eval=[[x["nev"] if x else 0 for x in r["res"]] for r in rows],
                 xh=[r["xh"] for r in rows], xh_total=_xh_total(rows),
                 stats=[r["stats"] for r in rows])
    p = _save(C.bundle_path(arm, b["seed"], b["set"]), _partial(args) or b["meta"].get("partial", False),
              arm=arm, seed=b["seed"], set_name=b["set"], refcodes=[r["refcode"] for r in rows], kind="rasym",
              R=R, valid=valid, sel=sel, meta=meta)
    print("wrote", p, "| crystals with no finite minimum (MISS):", len(meta["miss"]),
          "| normalised X-H bonds:", _xh_total(rows))
    print(_det_txt(meta))


def select_pick(values, valid):
    """§2.3 LJ-type pick: lowest value among valid draws; non-finite last; ties -> lowest index."""
    cand = [s for s in range(len(values)) if valid[s]]
    if not cand:
        return None
    key = lambda s: (0 if math.isfinite(values[s]) else 1, values[s] if math.isfinite(values[s]) else 0.0, s)
    return min(cand, key=key)


def _existing_bundle(arm, seed, set_name):
    p = C.bundle_path(arm, seed, set_name)
    return [q for q in (p, p[:-3] + ".partial.pt") if os.path.exists(q)]


def cmd_basin(args):
    b = C.load_bundle(args.draws)
    args.src_partial = bool(b["meta"].get("partial", False))
    partial = _smoke(args)
    if args.from_ie:
        # i-E already minimised every draw: the BASIN candidate is the i-E pose of the arm's pick (§3.3)
        if not b["arm"].startswith("i-E-"):
            raise SystemExit("--from-ie needs an i-E bundle")
        arm, mode = f"basin-{b['arm'][len('i-E-'):]}", "from_ie"
    else:
        if b["arm"].startswith("i-E-"):
            raise SystemExit("an i-E bundle is already minimised: use --from-ie (never minimise twice, §3.3)")
        ie = _existing_bundle(f"i-E-{b['arm']}", b["seed"], b["set"])
        if ie:
            raise SystemExit(f"{ie[0]} exists: {b['arm']}'s BASIN candidate is the i-E pose of its pick "
                             f"(§3.3, not minimised twice): run `basin --draws {ie[0]} --from-ie`")
        arm, mode = f"basin-{b['arm']}", "direct"
        _ff_resolve(args)
        _form_guard(args)
    out = C.bundle_path(arm, b["seed"], b["set"])
    out_eff = out[:-3] + ".partial.pt" if partial else out
    if os.path.exists(out_eff) and C.load_bundle(out_eff)["meta"].get("mode") != mode:
        raise SystemExit(f"{out_eff} was written by the other BASIN mode; refusing to overwrite it")
    idx = _pick_indices(len(b["refcodes"]), args)
    items = {a["refcode"]: a for a in C.load_set(b["set"])}
    n = len(idx)
    # the pick is made on the ARM's draws (valid = the arm's own validity; i-E keeps it as meta["in_valid"])
    key = ("in_" + args.selector) if args.from_ie else args.selector
    src_valid = b["meta"]["in_valid"] if args.from_ie else b["valid"]
    picks = [select_pick(b["sel"][key][j].tolist(), src_valid[j].tolist()) for j in idx]
    refcodes = [b["refcodes"][j] for j in idx]
    if args.from_ie:
        # a pick whose minimisation has no finite minimum is a non-match (valid False)
        R = torch.eye(3, dtype=torch.float64).expand(n, 1, 3, 3).clone()
        E = torch.full((n, 1), float("inf"), dtype=torch.float64)
        valid = torch.zeros(n, 1, dtype=torch.bool)
        conv = torch.zeros(n, 1, dtype=torch.bool)
        reason, rs = [], b["meta"].get("reason")
        for q, (j, s) in enumerate(zip(idx, picks)):
            why = None if s is None or rs is None else rs[j][s]
            reason.append(why)
            if s is not None and bool(b["valid"][j, s]):
                R[q, 0], E[q, 0], valid[q, 0] = b["R"][j, s], b["sel"]["ff"][j, s], True
                conv[q, 0] = bool(b["meta"]["converged"][j, s])
        m = b["meta"]
        rows_meta = {"from_ie": args.draws, "ffident": m.get("ffident"), "ff_state": m.get("ff_state"),
                     "fairchem_core": m.get("fairchem_core"), "before_form": m.get("before_form"),
                     "max_evals": m.get("max_evals"), "reason": [[w] for w in reason],
                     "det_fail": _det_summary(refcodes, [[None if w is None else {"why": w}] for w in reason]),
                     "ff_retries": 0, "ff_det_structures": 0}
    else:
        tasks = [{"refcode": b["refcodes"][j], "a": items[b["refcodes"][j]],
                  "R": [b["R"][j, s].reshape(-1).tolist()] if s is not None else [],
                  "valid": [True] if s is not None else [], "max_evals": args.max_evals}
                 for j, s in zip(idx, picks)]
        job = f"{arm}_s{b['seed']}_{b['set']}"
        done = _run(tasks, _w_ie, os.path.join(OUT, job + ".jsonl"), args.workers,
                    init=_ff_init, initargs=(_ff_cfg(args, job),),
                    cfg=_ffkey(args, source=os.path.basename(args.draws), selector=args.selector),
                    smoke=partial)
        rows = [done[b["refcodes"][j]] for j in idx]
        for r in rows:
            r["res"] = r["res"] or [None]
        R, E, valid, conv = _min_bundle(rows, b, 1)
        rows_meta = {"stats": [r["stats"] for r in rows], "xh": [r["xh"] for r in rows],
                     "max_evals": args.max_evals, **_ff_meta(args, job, rows),
                     "before_form": bool(getattr(args, "before_form", False)),
                     "reason": [[x["why"] if x else None for x in r["res"]] for r in rows],
                     "det_fail": _det_summary(refcodes, [r["res"] for r in rows])}
    meta = _meta(args, source=args.draws, selector=args.selector, pick=picks, converged=conv, mode=mode,
                 **rows_meta)
    p = _save(out, partial, arm=arm, seed=b["seed"], set_name=b["set"], refcodes=refcodes,
              kind="rasym", R=R, valid=valid, sel={"ff": E}, meta=meta)
    print("wrote", p, f"({mode}) | picks with no minimum:", int((~valid[:, 0]).sum()))
    print(_det_txt(meta))


# ------------------------------------------------------------------------------ synthetic self-test
class _ToyFF:
    """Stand-in for FFStar on a synthetic cell: E = sum_{i<j} w_ij |x_i - x_j - s_ij|^2 (float64), forces by
    autograd; `fail` = {(round, index)} or {"start"} makes those calls return None (deterministic failure)."""

    def __init__(self, n, seed=0, fail=()):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.rand(n, n, generator=g, dtype=torch.float64)
        self.s = torch.randn(n, n, 3, generator=g, dtype=torch.float64)
        self.fail, self.round, self.stats = set(fail), 0, {"sec_spawn": 0.0}

    def energy(self, X):
        D = X[:, None, :] - X[None, :, :] - self.s
        return 0.5 * (self.w * (D * D).sum(-1)).sum()

    def evaluate(self, structs, tag=""):
        out = []
        for k, (_, X, _) in enumerate(structs):
            if (self.round, k) in self.fail:
                out.append(None); continue
            X = torch.tensor(X, dtype=torch.float64, requires_grad=True)
            E = self.energy(X)
            out.append((float(E), (-torch.autograd.grad(E, X)[0]).numpy()))
        self.round += 1
        return out


def _toy_crystal():
    """Synthetic P-1-like cell (identity + inversion, det -1), carbon-only, no CSD content."""
    g = torch.Generator().manual_seed(1)
    local = torch.randn(6, 3, generator=g, dtype=torch.float64)
    local = local - local.mean(0)
    W = torch.stack([torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)])
    return {"L": torch.tensor([[7.0, 0, 0], [1.0, 8.0, 0], [0.5, 1.5, 9.0]], dtype=torch.float64), "W": W,
            "t": torch.zeros(2, 3, dtype=torch.float64), "c0": torch.tensor([0.2, 0.3, 0.1], dtype=torch.float64),
            "local": local, "Z": torch.full((6,), 6, dtype=torch.long), "K": 2}


def cmd_selftest(args):
    torch.manual_seed(0)
    a = _toy_crystal()
    p = ff_prep(a)
    n = len(p["numbers"])
    ok = True
    # 1. force-derived gradient == autograd of E(exp([delta]x) R) at delta = 0 (improper op included)
    ff = _ToyFF(n)
    R = haar(1, torch.Generator().manual_seed(5))[0]
    dlt = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    X = (p["base"][:, None, :] + torch.einsum("kij,aj->kai", p["Rc"] @ (_so3_exp(dlt) @ R), p["loc"])).reshape(-1, 3)
    g_auto = torch.autograd.grad(ff.energy(X), dlt)[0]
    E, F = ff.evaluate([ff_struct(p, R)])[0]
    g_ff = ff_grad(p, R, F)
    err = float((g_ff - g_auto).norm() / g_auto.norm())
    print(f"gradient formula vs autograd (det -1 op present): rel err {err:.2e}"); ok &= err < 1e-10
    # 2. L-BFGS converges from Haar starts (batched driver); converged iff |g|/K < tol at the returned pose
    starts = list(haar(6, torch.Generator().manual_seed(7)))
    res, info = ff_minimise(_ToyFF(n), p, starts, 100)
    print("lbfgs:", [(r["reason"], r["n_eval"], f"{r['gnorm_per_mol']:.1e}") for r in res], "rounds", info["rounds"])
    ok &= all(r["converged"] == (r["gnorm_per_mol"] < FF_TOL) for r in res)
    ok &= sum(r["converged"] for r in res) >= 5 and max(r["n_eval"] for r in res) <= 100
    # 3. cap: never more than max_evals evaluations
    res, _ = ff_minimise(_ToyFF(n), p, starts, 4)
    print("cap 4:", [(r["reason"], r["n_eval"]) for r in res]); ok &= all(r["n_eval"] <= 4 for r in res)
    # 4. deterministic failure at the start pose -> no minimum; mid-run -> last finite iterate, UNCONVERGED
    #    (start 0 fails in round 0, so from round 1 on start 1 is the only structure, batch index 0)
    res, _ = ff_minimise(_ToyFF(n, fail={(0, 0), (3, 0)}), p, starts[:2], 100)
    print("failures:", [(r["reason"], r["n_eval"], r["R"] is None) for r in res])
    ok &= res[0]["R"] is None and res[0]["reason"] == "ff_fail_start"
    ok &= res[1]["reason"] == "ff_fail" and res[1]["R"] is not None and math.isfinite(res[1]["E"])
    # 5. merge modulo a point group: R and R P (P a molecular 2-fold) are one minimum
    P = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))[None]
    Ps = torch.cat([torch.eye(3, dtype=torch.float64)[None], P])
    kept = merge_minima([R, R @ P[0], _so3_exp(torch.tensor([0.0, 0.0, 0.03], dtype=torch.float64)) @ R],
                        [1.0, 0.5, 2.0], Ps)
    print("merge:", kept); ok &= kept == [1]
    # 6. Haar generator = manifolds.random_so3 construction
    from symmc_flow import manifolds as M
    torch.manual_seed(3)
    ref = M.random_so3((4,), dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    same = bool(torch.equal(haar(4, g), ref))
    print("haar == random_so3 under the same seed:", same); ok &= same
    print("SELFTEST", "OK" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


# ------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s, set_default="val", ff=False):
        s.add_argument("--set", default=set_default, choices=list(C.SETS))
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--limit", type=int, default=None, help="smoke: first N crystals (-> .partial.pt)")
        s.add_argument("--only", default=None, help="smoke: comma-separated crystal indices")
        if ff:
            s.add_argument("--ff-config", default=None,
                           help="the fixed FF* configuration (default results/n3/private/classical/ffstar_config.json)")
            s.add_argument("--device", default=None, help="FF* device (default: the fixed configuration, else cpu)")
            s.add_argument("--ff-python", default=None, help="python of the fairchem venv")
            s.add_argument("--ckpt", default=None)
            s.add_argument("--refs", default=None)
            s.add_argument("--ff-threads", type=int, default=2, help="torch threads of each FF* server (CPU)")
            s.add_argument("--ff-settings", default=None,
                           help="fairchem inference_settings name (default: the fixed configuration, else 'default')")
            s.add_argument("--ff-dtype", default=None, choices=["float32", "float64"],
                           help="UMA base precision (default: the fixed configuration, else float32)")
            s.add_argument("--ff-max-atoms", type=int, default=None,
                           help="split a round into sub-batches of <= N atoms; 0 = off (default: the fixed "
                                "configuration, else off; needs an amendment)")
            s.add_argument("--max-evals", type=int, default=100)

    s = sub.add_parser("haar"); common(s); s.add_argument("--base", type=int, required=True, choices=[0, 1, 2])
    s = sub.add_parser("press"); common(s)
    s.add_argument("--base", type=int, default=0)
    s.add_argument("--draws", default=None, help="input bundle (default: the iii-u bundle of --base/--set)")
    s.add_argument("--arm", default=None); s.add_argument("--steps", type=int, default=80)
    s = sub.add_parser("p1basin"); common(s); s.add_argument("--steps", type=int, default=80)
    s = sub.add_parser("grid"); s.add_argument("--probes", type=int, default=200000)
    sub.add_parser("selftest")
    s = sub.add_parser("lbgate"); s.add_argument("--workers", type=int, default=2)
    s.add_argument("--n-crystals", type=int, default=20); s.add_argument("--n-orient", type=int, default=100)
    s.add_argument("--n-low", type=int, default=10)
    s = sub.add_parser("screen"); common(s)
    s.add_argument("--evaluator", default="auto", choices=["auto", "batched", "energy"])
    s.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("fdcheck"); common(s, ff=True); s.add_argument("--h", type=float, default=1e-4)
    s = sub.add_parser("fftime"); common(s, ff=True)
    s.add_argument("--batch", type=int, default=64); s.add_argument("--reps", type=int, default=3)
    s = sub.add_parser("ffmin"); common(s, ff=True)
    s.add_argument("--truth", action="store_true"); s.add_argument("--n-haar", type=int, default=0)
    s = sub.add_parser("ffconfig"); common(s, ff=True)
    s.add_argument("--force", action="store_true", help="replace a different fixed configuration")
    s = sub.add_parser("search"); common(s, ff=True)
    s.add_argument("--n-starts", type=int, default=N_KEEP, help="smoke only; the protocol value is 64")
    s.add_argument("--before-form", action="store_true", help="smoke: skip the H2-FORM-first guard")
    s.add_argument("--evaluator", default="auto", choices=["auto", "batched", "energy"])
    s = sub.add_parser("truths"); common(s, ff=True)
    s.set_defaults(max_evals=1000)
    s = sub.add_parser("anchor"); s.add_argument("--set", default="valsel", choices=list(C.SETS))
    s = sub.add_parser("ie"); common(s, ff=True); s.add_argument("--draws", required=True)
    s.add_argument("--before-form", action="store_true", help="smoke: skip the H2-FORM-first guard")
    s = sub.add_parser("basin"); common(s, ff=True); s.add_argument("--draws", required=True)
    s.add_argument("--selector", default="lj")
    s.add_argument("--from-ie", action="store_true", help="--draws is an i-E bundle: reuse its minimised pick")
    s.add_argument("--before-form", action="store_true", help="smoke: skip the H2-FORM-first guard")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    {"haar": cmd_haar, "press": cmd_press, "p1basin": cmd_p1basin, "grid": cmd_grid, "lbgate": cmd_lbgate,
     "screen": cmd_screen, "fdcheck": cmd_fdcheck, "fftime": cmd_fftime, "ffmin": cmd_ffmin,
     "ffconfig": cmd_ffconfig, "search": cmd_search,
     "truths": cmd_truths, "anchor": cmd_anchor, "ie": cmd_ie, "basin": cmd_basin,
     "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
