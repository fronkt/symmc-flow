"""N1 (Omega rebuild): pick ONE learned draw per crystal without the steric energy.

Stage 1 (`sample`, parallel, resumable): for each crystal of a split and each checkpoint, draw S
orientations from the learned flow (batched Euler integration) and record per draw
  R           the orientation (9 floats)
  exact       unrelaxed StructureMatcher match vs truth (the target the selector should pick)
  torque_end  |learned field| at t = 0.975 (a converged draw should feel little residual torque)
  e_lj        unrelaxed steric Lennard-Jones packing energy (the G2 ranking, for comparison)
Stage 2 (`select`, offline): score selectors on the recorded draws -- random pick (= match@1), lowest
e_lj, lowest torque_end, and endpoint-mode frequency (number of other draws whose asym molecule lies
within a Chamfer radius r, element-matched so molecular symmetry does not split a mode), per seed and
pooled over seeds. Selectors are CHOSEN on the val split and REPORTED on test (tasks/omega_rebuild_plan.md N1).

    python scripts/g2_rank.py sample --split val --ckpts results/vast_g2/eval_s0.pt ... --out results/n1_val.jsonl
    python scripts/g2_rank.py select --val results/n1_val.jsonl [--test results/n1_test.jsonl]
"""
import argparse
import json
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

_MODELS = {}


def _init(ckpts):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from g2_learned_flow import TorqueField
    for p in ckpts:
        m = TorqueField()
        ck = torch.load(p, weights_only=False)
        m.load_state_dict(ck.get("best_state", ck))
        m.eval()
        _MODELS[os.path.basename(p)] = m


@torch.no_grad()
def sample_batch(model, g, S, steps=40):
    from symmc_flow import manifolds as M
    R = M.random_so3((S,))
    for s in range(steps):
        t = torch.full((S,), s / steps)
        R = M.so3_exp(model.forward_batch(g, R, t) / steps) @ R
    return R


def _work(task):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from g2_asym_baselines import expand, energy, to_structure
    from g2_learned_flow import precompute
    a, S, seed = task
    torch.manual_seed(seed)
    sm = StructureMatcher()
    o = a["orig"]
    truth = to_structure(o["lattice"], o["centroid"], o["orient"], o["local"], o["Z"],
                         o["atom_mask"], o["mol_mask"])
    g = precompute(a, 8.0)
    row = {"refcode": a["refcode"], "sg": a["sg"]}
    for name, m in _MODELS.items():
        R = sample_batch(m, g, S)
        with torch.no_grad():
            tq = m.forward_batch(g, R, torch.full((S,), 0.975)).norm(dim=-1)
        draws = []
        for i in range(S):
            cell = expand(a, R[i].double())
            draws.append({"R": [round(x, 6) for x in R[i].reshape(-1).tolist()],
                          "exact": bool(sm.fit(to_structure(*cell), truth)),
                          "torque_end": round(float(tq[i]), 5), "e_lj": round(energy(*cell), 3)})
        row[name] = draws
    return row


def cmd_sample(args):
    warnings.filterwarnings("ignore")
    torch.multiprocessing.set_sharing_strategy("file_system")
    elig = torch.load(args.cache, weights_only=False)
    perm = torch.randperm(len(elig), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    idx = perm[:200] if args.split == "test" else perm[200:300]
    crystals = [elig[i] for i in idx]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["refcode"] for l in open(args.out)}
    todo = [(i, a) for i, a in enumerate(crystals) if a["refcode"] not in done]
    print(f"{args.split}: {len(done)} done, {len(todo)} to go", flush=True)
    from multiprocessing import Pool, TimeoutError as MPTimeout
    pool = Pool(args.workers, initializer=_init, initargs=(args.ckpts,))
    base = 50_000 if args.split == "test" else 40_000
    pend = [(a, pool.apply_async(_work, ((a, args.samples, base + i),))) for i, a in todo]
    with open(args.out, "a") as fh:
        for n_, (a, ar) in enumerate(pend):
            try:
                r = ar.get(timeout=args.timeout)
            except MPTimeout:
                r = {"refcode": a["refcode"], "timeout": True}
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (n_ + 1) % 10 == 0:
                print(f"{len(done) + n_ + 1}/{len(crystals)}", flush=True)
    pool.terminate(); pool.join()


# ------------------------------------------------------------------------------ selection
def _chamfer_matrix(draws, local, Z):
    """Pairwise element-matched Chamfer distance (A) between the asym molecules of the draws."""
    R = torch.tensor([d["R"] for d in draws]).reshape(-1, 3, 3)
    X = torch.einsum("sij,aj->sai", R, local.float())                    # (S,A,3), centroid fixed
    same = (Z[:, None] == Z[None, :])                                    # (A,A)
    D = (X[:, None, :, None] - X[None, :, None, :]).norm(dim=-1)         # (S,S,A,A)
    D = D.masked_fill(~same, float("inf"))
    return 0.5 * (D.min(-1).values.mean(-1) + D.min(-2).values.mean(-1))  # (S,S)


def _pick(draws, rule, local, Z, rng):
    if rule == "random":
        return int(torch.randint(len(draws), (1,), generator=rng))
    if rule == "lj":
        return min(range(len(draws)), key=lambda i: draws[i]["e_lj"])
    if rule == "torque":
        return min(range(len(draws)), key=lambda i: draws[i]["torque_end"])
    if rule.startswith("mode"):
        r = float(rule.split("@")[1])
        C = _chamfer_matrix(draws, local, Z)
        cnt = (C < r).sum(-1)
        best = int(cnt.max())
        cands = [i for i in range(len(draws)) if int(cnt[i]) == best]
        return min(cands, key=lambda i: draws[i]["torque_end"])      # tie-break: calmest draw
    raise ValueError(rule)


def score(path, rules, cache, seeds_pool=True):
    elig = {a["refcode"]: a for a in torch.load(cache, weights_only=False)}
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if not r.get("timeout")]
    arms = [k for k in rows[0] if k not in ("refcode", "sg")]
    out = {"n": len(rows)}
    rng = torch.Generator().manual_seed(0)
    for rule in rules:
        per_seed = []
        for arm in arms:
            hit = sum(r[arm][_pick(r[arm], rule, elig[r["refcode"]]["local"], elig[r["refcode"]]["Z"], rng)]["exact"]
                      for r in rows)
            per_seed.append(100 * hit / len(rows))
        out[rule] = {"per_seed": [round(v, 1) for v in per_seed],
                     "mean": round(sum(per_seed) / len(per_seed), 1)}
        if seeds_pool:
            hit = 0
            for r in rows:
                pooled = [d for arm in arms for d in r[arm]]
                a = elig[r["refcode"]]
                hit += pooled[_pick(pooled, rule, a["local"], a["Z"], rng)]["exact"]
            out[rule]["pooled_3seeds"] = round(100 * hit / len(rows), 1)
    out["oracle_any_draw"] = {"per_seed": [round(100 * sum(any(d["exact"] for d in r[arm]) for r in rows) / len(rows), 1)
                                           for arm in arms]}
    return out


def cmd_select(args):
    rules = ["random", "lj", "torque"] + [f"mode@{r}" for r in (0.25, 0.5, 0.75, 1.0, 1.5)]
    val = score(args.val, rules, args.cache)
    print("VAL:", json.dumps(val, indent=1))
    if args.test:
        chosen = args.rule or max(rules[1:], key=lambda r: val[r]["mean"])
        test = score(args.test, ["random", "lj", chosen], args.cache)
        print(f"TEST with val-chosen rule '{chosen}':", json.dumps(test, indent=1))
        if args.out:
            json.dump({"val": val, "chosen": chosen, "test": test}, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--split", choices=["val", "test"], required=True)
    s.add_argument("--ckpts", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cache", default="data/csd_mol/g2_asym.pt")
    s.add_argument("--split-seed", type=int, default=0)
    s.add_argument("--samples", type=int, default=16)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--timeout", type=float, default=3600.0)
    c = sub.add_parser("select")
    c.add_argument("--val", required=True)
    c.add_argument("--test", default=None)
    c.add_argument("--rule", default=None, help="force a rule (else best non-random on val)")
    c.add_argument("--out", default=None)
    c.add_argument("--cache", default="data/csd_mol/g2_asym.pt")
    args = ap.parse_args()
    cmd_sample(args) if args.cmd == "sample" else cmd_select(args)


if __name__ == "__main__":
    main()
