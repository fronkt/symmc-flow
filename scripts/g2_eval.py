"""G2 evaluation, parallel over test crystals: learned flow (one arm per checkpoint) vs Haar draws.

For each held-out crystal (same seeded split as g2_asym_baselines / g2_learned_flow; test = first
--n-test of the permutation) and each arm, draw S orientations and report
  match1 / matchS   unrelaxed exact StructureMatcher match vs truth (first draw / any draw)
  geo_min           smallest geodesic error to the true R_asym among the S draws (degrees)
  basin_best        relax all S (orientation-only press), keep lowest energy: in the relaxed-truth basin?
  basin_any         any relaxed draw in the relaxed-truth basin (search upper bound)
Per-crystal rows are appended to <out>.jsonl so a killed run resumes.

    python scripts/g2_eval.py --ckpts results/eval_s0.pt results/eval_s1.pt results/eval_s2.pt \
        --out results/g2_eval.json --samples 16 --workers 60
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


def _work(task):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from symmc_flow import manifolds as M
    from g2_asym_baselines import expand, press, energy, to_structure
    from g2_learned_flow import precompute, sample
    a, S, steps, seed = task
    torch.manual_seed(seed)
    sm = StructureMatcher()
    o = a["orig"]
    truth = to_structure(o["lattice"], o["centroid"], o["orient"], o["local"], o["Z"],
                         o["atom_mask"], o["mol_mask"])
    basin_ref = to_structure(*press(a, a["R0"], steps))
    g = precompute(a, 8.0)
    arms = {"haar": [M.random_so3(()) for _ in range(S)]}
    for name, m in _MODELS.items():
        arms[name] = sample(m, g, S)
    row = {"refcode": a["refcode"], "sg": a["sg"]}
    for name, draws in arms.items():
        draws = [R.double() for R in draws]
        ok = [bool(sm.fit(to_structure(*expand(a, R)), truth)) for R in draws]
        geo = [math.degrees(float(M.so3_angle(R.float(), a["R0"].float()))) for R in draws]
        rel = [press(a, R, steps) for R in draws]
        es = [energy(*c) for c in rel]
        inb = [bool(sm.fit(to_structure(*c), basin_ref)) for c in rel]
        bi = min(range(S), key=lambda i: es[i])
        row[name] = {"match1": ok[0], "matchS": any(ok), "geo_min": min(geo),
                     "basin_best": inb[bi], "basin_any": any(inb)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/csd_mol/g2_asym.pt")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--press-steps", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.multiprocessing.set_sharing_strategy("file_system")

    elig = torch.load(args.cache, weights_only=False)
    perm = torch.randperm(len(elig), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    test = [elig[i] for i in perm[:args.n_test]]

    part = args.out + ".jsonl"
    prev = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            prev[r["refcode"]] = r
    todo = [(i, a) for i, a in enumerate(test) if a["refcode"] not in prev]
    print(f"{len(prev)} done, {len(todo)} to go", flush=True)

    from multiprocessing import Pool, TimeoutError as MPTimeout
    pool = Pool(args.workers, initializer=_init, initargs=(args.ckpts,))
    pend = [(a, pool.apply_async(_work, ((a, args.samples, args.press_steps, 10_000 + i),)))
            for i, a in todo]
    with open(part, "a") as fh:
        for n_, (a, ar) in enumerate(pend):
            try:
                r = ar.get(timeout=args.timeout)
            except MPTimeout:
                r = {"refcode": a["refcode"], "timeout": True}
            prev[r["refcode"]] = r
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if (n_ + 1) % 20 == 0:
                print(f"{len(prev)}/{len(test)}", flush=True)
    pool.terminate(); pool.join()

    rows = [prev[a["refcode"]] for a in test if a["refcode"] in prev and not prev[a["refcode"]].get("timeout")]
    arms = [k for k in rows[0] if k not in ("refcode", "sg")] if rows else []
    summ = {"n": len(rows), "timeouts": sum(1 for r in prev.values() if r.get("timeout"))}
    for arm in arms:
        for k in ("match1", "matchS", "basin_best", "basin_any"):
            summ[f"{arm}:{k}"] = round(sum(r[arm][k] for r in rows) / len(rows), 4)
        summ[f"{arm}:geo_min_median"] = round(sorted(r[arm]["geo_min"] for r in rows)[len(rows) // 2], 1)
    json.dump({"args": vars(args), "summary": summ, "rows": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
