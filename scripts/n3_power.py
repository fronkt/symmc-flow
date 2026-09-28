"""N3 design sensitivity (protocol §2.4): power of the exact crystal-level sign-flip test.

Model (numpy default_rng(7), 4,000 replicates):
  OURS        per-crystal propensity p_i ~ Beta(1.056, 13.0) (fitted to N1's 161/33/6/0 crystals by number
              of seeds hit; mean 7.5%), 3 seeds, Bernoulli picks
  comparator  an independent Beta(1.056, 13.0) draw scaled by p_A / 0.075, 3 seeds (m = 1 for iii-s)
  test        n3_stats.signflip_p on d_i = mean_s OURS - mean_s ARM; for H1 p = max over two comparators
              at the same p_A (independent draws)
Output: power at alpha = 0.05 and 0.05/3 for Delta in {1.5, 2, 2.5, 3, 4} pp at the given n.

    python scripts/n3_power.py --n 1000 --out results/n3/power_testB.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_stats as S

A, B, BASE = 1.056, 13.0, 0.075


def sim(n, p_arm, m_arm, reps, rng, h1=False):
    out = np.empty(reps)
    for r in range(reps):
        p = rng.beta(A, B, n)
        ours = rng.random((n, 3)) < p[:, None]
        ps = []
        for _ in range(2 if h1 else 1):
            q = np.clip(rng.beta(A, B, n) * p_arm / BASE, 0, 1)
            arm = rng.random((n, m_arm)) < q[:, None]
            ps.append(S.signflip_p(ours.mean(1) - arm.mean(1)))
        out[r] = max(ps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    res = {"n": args.n, "reps": args.reps, "model": "Beta(1.056,13) OURS 7.5%, 3 seeds; exact sign-flip", "rows": []}
    for delta in (1.5, 2.0, 2.5, 3.0, 4.0):
        p_arm = BASE - delta / 100
        for label, m, h1 in (("3-seed arm", 3, False), ("iii-s (m=1)", 1, False), ("H1 (IUT, 2 arms)", 3, True)):
            p = sim(args.n, p_arm, m, args.reps, rng, h1)
            row = {"delta_pp": delta, "arm": label, "power_a05": float((p < 0.05).mean()),
                   "power_a05_3": float((p < 0.05 / 3).mean())}
            res["rows"].append(row)
            print(row, flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
