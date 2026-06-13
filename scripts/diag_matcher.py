"""Reconcile StructureMatcher.fit (break_on_match=True) vs get_rms_dist
(break_on_match=False) on ONE identical generated set, so the match-rate
difference is matcher-only (same ckpt, same seed, same generations).

    python scripts/diag_matcher.py --ckpt checkpoints/carbon24_big.pt \
        --eval-n 256 --match-k 20 --seed 0
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from symmc_flow.config import ModelConfig
from symmc_flow.model import SymMCFlow
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample
from scripts.train_carbon24 import _to_structures
from symmc_flow.carbon24 import Carbon24Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def both_rates(gens, ref, k):
    """gens: list of k structure-lists; ref: structure-list. Returns
    (fit_rate, rms_rate, mean_rmsd) for best-of-k."""
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    fit_hits = rms_hits = total = 0
    rmsds = []
    for i, r in enumerate(ref):
        if r is None:
            continue
        total += 1
        fit_ok = False
        best = None
        for draw in gens:
            g = draw[i]
            if g is None:
                continue
            if sm.fit(g, r):
                fit_ok = True
            try:
                rd = sm.get_rms_dist(g, r)
            except Exception:
                rd = None
            if rd is not None and (best is None or rd[0] < best):
                best = rd[0]
        fit_hits += fit_ok
        if best is not None:
            rms_hits += 1
            rmsds.append(best)
    n = max(total, 1)
    return fit_hits / n, rms_hits / n, (sum(rmsds) / len(rmsds) if rmsds else float("nan")), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "carbon24_big.pt"))
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--match-k", type=int, default=20)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--centroid-prior-std", type=float, default=0.30)
    ap.add_argument("--vol-per-atom", type=float, default=9.0)
    ap.add_argument("--max-mols", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = resolve_device("auto")
    val_ds = Carbon24Dataset(os.path.join(ROOT, "data", "raw", "carbon_val.csv"),
                             args.max_mols,
                             cache_path=os.path.join(ROOT, "data", "cache", "carbon_val.pt"))
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = SymMCFlow(ModelConfig(**ck["mcfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    n = min(args.eval_n, len(val_ds))
    batch = move_batch(collate([val_ds[i] for i in range(n)]), device)
    z1 = batch_to_state(batch)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    draws = []
    with torch.no_grad():
        for _ in range(max(args.match_k, 1)):
            z0 = sample_prior(z1, vol_per_atom=args.vol_per_atom,
                              centroid_prior_std=args.centroid_prior_std)
            draws.append(rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps))
    gens = [_to_structures(d) for d in draws]
    ref = _to_structures(z1)

    print(f"\n=== matcher reconciliation: {args.ckpt}  eval-n {n}  seed {args.seed} ===")
    f1, r1, rm1, tot = both_rates([gens[0]], ref, 1)
    print(f"  match@1   fit {100*f1:.1f}%   get_rms_dist {100*r1:.1f}%   ({tot} structs)")
    if args.match_k > 1:
        fk, rk, rmk, _ = both_rates(gens, ref, args.match_k)
        print(f"  match@{args.match_k}  fit {100*fk:.1f}%   get_rms_dist {100*rk:.1f}%   "
              f"RMSE(get_rms_dist) {rmk:.4f}")


if __name__ == "__main__":
    main()
