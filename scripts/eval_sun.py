"""SUN evaluation (Stable / Unique / Novel) for a trained SymMC-Flow checkpoint.

Generates one structure per val composition, then reports:
  validity (CDVAE structural validity), unique, novel (vs train), and -- with --stability --
  CHGNet E_above_hull stability and the combined SUN rate.

    MP_API_KEY=... python scripts/eval_sun.py --ckpt checkpoints/mp20.pt \
        --eval-n 256 --seed 0 --stability

Without --stability only U/N/validity are computed (no CHGNet / MP dependency).
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from symmc_flow.config import ModelConfig
from symmc_flow.model import SymMCFlow
from symmc_flow.mp20 import MP20Dataset
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample
from symmc_flow.sun import sun_summary
from scripts.train_mp20 import _to_structures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")


def _reduced_formula(Z_ints):
    from collections import Counter
    from pymatgen.core import Composition
    from pymatgen.core.periodic_table import Element
    c = Counter(Element.from_Z(int(z)).symbol for z in Z_ints)
    return Composition(dict(c)).reduced_formula


def build_train_structures_for(formulas, train_ds, device, batch=256):
    """Build pymatgen Structures only for train items whose reduced formula is in
    `formulas` (novelty only ever compares within a formula, so the rest are irrelevant)."""
    keep = []
    for i in range(len(train_ds)):
        it = train_ds[i]
        Z = it["Z"][..., 0][it["mol_mask"].bool()] if it["Z"].dim() == 2 else it["Z"][it["mol_mask"].bool()]
        if _reduced_formula(Z.tolist()) in formulas:
            keep.append(i)
    out = []
    for s in range(0, len(keep), batch):
        idx = keep[s:s + batch]
        b = move_batch(collate([train_ds[i] for i in idx]), device)
        st = batch_to_state(b)
        out += _to_structures(st, b["Z"][..., 0], st.mask)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "mp20.pt"))
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--centroid-prior-std", type=float, default=0.30)
    ap.add_argument("--vol-per-atom", type=float, default=20.0)
    ap.add_argument("--max-atoms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability", action="store_true", help="run CHGNet + MP E_above_hull")
    ap.add_argument("--ehull-threshold", type=float, default=0.1)
    ap.add_argument("--relax-steps", type=int, default=200)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = resolve_device("auto")
    val_ds = MP20Dataset(os.path.join(RAW, "mp20_val.csv"), args.max_atoms,
                         cache_path=os.path.join(CACHE, "mp20_val.pt"))
    train_ds = MP20Dataset(os.path.join(RAW, "mp20_train.csv"), args.max_atoms,
                           cache_path=os.path.join(CACHE, "mp20_train.pt"))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = SymMCFlow(ModelConfig(**ck["mcfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt}  device {device}")

    n = min(args.eval_n, len(val_ds))
    batch = move_batch(collate([val_ds[i] for i in range(n)]), device)
    z1 = batch_to_state(batch)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    with torch.no_grad():
        z0 = sample_prior(z1, vol_per_atom=args.vol_per_atom,
                          centroid_prior_std=args.centroid_prior_std)
        out = rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps)
    gen = _to_structures(out, batch["Z"][..., 0], out.mask)
    print(f"generated {sum(g is not None for g in gen)}/{len(gen)} structures")

    t0 = time.time()
    formulas = {g.composition.reduced_formula for g in gen if g is not None}
    train_structs = build_train_structures_for(formulas, train_ds, device)
    print(f"built {len(train_structs)} train refs for {len(formulas)} formulas "
          f"({time.time()-t0:.1f}s)")

    stable = None
    if args.stability:
        key = os.environ.get("MP_API_KEY")
        if not key:
            raise SystemExit("--stability needs MP_API_KEY in the environment")
        from symmc_flow.stability import stable_mask
        t1 = time.time()
        stable, relaxed, ehull = stable_mask(gen, key, threshold=args.ehull_threshold,
                                             steps=args.relax_steps)
        vals = [e for e in ehull if e is not None]
        print(f"stability: scored {len(vals)}/{len(gen)} (CHGNet relax + MP hull, "
              f"{time.time()-t1:.0f}s);  median E_hull "
              f"{sorted(vals)[len(vals)//2]:.3f} eV/atom" if vals else "stability: none scored")

    res = sun_summary(gen, train_structs, stable_mask=stable)
    print("\n=== SUN summary (mp20) ===")
    for k, v in res.items():
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
