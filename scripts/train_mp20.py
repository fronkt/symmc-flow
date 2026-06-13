"""Train + evaluate SymMC-Flow on the CDVAE MP-20 benchmark (multi-element CSP).

Same flow objective and match@k eval as carbon-24, but MP-20 is multi-element so the
periodic-table embedding / EGNN do real work and the StructureMatcher eval uses the real
per-atom species (carbon-24 assumed pure C). Atoms are single-atom blocks (lambda_orient=0).

    python scripts/train_mp20.py --steps 30000 --batch 256 --d-model 256 \
        --attn-layers 8 --eval-n 256 --match-k 20
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.mp20 import MP20Dataset
from symmc_flow.train import train, resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample
from symmc_flow.model import SymMCFlow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")


def min_image_nn_distances(L, frac, mask):
    """Mean nearest-neighbour distance per structure (minimum-image), Angstrom."""
    out = []
    for b in range(L.shape[0]):
        idx = mask[b].nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        f = frac[b, idx]
        d = f.unsqueeze(1) - f.unsqueeze(0)
        d = d - d.round()
        cart = d @ L[b]
        dist = cart.norm(dim=-1) + torch.eye(len(idx), device=L.device) * 1e3
        out.append(dist.min(dim=1).values.mean().item())
    return out


def _to_structures(state, Z, mask):
    """CrystalState (+ per-atom Z) -> list of pymatgen Structures with real species."""
    from pymatgen.core import Structure, Lattice
    from pymatgen.core.periodic_table import Element
    out = []
    for b in range(state.lattice.shape[0]):
        idx = mask[b].nonzero(as_tuple=True)[0]
        try:
            species = [Element.from_Z(int(z)) for z in Z[b, idx]]
            st = Structure(Lattice(state.lattice[b].cpu().numpy()), species,
                           state.centroid[b, idx].cpu().numpy())
        except Exception:
            st = None
        out.append(st)
    return out


def match_rate_topk(gen_states, ref_state, Z, mask):
    """Best-of-k CSP match rate + RMSE (the two standard CSP metrics; DiffCSP reports
    both). For each reference the best (lowest-RMSD) matching candidate among the k draws
    counts as the hit; RMSE is the mean of those StructureMatcher RMS distances (length-
    scale normalized) over matched references. Returns (rate, total, mean_rmsd)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    ref = _to_structures(ref_state, Z, mask)
    gens = [_to_structures(g, Z, mask) for g in gen_states]
    hits, total, rmsds = 0, 0, []
    for i, r in enumerate(ref):
        if r is None:
            continue
        total += 1
        best = None
        for draw in gens:
            g = draw[i]
            try:
                rms = sm.get_rms_dist(g, r) if g is not None else None
            except Exception:
                rms = None
            if rms is not None and (best is None or rms[0] < best):
                best = rms[0]
        if best is not None:
            hits += 1
            rmsds.append(best)
    mean_rmsd = sum(rmsds) / len(rmsds) if rmsds else float("nan")
    return hits / max(total, 1), total, mean_rmsd


def sample_and_eval(model, val_ds, device, sampler_steps, vol_per_atom,
                    centroid_prior_std, n_eval=256, match_k=1):
    model.eval()
    n_sample = min(n_eval, len(val_ds))
    batch = move_batch(collate([val_ds[i] for i in range(n_sample)]), device)
    z1 = batch_to_state(batch)
    Z = batch["Z"][..., 0]                              # (B, Mmax)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    draws = []
    with torch.no_grad():
        for _ in range(max(match_k, 1)):
            z0 = sample_prior(z1, vol_per_atom=vol_per_atom,
                              centroid_prior_std=centroid_prior_std)
            draws.append(rk4_sample(model, mol_emb, z0, batch["sg"], steps=sampler_steps))
    out = draws[0]

    gen_nn = torch.tensor(min_image_nn_distances(out.lattice, out.centroid, out.mask))
    ref_nn = torch.tensor(min_image_nn_distances(z1.lattice, z1.centroid, z1.mask))
    n = out.mask.sum(-1).clamp_min(1).float()
    vol = torch.linalg.det(out.lattice).abs()
    ref_vol = torch.linalg.det(z1.lattice).abs()
    print("\n=== MP-20 generated-structure sanity ===")
    print(f"  gen NN dist  mean {gen_nn.mean():.3f}  median {gen_nn.median():.3f}  "
          f"(ref mean {ref_nn.mean():.3f}  median {ref_nn.median():.3f}) A")
    print(f"  overlaps <0.9 A: {100*(gen_nn<0.9).float().mean():.0f}%")
    print(f"  vol/atom  gen {(vol/n).mean():.2f}  ref {(ref_vol/n).mean():.2f} A^3   "
          f"det>0 {(torch.linalg.det(out.lattice)>0).float().mean():.2f}")
    mr, total, rmsd = match_rate_topk(draws if match_k > 1 else [out], z1, Z, out.mask)
    tag = f"@{match_k} (best-of-{match_k})" if match_k > 1 else ""
    print(f"  StructureMatcher match rate{tag}: {100*mr:.1f}%  ({total} structures)  "
          f"RMSE {rmsd:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--max-atoms", type=int, default=20)
    ap.add_argument("--centroid-prior-std", type=float, default=0.30,
                    help="wrapped-normal centroid prior std (negative -> uniform)")
    ap.add_argument("--vol-per-atom", type=float, default=20.0,
                    help="mean cell volume per atom (A^3) of the lattice prior")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "mp20.pt"))
    ap.add_argument("--tag", default="mp20")
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--match-k", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--attn-layers", type=int, default=8)
    ap.add_argument("--egnn-hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed prior draws for reproducible match@k (RNG over k draws "
                         "otherwise jitters the rate)")
    args = ap.parse_args()
    if args.centroid_prior_std is not None and args.centroid_prior_std < 0:
        args.centroid_prior_std = None
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(CACHE, exist_ok=True)
    print("loading MP-20 (parse + cache) ...")
    t0 = time.time()
    train_ds = MP20Dataset(os.path.join(RAW, "mp20_train.csv"), args.max_atoms,
                           cache_path=os.path.join(CACHE, "mp20_train.pt"))
    val_ds = MP20Dataset(os.path.join(RAW, "mp20_val.csv"), args.max_atoms,
                         cache_path=os.path.join(CACHE, "mp20_val.pt"))
    print(f"  train {len(train_ds)}  val {len(val_ds)}  ({time.time()-t0:.1f}s)")

    device = resolve_device("auto")
    print(f"device: {device}  | centroid_prior_std={args.centroid_prior_std} "
          f"vol_per_atom={args.vol_per_atom}")

    if args.eval_only:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = SymMCFlow(ModelConfig(**ck["mcfg"])).to(device)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.ckpt} (eval-only)")
        sample_and_eval(model, val_ds, device, args.sampler_steps, args.vol_per_atom,
                        args.centroid_prior_std, args.eval_n, args.match_k)
        return

    mcfg = ModelConfig(d_model=args.d_model, n_heads=8, n_attn_layers=args.attn_layers,
                       egnn_hidden=args.egnn_hidden, egnn_layers=2, atom_embed_dim=64,
                       lambda_lattice=1.0, lambda_centroid=1.0, lambda_orient=0.0)
    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch, steps=args.steps,
                       log_every=200, sampler_steps=args.sampler_steps, device="auto",
                       use_ot_coupling=True, prior_vol_per_atom=args.vol_per_atom,
                       centroid_prior_std=args.centroid_prior_std)

    t0 = time.time()
    model, hist = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    print(f"trained {args.steps} steps in {time.time()-t0:.1f}s")
    first, last = sum(hist[:20]) / 20, sum(hist[-20:]) / 20
    print(f"loss  first-20 {first:.4f}  ->  last-20 {last:.4f}  ({100*(1-last/first):.1f}% drop)")

    ckdir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    out_ckpt = os.path.join(ckdir, f"{args.tag}.pt")
    torch.save({"model": model.state_dict(), "mcfg": mcfg.__dict__}, out_ckpt)
    print(f"saved checkpoint -> {out_ckpt}")

    sample_and_eval(model, val_ds, device, args.sampler_steps, args.vol_per_atom,
                    args.centroid_prior_std, args.eval_n, args.match_k)


if __name__ == "__main__":
    main()
