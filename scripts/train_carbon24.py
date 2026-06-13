"""Train + sample SymMC-Flow on the real carbon-24 benchmark.

    python scripts/train_carbon24.py --steps 6000 --batch 128

Expects CDVAE carbon-24 CSVs at data/raw/carbon_{train,val,test}.csv
(downloaded in the GPU phase). Parsed structures are cached to data/cache/.
Carbon atoms are single-atom rigid blocks, so orientation is disabled
(lambda_orient=0); the model learns the lattice + fractional-coordinate flow.
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.carbon24 import Carbon24Dataset
from symmc_flow.train import train, resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.model import SymMCFlow
from symmc_flow.sampler import rk4_sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")


def min_image_nn_distances(L, frac, mask):
    """Mean nearest-neighbour C-C distance per structure (minimum-image), Angstrom."""
    out = []
    for b in range(L.shape[0]):
        idx = mask[b].nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        f = frac[b, idx]                                   # (n,3)
        d = f.unsqueeze(1) - f.unsqueeze(0)                # (n,n,3)
        d = d - d.round()                                  # minimum image
        cart = d @ L[b]                                    # (n,n,3)
        dist = cart.norm(dim=-1)                           # (n,n)
        dist = dist + torch.eye(len(idx), device=L.device) * 1e3
        out.append(dist.min(dim=1).values.mean().item())
    return out


def _to_structures(state):
    """CrystalState -> list of pymatgen Structures (carbon only)."""
    from pymatgen.core import Structure, Lattice
    out = []
    for b in range(state.lattice.shape[0]):
        idx = state.mask[b].nonzero(as_tuple=True)[0]
        try:
            st = Structure(Lattice(state.lattice[b].cpu().numpy()),
                           ["C"] * len(idx), state.centroid[b, idx].cpu().numpy())
        except Exception:
            st = None
        out.append(st)
    return out


def match_rate(gen_state, ref_state):
    """CSP match rate + RMSE: fraction of generated structures that StructureMatcher
    (CDVAE tolerances) matches to their ground truth, and the mean RMS distance over
    matched pairs (length-scale normalized). Returns (rate, total, mean_rmsd)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    gen, ref = _to_structures(gen_state), _to_structures(ref_state)
    hits, total, rmsds = 0, 0, []
    for g, r in zip(gen, ref):
        if r is None:
            continue
        total += 1
        try:
            rms = sm.get_rms_dist(g, r) if g is not None else None
        except Exception:
            rms = None
        if rms is not None:
            hits += 1
            rmsds.append(rms[0])
    mean_rmsd = sum(rmsds) / len(rmsds) if rmsds else float("nan")
    return hits / max(total, 1), total, mean_rmsd


def match_rate_topk(gen_states, ref_state):
    """Best-of-k CSP match rate + RMSE (the two standard CSP metrics; DiffCSP/CDVAE
    report both). `gen_states` is a list of k independent generations for the SAME
    references; a reference is matched if any candidate matches, and RMSE averages the
    best (lowest) matching RMS distance per reference. Returns (rate, total, mean_rmsd)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    ref = _to_structures(ref_state)
    gens = [_to_structures(g) for g in gen_states]            # k x (n_struct)
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
                    centroid_prior_std, churn, n_eval=64, match_k=1):
    """Sample from the prior and report NN-distance / volume / match-rate stats.
    With match_k>1, draws k independent generations per reference and reports the
    best-of-k (match@k) rate; sanity stats use the first draw."""
    model.eval()
    n_sample = min(n_eval, len(val_ds))
    batch = move_batch(collate([val_ds[i] for i in range(n_sample)]), device)
    z1 = batch_to_state(batch)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    draws = []
    with torch.no_grad():
        for _ in range(max(match_k, 1)):
            z0 = sample_prior(z1, vol_per_atom=vol_per_atom,
                              centroid_prior_std=centroid_prior_std)
            draws.append(rk4_sample(model, mol_emb, z0, batch["sg"],
                                    steps=sampler_steps, churn=churn))
    out = draws[0]

    gen_nn = torch.tensor(min_image_nn_distances(out.lattice, out.centroid, out.mask))
    ref_nn = torch.tensor(min_image_nn_distances(z1.lattice, z1.centroid, z1.mask))
    n = out.mask.sum(-1).clamp_min(1).float()
    vol = torch.linalg.det(out.lattice).abs()
    ref_vol = torch.linalg.det(z1.lattice).abs()
    print("\n=== generated-structure sanity (carbon-carbon nearest neighbour) ===")
    print(f"  gen NN dist  mean {gen_nn.mean():.3f}  median {gen_nn.median():.3f}  "
          f"(ref mean {ref_nn.mean():.3f}  median {ref_nn.median():.3f}) A")
    print(f"  gen in [1.2,1.8] A: {100*((gen_nn>1.2)&(gen_nn<1.8)).float().mean():.1f}%   "
          f"overlaps <0.9 A: {100*(gen_nn<0.9).float().mean():.0f}%")
    print(f"  vol/atom  gen {(vol/n).mean():.2f}  ref {(ref_vol/n).mean():.2f} A^3   "
          f"det>0 {(torch.linalg.det(out.lattice)>0).float().mean():.2f}")
    if match_k <= 1:
        mr, total, rmsd = match_rate(out, z1)
        print(f"  StructureMatcher match rate: {100*mr:.1f}%  ({total} structures)  "
              f"RMSE {rmsd:.4f}")
    else:
        mr, total, rmsd = match_rate_topk(draws, z1)
        print(f"  StructureMatcher match rate @{match_k} (best-of-{match_k}): "
              f"{100*mr:.1f}%  ({total} structures)  RMSE {rmsd:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--max-mols", type=int, default=24)
    # position-flow ablation levers
    ap.add_argument("--churn", type=float, default=0.0, help="Langevin sampler noise")
    ap.add_argument("--centroid-prior-std", type=float, default=0.30,
                    help="wrapped-normal centroid prior std (the dispersion fix; "
                         "pass a negative value to use the uniform prior instead)")
    ap.add_argument("--fixed-prior", action="store_true",
                    help="cache one prior per structure (sharper OT target)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, load --ckpt and only sample+eval")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "carbon24.pt"))
    ap.add_argument("--tag", default="carbon24", help="checkpoint name tag")
    ap.add_argument("--eval-n", type=int, default=64, help="# val structures to evaluate")
    ap.add_argument("--match-k", type=int, default=1,
                    help="best-of-k CSP match rate: draw k generations per reference "
                         "and count a hit if any matches (standard CSP eval; 1 = match@1)")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--attn-layers", type=int, default=6)
    ap.add_argument("--egnn-hidden", type=int, default=96)
    args = ap.parse_args()
    if args.centroid_prior_std is not None and args.centroid_prior_std < 0:
        args.centroid_prior_std = None  # negative -> uniform prior

    os.makedirs(CACHE, exist_ok=True)
    print("loading carbon-24 (parse + cache) ...")
    t0 = time.time()
    train_ds = Carbon24Dataset(os.path.join(RAW, "carbon_train.csv"), args.max_mols,
                               cache_path=os.path.join(CACHE, "carbon_train.pt"))
    val_ds = Carbon24Dataset(os.path.join(RAW, "carbon_val.csv"), args.max_mols,
                             cache_path=os.path.join(CACHE, "carbon_val.pt"))
    print(f"  train {len(train_ds)}  val {len(val_ds)}  ({time.time()-t0:.1f}s)")

    device = resolve_device("auto")
    print(f"device: {device}  | churn={args.churn} centroid_prior_std="
          f"{args.centroid_prior_std} fixed_prior={args.fixed_prior}")

    if args.eval_only:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = SymMCFlow(ModelConfig(**ck["mcfg"])).to(device)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.ckpt} (eval-only)")
        sample_and_eval(model, val_ds, device, args.sampler_steps, 9.0,
                        args.centroid_prior_std, args.churn, args.eval_n, args.match_k)
        return

    mcfg = ModelConfig(d_model=args.d_model, n_heads=8, n_attn_layers=args.attn_layers,
                       egnn_hidden=args.egnn_hidden, egnn_layers=2, atom_embed_dim=64,
                       lambda_lattice=1.0, lambda_centroid=1.0, lambda_orient=0.0)
    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch, steps=args.steps,
                       log_every=200, sampler_steps=args.sampler_steps, device="auto",
                       use_ot_coupling=True, prior_vol_per_atom=9.0,  # graphite ~8.8 A^3/atom
                       centroid_prior_std=args.centroid_prior_std,
                       fixed_prior=args.fixed_prior, sampler_churn=args.churn)

    t0 = time.time()
    model, hist = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    print(f"trained {args.steps} steps in {time.time()-t0:.1f}s")
    first, last = sum(hist[:20]) / 20, sum(hist[-20:]) / 20
    print(f"loss  first-20 {first:.4f}  ->  last-20 {last:.4f}  ({100*(1-last/first):.1f}% drop)")

    ckpt = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    out_ckpt = os.path.join(ckpt, f"{args.tag}.pt")
    torch.save({"model": model.state_dict(), "mcfg": mcfg.__dict__}, out_ckpt)
    print(f"saved checkpoint -> {out_ckpt}")

    sample_and_eval(model, val_ds, device, args.sampler_steps, tcfg.prior_vol_per_atom,
                    args.centroid_prior_std, args.churn, args.eval_n, args.match_k)


if __name__ == "__main__":
    main()
