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
    """CSP match rate: fraction of generated structures that StructureMatcher
    (CDVAE tolerances) matches to their ground-truth counterpart."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    gen, ref = _to_structures(gen_state), _to_structures(ref_state)
    hits, total = 0, 0
    for g, r in zip(gen, ref):
        if r is None:
            continue
        total += 1
        try:
            hits += bool(g is not None and sm.fit(g, r))
        except Exception:
            pass
    return hits / max(total, 1), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--max-mols", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    print("loading carbon-24 (parse + cache) ...")
    t0 = time.time()
    train_ds = Carbon24Dataset(os.path.join(RAW, "carbon_train.csv"), args.max_mols,
                               cache_path=os.path.join(CACHE, "carbon_train.pt"))
    val_ds = Carbon24Dataset(os.path.join(RAW, "carbon_val.csv"), args.max_mols,
                             cache_path=os.path.join(CACHE, "carbon_val.pt"))
    print(f"  train {len(train_ds)}  val {len(val_ds)}  ({time.time()-t0:.1f}s)")

    mcfg = ModelConfig(d_model=192, n_heads=8, n_attn_layers=6,
                       egnn_hidden=96, egnn_layers=2, atom_embed_dim=64,
                       lambda_lattice=1.0, lambda_centroid=1.0, lambda_orient=0.0)
    tcfg = TrainConfig(lr=args.lr, batch_size=args.batch, steps=args.steps,
                       log_every=200, sampler_steps=args.sampler_steps, device="auto",
                       use_ot_coupling=True, prior_vol_per_atom=9.0)  # graphite ~8.8 A^3/atom

    device = resolve_device(tcfg.device)
    print(f"device: {device}")
    t0 = time.time()
    model, hist = train(mcfg, tcfg, verbose=True, train_dataset=train_ds, val_dataset=val_ds)
    print(f"trained {args.steps} steps in {time.time()-t0:.1f}s")
    first = sum(hist[:20]) / 20
    last = sum(hist[-20:]) / 20
    print(f"loss  first-20 {first:.4f}  ->  last-20 {last:.4f}  ({100*(1-last/first):.1f}% drop)")

    # save checkpoint
    ckpt = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    torch.save({"model": model.state_dict(), "mcfg": mcfg.__dict__},
               os.path.join(ckpt, "carbon24.pt"))
    print(f"saved checkpoint -> {os.path.join(ckpt, 'carbon24.pt')}")

    # sample & sanity-check generated structures
    model.eval()
    n_sample = min(64, len(val_ds))
    batch = move_batch(collate([val_ds[i] for i in range(n_sample)]), device)
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1, vol_per_atom=tcfg.prior_vol_per_atom)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    out = rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps)

    gen_nn = min_image_nn_distances(out.lattice, out.centroid, out.mask)
    ref_nn = min_image_nn_distances(z1.lattice, z1.centroid, z1.mask)
    n = out.mask.sum(-1).clamp_min(1).float()
    vol = torch.linalg.det(out.lattice).abs()
    ref_vol = torch.linalg.det(z1.lattice).abs()
    g = torch.tensor(gen_nn)
    print("\n=== generated-structure sanity (carbon-carbon nearest neighbour) ===")
    print(f"  generated mean NN dist: {g.mean():.3f} A   (ref: {torch.tensor(ref_nn).mean():.3f} A)")
    print(f"  generated in [1.2,1.8] A: {100*((g>1.2)&(g<1.8)).float().mean():.1f}%  "
          f"(graphite 1.42 / diamond 1.54)")
    print(f"  vol/atom  gen {(vol/n).mean():.2f} A^3  ref {(ref_vol/n).mean():.2f} A^3   "
          f"det>0 frac {(torch.linalg.det(out.lattice)>0).float().mean():.2f}")
    mr, total = match_rate(out, z1)
    print(f"  StructureMatcher match rate: {100*mr:.1f}%  ({total} structures, CDVAE tolerances)")


if __name__ == "__main__":
    main()
