"""DiffCSP-style diffusion baseline on carbon-24 — head-to-head vs the flow model.

Same data, same network (SymMCFlow), same match@k eval as scripts/train_carbon24.py;
only the objective changes (DDPM lattice + wrapped-normal score fractional). Lets us
attribute any difference to flow-matching vs diffusion rather than architecture.

    python scripts/train_carbon24_diffusion.py --steps 15000 --batch 128 \
        --d-model 256 --attn-layers 8 --tag carbon24_diff --eval-n 256 --match-k 20
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling scripts

import torch
from torch.utils.data import DataLoader
from symmc_flow.config import ModelConfig
from symmc_flow.carbon24 import Carbon24Dataset
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.model import SymMCFlow
from symmc_flow.diffusion import DiffusionProcess, diffusion_loss, diffusion_sample
# reuse the exact eval helpers from the flow script so the metrics are identical
from train_carbon24 import (min_image_nn_distances, match_rate, match_rate_topk)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")


def sample_and_eval(model, val_ds, device, proc, sampler_steps, n_eval=64, match_k=1,
                    corrector_steps=0, snr=0.16):
    model.eval()
    n_sample = min(n_eval, len(val_ds))
    batch = move_batch(collate([val_ds[i] for i in range(n_sample)]), device)
    z1 = batch_to_state(batch)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    draws = []
    with torch.no_grad():
        for _ in range(max(match_k, 1)):
            draws.append(diffusion_sample(model, mol_emb, z1, batch["sg"], proc,
                                          steps=sampler_steps,
                                          corrector_steps=corrector_steps, snr=snr))
    out = draws[0]

    gen_nn = torch.tensor(min_image_nn_distances(out.lattice, out.centroid, out.mask))
    ref_nn = torch.tensor(min_image_nn_distances(z1.lattice, z1.centroid, z1.mask))
    n = out.mask.sum(-1).clamp_min(1).float()
    vol = torch.linalg.det(out.lattice).abs()
    ref_vol = torch.linalg.det(z1.lattice).abs()
    print("\n=== diffusion generated-structure sanity (C-C nearest neighbour) ===")
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
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-mols", type=int, default=24)
    ap.add_argument("--diffusion-T", type=int, default=1000)
    ap.add_argument("--sampler-steps", type=int, default=250)
    ap.add_argument("--corrector-steps", type=int, default=0,
                    help="Langevin corrector steps per level (PC sampler; 0 = predictor-only)")
    ap.add_argument("--snr", type=float, default=0.16, help="corrector signal-to-noise ratio")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "carbon24_diff.pt"))
    ap.add_argument("--tag", default="carbon24_diff")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--match-k", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--attn-layers", type=int, default=6)
    ap.add_argument("--egnn-hidden", type=int, default=96)
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    print("loading carbon-24 ...")
    t0 = time.time()
    train_ds = Carbon24Dataset(os.path.join(RAW, "carbon_train.csv"), args.max_mols,
                               cache_path=os.path.join(CACHE, "carbon_train.pt"))
    val_ds = Carbon24Dataset(os.path.join(RAW, "carbon_val.csv"), args.max_mols,
                             cache_path=os.path.join(CACHE, "carbon_val.pt"))
    print(f"  train {len(train_ds)}  val {len(val_ds)}  ({time.time()-t0:.1f}s)")

    device = resolve_device("auto")
    proc = DiffusionProcess(T=args.diffusion_T)
    print(f"device: {device}  | diffusion T={args.diffusion_T} sampler_steps={args.sampler_steps}")

    if args.eval_only:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = SymMCFlow(ModelConfig(**ck["mcfg"])).to(device)
        model.load_state_dict(ck["model"])
        print(f"loaded {args.ckpt} (eval-only)")
        sample_and_eval(model, val_ds, device, proc, args.sampler_steps,
                        args.eval_n, args.match_k, args.corrector_steps, args.snr)
        return

    mcfg = ModelConfig(d_model=args.d_model, n_heads=8, n_attn_layers=args.attn_layers,
                       egnn_hidden=args.egnn_hidden, egnn_layers=2, atom_embed_dim=64,
                       lambda_lattice=1.0, lambda_centroid=1.0, lambda_orient=0.0)
    model = SymMCFlow(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    t0, step, hist = time.time(), 0, []
    model.train()
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            batch = move_batch(batch, device)
            loss, parts = diffusion_loss(model, batch, proc, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            hist.append(parts["total"].item())
            if step % 200 == 0:
                print(f"step {step:5d}  loss {parts['total']:.4f}  "
                      f"(lat {parts['lattice']:.3f}  frac {parts['frac']:.3f})")
            step += 1
    print(f"trained {args.steps} steps in {time.time()-t0:.1f}s")
    first, last = sum(hist[:20]) / 20, sum(hist[-20:]) / 20
    print(f"loss  first-20 {first:.4f}  ->  last-20 {last:.4f}  ({100*(1-last/first):.1f}% drop)")

    ckdir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    out_ckpt = os.path.join(ckdir, f"{args.tag}.pt")
    torch.save({"model": model.state_dict(), "mcfg": mcfg.__dict__}, out_ckpt)
    print(f"saved checkpoint -> {out_ckpt}")

    sample_and_eval(model, val_ds, device, proc, args.sampler_steps,
                    args.eval_n, args.match_k, args.corrector_steps, args.snr)


if __name__ == "__main__":
    main()
