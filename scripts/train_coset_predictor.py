"""Train the de-novo CosetPredictor (C4) and measure how much of the coset is recoverable from
PACKING ALONE -- the closable fraction of the inference-limited ceiling.

    python scripts/train_coset_predictor.py --cache data/csd_mol/ds.pt --steps 2000

Trains a packing-only classifier (conformer + centroid + lattice + space group; no orientation)
to predict the leak-free symmetry coset (`assign_symmetry_cosets`) of each non-reference copy,
on the true (clean) packing. Reports held-out top-1 accuracy against a majority-class baseline.
High accuracy => the template-free (de-novo) path can realize most of the coset gain; the saved
checkpoint feeds `eval_orient_matchrate.py --predictor-ckpt` to read the predicted-coset
reconstruction rate.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from symmc_flow.config import ModelConfig
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item,
                                    species_multiplicity, assign_symmetry_cosets)
from symmc_flow.coset_predictor import CosetPredictor
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate


def nonref_mask(batch):
    return (~batch["is_ref"]) & batch["mol_mask"]


@torch.no_grad()
def accuracy(model, dl, device):
    model.eval()
    hit, tot = 0, 0
    for batch in dl:
        batch = move_batch(batch, device)
        emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        logits = model(emb, batch["lattice"], batch["centroid"], batch["sg"], batch["mol_mask"])
        m = nonref_mask(batch)
        pred = logits.argmax(-1)
        hit += int(((pred == batch["coset"]) & m).sum())
        tot += int(m.sum())
    return hit / max(tot, 1)


def majority_baseline(items, val_idx):
    """Most-frequent non-ref coset overall -> the feature-free accuracy floor on val."""
    from collections import Counter
    cnt = Counter()
    train_set = set(range(len(items))) - set(val_idx)
    for i in train_set:
        it = items[i]
        m = (~it["is_ref"]) & it["mol_mask"]
        cnt.update(int(c) for c in it["coset"][m].tolist())
    top = cnt.most_common(1)[0][0] if cnt else 0
    hit, tot = 0, 0
    for i in val_idx:
        it = items[i]
        m = (~it["is_ref"]) & it["mol_mask"]
        hit += int((it["coset"][m] == top).sum()); tot += int(m.sum())
    return hit / max(tot, 1), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/coset_predictor.pt")
    args = ap.parse_args()
    if not os.path.exists(args.cache):
        sys.exit(f"no cache at {args.cache}")

    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    items = [it for it in rel if species_multiplicity(it) >= 2]
    items, n_cosets = assign_symmetry_cosets(items)
    full.items = items
    n = len(full)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(4, int(round(args.val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    device = resolve_device("auto")
    print(f"corpus {len(rel)} -> {n} multi-copy; cosets {n_cosets}; "
          f"split {len(train_idx)} train / {len(val_idx)} val; device {device}")

    mcfg = ModelConfig(n_cosets=n_cosets)
    model = CosetPredictor(mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    dl_tr = DataLoader(Subset(full, train_idx), batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate)
    dl_val = DataLoader(Subset(full, val_idx), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate)
    ce = torch.nn.CrossEntropyLoss()

    base_acc, top = majority_baseline(items, val_idx)
    print(f"majority-class baseline (coset {top}): {100*base_acc:.1f}% val accuracy\n")

    step = 0
    while step < args.steps:
        model.train()
        for batch in dl_tr:
            batch = move_batch(batch, device)
            emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
            logits = model(emb, batch["lattice"], batch["centroid"], batch["sg"], batch["mol_mask"])
            m = nonref_mask(batch)
            if int(m.sum()) == 0:
                continue
            loss = ce(logits[m], batch["coset"][m])
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 100 == 0:
                print(f"  step {step:5d}  loss {float(loss):.4f}", flush=True)
            if step >= args.steps:
                break

    acc = accuracy(model, dl_val, device)
    print(f"\n==== coset predictor (packing-only, clean packing) ====")
    print(f"  held-out top-1 accuracy: {100*acc:.1f}%   (majority baseline {100*base_acc:.1f}%)")
    print(f"  => {100*acc:.1f}% of the generating operation is recoverable from packing without a template")

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": mcfg.__dict__,
                "n_cosets": n_cosets, "val_idx": val_idx, "acc": acc,
                "baseline_acc": base_acc}, args.ckpt)
    print(f"checkpoint -> {args.ckpt}")


if __name__ == "__main__":
    main()
