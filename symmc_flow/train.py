"""CUDA-aware training loop for SymMC-Flow on the synthetic harness."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .data import SyntheticCrystalDataset, collate, batch_to_state
from .flow import sample_prior, interpolate, cfm_loss
from .model import SymMCFlow


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train(model_cfg: ModelConfig | None = None, train_cfg: TrainConfig | None = None,
          verbose: bool = True):
    mcfg = model_cfg or ModelConfig()
    tcfg = train_cfg or TrainConfig()
    torch.manual_seed(tcfg.seed)
    device = resolve_device(tcfg.device)

    ds = SyntheticCrystalDataset(tcfg.n_train, tcfg.max_mols,
                                 tcfg.max_atoms_per_mol, tcfg.seed)
    dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate)

    model = SymMCFlow(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)

    history, step = [], 0
    model.train()
    while step < tcfg.steps:
        for batch in dl:
            if step >= tcfg.steps:
                break
            batch = move_batch(batch, device)
            z1 = batch_to_state(batch)
            z0 = sample_prior(z1)
            t = torch.rand(z1.lattice.shape[0], device=device)
            z_t, targets = interpolate(z0, z1, t)

            mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
            pred = model(mol_emb, z_t.lattice, z_t.centroid, z_t.orient,
                         t, batch["sg"], z1.mask)
            loss, parts = cfm_loss(pred, targets, z1.mask, weights)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()

            history.append(parts["total"].item())
            if verbose and step % tcfg.log_every == 0:
                print(f"step {step:4d}  loss {parts['total']:.4f}  "
                      f"(L {parts['lattice']:.3f}  x {parts['centroid']:.3f}  "
                      f"R {parts['orient']:.3f})")
            step += 1

    return model, history


if __name__ == "__main__":
    train()
