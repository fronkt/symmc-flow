"""Demo: train briefly, then sample crystals with the 50-step RK4 ODE solver.

    python scripts/sample_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.train import train, resolve_device, move_batch
from symmc_flow.data import SyntheticCrystalDataset, collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample

if __name__ == "__main__":
    mcfg = ModelConfig(d_model=96, egnn_hidden=96, n_attn_layers=3, egnn_layers=3)
    tcfg = TrainConfig(steps=200, batch_size=16)
    model, _ = train(mcfg, tcfg, verbose=False)
    model.eval()

    device = resolve_device(tcfg.device)
    ds = SyntheticCrystalDataset(8, tcfg.max_mols, tcfg.max_atoms_per_mol, seed=1)
    batch = move_batch(collate([ds[i] for i in range(8)]), device)
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])

    out = rk4_sample(model, mol_emb, z0, batch["sg"], steps=50)
    dets = torch.linalg.det(out.orient)
    print("sampled lattice[0]:\n", out.lattice[0])
    print("centroids[0] in [0,1)? ",
          bool((out.centroid[0] >= 0).all() and (out.centroid[0] < 1.0001).all()))
    print("orientation dets (should be ~1):", dets[0].tolist())
    cell_vol = torch.linalg.det(out.lattice).abs()
    print("cell volumes:", cell_vol.tolist())
