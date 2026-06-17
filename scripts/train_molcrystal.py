"""Smoke train: SymMC-Flow on REAL rigid-body molecular crystals with lambda_orient>0.

This is the first time the SO(3) orientation head trains on multi-atom rigid blocks (the
real benchmarks mp20/carbon24 are single-atom, lambda_orient=0). It is a WIRING + GRADIENT
sanity check, not the CSD benchmark: we synthesize molecular crystals from known conformers
whose planted orientation is a smooth function of the molecule's centroid (the honest
analogue of "packing determines orientation"), feed them through `MolCrystalDataset`, and
assert the *orientation* loss component descends.

    python scripts/train_molcrystal.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pymatgen.core import Structure, Lattice

from symmc_flow import manifolds as M
from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.molcrystal import MolCrystalDataset
from symmc_flow.model import SymMCFlow
from symmc_flow.train import train, _step_loss, move_batch, resolve_device
from symmc_flow.data import collate, batch_to_state  # noqa: F401
from torch.utils.data import DataLoader

# A few rigid species (centred conformers + atomic numbers).
_SPECIES = {
    "cnoh": (np.array([[0., 0., 0.], [1.30, .1, 0.], [2.1, 1.05, 0.], [-.3, -1.05, .2]]),
             [6, 7, 8, 1]),
    "water": (np.array([[0., 0., 0.], [.757, .586, 0.], [-.757, .586, 0.]]), [8, 1, 1]),
    "co2": (np.array([[0., 0., 0.], [1.16, 0., 0.], [-1.16, 0., 0.]]), [6, 8, 8]),
}
_GRID = np.array([[.25, .25, .25], [.75, .25, .25], [.25, .75, .25], [.75, .75, .75]])


def _orient_from_centroid(c):
    """Smooth, deterministic SO(3) target from fractional centroid -> learnable signal."""
    w = np.array([np.sin(3 * c[0]), np.cos(2 * c[1]), 0.8 * (c[2] - 0.5)]) * 1.4
    return M.so3_exp(torch.tensor(w, dtype=torch.float64)).numpy()


def _make_corpus(n_structs=40, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(_SPECIES)
    out = []
    for s in range(n_structs):
        conf, Z = _SPECIES[keys[s % len(keys)]]
        conf = conf - conf.mean(0)
        a = 24.0 + rng.uniform(0, 6, size=3)                 # big cell: copies stay apart
        L = np.array(Lattice.from_parameters(a[0], a[1], a[2], 90, 90, 90).matrix)
        Linv = np.linalg.inv(L)
        n = int(rng.integers(2, 5))
        cents = _GRID[:n] + rng.normal(0, 0.02, size=(n, 3))  # jittered, well separated
        species, frac = [], []
        for c in cents:
            R = _orient_from_centroid(c)
            cart_c = c @ L
            for z, xyz in zip(Z, conf):
                f = (cart_c + R @ xyz) @ Linv
                species.append(int(z)); frac.append(f - np.floor(f))
        out.append(Structure(Lattice(L), species, frac))
    return out


@torch.no_grad()
def _mean_orient_loss(model, ds, device, weights, n_batches=8):
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    tot, k = 0.0, 0
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batch = move_batch(batch, device)
        _, parts = _step_loss(model, batch, weights, device)
        tot += float(parts["orient"]); k += 1
    return tot / max(k, 1)


if __name__ == "__main__":
    structs = _make_corpus()
    ds = MolCrystalDataset(structures=structs, max_mols=8, max_atoms=8)
    print(f"corpus: {len(structs)} structures -> {len(ds)} kept, {len(ds.skipped)} skipped")
    if ds.skipped:
        print("  skipped:", ds.skipped[:5])
    assert len(ds) > 20, "too few structures survived detection"

    mcfg = ModelConfig(d_model=96, egnn_hidden=96, n_attn_layers=3, egnn_layers=3,
                       lambda_orient=1.0)
    tcfg = TrainConfig(steps=300, batch_size=16, log_every=50, seed=0)
    device = resolve_device(tcfg.device)
    weights = (mcfg.lambda_lattice, mcfg.lambda_centroid, mcfg.lambda_orient)

    torch.manual_seed(0)
    pre = _mean_orient_loss(SymMCFlow(mcfg).to(device), ds, device, weights)
    model, _ = train(mcfg, tcfg, verbose=True, train_dataset=ds)
    post = _mean_orient_loss(model, ds, device, weights)

    drop = 100 * (1 - post / pre)
    print(f"\norient loss  untrained {pre:.4f}  ->  trained {post:.4f}  ({drop:.1f}% drop)")
    assert post < pre, "orientation flow did not learn (orient loss did not decrease)"
    print("OK: lambda_orient>0 SO(3) flow trains on real rigid-body molecular crystals.")
