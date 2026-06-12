"""Synthetic molecular-crystal dataset + hooks for real datasets.

Each sample is a small unit cell holding a few rigid molecular blocks. Atom
identities, intramolecular conformer geometry, lattice, centroids and
orientations are drawn from simple structured distributions so the flow has a
non-trivial but learnable target. The real-data loaders (MP-20, carbon, WBM)
are left as documented stubs for the GPU benchmark phase.
"""
from __future__ import annotations
import math
import torch
from torch.utils.data import Dataset

from .flow import CrystalState
from . import manifolds as M

# A handful of tiny "molecules": list of atomic numbers + a canonical conformer.
_MOTIFS = [
    # (atomic numbers, local coords)   — coords are arbitrary rigid conformers
    ([6, 1, 1, 1, 1], None),            # methane-like
    ([8, 1, 1], None),                  # water-like
    ([6, 6, 1, 1, 1, 1], None),         # ethylene-like
    ([7, 1, 1, 1], None),               # ammonia-like
    ([6, 8, 8], None),                  # CO2-like
]
_SG_CHOICES = [1, 2, 3, 6, 16, 25]


def _conformer(z_list, gen):
    n = len(z_list)
    coords = torch.randn(n, 3, generator=gen) * 0.6
    coords = coords - coords.mean(0, keepdim=True)  # centroid at origin
    return coords


def _target_pose(z_list, gen):
    """Deterministic target (centroid, orientation) from composition + tiny noise.
    Ties the generative target to the molecule so the conditional flow is learnable
    (pure-noise targets would make the orientation field unpredictable by design)."""
    s = float(sum(z_list))
    n = float(len(z_list))
    w = torch.tensor([math.sin(0.3 * s), math.cos(0.5 * n), 0.4 * ((s % 7) / 7 - 0.5)])
    w = w / (w.norm() + 1e-6) * (0.5 + (s % 5) / 5 * 1.8)   # angle in [0.5, 2.3]
    w = w + 0.03 * torch.randn(3, generator=gen)
    R = M.so3_exp(w)
    c = torch.tensor([(s % 11) / 11, (n % 5) / 5, ((s * 7) % 13) / 13])
    c = c + 0.02 * torch.randn(3, generator=gen)
    return (c - torch.floor(c)), R


class SyntheticCrystalDataset(Dataset):
    def __init__(self, n: int = 512, max_mols: int = 4, max_atoms: int = 12, seed: int = 0):
        self.max_mols = max_mols
        self.max_atoms = max_atoms
        g = torch.Generator().manual_seed(seed)
        self.items = [self._make(g) for _ in range(n)]

    def _make(self, gen):
        n_mol = int(torch.randint(1, self.max_mols + 1, (1,), generator=gen))
        sg = _SG_CHOICES[int(torch.randint(0, len(_SG_CHOICES), (1,), generator=gen))]

        Z = torch.zeros(self.max_mols, self.max_atoms, dtype=torch.long)
        local = torch.zeros(self.max_mols, self.max_atoms, 3)
        atom_mask = torch.zeros(self.max_mols, self.max_atoms, dtype=torch.bool)
        centroid = torch.rand(self.max_mols, 3, generator=gen)
        orient = M.random_so3((self.max_mols,))
        for m in range(n_mol):
            motif = _MOTIFS[int(torch.randint(0, len(_MOTIFS), (1,), generator=gen))]
            z_list = motif[0][: self.max_atoms]
            na = len(z_list)
            Z[m, :na] = torch.tensor(z_list)
            local[m, :na] = _conformer(z_list, gen)
            atom_mask[m, :na] = True
            centroid[m], orient[m] = _target_pose(z_list, gen)  # learnable target pose

        mol_mask = torch.zeros(self.max_mols, dtype=torch.bool)
        mol_mask[:n_mol] = True

        # target lattice: near-cubic with structured jitter
        a = 4.0 + 2.0 * torch.rand(1, generator=gen).item()
        lattice = torch.eye(3) * a + 0.3 * torch.randn(3, 3, generator=gen)

        return {
            "Z": Z, "local": local, "atom_mask": atom_mask, "mol_mask": mol_mask,
            "lattice": lattice, "centroid": centroid, "orient": orient,
            "sg": torch.tensor(sg, dtype=torch.long),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {**self.items[i], "idx": torch.tensor(i, dtype=torch.long)}


def collate(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], 0) for k in keys}
    return out


def batch_to_state(batch) -> CrystalState:
    return CrystalState(
        lattice=batch["lattice"], centroid=batch["centroid"],
        orient=batch["orient"], mask=batch["mol_mask"],
    )


# --- Real-dataset hooks (GPU benchmark phase) --------------------------------
def load_mp20(*args, **kwargs):
    raise NotImplementedError(
        "GPU phase: build with pymatgen/mp-api. Parse CIFs -> rigid molecular "
        "blocks via connected-component detection, PCA orientations, fractional "
        "centroids, and space-group number from SpacegroupAnalyzer."
    )


def load_carbon(*args, **kwargs):
    raise NotImplementedError("GPU phase: carbon-24 loader stub.")
