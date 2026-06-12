"""Carbon-24 dataset loader (CDVAE benchmark: ~10k carbon allotropes, 1-24 atoms/cell).

Each carbon atom is treated as a single-atom rigid block (orientation is trivial for
a point, so train with `lambda_orient=0`). Lattices are rebuilt from their 6 cell
parameters into a canonical matrix to remove the global-rotation ambiguity of raw
matrices; space groups are recovered with pymatgen since the CDVAE CIFs are stored P1.

Yields the same batch dict as `SyntheticCrystalDataset` so the model/training code is
unchanged. Requires `pymatgen` + `pandas` (GPU benchmark phase deps).
"""
from __future__ import annotations
import os
import torch
from torch.utils.data import Dataset

_C = 6  # carbon atomic number


def _parse_rows(csv_path: str, max_mols: int, symprec: float):
    import pandas as pd
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    df = pd.read_csv(csv_path)
    rows = []
    for cif in df["cif"]:
        st = Structure.from_str(cif, fmt="cif")
        n = len(st)
        if n == 0 or n > max_mols:
            continue
        a, b, c, al, be, ga = st.lattice.parameters
        L = torch.tensor(Lattice.from_parameters(a, b, c, al, be, ga).matrix,
                         dtype=torch.float32)
        frac = torch.tensor(st.frac_coords, dtype=torch.float32)
        frac = frac - torch.floor(frac)  # wrap into [0,1)
        try:
            sg = SpacegroupAnalyzer(st, symprec=symprec).get_space_group_number()
        except Exception:
            sg = 1
        rows.append((n, L, frac, int(sg)))
    return rows


def _to_item(n, L, frac, sg, max_mols):
    Z = torch.zeros(max_mols, 1, dtype=torch.long)
    local = torch.zeros(max_mols, 1, 3)
    atom_mask = torch.zeros(max_mols, 1, dtype=torch.bool)
    centroid = torch.zeros(max_mols, 3)
    orient = torch.eye(3).expand(max_mols, 3, 3).clone()
    mol_mask = torch.zeros(max_mols, dtype=torch.bool)
    Z[:n, 0] = _C
    atom_mask[:n, 0] = True
    mol_mask[:n] = True
    centroid[:n] = frac
    return {"Z": Z, "local": local, "atom_mask": atom_mask, "mol_mask": mol_mask,
            "lattice": L, "centroid": centroid, "orient": orient,
            "sg": torch.tensor(sg, dtype=torch.long)}


class Carbon24Dataset(Dataset):
    def __init__(self, csv_path: str, max_mols: int = 24, symprec: float = 0.1,
                 cache_path: str | None = None):
        self.max_mols = max_mols
        if cache_path and os.path.exists(cache_path):
            self.items = torch.load(cache_path, weights_only=False)
        else:
            rows = _parse_rows(csv_path, max_mols, symprec)
            self.items = [_to_item(*r, max_mols) for r in rows]
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                torch.save(self.items, cache_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {**self.items[i], "idx": torch.tensor(i, dtype=torch.long)}
