"""MP-20 dataset loader (CDVAE benchmark: ~45k Materials Project structures, <=20
atoms/cell, 89 elements, diverse compositions).

Like carbon-24 each atom is a single-atom rigid block (orientation trivial, train with
`lambda_orient=0`), but compositions are multi-element so per-atom atomic numbers Z are
read from the structure (carbon-24 hardcodes Z=6). The periodic-table embedding + EGNN
encoder therefore do real work here. Lattices are rebuilt from the 6 cell parameters to
remove global-rotation ambiguity; space groups recovered with pymatgen.

Yields the same batch dict as the other loaders so the model/training code is unchanged.
Requires `pymatgen` + `pandas`.
"""
from __future__ import annotations
import os
import torch
from torch.utils.data import Dataset


def _parse_rows(csv_path: str, max_atoms: int, symprec: float):
    import pandas as pd
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    df = pd.read_csv(csv_path)
    rows = []
    for cif in df["cif"]:
        st = Structure.from_str(cif, fmt="cif")
        n = len(st)
        if n == 0 or n > max_atoms:
            continue
        a, b, c, al, be, ga = st.lattice.parameters
        L = torch.tensor(Lattice.from_parameters(a, b, c, al, be, ga).matrix,
                         dtype=torch.float32)
        frac = torch.tensor(st.frac_coords, dtype=torch.float32)
        frac = frac - torch.floor(frac)  # wrap into [0,1)
        Z = torch.tensor([site.specie.Z for site in st], dtype=torch.long)  # (n,)
        try:
            sg = SpacegroupAnalyzer(st, symprec=symprec).get_space_group_number()
        except Exception:
            sg = 1
        rows.append((n, L, frac, Z, int(sg)))
    return rows


def _to_item(n, L, frac, Zatoms, sg, max_atoms):
    Z = torch.zeros(max_atoms, 1, dtype=torch.long)
    local = torch.zeros(max_atoms, 1, 3)
    atom_mask = torch.zeros(max_atoms, 1, dtype=torch.bool)
    centroid = torch.zeros(max_atoms, 3)
    orient = torch.eye(3).expand(max_atoms, 3, 3).clone()
    mol_mask = torch.zeros(max_atoms, dtype=torch.bool)
    Z[:n, 0] = Zatoms
    atom_mask[:n, 0] = True
    mol_mask[:n] = True
    centroid[:n] = frac
    return {"Z": Z, "local": local, "atom_mask": atom_mask, "mol_mask": mol_mask,
            "lattice": L, "centroid": centroid, "orient": orient,
            "sg": torch.tensor(sg, dtype=torch.long)}


class MP20Dataset(Dataset):
    def __init__(self, csv_path: str, max_atoms: int = 20, symprec: float = 0.1,
                 cache_path: str | None = None):
        self.max_atoms = max_atoms
        if cache_path and os.path.exists(cache_path):
            self.items = torch.load(cache_path, weights_only=False)
        else:
            rows = _parse_rows(csv_path, max_atoms, symprec)
            self.items = [_to_item(*r, max_atoms) for r in rows]
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                torch.save(self.items, cache_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {**self.items[i], "idx": torch.tensor(i, dtype=torch.long)}
