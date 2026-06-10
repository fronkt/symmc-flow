"""2D periodic-table embedding (CrystalDiT-style).

Each element Z is mapped to a (period/row, group/column, f-block index) triple in
the 18-column IUPAC layout. The f_index disambiguates lanthanides/actinides, which
otherwise collapse onto group 3. Embedding = sum of row/col/f embeddings.
"""
from __future__ import annotations
import torch
import torch.nn as nn

# --- Build Z -> (row, col, f_index) for Z = 1..118 ---------------------------
# col in 1..18 (IUPAC groups); f_index in 0..14 (0 = not an inner f-block atom).


def _build_table() -> dict[int, tuple[int, int, int]]:
    t: dict[int, tuple[int, int, int]] = {}
    t[1] = (1, 1, 0)
    t[2] = (1, 18, 0)
    # periods 2 & 3: s-block (1,2) then p-block (13..18)
    for z, col in zip(range(3, 11), [1, 2, 13, 14, 15, 16, 17, 18]):
        t[z] = (2, col, 0)
    for z, col in zip(range(11, 19), [1, 2, 13, 14, 15, 16, 17, 18]):
        t[z] = (3, col, 0)
    # periods 4 & 5: s(1,2) d(3..12) p(13..18)
    for z, col in zip(range(19, 37), list(range(1, 13)) + list(range(13, 19))):
        t[z] = (4, col, 0)
    for z, col in zip(range(37, 55), list(range(1, 13)) + list(range(13, 19))):
        t[z] = (5, col, 0)
    # period 6: Cs Ba | La (group3) | Ce..Lu f-block | Hf..Hg (4..12) | Tl..Rn
    t[55], t[56] = (6, 1, 0), (6, 2, 0)
    t[57] = (6, 3, 0)                       # La in group 3
    for i, z in enumerate(range(58, 72)):   # Ce..Lu -> f_index 1..14
        t[z] = (6, 3, i + 1)
    for z, col in zip(range(72, 87), list(range(4, 13)) + list(range(13, 19))):
        t[z] = (6, col, 0)
    # period 7: mirror of period 6
    t[87], t[88] = (7, 1, 0), (7, 2, 0)
    t[89] = (7, 3, 0)                       # Ac in group 3
    for i, z in enumerate(range(90, 104)):  # Th..Lr -> f_index 1..14
        t[z] = (7, 3, i + 1)
    for z, col in zip(range(104, 119), list(range(4, 13)) + list(range(13, 19))):
        t[z] = (7, col, 0)
    return t


_TABLE = _build_table()
_MAX_Z = 118
# index 0 is reserved as a padding atom (row=col=f=0)
_ROW = torch.zeros(_MAX_Z + 1, dtype=torch.long)
_COL = torch.zeros(_MAX_Z + 1, dtype=torch.long)
_FID = torch.zeros(_MAX_Z + 1, dtype=torch.long)
for _z, (_r, _c, _f) in _TABLE.items():
    _ROW[_z], _COL[_z], _FID[_z] = _r, _c, _f


def z_to_rcf(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map atomic numbers (any shape, long) to (row, col, f_index) tensors."""
    z = z.long().clamp(0, _MAX_Z)
    dev = z.device
    return _ROW.to(dev)[z], _COL.to(dev)[z], _FID.to(dev)[z]


class PeriodicTableEmbedding(nn.Module):
    """Sum of learned row / col / f-block embeddings for each atom (Z=0 -> pad)."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.row_emb = nn.Embedding(8, dim, padding_idx=0)   # periods 0..7
        self.col_emb = nn.Embedding(19, dim, padding_idx=0)  # groups 0..18
        self.f_emb = nn.Embedding(15, dim, padding_idx=0)    # f_index 0..14
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r, c, f = z_to_rcf(z)
        return self.row_emb(r) + self.col_emb(c) + self.f_emb(f)
