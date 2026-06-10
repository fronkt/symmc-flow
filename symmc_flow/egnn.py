"""Equivariant Graph Neural Network (Satorras et al. 2021) producing E(3)-invariant
per-atom embeddings for each rigid molecular block.

We only need the invariant node features h (the coordinate update is kept so the
message passing is geometry-aware, but the exported embedding is invariant).
Operates per-molecule on a dense atom set with a padding mask.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, h, x, mask):
        """h:(B,N,H) x:(B,N,3) mask:(B,N) bool. Returns updated (h, x)."""
        B, N, H = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        diff = x.unsqueeze(2) - x.unsqueeze(1)            # (B,N,N,3)
        dist2 = (diff ** 2).sum(-1, keepdim=True)          # (B,N,N,1)
        m = self.edge_mlp(torch.cat([hi, hj, dist2], dim=-1))  # (B,N,N,H)

        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()  # (B,N,N)
        eye = torch.eye(N, device=h.device, dtype=pair_mask.dtype).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - eye)                # no self edges
        pm = pair_mask.unsqueeze(-1)                        # (B,N,N,1)

        agg = (m * pm).sum(2)                               # (B,N,H)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # coordinate update (equivariant) — keeps message passing geometry-aware
        coeff = self.coord_mlp(m)                           # (B,N,N,1)
        denom = pair_mask.sum(2, keepdim=True).clamp_min(1.0)  # (B,N,1)
        dx = (diff * coeff * pm).sum(2) / denom             # (B,N,3)
        x = x + dx * mask.unsqueeze(-1).float()
        return h, x


class EGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.embed_in = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(EGNNLayer(hidden) for _ in range(layers))
        self.out = nn.Linear(hidden, hidden)
        self.hidden = hidden

    def forward(self, atom_feats, coords, mask):
        """atom_feats:(B,N,F) coords:(B,N,3) mask:(B,N).
        Returns per-atom invariant h (B,N,hidden) and pooled molecule emb (B,hidden)."""
        h = self.embed_in(atom_feats)
        x = coords
        for layer in self.layers:
            h, x = layer(h, x, mask)
        h = self.out(h)
        h = h * mask.unsqueeze(-1).float()
        denom = mask.sum(1, keepdim=True).clamp_min(1.0).float()
        pooled = h.sum(1) / denom
        return h, pooled
