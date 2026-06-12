"""SymMCFlow — the combined vector-field network.

Pipeline per forward pass:
  atoms -> 2D periodic-table embed -> EGNN (per-molecule invariant emb)
        -> molecule tokens augmented with current (centroid, orient, t, space group)
        -> pair-bias attention over tokens (pair features from torus distances)
        -> three heads predict the manifold velocities (lattice, centroid, orient).
Optionally the centroid field is symmetrized over the space group (SGFM).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .config import ModelConfig
from .periodic_table import PeriodicTableEmbedding
from .egnn import EGNNEncoder
from .pair_bias_attention import PairBiasStack
from . import manifolds as M
from .space_group import get_ops


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t:(B,) -> (B,dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    a = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class SymMCFlow(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.atom_embed = PeriodicTableEmbedding(c.atom_embed_dim)
        self.egnn = EGNNEncoder(c.atom_embed_dim, c.egnn_hidden, c.egnn_layers)

        self.mol_proj = nn.Linear(c.egnn_hidden, c.d_model)
        self.centroid_in = nn.Linear(3, c.d_model)
        self.orient_in = nn.Linear(9, c.d_model)
        self.lattice_in = nn.Linear(10, c.d_model)  # current lattice in param space
        self.time_mlp = nn.Sequential(
            nn.Linear(c.time_embed_dim, c.d_model), nn.SiLU(), nn.Linear(c.d_model, c.d_model)
        )
        self.sg_embed = nn.Embedding(c.n_space_groups + 1, c.d_model)

        # pair features: frac diff (3) + frac dist (1) + Cartesian diff (3) +
        # Cartesian dist (1) + Fourier(frac diff) (3*2*n_freq). Cartesian terms give
        # the field real interatomic geometry at the current lattice; Fourier terms
        # (DiffCSP-style) sharpen short-range fractional resolution.
        self.pair_freqs = torch.arange(1, c.pair_n_freq + 1, dtype=torch.float32)
        self.pair_dim = 4 + 4 + 3 * 2 * c.pair_n_freq
        self.attn = PairBiasStack(c.d_model, c.n_heads, self.pair_dim,
                                  c.n_attn_layers, c.ffn_mult, c.dropout)

        self.head_centroid = nn.Sequential(
            nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model), nn.SiLU(),
            nn.Linear(c.d_model, 3))
        self.head_orient = nn.Sequential(
            nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model), nn.SiLU(),
            nn.Linear(c.d_model, 3))
        self.head_lattice = nn.Sequential(
            nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model), nn.SiLU(),
            nn.Linear(c.d_model, 10))

    # -- molecule encoding (geometry-invariant, computed once per crystal) -----
    def encode_molecules(self, Z, local, atom_mask):
        """Z:(B,M,A) local:(B,M,A,3) atom_mask:(B,M,A) -> mol_emb (B,M,d_model)."""
        B, Mm, A = Z.shape
        feats = self.atom_embed(Z.reshape(B * Mm, A))            # (B*M,A,emb)
        coords = local.reshape(B * Mm, A, 3)
        amask = atom_mask.reshape(B * Mm, A)
        # avoid all-False masks (padding molecules) -> set one fake atom active
        safe = amask.clone()
        empty = ~safe.any(dim=1)
        safe[empty, 0] = True
        _, pooled = self.egnn(feats, coords, safe)              # (B*M,hidden)
        pooled = pooled.reshape(B, Mm, -1)
        return self.mol_proj(pooled)

    def _pair_features(self, centroid, lattice):
        d = M.torus_diff(centroid.unsqueeze(2), centroid.unsqueeze(1))  # (B,M,M,3) frac
        dist = torch.linalg.norm(d, dim=-1, keepdim=True)
        cart = torch.einsum("bijc,bcd->bijd", d, lattice)              # (B,M,M,3) Angstrom
        cart_dist = torch.linalg.norm(cart, dim=-1, keepdim=True)
        freqs = self.pair_freqs.to(d.device, d.dtype)                  # (F,)
        ang = 2.0 * math.pi * d.unsqueeze(-1) * freqs                  # (B,M,M,3,F)
        fourier = torch.cat([ang.sin(), ang.cos()], dim=-1)           # (B,M,M,3,2F)
        fourier = fourier.flatten(-2)                                  # (B,M,M,3*2F)
        return torch.cat([d, dist, cart, cart_dist, fourier], dim=-1)

    def forward(self, mol_emb, lattice, centroid, orient, t, sg, mol_mask):
        """Predict velocity fields.
        mol_emb:(B,M,d) lattice:(B,3,3) centroid:(B,M,3) orient:(B,M,3,3)
        t:(B,) sg:(B,) mol_mask:(B,M). Returns (v_L (B,10) in lattice param space,
        v_x (B,M,3), v_R (B,M,3))."""
        B, Mm, _ = centroid.shape
        n = mol_mask.sum(-1).clamp_min(1)
        k = M.lattice_to_param(lattice, n)                                     # (B,10)
        tok = mol_emb + self.centroid_in(centroid) + self.orient_in(orient.reshape(B, Mm, 9))
        temb = self.time_mlp(timestep_embedding(t, self.cfg.time_embed_dim))  # (B,d)
        sgemb = self.sg_embed(sg.clamp(0, self.cfg.n_space_groups))            # (B,d)
        tok = tok + temb.unsqueeze(1) + sgemb.unsqueeze(1) + self.lattice_in(k).unsqueeze(1)

        pair = self._pair_features(centroid, lattice)
        h = self.attn(tok, pair, mol_mask)                                    # (B,M,d)

        v_x = self.head_centroid(h)
        v_R = self.head_orient(h)
        pooled = (h * mol_mask.unsqueeze(-1).float()).sum(1) / \
                 mol_mask.sum(1, keepdim=True).clamp_min(1.0).float()
        v_L = self.head_lattice(pooled)
        m = mol_mask.unsqueeze(-1).float()
        return v_L, v_x * m, v_R * m

    # -- SGFM-symmetrized centroid field --------------------------------------
    def symmetrized_velocity(self, mol_emb, lattice, centroid, orient, t, sg, mol_mask):
        """Per-sample group-averaged centroid field; lattice/orient unaveraged.
        Falls back to plain forward when all samples share trivial symmetry."""
        v_L, v_x, v_R = self.forward(mol_emb, lattice, centroid, orient, t, sg, mol_mask)
        B = centroid.shape[0]
        out = v_x.clone()
        for b in range(B):
            ops = get_ops(int(sg[b]), device=centroid.device, dtype=centroid.dtype)
            if ops.order == 1:
                continue

            def predict_centroid(xq):  # xq:(1,M*K,3) -> velocity (1,M*K,3)
                Mq = xq.shape[1]
                # tile per-molecule conditioning to match the expanded centroid set
                k = Mq // mol_mask.shape[1]
                me = mol_emb[b:b + 1].repeat(1, k, 1)
                ori = orient[b:b + 1].repeat(1, k, 1, 1)
                mm = mol_mask[b:b + 1].repeat(1, k)
                _, vxx, _ = self.forward(me, lattice[b:b + 1], xq, ori,
                                         t[b:b + 1], sg[b:b + 1], mm)
                return vxx

            out[b:b + 1] = ops.symmetrize_field(predict_centroid, centroid[b:b + 1])
        return v_L, out * mol_mask.unsqueeze(-1).float(), v_R
