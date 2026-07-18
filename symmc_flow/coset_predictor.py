"""CosetPredictor -- the de-novo inference module (C4).

The main-text finding is that coset conditioning nearly doubles the learnable relative
orientation, so the residual ceiling is INFERENCE-limited: the hard part is knowing which
space-group operation generated each copy. When a template supplies that operation (the
`assign_symmetry_cosets` / `eval_templated_matchrate` setting) the gain is realized directly.
This module handles the complementary DE-NOVO case, where no template is given: it predicts the
per-molecule coset id from PACKING ONLY -- the conformer, the fractional centroids, the lattice,
and the space group -- and never sees the orientation it will help predict, so it is leak-free
and available at generation time. Its held-out top-1 accuracy quantifies how much of the
inference gap is closable from packing; feeding its argmax to the coset-conditioned orientation
flow realizes that fraction of the coset gain without a template.

Architecture mirrors SymMCFlow's encoder + pair-bias attention, minus the orientation input and
the time/velocity heads, plus a per-molecule classification head over the coset codebook.
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


class CosetPredictor(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        if c.n_cosets <= 0:
            raise ValueError("CosetPredictor needs cfg.n_cosets > 0 (the coset codebook size)")
        self.atom_embed = PeriodicTableEmbedding(c.atom_embed_dim)
        self.egnn = EGNNEncoder(c.atom_embed_dim, c.egnn_hidden, c.egnn_layers)
        self.mol_proj = nn.Linear(c.egnn_hidden, c.d_model)
        self.centroid_in = nn.Linear(3, c.d_model)
        self.lattice_in = nn.Linear(10, c.d_model)
        self.sg_embed = nn.Embedding(c.n_space_groups + 1, c.d_model)
        self.pair_freqs = torch.arange(1, c.pair_n_freq + 1, dtype=torch.float32)
        self.pair_dim = 4 + 4 + 3 * 2 * c.pair_n_freq
        self.attn = PairBiasStack(c.d_model, c.n_heads, self.pair_dim,
                                  c.n_attn_layers, c.ffn_mult, c.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model), nn.SiLU(),
            nn.Linear(c.d_model, c.n_cosets + 1))

    # identical geometry-invariant molecule encoding as SymMCFlow (own weights)
    def encode_molecules(self, Z, local, atom_mask):
        B, Mm, A = Z.shape
        feats = self.atom_embed(Z.reshape(B * Mm, A))
        coords = local.reshape(B * Mm, A, 3)
        amask = atom_mask.reshape(B * Mm, A)
        safe = amask.clone()
        empty = ~safe.any(dim=1)
        safe[empty, 0] = True
        _, pooled = self.egnn(feats, coords, safe)
        return self.mol_proj(pooled.reshape(B, Mm, -1))

    def _pair_features(self, centroid, lattice):
        d = M.torus_diff(centroid.unsqueeze(2), centroid.unsqueeze(1))
        dist = torch.linalg.norm(d, dim=-1, keepdim=True)
        cart = torch.einsum("bijc,bcd->bijd", d, lattice)
        cart_dist = torch.linalg.norm(cart, dim=-1, keepdim=True)
        freqs = self.pair_freqs.to(d.device, d.dtype)
        ang = 2.0 * math.pi * d.unsqueeze(-1) * freqs
        fourier = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)
        return torch.cat([d, dist, cart, cart_dist, fourier], dim=-1)

    def forward(self, mol_emb, lattice, centroid, sg, mol_mask):
        """mol_emb:(B,M,d) lattice:(B,3,3) centroid:(B,M,3) sg:(B,) mol_mask:(B,M).
        Returns coset logits (B,M,n_cosets+1). No orientation input by construction."""
        B, Mm, _ = centroid.shape
        n = mol_mask.sum(-1).clamp_min(1)
        k = M.lattice_to_param(lattice, n)                                    # (B,10)
        tok = (mol_emb + self.centroid_in(centroid)
               + self.lattice_in(k).unsqueeze(1)
               + self.sg_embed(sg.clamp(0, self.cfg.n_space_groups)).unsqueeze(1))
        pair = self._pair_features(centroid, lattice)
        h = self.attn(tok, pair, mol_mask)
        return self.head(h)                                                   # (B,M,n_cosets+1)

    @torch.no_grad()
    def predict(self, mol_emb, lattice, centroid, sg, mol_mask):
        """Argmax coset id per molecule (B,M) long -- the template-free coset for the sampler."""
        return self.forward(mol_emb, lattice, centroid, sg, mol_mask).argmax(-1)
