"""Clari-style pair-bias attention.

Plain multi-head self-attention over molecular-block tokens, with a learned
additive pairwise bias injected into the attention logits. This replaces
triangle attention/triangle multiplication: O(N^2) memory instead of O(N^3),
no axial triangle updates.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class PairBiasAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pair_dim: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.pair_to_bias = nn.Linear(pair_dim, n_heads)  # per-head scalar bias
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pair, mask=None):
        """x:(B,N,D) pair:(B,N,N,pair_dim) mask:(B,N) bool. Returns (B,N,D)."""
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)  # (B,h,N,dh)
        k = self.k(x).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.dh).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)     # (B,h,N,N)
        bias = self.pair_to_bias(pair).permute(0, 3, 1, 2)          # (B,h,N,N)
        logits = logits + bias
        if mask is not None:
            kmask = (~mask).view(B, 1, 1, N)
            logits = logits.masked_fill(kmask, float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=-1))
        out = attn @ v                                              # (B,h,N,dh)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.o(out)


class PairBiasBlock(nn.Module):
    def __init__(self, d_model, n_heads, pair_dim, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = PairBiasAttention(d_model, n_heads, pair_dim, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, pair, mask=None):
        x = x + self.attn(self.n1(x), pair, mask)
        x = x + self.ffn(self.n2(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


class PairBiasStack(nn.Module):
    def __init__(self, d_model, n_heads, pair_dim, n_layers=4, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            PairBiasBlock(d_model, n_heads, pair_dim, ffn_mult, dropout)
            for _ in range(n_layers)
        )

    def forward(self, x, pair, mask=None):
        for blk in self.blocks:
            x = blk(x, pair, mask)
        return x
