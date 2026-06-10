"""Space-group symmetry operations and SGFM group averaging.

A symmetry operation acts on fractional coordinates as g.x = (W x + t) mod 1.
The averaged (group-equivariant) vector field is

    v^G(x) = (1/|G|) sum_g  W_g^T  v_theta(g.x)

so that v^G(h.x) = W_h v^G(x) for every h in G (the pushforward W_g^T = W_g^{-1}
re-expresses each transported velocity in the frame of x).

This reference ships a curated op subset (P1, P-1, 2-fold, mm2, ...). For the GPU
benchmark phase, replace `get_ops` with full operations from
`pymatgen.symmetry.groups.SpaceGroup(n).symmetry_ops`.
"""
from __future__ import annotations
import torch

_I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
_INV = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
_C2Z = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]      # 2-fold about z
_C2X = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]      # 2-fold about x
_C2Y = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]      # 2-fold about y
_MZ = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]        # mirror perpendicular to z

# space-group number -> list of (W, t). Curated subset; identity fallback.
_REGISTRY: dict[int, list[tuple[list, list]]] = {
    1: [(_I, [0, 0, 0])],                                              # P1
    2: [(_I, [0, 0, 0]), (_INV, [0, 0, 0])],                           # P-1
    3: [(_I, [0, 0, 0]), (_C2Z, [0, 0, 0])],                           # P2
    6: [(_I, [0, 0, 0]), (_MZ, [0, 0, 0])],                            # Pm
    16: [(_I, [0, 0, 0]), (_C2Z, [0, 0, 0]),                           # P222
         (_C2X, [0, 0, 0]), (_C2Y, [0, 0, 0])],
    25: [(_I, [0, 0, 0]), (_C2Z, [0, 0, 0]),                           # Pmm2
         (_MZ, [0, 0, 0]), ([[1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, 0, 0])],
}


class SpaceGroupOps:
    """Holds W:(K,3,3), t:(K,3) for one space group."""

    def __init__(self, W: torch.Tensor, t: torch.Tensor):
        self.W = W
        self.t = t

    @property
    def order(self) -> int:
        return self.W.shape[0]

    def to(self, device=None, dtype=None):
        return SpaceGroupOps(self.W.to(device=device, dtype=dtype),
                             self.t.to(device=device, dtype=dtype))

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """g.x for all ops. x:(...,3) -> (...,K,3) wrapped into [0,1)."""
        # (...,1,3) @ (K,3,3)^T -> (...,K,3)
        xg = torch.einsum("kij,...j->...ki", self.W, x) + self.t
        return xg - torch.floor(xg)

    def symmetrize_field(self, predict_fn, x: torch.Tensor) -> torch.Tensor:
        """v^G(x) = mean_g W_g^T predict_fn(g.x). x:(B,M,3) -> (B,M,3)."""
        B, M, _ = x.shape
        K = self.order
        xg = self.act(x)                          # (B,M,K,3)
        xg = xg.reshape(B, M * K, 3)
        vg = predict_fn(xg).reshape(B, M, K, 3)   # (B,M,K,3)
        # pull back: W_g^T v(g.x)
        v_pb = torch.einsum("kji,bmkj->bmki", self.W, vg)
        return v_pb.mean(dim=2)

    def symmetrize_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Cheap output-side symmetrization: average the point-group images
        mean_g W_g v of a per-molecule velocity. v:(B,M,3) -> (B,M,3).
        Used when only the predicted field (not the network input) is symmetrized."""
        return torch.einsum("kij,bmj->bmki", self.W, v).mean(dim=2)


def get_ops(sg_number: int, device=None, dtype=torch.float32) -> SpaceGroupOps:
    ops = _REGISTRY.get(int(sg_number), _REGISTRY[1])
    W = torch.tensor([w for w, _ in ops], dtype=dtype, device=device)
    t = torch.tensor([tt for _, tt in ops], dtype=dtype, device=device)
    return SpaceGroupOps(W, t)
