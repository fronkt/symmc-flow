"""Space-group symmetry operations (full, via pymatgen) and SGFM group averaging.

A symmetry operation acts on fractional coordinates as g.x = (W x + t) mod 1.
The averaged (group-equivariant) vector field is

    v^G(x) = (1/|G|) sum_g  W_g^T  v_theta(g.x)

so that v^G(h.x) = W_h v^G(x) for every h in G (the pushforward W_g^T = W_g^{-1}
re-expresses each transported velocity in the frame of x).

Operations are the full general-position operators of the conventional cell, read from
`pymatgen.symmetry.groups.SpaceGroup.from_int_number(n).symmetry_ops` and cached per space
group in a deterministic order (identity first). This replaces the earlier six-group stub
that fell back to P1 for everything else, so group averaging is now active for every space
group, and the Cartesian rotation parts needed for symmetry-coset labelling are available
(`cartesian_rotations`). If pymatgen is unavailable or a number is invalid, `get_ops` falls
back to the P1 identity.

Convention: the rest of the codebase uses cart = frac @ lattice (lattice rows are the
lattice vectors), i.e. x_cart = L^T x_frac for column vectors, so the Cartesian linear part
of a fractional operation W is R_cart = L^T W L^{-T}.
"""
from __future__ import annotations
import functools

import torch


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


def _is_identity(W, t) -> bool:
    import numpy as np
    return bool(np.allclose(W, np.eye(3)) and np.allclose(np.mod(t, 1.0), 0.0))


@functools.lru_cache(maxsize=512)
def _ops_frac_np(sg_number: int):
    """(W (K,3,3), t (K,3)) float64 numpy arrays for the full space group, deterministically
    ordered (identity first, then lexicographic). Cached. P1 identity on any failure."""
    import numpy as np
    try:
        from pymatgen.symmetry.groups import SpaceGroup
        ops = list(SpaceGroup.from_int_number(int(sg_number)).symmetry_ops)
        Ws = [np.asarray(o.rotation_matrix, dtype=np.float64) for o in ops]
        ts = [np.mod(np.asarray(o.translation_vector, dtype=np.float64), 1.0) for o in ops]
        if not Ws:
            raise ValueError("no ops")
    except Exception:
        Ws, ts = [np.eye(3)], [np.zeros(3)]

    def key(i):
        return (0 if _is_identity(Ws[i], ts[i]) else 1,
                tuple(np.round(Ws[i].ravel(), 3).tolist()),
                tuple(np.round(ts[i], 3).tolist()))

    order = sorted(range(len(Ws)), key=key)
    W = np.stack([Ws[i] for i in order])
    t = np.stack([ts[i] for i in order])
    return W, t


def get_ops(sg_number: int, device=None, dtype=torch.float32) -> SpaceGroupOps:
    """Full general-position operators of space group `sg_number` (identity first)."""
    W, t = _ops_frac_np(int(sg_number))
    return SpaceGroupOps(torch.as_tensor(W, dtype=dtype, device=device),
                         torch.as_tensor(t, dtype=dtype, device=device))


def n_ops(sg_number: int) -> int:
    """Number of general-position operators of `sg_number`."""
    return int(_ops_frac_np(int(sg_number))[0].shape[0])


def cartesian_rotations(sg_number: int, lattice: torch.Tensor) -> torch.Tensor:
    """Cartesian linear parts R_cart(g) = L^T W_g L^{-T} of every operator, in the codebase's
    cart = frac @ lattice convention. `lattice` is (3,3) (rows = lattice vectors). Returns
    (K,3,3) on lattice's device/dtype, ordered identity-first to match `get_ops`. These are
    orthogonal (det +-1); the proper ones are the relative rotations between symmetry copies."""
    W, _ = _ops_frac_np(int(sg_number))
    Wt = torch.as_tensor(W, dtype=lattice.dtype, device=lattice.device)   # (K,3,3)
    Lt = lattice.transpose(-1, -2)
    LtInv = torch.linalg.inv(Lt)
    return Lt @ Wt @ LtInv
