"""Riemannian geometry for the three state manifolds.

- SO(3): orientations. exp/log via Rodrigues, geodesic interpolation, SVD projection.
- T^3 : centroid fractional coords. wrapped difference / geodesic on the torus.
- Lattice R^{3x3}: Euclidean, optimal-transport (straight-line) path.

All ops are batched over a leading (...) shape and run on CPU or CUDA.
"""
from __future__ import annotations
import torch

_EPS = 1e-7


# ===================== SO(3) =================================================
def hat(w: torch.Tensor) -> torch.Tensor:
    """so(3) vector (...,3) -> skew matrix (...,3,3)."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    row0 = torch.stack([O, -wz, wy], dim=-1)
    row1 = torch.stack([wz, O, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, O], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def vee(M: torch.Tensor) -> torch.Tensor:
    """skew matrix (...,3,3) -> so(3) vector (...,3)."""
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], dim=-1)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Exponential map so(3) (...,3) -> SO(3) (...,3,3) via Rodrigues."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(_EPS)  # (...,1)
    K = hat(w)
    t = theta[..., None]  # (...,1,1)
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    a = sin_t / t
    b = (1.0 - cos_t) / (t * t)
    return I + a * K + b * (K @ K)


def _rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """SO(3) (...,3,3) -> unit quaternion (...,4) as (w,x,y,z), Shepperd's method.
    Numerically stable for all rotations including theta = pi."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                     1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1)  # (...,4)

    def branch(ti):
        r = torch.sqrt(ti.clamp_min(_EPS))
        s = 0.5 / r
        return r, s

    r0, s0 = branch(t[..., 0])
    q0 = torch.stack([0.5 * r0, (m21 - m12) * s0, (m02 - m20) * s0, (m10 - m01) * s0], -1)
    r1, s1 = branch(t[..., 1])
    q1 = torch.stack([(m21 - m12) * s1, 0.5 * r1, (m01 + m10) * s1, (m02 + m20) * s1], -1)
    r2, s2 = branch(t[..., 2])
    q2 = torch.stack([(m02 - m20) * s2, (m01 + m10) * s2, 0.5 * r2, (m12 + m21) * s2], -1)
    r3, s3 = branch(t[..., 3])
    q3 = torch.stack([(m10 - m01) * s3, (m02 + m20) * s3, (m12 + m21) * s3, 0.5 * r3], -1)

    cand = torch.stack([q0, q1, q2, q3], dim=-2)            # (...,4,4)
    idx = torch.argmax(t, dim=-1)                            # (...)
    q = torch.gather(cand, -2, idx[..., None, None].expand(*idx.shape, 1, 4)).squeeze(-2)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(_EPS)
    # canonicalize to w >= 0 (q and -q are the same rotation)
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Log map SO(3) (...,3,3) -> so(3) vector (...,3). Stable near 0 and pi."""
    q = _rot_to_quat(R)
    w_, v = q[..., 0], q[..., 1:]
    vnorm = torch.linalg.norm(v, dim=-1, keepdim=True)       # = |sin(theta/2)|
    theta = 2.0 * torch.atan2(vnorm.squeeze(-1), w_.abs())   # in [0, pi]
    small = vnorm.squeeze(-1) < 1e-6
    # axis = v/|v|; for small angle, w ~ 2*v (since theta/|v| -> 2)
    coef = torch.where(small, 2.0 * torch.ones_like(theta),
                       theta / vnorm.squeeze(-1).clamp_min(_EPS))
    return coef.unsqueeze(-1) * v


def so3_geodesic(R0: torch.Tensor, R1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Point at parameter t on the geodesic from R0 to R1 (t broadcastable to (...))."""
    rel = so3_log(R0.transpose(-1, -2) @ R1)  # body-frame tangent (...,3)
    if t.dim() < rel.dim():
        t = t.reshape(t.shape + (1,) * (rel.dim() - t.dim()))
    return R0 @ so3_exp(t * rel)


def so3_velocity(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Constant body-frame velocity of the geodesic R0 -> R1, as so(3) vector (...,3)."""
    return so3_log(R0.transpose(-1, -2) @ R1)


def project_so3(M: torch.Tensor) -> torch.Tensor:
    """Nearest rotation matrix to M (SVD; fixes det sign)."""
    U, _, Vh = torch.linalg.svd(M)
    d = torch.linalg.det(U @ Vh)
    D = torch.eye(3, device=M.device, dtype=M.dtype).expand_as(M).clone()
    D[..., 2, 2] = d
    return U @ D @ Vh


def random_so3(shape: tuple[int, ...], device=None, dtype=torch.float32) -> torch.Tensor:
    """Uniform random rotations (Haar) via exp of small-to-large axis-angle, then QR."""
    A = torch.randn(*shape, 3, 3, device=device, dtype=dtype)
    Q, Rm = torch.linalg.qr(A)
    # make Haar-uniform: multiply by sign of diag(R)
    sign = torch.sign(torch.diagonal(Rm, dim1=-2, dim2=-1))
    Q = Q * sign.unsqueeze(-2)
    # ensure det +1
    det = torch.linalg.det(Q)
    Q[..., :, 0] = Q[..., :, 0] * det.unsqueeze(-1)
    return Q


def so3_angle(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between rotations (...,)."""
    return torch.linalg.norm(so3_log(R0.transpose(-1, -2) @ R1), dim=-1)


# ===================== Torus T^3 =============================================
def wrap(x: torch.Tensor) -> torch.Tensor:
    """Wrap fractional coords into [0,1)."""
    return x - torch.floor(x)


def torus_diff(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Shortest signed displacement x0 -> x1 on the torus, in (-0.5, 0.5]."""
    d = x1 - x0
    return d - torch.round(d)


def torus_geodesic(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    w = torus_diff(x0, x1)
    if t.dim() < w.dim():
        t = t.reshape(t.shape + (1,) * (w.dim() - t.dim()))
    return wrap(x0 + t * w)


def torus_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return torus_diff(x0, x1)


# ===================== Lattice (Euclidean) ===================================
def lattice_geodesic(L0: torch.Tensor, L1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.dim() < L0.dim():
        t = t.reshape(t.shape + (1,) * (L0.dim() - t.dim()))
    return (1.0 - t) * L0 + t * L1


def lattice_velocity(L0: torch.Tensor, L1: torch.Tensor) -> torch.Tensor:
    return L1 - L0


# ===================== Priors ================================================
def prior_centroid(shape, device=None, dtype=torch.float32):
    return torch.rand(*shape, 3, device=device, dtype=dtype)


def prior_orientation(shape, device=None, dtype=torch.float32):
    return random_so3(tuple(shape), device=device, dtype=dtype)


def prior_lattice(shape, device=None, dtype=torch.float32):
    """Identity lattice + Gaussian noise (a neutral starting lattice)."""
    I = torch.eye(3, device=device, dtype=dtype).expand(*shape, 3, 3)
    return I + 0.1 * torch.randn(*shape, 3, 3, device=device, dtype=dtype)
