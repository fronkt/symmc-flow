"""Riemannian geometry for the three state manifolds.

- SO(3): orientations. exp/log via Rodrigues, geodesic interpolation, SVD projection.
- T^3 : centroid fractional coords. wrapped difference / geodesic on the torus.
- Lattice R^{3x3}: Euclidean, optimal-transport (straight-line) path.

All ops are batched over a leading (...) shape and run on CPU or CUDA.
"""
from __future__ import annotations
import functools
import math
import torch

from .space_group import family_of

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


# ----- log-volume + shape parametrization ------------------------------------
# The lattice is flowed in k = (s, vec(S)) in R^10 where s = log(V / n) is the
# per-atom log volume (cell volume scales ~linearly with atom count) and
# S = L / V^(1/3) is the det-1 shape. Decoding always yields det > 0; real
# lattices are stored right-handed so no handedness information is lost.
def _signed_cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * x.abs().clamp_min(_EPS).pow(1.0 / 3.0)


def lattice_to_param(L: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """L:(B,3,3), n:(B,) atom count -> k:(B,10)."""
    V = torch.linalg.det(L)
    s = torch.log(V.abs().clamp_min(_EPS) / n.to(L.dtype).clamp_min(1.0))
    S = L / _signed_cbrt(V)[..., None, None]
    return torch.cat([s.unsqueeze(-1), S.reshape(*L.shape[:-2], 9)], dim=-1)


def param_to_lattice(k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """k:(B,10), n:(B,) -> L:(B,3,3) with det(L) = n * exp(s) > 0."""
    s, K = k[..., 0], k[..., 1:].reshape(*k.shape[:-1], 3, 3)
    S = K / _signed_cbrt(torch.linalg.det(K))[..., None, None]  # re-impose det = 1
    V = n.to(k.dtype).clamp_min(1.0) * torch.exp(s)
    return V.pow(1.0 / 3.0)[..., None, None] * S


def prior_lattice_param(n: torch.Tensor, vol_per_atom: float = 10.0,
                        logvol_std: float = 0.3, shape_std: float = 0.15) -> torch.Tensor:
    """Lattice prior in param space: s ~ N(log v0, sigma^2), shape ~ I + noise.
    n:(B,) -> k:(B,10)."""
    B = n.shape[0]
    dev = n.device
    s = math.log(vol_per_atom) + logvol_std * torch.randn(B, device=dev)
    K = torch.eye(3, device=dev) + shape_std * torch.randn(B, 3, 3, device=dev)
    return torch.cat([s.unsqueeze(-1), K.reshape(B, 9)], dim=-1)


# ===================== Lattice (log-metric k in R^6, Phase F) =================
# Rotation-invariant cell repr for space-group family masking. The metric is G = L L^T (rows of
# L are lattice vectors, cart = frac @ L), S = logm(G) symmetric, expanded in an ORTHONORMAL
# symmetric basis {B_0..B_5}: k_i = <S, B_i>. Then:
#   - k_0 = tr(S)/sqrt(3) = 2 ln(V)/sqrt(3)  is the sole VOLUME coordinate (always free)
#   - crystal-family constraints become "freeze a subset of k to a canonical value" (verified in
#     tasks/phaseF_spec.md): monoclinic freezes k3,k5; orthorhombic k3,k4,k5; tetragonal k2..k5;
#     hexagonal k2,k3,k4 and k5=CANON_HEX (gamma=120, an a-independent constant in log-metric
#     space); cubic k1..k5. This is the DiffCSP++/SGFM mechanism, ported to molecular flow.
def _sym_basis(device=None, dtype=torch.float32) -> torch.Tensor:
    s3, s6, s2 = math.sqrt(3.0), math.sqrt(6.0), math.sqrt(2.0)
    B = torch.zeros(6, 3, 3, device=device, dtype=dtype)
    B[0] = torch.eye(3, device=device, dtype=dtype) / s3
    B[1] = torch.diag(torch.tensor([1.0, 1.0, -2.0], device=device, dtype=dtype)) / s6
    B[2] = torch.diag(torch.tensor([1.0, -1.0, 0.0], device=device, dtype=dtype)) / s2
    B[3, 1, 2] = B[3, 2, 1] = 1.0 / s2       # S_23  (alpha)
    B[4, 0, 2] = B[4, 2, 0] = 1.0 / s2       # S_13  (beta)
    B[5, 0, 1] = B[5, 1, 0] = 1.0 / s2       # S_12  (gamma)
    return B


SYM_BASIS = _sym_basis()
CANON_HEX = -math.log(3.0) / math.sqrt(2.0)   # k5 for gamma=120 (a-independent); ~ -0.776836


def _logm_spd(G: torch.Tensor) -> torch.Tensor:
    """Matrix log of a symmetric positive-definite (...,3,3) via eigendecomposition."""
    G = 0.5 * (G + G.transpose(-1, -2))
    w, V = torch.linalg.eigh(G)
    w = torch.log(w.clamp_min(_EPS))
    return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)


def _expm_sym(S: torch.Tensor) -> torch.Tensor:
    """Matrix exp of a symmetric (...,3,3) via eigendecomposition (result is SPD)."""
    S = 0.5 * (S + S.transpose(-1, -2))
    w, V = torch.linalg.eigh(S)
    return (V * torch.exp(w).unsqueeze(-2)) @ V.transpose(-1, -2)


def lattice_to_logmetric(L: torch.Tensor) -> torch.Tensor:
    """L:(...,3,3) rows=lattice vectors -> k:(...,6) log-metric coordinates."""
    Bmat = SYM_BASIS.to(L.device, L.dtype)
    G = L @ L.transpose(-1, -2)
    S = _logm_spd(G)
    return torch.einsum("...ij,aij->...a", S, Bmat)


def logmetric_to_lattice(k: torch.Tensor) -> torch.Tensor:
    """k:(...,6) -> L:(...,3,3) lower-triangular (rows=lattice vectors), det>0.
    Frame differs from a from_parameters matrix but the METRIC L L^T is identical, so it is
    gauge-consistent everywhere the codebase uses the lattice (round-tripped through k; pair
    distances are frac^T G frac; StructureMatcher is frame-invariant)."""
    Bmat = SYM_BASIS.to(k.device, k.dtype)
    S = torch.einsum("...a,aij->...ij", k, Bmat)
    G = _expm_sym(S)
    return torch.linalg.cholesky(G)


@functools.lru_cache(maxsize=256)
def _family_free_canon(sg: int) -> tuple[tuple[bool, ...], tuple[float, ...]]:
    """(free[6], canon[6]) for a space group: free=True dims are generated; the rest are frozen
    at canon. k0 (volume) is always free. See the table in tasks/phaseF_spec.md."""
    fam = family_of(int(sg))
    free = [True] * 6
    canon = [0.0] * 6
    if fam == 1:                    # monoclinic: freeze k3 (alpha), k5 (gamma)
        free[3] = free[5] = False
    elif fam == 2:                  # orthorhombic: freeze all off-diagonals
        free[3] = free[4] = free[5] = False
    elif fam == 3:                  # tetragonal: a=b, all 90
        free[2] = free[3] = free[4] = free[5] = False
    elif fam in (4, 5):             # trigonal/hexagonal: a=b, gamma=120
        free[2] = free[3] = free[4] = free[5] = False
        canon[5] = CANON_HEX
    elif fam == 6:                  # cubic: a=b=c, all 90
        free[1] = free[2] = free[3] = free[4] = free[5] = False
    return tuple(free), tuple(canon)


def _free_canon_batch(sg: torch.Tensor):
    """sg:(B,) long -> (free (B,6) bool, canon (B,6) float) on sg.device."""
    frees, canons = [], []
    for s in sg.tolist():
        f, c = _family_free_canon(int(s))
        frees.append(f)
        canons.append(c)
    free = torch.tensor(frees, dtype=torch.bool, device=sg.device)
    canon = torch.tensor(canons, dtype=torch.float32, device=sg.device)
    return free, canon


def apply_family_mask(k: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
    """Freeze crystal-family-constrained coords of k:(B,6) to their canonical values."""
    free, canon = _free_canon_batch(sg)
    return torch.where(free, k, canon.to(k.dtype))


def mask_velocity(v: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
    """Zero the velocity on frozen (non-free) lattice coords so integration never moves them."""
    free, _ = _free_canon_batch(sg)
    return v * free.to(v.dtype)


def prior_logmetric(n: torch.Tensor, sg: torch.Tensor, vol_per_atom: float = 10.0,
                    logvol_std: float = 0.3, dev_std: float = 0.3,
                    family_mask: bool = True) -> torch.Tensor:
    """Data-informed lattice prior in log-metric space. n:(B,) sg:(B,) -> k:(B,6).
    k0 (volume) ~ N(2 ln(vol_per_atom*n)/sqrt3, (2 logvol_std/sqrt3)^2); deviatoric dims ~ N(0,
    dev_std^2); with family_mask the crystal-family mask is applied so the cell starts on-family."""
    B = n.shape[0]
    dev = n.device
    lnV = math.log(vol_per_atom) + torch.log(n.to(torch.float32).clamp_min(1.0)) \
        + logvol_std * torch.randn(B, device=dev)
    k0 = 2.0 * lnV / math.sqrt(3.0)
    kdev = dev_std * torch.randn(B, 5, device=dev)
    k = torch.cat([k0.unsqueeze(-1), kdev], dim=-1)
    return apply_family_mask(k, sg) if family_mask else k


# ----- repr dispatch (shape10 default vs logmetric6 Phase F) ------------------
def lat_to_k(L: torch.Tensor, n: torch.Tensor, repr: str = "shape10") -> torch.Tensor:
    return lattice_to_logmetric(L) if repr == "logmetric6" else lattice_to_param(L, n)


def k_to_lat(k: torch.Tensor, n: torch.Tensor, repr: str = "shape10") -> torch.Tensor:
    return logmetric_to_lattice(k) if repr == "logmetric6" else param_to_lattice(k, n)


def lattice_param_dim(repr: str = "shape10") -> int:
    return 6 if repr == "logmetric6" else 10


# ===================== Priors ================================================
def prior_centroid(shape, device=None, dtype=torch.float32, std=None):
    """Fractional-centroid prior. std=None -> uniform U[0,1)^3 (max-entropy torus
    prior); std set -> wrapped normal around the cell center 0.5 (softer start that
    asks the field to learn a smaller mean displacement)."""
    if std is None:
        return torch.rand(*shape, 3, device=device, dtype=dtype)
    return wrap(0.5 + std * torch.randn(*shape, 3, device=device, dtype=dtype))


def prior_orientation(shape, device=None, dtype=torch.float32):
    return random_so3(tuple(shape), device=device, dtype=dtype)
