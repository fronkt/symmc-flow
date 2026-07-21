"""Phase F3: symmetry- and rigidity-preserving rigid-press packing finisher.

An UNSUPERVISED, post-hoc physical relaxer for the flow's generated molecular crystals. It never
sees the reference structure -- only the generated draw plus the space-group template that already
conditioned the sampler. It answers F2's open question: once the flow puts a crystal in the right
basin (right family via the F1 mask, right-ish orientation via the coset), can a local physical
polish push it to an exact match, the way MolCrystalFlow / MOFFlow reach match through a hard-sphere
rigid-press rather than through the flow itself?

Design (see tasks/phaseF3_spec.md):
  - DOF = per asymmetric-unit (is_ref) molecule of each species group: fractional centroid c0 and an
    so(3) tangent delta with R0 = so3_exp(delta) @ R0_init; plus the cell as logmetric6 k, re-masked
    to the crystal family every step so it stays on-family.
  - Symmetry copies are REGENERATED every step from the ASU pose via the space-group ops
    (c_g = wrap(W_h c0 + t_h); R_g = Rcart_h @ R0), so symmetry is exact by construction. Rigid
    molecular geometry is preserved because atoms = centroid + R @ local with `local` frozen.
  - Objective = intermolecular Lennard-Jones packing energy over atom pairs across molecule instances
    and periodic images (intramolecular same-cell pairs excluded -- rigid, fixed), plus a light
    volume prior anchoring density. Minimised with L-BFGS. CPU-fine (tens-hundreds of atoms).
"""
from __future__ import annotations

import math
import torch

from . import manifolds as M
from .space_group import get_ops, cartesian_rotations
from .molcrystal import _species_groups

# van der Waals radii (Angstrom), common organic + a few heavier; fallback 1.7 (carbon).
_VDW = {1: 1.10, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 11: 2.27, 12: 1.73,
        14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 19: 2.75, 35: 1.85, 53: 1.98}


def _radii(Z_flat: torch.Tensor) -> torch.Tensor:
    return torch.tensor([_VDW.get(int(z), 1.70) for z in Z_flat.tolist()],
                        dtype=torch.float32, device=Z_flat.device)


def symmetry_op_indices(centroid: torch.Tensor, sg: int, groups) -> list[int]:
    """Per molecule slot, the space-group op index h with wrap(W_h c0 + t_h) == c_m (min image).

    Derived from the (clean) template fractional centroids -- the same centroid matching
    `assign_symmetry_cosets` uses -- so the finisher regenerates each copy from its reference with
    the correct operation. Reference slots map to op 0 (identity is first).
    """
    ops = get_ops(int(sg))
    W, t = ops.W.to(centroid), ops.t.to(centroid)          # (K,3,3), (K,3)
    h = [0] * centroid.shape[0]
    for g in groups:
        c0 = centroid[g[0]]
        cand = torch.einsum("kij,j->ki", W, c0) + t         # (K,3)
        cand = cand - torch.floor(cand)
        for m in g:
            d = cand - centroid[m]
            d = d - torch.round(d)                          # fractional min image
            h[m] = int(torch.argmin((d * d).sum(-1)))
    return h


def _image_shifts(L: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Fractional lattice-translation vectors whose cells lie within `cutoff` of the origin cell.

    n_max per axis from the interplanar spacing d_i = V / |a_j x a_k| so the shell always covers the
    cutoff regardless of cell shape.
    """
    V = torch.det(L).abs()
    cross = torch.stack([torch.linalg.cross(L[(i + 1) % 3], L[(i + 2) % 3]) for i in range(3)])
    d = V / cross.norm(dim=-1).clamp_min(1e-6)              # (3,) interplanar spacings
    nmax = torch.clamp(torch.ceil(cutoff / d), min=1, max=3).long().tolist()
    rng = [torch.arange(-n, n + 1, device=L.device) for n in nmax]
    grid = torch.stack(torch.meshgrid(*rng, indexing="ij"), dim=-1).reshape(-1, 3).float()
    return grid                                            # (nS,3) fractional


def packing_energy(X: torch.Tensor, r: torch.Tensor, mol_id: torch.Tensor, L: torch.Tensor,
                   contact: float = 0.90, eps: float = 1.0, d_floor: float = 0.7,
                   cutoff: float = 6.0) -> torch.Tensor:
    """Smooth Lennard-Jones intermolecular packing energy with periodic images.

    X:(N,3) Cartesian atoms, r:(N,) vdW radii, mol_id:(N,) molecule-instance id, L:(3,3) cell.
    phi(d) = eps[(R/d~)^12 - 2(R/d~)^6], R = contact*(r_i+r_j), d~ = clamp(d, d_floor). Intramolecular
    same-cell pairs (same mol_id, zero shift) and self pairs are excluded; a molecule with its own
    periodic image is a real contact and is kept. Each unordered pair is counted once (0.5 factor).
    """
    N = X.shape[0]
    shifts = _image_shifts(L, cutoff)                      # (nS,3) frac
    Scart = shifts @ L                                     # (nS,3) cartesian
    Rij = contact * (r[:, None] + r[None, :])              # (N,N)
    same_mol = mol_id[:, None] == mol_id[None, :]          # (N,N)
    eye = torch.eye(N, dtype=torch.bool, device=X.device)
    e = X.new_zeros(())
    for s, shift in enumerate(Scart):
        diff = X[:, None, :] - (X[None, :, :] + shift)     # (N,N,3)
        dist = diff.norm(dim=-1)
        is_zero = bool((shifts[s].abs().sum() == 0).item())
        if is_zero:
            excl = same_mol | eye                          # drop intramolecular + self
        else:
            excl = torch.zeros_like(eye)
        within = (dist < cutoff) & (~excl)
        dt = dist.clamp_min(d_floor)
        ratio6 = (Rij / dt) ** 6
        phi = eps * (ratio6 * ratio6 - 2.0 * ratio6)
        e = e + (phi * within).sum()
    return 0.5 * e


def build_neighbor_list(X0, r, mol_id, L, contact, cutoff, margin=1.5):
    """Enumerate intermolecular atom pairs within `cutoff+margin` from the (detached) positions X0.

    Returns (I, J, Sfrac, Rij): flat pair index tensors, per-pair integer lattice shift, and per-pair
    contact distance. The margin lets atoms drift into range between rebuilds; the list is O(pairs)
    rather than O(N^2 * shifts), which is what makes the relaxation tractable for big cells.
    """
    with torch.no_grad():
        shifts = _image_shifts(L, cutoff + margin)          # (nS,3) integer frac vectors
        Scart = shifts @ L
        I_list, J_list, S_list = [], [], []
        for s in range(shifts.shape[0]):
            diff = X0[:, None, :] - (X0[None, :, :] + Scart[s])
            dist = diff.norm(dim=-1)
            mask = dist < (cutoff + margin)
            if bool((shifts[s].abs().sum() == 0).item()):
                mask = mask & (mol_id[:, None] != mol_id[None, :])   # drop intramolecular + self
            ii, jj = torch.where(mask)
            if ii.numel():
                I_list.append(ii); J_list.append(jj)
                S_list.append(shifts[s].expand(ii.shape[0], 3))
        if not I_list:
            z = torch.zeros(0, dtype=torch.long, device=X0.device)
            return z, z, X0.new_zeros(0, 3), X0.new_zeros(0)
        I = torch.cat(I_list); J = torch.cat(J_list); Sfrac = torch.cat(S_list).to(L)
    return I, J, Sfrac, contact * (r[I] + r[J])


def packing_energy_nbr(X, L, I, J, Sfrac, Rij, eps=1.0, d_floor=0.7, cutoff=6.0):
    """Smooth LJ over a prebuilt neighbor list; differentiable in X and L. See `packing_energy`."""
    if I.numel() == 0:
        return X.new_zeros(())
    Scart = Sfrac @ L
    dist = (X[I] - (X[J] + Scart)).norm(dim=-1)
    within = dist < cutoff
    r6 = (Rij / dist.clamp_min(d_floor)) ** 6
    phi = eps * (r6 * r6 - 2.0 * r6)
    return 0.5 * (phi * within).sum()


def finish_structure(lattice, centroid, orient, local, Z, atom_mask, mol_mask, sg,
                     groups=None, op_idx=None, steps: int = 60, lr: float = 0.03,
                     contact: float = 0.90, eps: float = 1.0, d_floor: float = 0.7,
                     cutoff: float = 6.0, lam_vol: float = 0.05, vol_per_atom: float = 10.0,
                     relax_cell: bool = True, family_mask: bool = True):
    """Relax one crystal's ASU poses (+ optionally the cell) under `packing_energy`, keeping symmetry,
    rigid geometry and crystal family exact. Returns finished (lattice, centroid, orient) for ALL
    real slots (copies regenerated from the optimised ASU poses). Never reads the reference.

    Single-crystal tensors: lattice (3,3); centroid (M,3) frac; orient (M,3,3); local (M,A,3);
    Z (M,A); atom_mask (M,A); mol_mask (M,); sg scalar. `op_idx` (per-slot op index) should come from
    `symmetry_op_indices` on the clean template; if None it is derived from `centroid`.
    """
    dev = lattice.device
    sg_i = int(sg)
    real = [m for m in range(mol_mask.shape[0]) if bool(mol_mask[m])]
    if groups is None:
        # group real slots by shared conformer (species); group[0] = reference of that species
        item = {"local": local, "mol_mask": mol_mask, "atom_mask": atom_mask}
        groups = [[m for m in g if bool(mol_mask[m])] for g in _species_groups(item)]
        groups = [g for g in groups if g]
    if op_idx is None:
        op_idx = symmetry_op_indices(centroid, sg_i, groups)

    ops = get_ops(sg_i)
    W, t = ops.W.to(lattice), ops.t.to(lattice)            # (K,3,3),(K,3) fractional
    ref_of = {}                                            # slot -> its group's reference slot
    for g in groups:
        for m in g:
            ref_of[m] = g[0]
    refs = [g[0] for g in groups]
    ref_pos = {m: i for i, m in enumerate(refs)}           # reference slot -> DOF row

    # --- DOF ---
    R0_init = torch.stack([orient[m] for m in refs]).to(dev)          # (G,3,3)
    c0 = torch.stack([centroid[m] for m in refs]).to(dev).clone().requires_grad_(True)   # (G,3)
    delta = torch.zeros(len(refs), 3, device=dev, requires_grad=True)                     # (G,3)
    n_dummy = atom_mask.sum().to(dev).view(1)
    k_init = M.lat_to_k(lattice.unsqueeze(0), n_dummy, "logmetric6")[0].to(dev)
    if family_mask:
        k_init = M.apply_family_mask(k_init.unsqueeze(0), torch.tensor([sg_i], device=dev))[0]
    k = k_init.clone().requires_grad_(relax_cell)

    # precompute per-atom element radii + molecule id + local coords (fixed, rigid)
    a_local, a_r, a_mid, a_ref, a_h = [], [], [], [], []
    for m in real:
        for a in range(atom_mask.shape[1]):
            if bool(atom_mask[m, a]):
                a_local.append(local[m, a]); a_r.append(_VDW.get(int(Z[m, a]), 1.70))
                a_mid.append(m); a_ref.append(ref_pos[ref_of[m]]); a_h.append(op_idx[m])
    a_local = torch.stack(a_local).to(dev)                 # (N,3)
    a_r = torch.tensor(a_r, dtype=torch.float32, device=dev)
    a_mid = torch.tensor(a_mid, device=dev)
    a_ref = torch.tensor(a_ref, device=dev)                # which DOF row each atom belongs to
    a_h = torch.tensor(a_h, device=dev)                    # which op generates each atom's copy
    Wsel, tsel = W[a_h], t[a_h]                            # (N,3,3),(N,3)

    def build():
        km = M.apply_family_mask(k.unsqueeze(0), torch.tensor([sg_i], device=dev))[0] \
            if family_mask else k
        L = M.k_to_lat(km.unsqueeze(0), n_dummy, "logmetric6")[0]
        Rcart = cartesian_rotations(sg_i, L).to(dev)       # (K,3,3), differentiable in L
        R0 = torch.bmm(M.so3_exp(delta), R0_init)          # (G,3,3)
        # per-atom copy centroid (fractional) and orientation
        c_ref = c0[a_ref]                                  # (N,3)
        c_cp = torch.einsum("nij,nj->ni", Wsel, c_ref) + tsel      # (N,3) frac
        c_cp = c_cp - torch.floor(c_cp)
        Rcp = torch.bmm(Rcart[a_h], R0[a_ref])             # (N,3,3)
        off = torch.einsum("nij,nj->ni", Rcp, a_local)     # (N,3) cartesian
        X = c_cp @ L + off                                 # (N,3) cartesian
        return L, X

    params = [c0, delta] + ([k] if relax_cell else [])
    opt = torch.optim.Adam(params, lr=lr)                  # one energy eval per step (LBFGS+
    V0 = vol_per_atom * float(a_local.shape[0])            # strong-wolfe was ~20x costlier)
    lnV0 = math.log(max(V0, 1.0))
    nbr = (None, None, None, None)
    nbr_every = 15
    for step in range(steps):
        if step % nbr_every == 0:                          # refresh neighbor list as atoms drift
            with torch.no_grad():
                L0, X0 = build()
            nbr = build_neighbor_list(X0, a_r, a_mid, L0, contact, cutoff)
        opt.zero_grad()
        L, X = build()
        e = packing_energy_nbr(X, L, *nbr, eps=eps, d_floor=d_floor, cutoff=cutoff)
        if relax_cell and lam_vol > 0:
            e = e + lam_vol * (torch.log(torch.det(L).abs().clamp_min(1e-3)) - lnV0) ** 2
        if not torch.isfinite(e):
            break                                          # degenerate cell -> keep last good pose
        e.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    with torch.no_grad():
        km = M.apply_family_mask(k.unsqueeze(0), torch.tensor([sg_i], device=dev))[0] \
            if family_mask else k
        L = M.k_to_lat(km.unsqueeze(0), n_dummy, "logmetric6")[0]
        Rcart = cartesian_rotations(sg_i, L).to(dev)
        R0 = torch.bmm(M.so3_exp(delta), R0_init)
        out_c = centroid.clone(); out_R = orient.clone()
        for m in real:
            g_row = ref_pos[ref_of[m]]; h = op_idx[m]
            cm = W[h] @ c0[g_row] + t[h]
            out_c[m] = cm - torch.floor(cm)
            out_R[m] = Rcart[h] @ R0[g_row]
        if not torch.isfinite(L).all():
            L = lattice
    return L, out_c, out_R
