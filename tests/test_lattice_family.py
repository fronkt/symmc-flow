"""Phase F: log-metric lattice repr + crystal-family masking.

Verifies the math (orthonormal basis, metric-preserving round-trip, volume coordinate),
the per-family masks (reconstructed cells obey their crystal-system constraints exactly),
the frozen-DOF velocity/prior handling, and that nothing regresses the shape10 default.
"""
import math
import torch

from symmc_flow import manifolds as M
from symmc_flow.space_group import family_of, family_name
from symmc_flow.config import ModelConfig
from symmc_flow.model import SymMCFlow
from symmc_flow.flow import CrystalState, sample_prior, interpolate
from symmc_flow.data import SyntheticCrystalDataset, collate, batch_to_state
from symmc_flow.sampler import rk4_sample


# ---- helpers ---------------------------------------------------------------
def L_from_params(a, b, c, al, be, ga, dtype=torch.float64):
    al, be, ga = (math.radians(x) for x in (al, be, ga))
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 1e-9))
    return torch.tensor([[a, 0.0, 0.0],
                         [b * math.cos(ga), b * math.sin(ga), 0.0],
                         [cx, cy, cz]], dtype=dtype)


def cell_params(L):
    G = L @ L.transpose(-1, -2)
    a, b, c = torch.sqrt(torch.diag(G))
    al = torch.rad2deg(torch.arccos(G[1, 2] / (b * c)))
    be = torch.rad2deg(torch.arccos(G[0, 2] / (a * c)))
    ga = torch.rad2deg(torch.arccos(G[0, 1] / (a * b)))
    return [float(x) for x in (a, b, c, al, be, ga)]


# representative space group per crystal system
SG = {"monoclinic": 14, "orthorhombic": 19, "tetragonal": 76,
      "hexagonal": 168, "trigonal": 143, "cubic": 195}


# ---- math ------------------------------------------------------------------
def test_sym_basis_orthonormal():
    B = M.SYM_BASIS.double()
    gram = torch.einsum("aij,bij->ab", B, B)
    assert torch.allclose(gram, torch.eye(6, dtype=torch.float64), atol=1e-12)


def test_logmetric_roundtrip_preserves_metric():
    for p in [(5, 7, 9, 90, 110, 90), (8, 8, 12, 90, 90, 120), (6, 6, 6, 70, 70, 70),
              (5, 6, 7, 80, 95, 100), (4, 4, 4, 90, 90, 90)]:
        L = L_from_params(*p)
        k = M.lattice_to_logmetric(L)
        L2 = M.logmetric_to_lattice(k)
        g1, g2 = L @ L.T, L2 @ L2.T
        assert torch.allclose(g1, g2, atol=1e-8), p
        # cell parameters recovered (frame-invariant)
        assert max(abs(x - y) for x, y in zip(cell_params(L), cell_params(L2))) < 1e-4, p


def test_volume_is_k0_coordinate():
    L = L_from_params(5, 7, 9, 90, 110, 90)
    k = M.lattice_to_logmetric(L)
    V = float(torch.det(L).abs())
    assert abs(math.exp(math.sqrt(3) * float(k[0]) / 2) - V) < 1e-6 * V


def test_family_of_ranges():
    assert family_of(1) == 0 and family_of(2) == 0
    assert family_of(14) == 1 and family_of(15) == 1
    assert family_of(19) == 2 and family_of(74) == 2
    assert family_of(76) == 3 and family_of(142) == 3
    assert family_of(143) == 4 and family_of(168) == 5
    assert family_of(195) == 6 and family_of(230) == 6


def test_canon_hex_value_and_a_independence():
    assert abs(M.CANON_HEX - (-math.log(3.0) / math.sqrt(2.0))) < 1e-12
    k1 = M.lattice_to_logmetric(L_from_params(5, 5, 9, 90, 90, 120))
    k2 = M.lattice_to_logmetric(L_from_params(8, 8, 3, 90, 90, 120))
    assert abs(float(k1[5]) - M.CANON_HEX) < 1e-4
    assert abs(float(k1[5]) - float(k2[5])) < 1e-6   # a-independent


# ---- family masks ----------------------------------------------------------
def _mask_reconstruct(sg):
    """Mask a GENERIC triclinic cell into sg's family, return reconstructed cell params."""
    k = M.lattice_to_logmetric(L_from_params(5, 6, 7, 80, 95, 100))
    km = M.apply_family_mask(k.unsqueeze(0), torch.tensor([sg]))[0]
    return cell_params(M.logmetric_to_lattice(km))


def test_mask_monoclinic():
    a, b, c, al, be, ga = _mask_reconstruct(SG["monoclinic"])
    assert abs(al - 90) < 1e-2 and abs(ga - 90) < 1e-2 and abs(be - 90) > 1.0


def test_mask_orthorhombic():
    a, b, c, al, be, ga = _mask_reconstruct(SG["orthorhombic"])
    assert all(abs(x - 90) < 1e-2 for x in (al, be, ga))


def test_mask_tetragonal():
    a, b, c, al, be, ga = _mask_reconstruct(SG["tetragonal"])
    assert abs(a - b) < 1e-3 and all(abs(x - 90) < 1e-2 for x in (al, be, ga))


def test_mask_hexagonal():
    a, b, c, al, be, ga = _mask_reconstruct(SG["hexagonal"])
    assert abs(a - b) < 1e-3 and abs(al - 90) < 1e-2 and abs(be - 90) < 1e-2
    assert abs(ga - 120) < 1e-2


def test_mask_trigonal_uses_hex_setting():
    a, b, c, al, be, ga = _mask_reconstruct(SG["trigonal"])
    assert abs(a - b) < 1e-3 and abs(ga - 120) < 1e-2


def test_mask_cubic():
    a, b, c, al, be, ga = _mask_reconstruct(SG["cubic"])
    assert abs(a - b) < 1e-3 and abs(b - c) < 1e-3
    assert all(abs(x - 90) < 1e-2 for x in (al, be, ga))


def test_apply_family_mask_idempotent():
    k = M.lattice_to_logmetric(L_from_params(5, 6, 7, 80, 95, 100)).unsqueeze(0)
    sg = torch.tensor([SG["orthorhombic"]])
    k1 = M.apply_family_mask(k, sg)
    k2 = M.apply_family_mask(k1, sg)
    assert torch.allclose(k1, k2, atol=1e-7)


def test_mask_velocity_zeros_frozen_dims():
    v = torch.ones(1, 6)
    vm = M.mask_velocity(v, torch.tensor([SG["orthorhombic"]]))
    # orthorhombic frees dims 0,1,2 ; freezes 3,4,5
    assert torch.allclose(vm[0, :3], torch.ones(3))
    assert torch.allclose(vm[0, 3:], torch.zeros(3))


# ---- model wiring / backward compat ---------------------------------------
def test_shape10_default_head_dim():
    m = SymMCFlow(ModelConfig(d_model=32, egnn_hidden=32, atom_embed_dim=16,
                              n_attn_layers=1, egnn_layers=1, n_heads=4))
    assert m.lattice_in.in_features == 10
    assert m.head_lattice[-1].out_features == 10


def test_logmetric6_head_dim():
    m = SymMCFlow(ModelConfig(d_model=32, egnn_hidden=32, atom_embed_dim=16,
                              n_attn_layers=1, egnn_layers=1, n_heads=4,
                              lattice_repr="logmetric6"))
    assert m.lattice_in.in_features == 6
    assert m.head_lattice[-1].out_features == 6


def test_interpolate_logmetric_freezes_velocity():
    n = 3
    z0 = CrystalState(L_from_params(5, 5, 9, 90, 90, 90, dtype=torch.float32).unsqueeze(0),
                      torch.rand(1, n, 3), torch.eye(3).expand(1, n, 3, 3).clone(),
                      torch.ones(1, n, dtype=torch.bool))
    z1 = CrystalState(L_from_params(6, 7, 8, 80, 95, 100, dtype=torch.float32).unsqueeze(0),
                      torch.rand(1, n, 3), torch.eye(3).expand(1, n, 3, 3).clone(),
                      torch.ones(1, n, dtype=torch.bool))
    sg = torch.tensor([SG["orthorhombic"]])
    _, (u_L, _, _) = interpolate(z0, z1, torch.tensor([0.5]), sg=sg,
                                 lattice_repr="logmetric6", family_mask=True)
    assert u_L.shape[-1] == 6
    assert torch.allclose(u_L[0, 3:], torch.zeros(3), atol=1e-5)   # frozen dims: no target


def test_sampler_stays_on_family():
    cfg = ModelConfig(d_model=48, egnn_hidden=48, atom_embed_dim=32, n_attn_layers=2,
                      egnn_layers=2, n_heads=4, lattice_repr="logmetric6",
                      lattice_family_mask=True)
    model = SymMCFlow(cfg).eval()
    ds = SyntheticCrystalDataset(4, max_mols=4, max_atoms=12, seed=5)
    batch = collate([ds[i] for i in range(4)])
    fams = ["orthorhombic", "tetragonal", "hexagonal", "cubic"]
    batch["sg"] = torch.tensor([SG[f] for f in fams])
    z1 = batch_to_state(batch)
    z0 = sample_prior(z1, sg=batch["sg"], lattice_repr="logmetric6", family_mask=True)
    mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
    out = rk4_sample(model, mol_emb, z0, batch["sg"], steps=30)
    for b, fam in enumerate(fams):
        a, bb, c, al, be, ga = cell_params(out.lattice[b].double())
        assert all(abs(x - 90) < 1.0 for x in (al, be)), (fam, al, be)
        if fam == "hexagonal":
            assert abs(ga - 120) < 1.0 and abs(a - bb) < 1e-2, (fam, ga, a, bb)
        else:
            assert abs(ga - 90) < 1.0, (fam, ga)
        if fam in ("tetragonal", "hexagonal", "cubic"):
            assert abs(a - bb) < 1e-2, (fam, a, bb)
        if fam == "cubic":
            assert abs(bb - c) < 1e-2, (fam, bb, c)
