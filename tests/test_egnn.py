import torch
from symmc_flow.egnn import EGNNEncoder
from symmc_flow import manifolds as M


def test_shapes_and_pool():
    enc = EGNNEncoder(in_dim=16, hidden=32, layers=3)
    B, N, F = 4, 6, 16
    feats = torch.randn(B, N, F)
    coords = torch.randn(B, N, 3)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 4:] = False
    h, pooled = enc(feats, coords, mask)
    assert h.shape == (B, N, 32) and pooled.shape == (B, 32)
    # padded atoms produce zero rows
    assert torch.allclose(h[:, 4:], torch.zeros(B, 2, 32), atol=1e-6)


def test_invariance_to_rotation_translation():
    torch.manual_seed(0)
    enc = EGNNEncoder(in_dim=8, hidden=24, layers=3).eval()
    B, N = 2, 5
    feats = torch.randn(B, N, 8)
    coords = torch.randn(B, N, 3)
    mask = torch.ones(B, N, dtype=torch.bool)
    _, pooled = enc(feats, coords, mask)

    Rg = M.random_so3((1,))[0]
    tg = torch.randn(3)
    coords2 = coords @ Rg.T + tg
    _, pooled2 = enc(feats, coords2, mask)
    assert torch.allclose(pooled, pooled2, atol=1e-4)
