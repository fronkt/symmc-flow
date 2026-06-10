import torch
from symmc_flow.periodic_table import z_to_rcf, PeriodicTableEmbedding, _TABLE


def test_known_elements():
    z = torch.tensor([1, 6, 8, 26, 79])  # H, C, O, Fe, Au
    r, c, f = z_to_rcf(z)
    assert r.tolist() == [1, 2, 2, 4, 6]
    assert c.tolist() == [1, 14, 16, 8, 11]
    assert f.tolist() == [0, 0, 0, 0, 0]


def test_fblock_disambiguated():
    # Ce(58)..Lu(71) all sit in group 3 but get distinct f_index 1..14
    z = torch.arange(58, 72)
    _, c, f = z_to_rcf(z)
    assert (c == 3).all()
    assert f.tolist() == list(range(1, 15))


def test_full_coverage():
    assert set(_TABLE.keys()) == set(range(1, 119))


def test_padding_index_zero():
    z = torch.tensor([0, 0])
    r, c, f = z_to_rcf(z)
    assert r.tolist() == [0, 0] and c.tolist() == [0, 0]


def test_embedding_shape_and_pad():
    emb = PeriodicTableEmbedding(32)
    z = torch.tensor([[1, 6, 0], [8, 0, 0]])
    out = emb(z)
    assert out.shape == (2, 3, 32)
    # padding atom (Z=0) embeds to zero
    assert torch.allclose(out[0, 2], torch.zeros(32))
