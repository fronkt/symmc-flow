import torch
from symmc_flow.pair_bias_attention import PairBiasStack, PairBiasAttention


def test_shapes_and_grad():
    B, N, D, P = 3, 7, 32, 4
    stack = PairBiasStack(D, n_heads=4, pair_dim=P, n_layers=2)
    x = torch.randn(B, N, D, requires_grad=True)
    pair = torch.randn(B, N, N, P)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 5:] = False
    out = stack(x, pair, mask)
    assert out.shape == (B, N, D)
    out.sum().backward()
    assert x.grad is not None
    # masked tokens zeroed
    assert torch.allclose(out[:, 5:], torch.zeros(B, 2, D), atol=1e-6)


def test_mask_blocks_attention():
    torch.manual_seed(0)
    B, N, D, P = 1, 4, 16, 4
    attn = PairBiasAttention(D, 2, P).eval()
    x = torch.randn(B, N, D)
    pair = torch.zeros(B, N, N, P)
    mask = torch.tensor([[True, True, False, False]])
    out_a = attn(x, pair, mask)
    # change the masked-out tokens; attended output must not move
    x2 = x.clone()
    x2[:, 2:] += 5.0
    out_b = attn(x2, pair, mask)
    assert torch.allclose(out_a[:, :2], out_b[:, :2], atol=1e-5)
