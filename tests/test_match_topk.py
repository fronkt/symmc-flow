"""Best-of-k CSP match metric (match@k). Verifies that a reference matched only
by a later draw is counted as a hit at k>1 but missed at k=1 — i.e. the OR across
the k independent generations is plumbed correctly."""
import os
import sys
import importlib.util
import pytest
import torch

pytest.importorskip("pymatgen")

from symmc_flow.flow import CrystalState

# Load the training script as a module to reach match_rate_topk / match_rate.
_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_carbon24.py")
_spec = importlib.util.spec_from_file_location("train_carbon24", _SCRIPT)
tc = importlib.util.module_from_spec(_spec)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec.loader.exec_module(tc)


def _state(lattice, frac):
    """Single-structure CrystalState from a 3x3 lattice and (n,3) fractional coords."""
    n = frac.shape[0]
    mask = torch.ones(1, n, dtype=torch.bool)
    orient = torch.eye(3).expand(1, n, 3, 3).contiguous()
    return CrystalState(lattice.unsqueeze(0), frac.unsqueeze(0), orient, mask)


def test_match_topk_ors_across_draws():
    L = torch.eye(3) * 3.5
    frac = torch.tensor([[0.0, 0.0, 0.0],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.0, 0.5],
                         [0.0, 0.5, 0.0]])
    ref = _state(L, frac)

    exact = _state(L.clone(), frac.clone())                 # matches ref
    collapsed = _state(L.clone(), torch.zeros_like(frac))   # all atoms overlap -> miss

    # k=1 with only the miss draw -> 0%
    mr1, total = tc.match_rate_topk([collapsed], ref)
    assert total == 1 and mr1 == 0.0

    # best-of-3 where the matching candidate is the LAST draw -> hit
    mr3, total = tc.match_rate_topk([collapsed, collapsed, exact], ref)
    assert total == 1 and mr3 == 1.0

    # sanity: the exact copy matches under the one-shot metric too
    mr_one, _ = tc.match_rate(exact, ref)
    assert mr_one == 1.0
