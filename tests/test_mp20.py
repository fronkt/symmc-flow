"""MP-20 loader: multi-element per-atom Z + Z-aware StructureMatcher eval."""
import os
import sys
import pytest
import torch

pytest.importorskip("pymatgen")
pytest.importorskip("pandas")

from symmc_flow.mp20 import _to_item, MP20Dataset
from symmc_flow.data import collate, batch_to_state

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_to_item_keeps_per_atom_species():
    L = torch.eye(3) * 4.0
    frac = torch.rand(3, 3)
    Zatoms = torch.tensor([11, 17, 8])           # Na, Cl, O — multi-element
    item = _to_item(3, L, frac, Zatoms, 1, max_atoms=20)
    assert item["Z"].shape == (20, 1)
    assert (item["Z"][:3, 0] == Zatoms).all()    # real Z preserved (not hardcoded)
    assert item["mol_mask"].sum() == 3
    assert torch.allclose(item["centroid"][:3], frac)


def test_z_aware_structures_use_real_species():
    import train_mp20 as tm
    L = torch.eye(3).unsqueeze(0) * 4.0
    centroid = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]])
    orient = torch.eye(3).expand(1, 2, 3, 3).contiguous()
    mask = torch.ones(1, 2, dtype=torch.bool)
    from symmc_flow.flow import CrystalState
    state = CrystalState(L, centroid, orient, mask)
    Z = torch.tensor([[11, 17]])                 # NaCl
    structs = tm._to_structures(state, Z, mask)
    assert structs[0] is not None
    assert {str(s) for s in structs[0].species} == {"Na", "Cl"}
    # a NaCl-vs-NaCl identity match should score a hit under the topk metric
    mr, total = tm.match_rate_topk([state], state, Z, mask)
    assert total == 1 and mr == 1.0


def test_dataset_collate_roundtrip(tmp_path):
    csv = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "mp20_val.csv")
    if not os.path.exists(csv):
        pytest.skip("MP-20 CSV not present (GPU phase only)")
    ds = MP20Dataset(csv, max_atoms=20, cache_path=str(tmp_path / "mp20.pt"))
    assert len(ds) > 0
    batch = collate([ds[i] for i in range(min(4, len(ds)))])
    state = batch_to_state(batch)
    assert state.lattice.shape[1:] == (3, 3)
    assert (state.centroid[state.mask] >= 0).all() and (state.centroid[state.mask] < 1.0001).all()
    assert (batch["Z"][batch["atom_mask"]] > 0).all()   # real atomic numbers
