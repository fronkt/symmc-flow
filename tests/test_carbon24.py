import os
import pytest
import torch

pytest.importorskip("pymatgen")
pytest.importorskip("pandas")

from symmc_flow.carbon24 import _to_item, Carbon24Dataset
from symmc_flow.data import collate, batch_to_state


def test_to_item_shapes():
    L = torch.eye(3) * 3.0
    frac = torch.rand(5, 3)
    item = _to_item(5, L, frac, 1, max_mols=24)
    assert item["Z"].shape == (24, 1)
    assert item["mol_mask"].sum() == 5
    assert (item["Z"][:5, 0] == 6).all()
    assert torch.allclose(item["centroid"][:5], frac)
    # padded molecules carry identity orientation
    assert torch.allclose(item["orient"][0], torch.eye(3))


def test_dataset_collate_roundtrip(tmp_path):
    csv = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "carbon_val.csv")
    if not os.path.exists(csv):
        pytest.skip("carbon-24 CSV not present (GPU phase only)")
    ds = Carbon24Dataset(csv, max_mols=24, cache_path=str(tmp_path / "c.pt"))
    assert len(ds) > 0
    batch = collate([ds[i] for i in range(min(4, len(ds)))])
    state = batch_to_state(batch)
    assert state.lattice.shape[1:] == (3, 3)
    assert (state.centroid[state.mask] >= 0).all() and (state.centroid[state.mask] < 1.0001).all()
