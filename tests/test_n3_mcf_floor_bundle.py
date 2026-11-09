"""MCF-A prior floor -> to_bundle (2026-11-09 Anvil step-7 smoke failure, tasks/n3_step7_codefix_2026-11-09.md).
Amendment A3 (a25a444) made to_bundle read chk["info"] for every arm, but run_check_floor returned no "info".
Two parts:
  - hermetic: run_check_floor returns and writes "info";
  - end to end: to_bundle on a tmp_path copy of the 09-28 laptop floor smoke run, skipped where it is absent.
Plus a check that the step-7 pack re-ships the fixed script. Nothing is written outside tmp_path."""
import json
import os
import shutil
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_mcf_run as R  # noqa: E402
import n3_mcf_lib as ML  # noqa: E402

LAPTOP_FLOOR_SMOKE = os.path.join(REPO, "n3_mcf", "_smoke", "infer", "mcfA_floor", "smoke", "s0")


def test_run_check_floor_returns_info(tmp_path, monkeypatch):
    run = tmp_path / "s0"
    (run / "out").mkdir(parents=True)
    json.dump({"arm": "floor", "set": "fake", "num_samples": 2, "pickle_sha256": "abc"},
              open(run / "n3_run.json", "w"))
    torch.save({"num_atoms": torch.tensor([[3, 4], [3, 4]])}, run / "out" / "predictions_2.pt")
    entries = [{"local_coords": torch.zeros(3, 3)}, {"local_coords": torch.zeros(4, 3)}]
    monkeypatch.setattr(ML, "export_set", lambda set_name: (entries, {"sha256": "abc"}))
    res = R.run_check_floor(str(run))
    assert res["pass"] and res["info"] == {}
    assert json.load(open(run / "check.json"))["info"] == {}


@pytest.mark.skipif(not os.path.isfile(os.path.join(LAPTOP_FLOOR_SMOKE, "n3_run.json")),
                    reason="no laptop floor smoke run (CSD-derived, local only)")
def test_floor_to_bundle_end_to_end(tmp_path):
    run = tmp_path / "s0"
    shutil.copytree(LAPTOP_FLOOR_SMOKE, run)
    out = tmp_path / "mcfAfloor_s0_val2.pt"
    args = type("A", (), {"run_dir": str(run), "arm": None, "out": str(out), "workers": 1})()
    R.cmd_to_bundle(args)
    b = torch.load(out, map_location="cpu", weights_only=False)
    assert b["arm"] == "mcfAfloor"
    assert b["meta"]["arm_failures"]["resid_nonfinite_on_finite_draws"] == 0
    assert b["meta"]["check"]["pass"] is True and b["meta"]["check"]["resid_sha256"] is None
    assert torch.isnan(b["sel"]["resid"]).all()


def test_step7_pack_reships_the_fixed_script():
    import n3_box_manifest as M
    g = {name: (csd, required, paths) for name, csd, required, paths in M.groups(7)}
    assert g["code_fix_7"] == (False, True, ["scripts/n3_mcf_run.py"])
