"""Diagnostic d must consume current score rows, with no new fitting in this test."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
import n3_common as C
import n3_mcf_diag as D
import n3_score as SC


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "MATCH_DIR", str(tmp_path / "match"))
    monkeypatch.setattr(C, "load_set", lambda name: [{"refcode": "VALTST", "value": torch.zeros(1)}])
    path = str(tmp_path / "mcfR_s0_val.pt")
    bundle = {"arm": "mcfR", "kind": "rasym", "set": "val", "seed": 0, "refcodes": ["VALTST"],
              "R": torch.eye(3).repeat(1, 2, 1, 1), "valid": torch.ones(1, 2, dtype=torch.bool),
              "sel": {"lj": torch.zeros(1, 2)}}
    torch.save(bundle, path)
    SC._meta_check(path, "primary", {"selection": None, "step9": False,
                                     "scoring_identity": SC.scoring_identity("val")})
    with open(SC.match_path(path, "primary"), "w") as f:
        for j in range(2):
            f.write(json.dumps({"i": 0, "j": j, "ok": bool(j)}) + "\n")
    ceiling = str(tmp_path / "ceiling.json")
    with open(ceiling, "w") as f:
        json.dump({"tag": "val", "per_crystal": {"VALTST": {"z_MCF": 2}},
                   "summary": {"primary": {"z_MCF=2": {"mean_c_draw1_ceiling": .6}}}}, f)
    source = str(tmp_path / "export.pt")
    with open(source, "wb") as f:
        f.write(b"synthetic export identity")
    with open(ceiling + ".meta.json", "w") as f:
        json.dump({"sha256": C.sha256(ceiling), "identity": D.ceiling_identity("val"),
                   "selection_sha256": None, "artifacts": {source: C.sha256(source)}}, f)
    return SimpleNamespace(ceiling=ceiling, bundles=[path], matcher="primary", n_boot=50,
                           out=str(tmp_path / "diagnostic.json")), bundle


def test_diagnostic_d_accepts_current_rows(diagnostic):
    args, _ = diagnostic
    D.cmd_d(args)
    result = json.load(open(args.out))
    assert result["strata"]["z_MCF=2"]["draw1"] == .5
    assert result["consistent"]


def test_diagnostic_d_refuses_rows_from_changed_bundle(diagnostic):
    args, bundle = diagnostic
    bundle["valid"][0, 0] = False
    torch.save(bundle, args.bundles[0])
    with pytest.raises(SystemExit, match="changed since"):
        D.cmd_d(args)
    assert not os.path.exists(args.out)


def test_diagnostic_d_refuses_changed_ceiling(diagnostic):
    args, _ = diagnostic
    with open(args.ceiling, "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit, match="stale ceiling"):
        D.cmd_d(args)


def test_diagnostic_d_refuses_changed_ceiling_source(diagnostic):
    args, _ = diagnostic
    meta = json.load(open(args.ceiling + ".meta.json"))
    with open(next(iter(meta["artifacts"])), "ab") as f:
        f.write(b"changed")
    with pytest.raises(SystemExit, match="source artifact changed"):
        D.cmd_d(args)


def test_diagnostic_d_refuses_changed_rigid_structure_adapter(diagnostic, monkeypatch):
    args, _ = diagnostic
    original_sha256 = C.sha256

    def changed_sha256(path):
        if str(path).replace("\\", "/").endswith("/symmc_flow/molcrystal.py"):
            return "0" * 64
        return original_sha256(path)

    monkeypatch.setattr(C, "sha256", changed_sha256)
    with pytest.raises(SystemExit, match="stale ceiling"):
        D.cmd_d(args)
    assert not os.path.exists(args.out)


def test_haar_floor_provenance_checks_source_artifacts(diagnostic, tmp_path):
    args, _ = diagnostic
    floor = str(tmp_path / "floor.json")
    with open(floor, "w") as f:
        json.dump({"VALTST": {"primary": .5, "mcf05": .5, "mcf08": .5}}, f)
    meta = {"set": "val", "sha256": C.sha256(floor), "scoring_identity": SC.scoring_identity("val"),
            "seeds": [0, 1, 2], "draws_per_seed": 16, "selection": None,
            "bundles": {args.bundles[0]: C.sha256(args.bundles[0])},
            "match_rows": {SC.match_path(args.bundles[0], "primary"): C.sha256(SC.match_path(args.bundles[0], "primary"))}}
    with open(floor + ".meta.json", "w") as f:
        json.dump(meta, f)
    assert D.checked_haar(floor, "val", None)["VALTST"]["primary"] == .5
    with open(args.bundles[0], "ab") as f:
        f.write(b"changed")
    with pytest.raises(SystemExit, match="source artifact changed"):
        D.checked_haar(floor, "val", None)
