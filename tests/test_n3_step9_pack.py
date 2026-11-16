"""The step-9 upload pack (scripts/n3_box_manifest.py --step 9): its input list comes from the frozen commands, it
carries the audit's implicit inputs, and its manifest gates (frozen code, blinding, the selection) are enforced.
File contents are hashed only by the real-tree test; nothing is packed or uploaded."""
import copy
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_box_manifest as M  # noqa: E402
import n3_step9 as S9  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(REPO, S9.SELECTION)), reason="no published freeze")


def test_inputs_from_the_frozen_commands():
    files, runs = M.step9_inputs()
    bundles = [f for f in files if f.startswith("results/n3/private/draws/")]
    assert len(bundles) == 96 and sum(f.endswith("_testB.pt") for f in bundles) == 48
    assert "results/n3/private/labels_testB.json" in files and "results/n3/private/labels_devtest.json" in files
    assert not any(os.path.basename(f).startswith("mcfRoal_") for f in files)          # made by (c) at step 9
    assert not any(f.endswith(".json") and "/mcf/" in f for f in files)                 # haar floor / (b): made
    assert len(runs) == 12 and all(r.startswith("n3_mcf/infer/mcfR/") for r in runs)
    assert sum(r.endswith("/out/predictions_16.pt") for r in runs) == 6


def test_groups_carry_the_implicit_inputs():
    g = {name: paths for name, _, _, paths in M.groups(9)}
    assert "data/n3_sets/val.pt" in g["set_caches_9"]                                    # anchor loads VAL
    assert {"n3_mcf/test/test_sidecar.json", "n3_mcf/devtest/test_molcrystal_normalized.pkl.gz"} <= set(g["mcf_exports_9"])
    assert {"external/mcf/data-preprocess/lattice_fit.py",
            "external/mcf/data-preprocess/thurlemann23/generate_features.py"} <= set(g["fork_9"])
    code = set(g["code_9"])
    assert set(M.STEP9_SCORING_CODE) <= code and set(M.STEP9_NEW) <= code
    assert {"scripts/n3_box_cleanup.sh", "symmc_flow/molcrystal.py", "scripts/n3_mcf_run.py"} <= code
    assert "results/n3/n3_selection.json" in g["frozen_lists_9"]


def gated(man):
    m = copy.deepcopy(man)
    m["checks"].update(n3_selection_complete=True, n3_selection_tracked=True, n3_selection_unchanged=True,
                       head_on_remote=True, scoring_code_as_frozen=True, blinded_outputs_packed=[])
    return m


def test_manifest_gates(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    base = {"format": M.FORMAT, "label": "step9", "step": 9, "deployable": True, "problems": [], "missing": [],
            "files": [{"group": "g", "csd": False, "path": "a.txt", "bytes": 1, "sha256": M.sha256(str(f))}],
            "checks": {}}
    assert M.validate_manifest(gated(base), str(tmp_path))
    for k, v in (("scoring_code_as_frozen", False), ("blinded_outputs_packed", ["x"]), ("head_on_remote", False)):
        bad = gated(base)
        bad["checks"][k] = v
        with pytest.raises(ValueError):
            M.validate_manifest(bad, str(tmp_path))


def test_checks_flag_blinded_and_changed(tmp_path, monkeypatch):
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    json.dump({"files": [{"path": "results/n3/private/draws/ours_s0_testB.pt", "sha256": "a" * 64}]},
              open(mdir / "pullback_step7.manifest.json", "w"))
    monkeypatch.setenv("N3_MANIFEST_DIR", str(mdir))
    files = [{"group": "inputs_9", "csd": True, "path": "results/n3/private/draws/ours_s0_testB.pt", "bytes": 1,
              "sha256": "b" * 64},
             {"group": "inputs_9", "csd": True, "path": "results/n3/private/match/iii-u_s0_testB__primary.jsonl",
              "bytes": 1, "sha256": "c" * 64},
             {"group": "inputs_9", "csd": True, "path": "results/n3/report_devtest.out.json", "bytes": 1,
              "sha256": "d" * 64}]
    problems, checks = M.step9_checks(files)
    assert checks["csd_files_changed_since_receipt"] == ["results/n3/private/draws/ours_s0_testB.pt"]
    assert checks["blinded_outputs_packed"] == ["results/n3/private/match/iii-u_s0_testB__primary.jsonl",
                                                "results/n3/report_devtest.out.json"]
    assert any("receipts" in p for p in problems) and any("would be packed" in p for p in problems)


def test_pullback_skips_git():
    assert ".git" in M.PULL_SKIP and "__pycache__" in M.PULL_SKIP
