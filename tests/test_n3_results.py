"""results/n3/n3_results.json assembly (scripts/n3_results.py) on a synthetic step-9 tree in tmp_path (the real,
public freeze is copied in; every step-9 output is a made-up stand-in). Nothing is scored."""
import json
import os
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_results as R  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(REPO, R.SELECTION)), reason="no published freeze")
ENV = {"python": "3.12.15", "libraries": {"torch": "2.11.0+cu128"}}


def put(root, rel, obj):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if isinstance(obj, str):
        open(p, "w").write(obj)
    else:
        json.dump(obj, open(p, "w"))


def tree(tmp_path, monkeypatch, report_extra=None):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "results/n3"), exist_ok=True)
    shutil.copyfile(os.path.join(REPO, R.SELECTION), os.path.join(root, R.SELECTION))
    for st in R.SETS:
        put(root, f"results/n3/report_{st}.out.json", {"set": st, "table1": {"OURS": {"picked1_mean_pct": 1.0}},
                                                       **(report_extra or {})})
        put(root, f"{R.P}/mcf/diag_b_{st}.json", {"diagnostic": "b", "summary": {"primary": {}}, "per_crystal": {}})
        put(root, f"{R.P}/mcf/diag_c_{st}.json", {"diagnostic": "c", "z_MCF = 1": {"n": 3}, "matchers": {},
                                                  "selector": "lj"})
        for m in R.MATCHERS_BD:
            put(root, f"{R.P}/mcf/diag_d_{st}_{m}.json", {"strata": {}, "consistent": True})
        put(root, f"{R.P}/mcf/haar_floor_{st}.json", {"A": {"primary": 0.1}, "B": {"primary": 0.2}})
        put(root, f"{R.P}/step9/started_{st}.json", {"environment": ENV, "host": "g001",
                                                     "selection_sha256": R.FREEZE_SHA256})
        put(root, f"{R.P}/step9/events_{st}.jsonl",
            "\n".join(json.dumps({"event": e, "unit": "-", "command": None, "time": "t", "slurm_job": "1"})
                      for e in ("first_start", "start", "done", "set_done")) + "\n")
        put(root, f"{R.P}/step9/{st}_checksums.tsv", "a\tx\nb\ty\n")
    monkeypatch.setattr(R, "REPO", root)
    return root


def test_assemble(tmp_path, monkeypatch):
    tree(tmp_path, monkeypatch)
    res = R.assemble()
    assert res["selection"]["sha256"] == R.FREEZE_SHA256 and res["h2_form"]["form"] == "EXACT"
    assert res["mcf_diagnostic_a_passed"] is True and res["one_scoring_environment"] is True
    t, d = res["sets"]["testB"], res["sets"]["devtest"]
    assert t["role"].startswith("confirmatory") and d["role"] == R.DEV_LABEL
    assert t["report"]["table1"]["OURS"]["picked1_mean_pct"] == 1.0
    assert t["haar_floor"]["crystals"] == 2 and t["provenance"]["output_files"] == 2
    assert "per_crystal" not in json.dumps(t["mcf_diagnostics"])
    assert t["mcf_diagnostics"]["d_consistent_all"] is True


@pytest.mark.parametrize("extra", [{"x": {"cart": [1.0]}},
                                   {"x": "data_ABC\n_cell_length_a 5.0"},
                                   {"x": [[0.1, 0.2, 0.3]] * 4}])
def test_hygiene_stops(tmp_path, monkeypatch, extra):
    tree(tmp_path, monkeypatch, extra)
    with pytest.raises(SystemExit, match="hygiene"):
        R.assemble()


def test_missing_output_stops(tmp_path, monkeypatch):
    root = tree(tmp_path, monkeypatch)
    os.remove(os.path.join(root, R.P, "mcf", "diag_c_devtest.json"))
    with pytest.raises(SystemExit, match="missing"):
        R.assemble()


def test_unfinished_set_stops(tmp_path, monkeypatch):
    root = tree(tmp_path, monkeypatch)
    put(root, f"{R.P}/step9/events_testB.jsonl", json.dumps({"event": "start", "unit": "u", "command": 0,
                                                             "time": "t"}) + "\n")
    with pytest.raises(SystemExit, match="set_done"):
        R.assemble()


def test_dry_run_report_refused(tmp_path, monkeypatch):
    tree(tmp_path, monkeypatch, {"dry_run": "DRY RUN"})
    with pytest.raises(SystemExit, match="confirmatory"):
        R.assemble()


def test_diag_a_failed_withholds_b_to_d():
    assert R.mcf_block("testB", False)["reported"] is False
