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


REPORT = {"table1": {"OURS": {"picked1_mean_pct": 1.0}}, "selection": {"sha256": R.FREEZE_SHA256},
          "missing_arms": {}}
# the frozen n3_score.py report makes the H1-H3 families (and the wording) on TEST-B only
FAMILIES = {"family_primary": {"complete": True}, "family_n2": {"complete": True}, "wording": {}}


def tree(tmp_path, monkeypatch, report_extra=None, skip_tsv=(), dev_extra=None):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "results/n3"), exist_ok=True)
    shutil.copyfile(os.path.join(REPO, R.SELECTION), os.path.join(root, R.SELECTION))
    for st in R.SETS:
        put(root, f"results/n3/report_{st}.out.json",
            {"set": st, **REPORT, **(FAMILIES if st == "testB" else (dev_extra or {})), **(report_extra or {})})
        put(root, f"{R.P}/classical/anchor_{st}.json", {"summary": {"n": 2, "anchoring": 1.0, "ff_state": "x"},
                                                        "rows": [{"refcode": "A"}]})
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
        outs = [f"results/n3/report_{st}.out.json", f"{R.P}/classical/anchor_{st}.json",
                f"{R.P}/mcf/diag_b_{st}.json", f"{R.P}/mcf/diag_c_{st}.json", f"{R.P}/mcf/haar_floor_{st}.json"] + \
            [f"{R.P}/mcf/diag_d_{st}_{m}.json" for m in R.MATCHERS_BD]
        put(root, f"{R.P}/step9/{st}_checksums.tsv",
            "".join(f"{R.sha256(os.path.join(root, o))}\t{o}\n" for o in outs if o not in skip_tsv))
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
    assert t["haar_floor"]["crystals"] == 2 and t["provenance"]["output_files"] == 8
    assert "per_crystal" not in json.dumps(t["mcf_diagnostics"]) and '"rows"' not in json.dumps(t["ff_truth_anchoring"])
    m = t["mcf_diagnostics"]
    assert m["d_consistent_all"] is True and m["approximate"] is True and "approximate" in m["c"]["label"]
    assert t["ff_truth_anchoring"]["anchoring"] == 1.0 and "ff_state" not in t["ff_truth_anchoring"]


@pytest.mark.parametrize("extra", [{"x": {"cart": [1.0]}},
                                   {"x": "data_ABC\n_cell_length_a 5.0"},
                                   {"x": [[0.1, 0.2, 0.3]] * 4},
                                   {"x": [[0.1, 0.2, 0.3], [1, 2, 3]]},
                                   {"x": {"Lattice_1": [5.0, 6.0]}},
                                   {"x": {"gt_coords": []}},
                                   {"x": "C 0.10 0.20 0.30\nH 1.10 0.20 0.30\nH 0.10 1.20 0.30\n"}])
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


def test_hygiene_allows_statistics():
    assert R.hygiene({"R": 6e-7, "lattice": 1e-13, "h": [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                      "per_seed": [12.5, 33.3, 50.0], "ci95": [[0.1, 0.2], [0.3, 0.4]],
                      "det_fail_list": [["ABCDEF", 0, 3]]}) == []


def test_diag_a_failed_withholds_b_to_d():
    assert R.mcf_block("testB", {"verdict": "FAILED"}, {})["reported"] is False


@pytest.mark.parametrize("extra,match", [({"missing_arms": {"iv-P": "no rows"}}, "missing arms"),
                                         ({"family_primary": {"complete": False}}, "not complete"),
                                         ({"selection": {"sha256": "0" * 64}}, "another freeze")])
def test_incomplete_or_foreign_report_stops(tmp_path, monkeypatch, extra, match):
    tree(tmp_path, monkeypatch, extra)
    with pytest.raises(SystemExit, match=match):
        R.assemble()


def test_devtest_has_no_families(tmp_path, monkeypatch):
    tree(tmp_path, monkeypatch)
    d = R.assemble()["sets"]["devtest"]
    assert "family_primary" not in d["report"] and d["families"].startswith("none")


def test_devtest_report_with_families_stops(tmp_path, monkeypatch):
    tree(tmp_path, monkeypatch, dev_extra=FAMILIES)
    with pytest.raises(SystemExit, match="TEST-B only"):
        R.assemble()


def test_file_not_in_the_box_checksums_stops(tmp_path, monkeypatch):
    tree(tmp_path, monkeypatch, skip_tsv=(f"{R.P}/mcf/diag_c_testB.json",))
    with pytest.raises(SystemExit, match="checksum list"):
        R.assemble()


def test_changed_after_the_box_stops(tmp_path, monkeypatch):
    root = tree(tmp_path, monkeypatch)
    put(root, "results/n3/report_devtest.out.json", {"set": "devtest", **REPORT, "edited": True})
    with pytest.raises(SystemExit, match="checksum list"):
        R.assemble()


def test_restarts_are_listed(tmp_path, monkeypatch):
    root = tree(tmp_path, monkeypatch)
    put(root, f"{R.P}/step9/events_testB.jsonl", "\n".join(json.dumps(
        {"event": e, "unit": "u", "command": 0, "time": "t", "slurm_job": j})
        for e, j in (("first_start", "1"), ("start", "1"), ("restart", "2"), ("resume", "2"), ("done", "2"),
                     ("set_done", "2"))) + "\n")
    p = R.assemble()["sets"]["testB"]["provenance"]
    assert [x["event"] for x in p["stops"]] == ["restart", "resume"] and p["slurm_jobs"] == ["1", "2"]
