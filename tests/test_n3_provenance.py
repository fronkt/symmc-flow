"""Scoring provenance and frozen scientific choices, using only synthetic existing rows."""
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
import n3_score as S


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    items = [{"refcode": "SYN001", "K": 1}, {"refcode": "SYN002", "K": 2}]
    monkeypatch.setattr(S.C, "load_set", lambda name: items)
    monkeypatch.setattr(S, "MATCH_DIR", str(tmp_path / "match"))
    monkeypatch.setattr(S, "_IDENTITIES", {})

    def bundle(name, arm="ours", seed=0, kind="rasym", draws=2, valid=None, conv=None):
        path = str(tmp_path / (name + ".pt"))
        b = {"arm": arm, "seed": seed, "set": "val", "kind": kind,
             "refcodes": [a["refcode"] for a in items], "valid": torch.ones(2, draws, dtype=torch.bool),
             "sel": {k: torch.arange(draws).repeat(2, 1).double() for k in ("lj", "ff", "resid")}, "meta": {}}
        if valid is not None:
            b["valid"] = torch.tensor(valid, dtype=torch.bool)
        if conv is not None:
            b["meta"]["converged"] = torch.tensor(conv)
        if kind == "rasym":
            b["R"] = torch.eye(3).expand(2, draws, 3, 3).clone().double()
        else:
            c = {"lattice": torch.eye(3).double(), "cart": torch.zeros(1, 3).double(),
                 "species": torch.tensor([6]), "mol": torch.tensor([0])}
            b["cells"] = [[copy.deepcopy(c) for _ in range(draws)] for _ in range(2)]
        torch.save(b, path)
        return path

    def rows(path, matcher="primary", hits=((0, 0),), reference=None):
        b = S.C.load_bundle(path)
        ref = S.ref_info(reference) if reference else None
        key = S.mkey(matcher, ref)
        extra = {"scoring_identity": S.scoring_identity("val"), "selection": None, "step9": False}
        if ref:
            extra.update(reference_sha256=ref["sha256"], reference_order=ref["order"])
        S._meta_check(path, key, extra)
        with open(S.match_path(path, key), "w") as f:
            for i in range(2):
                for j in range(b["valid"].shape[1]):
                    invalid = ref is not None and not S.ref_ok(ref, i)
                    json.dump({"i": i, "j": j, "ok": (i, j) in hits and not invalid,
                               "rms": None, "status": "ref_invalid" if invalid else "ok"}, f)
                    f.write("\n")
        return key
    return items, bundle, rows


def test_bundle_duplicates_mixed_arms_and_wrong_order_fail(synthetic):
    _, make, _ = synthetic
    b = S.C.load_bundle(make("s0"))
    with pytest.raises(SystemExit, match="unique"):
        S.validate_bundles([b, b])
    other = copy.deepcopy(b)
    other.update(arm="iii-u", seed=1)
    with pytest.raises(SystemExit, match="one arm"):
        S.validate_bundles([b, other])
    other = copy.deepcopy(b)
    other["refcodes"].reverse()
    with pytest.raises(SystemExit, match="frozen"):
        S.validate_bundles([other], allow_prefix=True)
    prefix = copy.deepcopy(b)
    prefix["refcodes"] = prefix["refcodes"][:1]
    prefix["valid"] = prefix["valid"][:1]
    prefix["R"] = prefix["R"][:1]
    prefix["sel"] = {k: v[:1] for k, v in prefix["sel"].items()}
    S.validate_bundles([prefix], allow_prefix=True)
    prefix["set"] = "valsel"
    with pytest.raises(SystemExit, match="frozen"):
        S.validate_bundles([prefix], allow_prefix=True)


def test_valid_seed_specific_checkpoints_share_arm_identity(synthetic):
    _, make, _ = synthetic
    for arms in (("mcfR_ep10_st100_u3", "mcfR_ep20_st200_u3"),
                 ("mcfRG_ep10_st100_u3", "mcfRG_ep20_st200_u3"),
                 ("mcfA_ep10_st100_u9_f3", "mcfA_ep20_st200_u9_f3"),
                 ("ivP_lr3e-4_ep12", "ivP_lr3e-4_ep09")):
        bs = [S.C.load_bundle(make(f"checkpoint{s}", arm, s)) for s, arm in enumerate(arms)]
        S.validate_bundles(bs)
        bs[1]["arm"] = "mcfR_ep20_st200_u1"
        with pytest.raises(SystemExit, match="one arm"):
            S.validate_bundles(bs)


def test_same_selection_can_move_between_git_and_sha_only_workers():
    assert S.same_selection({"sha256": "a", "commit": "commit1"}, {"sha256": "a", "commit": None})
    assert not S.same_selection({"sha256": "a"}, {"sha256": "b"})


def test_stale_rows_changed_bundle_or_truth_fail(synthetic, monkeypatch):
    items, make, rows = synthetic
    p = make("ours")
    rows(p)
    assert S.rows_current(p, "primary")
    changed = copy.deepcopy(items)
    changed[0]["K"] = 5
    monkeypatch.setattr(S.C, "load_set", lambda name: changed)
    with pytest.raises(SystemExit, match="stale scoring provenance"):
        S.rows_current(p, "primary")
    monkeypatch.setattr(S.C, "load_set", lambda name: items)
    b = S.C.load_bundle(p)
    b["R"][0, 0] *= 2
    torch.save(b, p)
    with pytest.raises(SystemExit, match="changed since"):
        S.rows_current(p, "primary")


def test_cached_rows_refuse_changed_rigid_structure_adapter(synthetic, tmp_path, monkeypatch):
    _, make, rows = synthetic
    source_repo = Path(S.C.REPO)
    repo = tmp_path / "scoring_repo"
    sources = [f"scripts/{name}" for name in
               ("n3_score.py", "n3_common.py", "n3_set_cache.py", "n3_harness.py", "g2_asym_baselines.py")]
    sources += ["symmc_flow/molcrystal.py", "tasks/n3_protocol.md"]
    for relative in sources:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_repo / relative, destination)
    monkeypatch.setattr(S.C, "REPO", str(repo))
    path = make("ours")
    rows(path)
    assert S.rows_current(path, "primary")
    adapter = repo / "symmc_flow/molcrystal.py"
    with adapter.open("a") as f:
        f.write("\n# synthetic coordinate adapter change after matching\n")
    with pytest.raises(SystemExit, match="stale scoring provenance"):
        S.rows_current(path, "primary")


def test_resume_refuses_missing_implementation_provenance(synthetic):
    _, make, rows = synthetic
    p = make("ours")
    rows(p)
    meta = S.match_path(p, "primary")[:-6] + ".meta.json"
    d = json.load(open(meta))
    d.pop("scoring_identity")
    json.dump(d, open(meta, "w"))
    with pytest.raises(SystemExit, match="stale scoring provenance"):
        S.rows_current(p, "primary")
    with pytest.raises(SystemExit, match="stale scoring provenance"):
        S._meta_check(p, "primary", {"scoring_identity": S.scoring_identity("val"),
                                      "selection": None, "step9": False})


def test_existing_rows_without_metadata_cannot_be_relabelled_current(synthetic):
    _, make, rows = synthetic
    p = make("ours")
    rows(p)
    os.remove(S.match_path(p, "primary")[:-6] + ".meta.json")
    with pytest.raises(SystemExit, match="lack scoring metadata"):
        S._meta_check(p, "primary", {"scoring_identity": S.scoring_identity("val"),
                                      "selection": None, "step9": False})


def test_basin_recomputes_rule_iii_and_retains_finite_unconverged(synthetic, tmp_path):
    _, make, rows = synthetic
    ref = make("fftruth", "fftruth", -1, draws=1, valid=[[True], [False]], conv=[[False], [False]])
    cand = make("basin", "basin-ours", draws=1)
    rows(cand, reference=ref, hits=((0, 0), (1, 0)))
    d = S.basin_outcomes([cand], ref, "ff")
    assert d["h"] == [[1.0], [0.0]]
    assert d["ref_unconverged_refcodes"] == ["SYN001"]
    p = str(tmp_path / "basin.json")
    json.dump(d, open(p, "w"))
    arr, _ = S._basin_jsons({"OURS": p}, d["refcodes"], ("fftruth",), "h2_basin")
    assert arr["OURS"].tolist() == d["h"]
    d["h"][1][0] = 1.
    json.dump(d, open(p, "w"))
    with pytest.raises(SystemExit, match="altered BASIN h"):
        S._basin_jsons({"OURS": p}, d["refcodes"], ("fftruth",), "h2_basin")


def test_basin_rejects_changed_sources_and_wrong_frozen_selector(synthetic, tmp_path):
    _, make, rows = synthetic
    ref = make("fftruth", "fftruth", -1, draws=1)
    cand = make("basin", "basin-ours", draws=1)
    rows(cand, reference=ref)
    d = S.basin_outcomes([cand], ref, "ff")
    p = str(tmp_path / "basin.json")
    json.dump(d, open(p, "w"))
    with pytest.raises(SystemExit, match="frozen BASIN choice"):
        S._basin_jsons({"OURS": p}, d["refcodes"], ("fftruth",), "h2_basin", {"OURS": {"selector": "lj"}})
    b = S.C.load_bundle(ref)
    b["valid"][0] = False
    torch.save(b, ref)
    with pytest.raises(SystemExit, match="another reference"):
        S._basin_jsons({"OURS": p}, d["refcodes"], ("fftruth",), "h2_basin")


def test_report_checks_row_freshness(synthetic, tmp_path):
    _, make, rows = synthetic
    p = make("cells", kind="cells")
    rows(p)
    b = S.C.load_bundle(p)
    b["sel"]["lj"] += 1
    torch.save(b, p)
    cfg = {"set": "val", "arms": {"OURS": {"bundles": [p], "selector": "lj"}},
           "out": str(tmp_path / "report.json")}
    q = str(tmp_path / "report_cfg.json")
    json.dump(cfg, open(q, "w"))
    with pytest.raises(SystemExit, match="changed since"):
        S.cmd_report(SimpleNamespace(config=q, step9=False, selection=S.SELECTION))


def test_frozen_report_allows_only_output_changes(tmp_path):
    cfg = {"set": "testB", "h2_form": "EXACT", "arms": {"OURS": {"bundles": ["b.pt"],
           "selector": "lj", "kind": "rasym", "seed_ids": [0, 1, 2]}}, "out": "a.json", "n_boot": 10000}
    selection = {"schema_version": 1, "complete": True,
                 "protocol_sha256": S.C.sha256(os.path.join(S.C.REPO, "tasks", "n3_protocol.md")),
                 "h2_form": {"form": "EXACT", "anchoring": .6}, "report_configs": {"testB": cfg}}
    p = str(tmp_path / "selection.json")
    json.dump(selection, open(p, "w"))
    other = copy.deepcopy(cfg)
    other.update(out="b.json")
    assert S.frozen_report(other, p)
    other["n_boot"] = 500
    with pytest.raises(SystemExit, match="scientific configuration"):
        S.frozen_report(other, p)
    other["n_boot"] = 10000
    other["arms"]["OURS"]["selector"] = "resid"
    with pytest.raises(SystemExit, match="scientific configuration"):
        S.frozen_report(other, p)
    other = copy.deepcopy(cfg)
    other["h2_form"] = "BASIN"
    with pytest.raises(SystemExit, match="scientific configuration"):
        S.frozen_report(other, p)
    selection["complete"] = False
    json.dump(selection, open(p, "w"))
    with pytest.raises(SystemExit, match="complete schema-version-1"):
        S.selection_data(p)


def test_haar_floor_uses_all_seeds_and_requires_complete_rows(synthetic, tmp_path):
    _, make, rows = synthetic
    paths = [make(f"haar{s}", "iii-u", s, draws=16) for s in range(3)]
    for matcher in ("primary", "mcf05", "mcf08"):
        for seed, p in enumerate(paths):
            rows(p, matcher, hits=tuple((0, j) for j in range(seed + 1)))
    args = SimpleNamespace(bundles=paths, step9=False, selection=S.SELECTION,
                           selection_sha256=None, out=str(tmp_path / "floor.json"))
    d = S.cmd_haar_floor(args)
    assert d["SYN001"] == {m: pytest.approx(6 / 48) for m in ("primary", "mcf05", "mcf08")}
    assert d["SYN002"]["primary"] == 0
    mp = S.match_path(paths[0], "mcf08")
    lines = open(mp).readlines()
    with open(mp, "w") as f:
        f.writelines(lines[:-1])
    with pytest.raises(SystemExit, match="incomplete"):
        S.cmd_haar_floor(args)
