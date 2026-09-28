"""N3 scorer / selection engine / statistics (scripts/n3_score.py, n3_stats.py, n3_harness.py).

Fast tests use synthetic bundles. The stored-flag statistics test reads only the stored N1/N2 DEV-TEST records
(no new match is made). Tests on real match results are opt-in (N3_HEAVY=1) and read files written by

    python scripts/n3_ours.py convert --arm ours --set val
    python scripts/n3_score.py match --bundles results/n3/private/draws/ours_s{0,1,2}_val.pt --matcher primary
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_score as SC  # noqa: E402
import n3_stats as ST  # noqa: E402

HEAVY = os.environ.get("N3_HEAVY") == "1"
N1_VAL = os.path.join(REPO, "results", "vast_n12", "n1_val.jsonl")
N1_TEST = os.path.join(REPO, "results", "vast_n12", "n1_test.jsonl")
N2_TEST = os.path.join(REPO, "results", "vast_n12", "n2_test.jsonl")


# ------------------------------------------------------------------ selectors
def test_argmin_nonfinite_last_ties_lowest_index():
    nan, inf = float("nan"), float("inf")
    assert SC.argmin_pick([True] * 4, [nan, 2.0, 1.0, 1.0]) == 2
    assert SC.argmin_pick([True] * 3, [nan, inf, -inf]) == 0          # every value non-finite -> index order
    assert SC.argmin_pick([True, True, False], [5.0, nan, -9.0]) == 0  # invalid draws never picked
    assert SC.argmin_pick([False, False], [1.0, 2.0]) is None


def test_mode_pick_counts_and_tiebreak():
    Z = np.array([6, 6, 1])
    base = np.array([[0.0, 0, 0], [1.4, 0, 0], [0, 1.0, 0]])
    far = base + np.array([0, 0, 3.0])
    coords = [(base, Z), (base + 0.05, Z), (far, Z), (far + 0.05, Z), (base + 0.1, Z)]
    valid = [True] * 5
    tb = [0.3, 0.2, 0.0, 0.0, 0.9]
    assert SC.mode_pick(coords, valid, 0.5, tb) == 1                  # 3-draw mode, calmest member
    assert SC.mode_pick(coords, [True, True, True, True, False], 0.5, tb) == 2   # 2 vs 2: lowest tie-break
    # element matching: an H cannot stand in for a C
    Zs = np.array([6, 1])
    a, b = np.array([[0.0, 0, 0], [1.0, 0, 0]]), np.array([[1.0, 0, 0], [0.0, 0, 0]])
    assert SC.chamfer(a, Zs, b, Zs) == pytest.approx(1.0)
    assert SC.chamfer(a, Zs, a, Zs) == 0.0


def test_selector_preference_order():
    names = ["random", "mode@1.5", "mode@0.25", "resid", "lj"]
    assert sorted(names, key=SC.sel_pref) == ["lj", "resid", "mode@0.25", "mode@1.5", "random"]


def _bundle(R_valid, lj, resid, set_name="val"):
    n, S = lj.shape
    return {"set": set_name, "kind": "rasym", "refcodes": [f"X{i}" for i in range(n)],
            "R": torch.eye(3, dtype=torch.float64).expand(n, S, 3, 3).clone(), "valid": R_valid,
            "sel": {"lj": lj, "resid": resid}, "meta": {}}


def test_arm_outcomes_random_expectation_and_picks():
    n, S = 3, 4
    valid = torch.ones(n, S, dtype=torch.bool)
    valid[2] = False                                                  # crystal 2: arm MISS
    lj = torch.tensor([[3.0, 1.0, 2.0, 5.0], [float("nan"), 4.0, 4.0, 9.0], [1.0, 1.0, 1.0, 1.0]])
    resid = torch.tensor([[0.1, 0.2, 0.0, 0.3], [0.5, 0.4, 0.3, 0.2], [0.0] * 4])
    b0 = _bundle(valid, lj, resid)
    b1 = _bundle(valid, lj.flip(1), resid)
    ok = {(0, 1): True, (1, 1): True, (1, 3): True}
    res0 = {(i, j): {"ok": ok.get((i, j), False)} for i in range(n) for j in range(S)}
    res1 = {(i, j): {"ok": ok.get((i, S - 1 - j), False)} for i in range(n) for j in range(S)}
    items = [{"refcode": f"X{i}"} for i in range(n)]
    o = SC.arm_outcomes([b0, b1], [res0, res1], items, "lj")
    assert o["picked"][:, 0].tolist() == [1.0, 1.0, 0.0]              # NaN ranks last; tie 4.0/4.0 -> index 1
    assert o["picked"][:, 1].tolist() == [1.0, 0.0, 0.0]              # flipped: tie -> index 1 = original 2
    assert o["draw1"][:, 0].tolist() == [0.25, 0.5, 0.0]
    assert o["miss_refcodes"] == ["X2"]
    assert o["ens"].tolist() == [1.0, 1.0, 0.0]                       # ensemble tie -> bundle 0 first
    r = SC.arm_outcomes([b0, b1], [res0, res1], items, "random")
    assert r["picked"][:, 0].tolist() == [0.25, 0.5, 0.0] and r["ens"].tolist() == [0.25, 0.5, 0.0]
    with pytest.raises(SC.Missing):
        SC.arm_outcomes([b0], [{}], items, "lj")


def test_metric_spec():
    assert SC.metric_spec("hits", "primary") == ("picked", None, "primary")
    assert SC.metric_spec("picked@mcf10", "primary") == ("picked", None, "mcf10")
    assert SC.metric_spec("any16", "primary") == ("any", None, "primary")
    assert SC.metric_spec("any10@mcf08", "primary") == ("any", 10, "mcf08")
    assert SC.metric_spec("draw1@mcf09", "primary") == ("draw1", None, "mcf09")
    assert SC.metric_spec("val_loss", "primary") is None and SC.metric_spec("selector", "primary") is None


def _res(n, S, hits):
    return {(i, j): {"ok": (i, j) in hits} for i in range(n) for j in range(S)}


def test_select_mcf_a_chain():
    """§3.2 (ii-A): stage 1 checkpoint by any-of-10 (FIRST 10 of 16 draws) at stol 1.0 -> stol 0.8 -> val loss;
    stage 2 knob cell by any-of-10 at stol 1.0, ties -> (9,3); selector from {RANDOM, LJ, RESID} by picked@1 at
    stol 1.0 (not the primary matcher), ties -> selector preference."""
    n, S = 6, 16
    items = [{"refcode": f"X{i}"} for i in range(n)]
    lj = torch.arange(S, dtype=torch.float64).repeat(n, 1)             # LJ picks draw 0
    resid = torch.arange(S, 0, -1, dtype=torch.float64).repeat(n, 1)   # RESID picks draw 15
    mk = lambda: _bundle(torch.ones(n, S, dtype=torch.bool), lj.clone(), resid.clone())
    loader = lambda res: (lambda k, m: res[k].get(m, {}))              # no file = no rows, as load_match
    # stage 1 (one seed): A and B tie on any10@mcf10 (2 each); B wins on any10@mcf08; C has 5 hits, all at draw
    # index >= 10, so any16 would pick C but any-of-10 must not
    cand = {"A": {"mcf10": _res(n, S, {(0, 1), (1, 2)}), "mcf08": _res(n, S, {(0, 1)})},
            "B": {"mcf10": _res(n, S, {(2, 3), (3, 9)}), "mcf08": _res(n, S, {(2, 3), (3, 9)})},
            "C": {"mcf10": _res(n, S, {(i, 12) for i in range(5)}), "mcf08": _res(n, S, set())}}
    seeds = {"0": [(k, f"{k}.pt", mk(), loader([cand[k]]), items, vl) for k, vl in (("A", 0.1), ("B", 0.3), ("C", 0.2))]}
    rec = SC.stage_b(seeds, {"rank": "any10@mcf10", "tie": ["any10@mcf08", "val_loss"]}, "primary", None)
    tab = {r["label"]: r for r in rec["seeds"]["0"]["table"]}
    assert tab["C"]["any10@mcf10"] == 0.0 and tab["A"]["any10@mcf10"] == tab["B"]["any10@mcf10"] == 2.0
    assert rec["seeds"]["0"]["chosen"] == "B" and rec["seeds"]["0"]["decided_by"] == "any10@mcf08"
    # stage 2: knob cells x 3 seeds, no selector; (5,1) and (9,3) tie on any10@mcf10 -> (9,3)
    hit2 = {(0, 0), (1, 4)}
    cells = []
    for lab, h in (("(5,1)", hit2), ("(9,3)", hit2), ("(13,2)", {(0, 0)})):
        res = [{"mcf10": _res(n, S, h)} for _ in range(3)]
        cells.append((lab, [mk() for _ in range(3)], loader(res), items))
    rec = SC.stage_a(cells, {"rank": "any10@mcf10", "tie": ["prefer_cell"], "prefer_cell": "(9,3)"}, "primary", None)
    assert rec["chosen"] == {"cell": "(9,3)", "selector": None} and rec["decided_by"] == "prefer_cell"
    # selector: LJ (draw 0) and RESID (draw 15) both hit twice at stol 1.0 -> LJ by preference; the primary
    # matcher (where only RESID's draw matches) must not be used
    res = [{"mcf10": _res(n, S, {(0, 0), (1, 0), (2, 15), (3, 15)}), "primary": _res(n, S, {(2, 15)})}
           for _ in range(3)]
    rec = SC.stage_a([("(9,3)", [mk() for _ in range(3)], loader(res), items)],
                     {"rank": "picked@mcf10", "tie": ["selector"]}, "primary", ["random", "lj", "resid"])
    tab = {r["selector"]: r for r in rec["table"]}
    assert tab["lj"]["picked@mcf10"] == tab["resid"]["picked@mcf10"] == 6.0
    assert tab["random"]["picked@mcf10"] == pytest.approx(3 * 4 / 16)
    assert rec["chosen"]["selector"] == "lj" and rec["decided_by"] == "selector"
    # a key without its match rows stops the run instead of ranking on partial data
    with pytest.raises(SystemExit, match="no mcf05 match row"):
        SC.stage_a([("(9,3)", [mk() for _ in range(3)], loader(res), items)],
                   {"rank": "picked@mcf05"}, "primary", ["lj"])


def test_stage_ranking_tie_orders():
    rows = [{"selector": "resid", "hits": 5, "draw1": 1.0, "any16": 9, "order": 0, "preferred": False},
            {"selector": "lj", "hits": 5, "draw1": 1.0, "any16": 9, "order": 1, "preferred": False},
            {"selector": "lj", "hits": 5, "draw1": 1.0, "any16": 9, "order": 2, "preferred": True},
            {"selector": "random", "hits": 4.5, "draw1": 2.0, "any16": 9, "order": 3, "preferred": True}]
    srt, by = SC._rank([rows[0], rows[1], rows[3]], ["hits", "selector"])
    assert srt[0]["order"] == 1 and by == "selector"
    srt, by = SC._rank(rows, ["hits", "selector"])
    assert srt[0]["order"] == 1 and by == "listed order"
    srt, by = SC._rank(rows, ["hits", "draw1", "any16", "prefer_cell", "selector"])
    assert srt[0]["order"] == 2 and by == "prefer_cell"
    b = [{"picked": 7, "draw1": 1.0, "any16": 12, "val_loss": 0.30, "order": 0},
         {"picked": 7, "draw1": 1.0, "any16": 12, "val_loss": 0.20, "order": 1}]
    srt, by = SC._rank(b, ["picked", "draw1", "any16", "val_loss"])
    assert srt[0]["order"] == 1 and by == "val_loss"


def test_blinding_guard(monkeypatch):
    for s in ("sel", "testB", "devtest"):
        with pytest.raises(SystemExit):
            SC.guard(s, False, SC.SELECTION)
    with pytest.raises(SystemExit):
        SC.guard("testB", True, os.path.join(REPO, "no_such_selection.json"))
    with pytest.raises(SystemExit):
        SC.guard("val", True, SC.SELECTION)                           # --step9 is for TEST-B / DEV-TEST only
    assert SC.guard("val", False, SC.SELECTION) is None
    assert SC.guard("valsel", False, SC.SELECTION) is None
    # an existing but uncommitted (gitignored) selection file does not open step 9
    fake = os.path.join(REPO, "results", "n3", "private", "pytest_selection.json")
    os.makedirs(os.path.dirname(fake), exist_ok=True)
    json.dump({"x": 1}, open(fake, "w"))
    try:
        with pytest.raises(SystemExit, match="not committed"):
            SC.guard("testB", True, fake)
        # a committed, unchanged, pushed file passes (the A1-frozen SEL list stands in for it; the protocol file
        # itself gains dated amendments in the working tree, so it is not a stable stand-in)
        rec = SC.selection_record(os.path.join(REPO, "tasks", "n3_sets", "sel_refcodes.txt"))
        assert rec["commit"] and rec["checked"].startswith("git")
        # without git: only a matching --selection-sha256 opens step 9
        monkeypatch.setattr(SC, "_git", lambda *a: None)
        with pytest.raises(SystemExit, match="without git"):
            SC.guard("testB", True, fake)
        with pytest.raises(SystemExit, match="SHA-256"):
            SC.guard("testB", True, fake, "0" * 64)
        assert SC.guard("devtest", True, fake, SC.C.sha256(fake))["commit"] is None
    finally:
        os.remove(fake)


def test_allow_blind_only_for_oracle_bundles(tmp_path):
    """run_match's oracle bypass refuses any bundle that is not a truth-derived oracle bundle."""
    b = _bundle(torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, 2), torch.zeros(2, 2), set_name="devtest")
    b.update(arm="ours", seed=0)
    p = str(tmp_path / "ours_s0_devtest.pt")
    torch.save(b, p)
    with pytest.raises(SystemExit, match="oracle"):
        SC.run_match([p], "primary", allow_blind=True, quiet=True)
    with pytest.raises(SystemExit, match="BLINDING"):
        SC.run_match([p], "primary", quiet=True)


# ------------------------------------------------------------------ reference-bundle matching (§3.3 BASIN forms)
def _rz(deg):
    import math
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def test_reference_matching_rule_iii_order_and_select(tmp_path, monkeypatch):
    """match --reference: candidates are matched to the reference's draw 0 (VAL, truth as reference), in the order
    the reference arm fixes (p1ref: fit(candidate, reference)); an invalid reference gives 'ref_invalid' rows and
    h = 0 for the crystal (rule iii); unconverged references are listed; `select` with "reference" reads the
    reference-keyed rows; a changed reference is refused."""
    import n3_common as C
    monkeypatch.setattr(SC, "MATCH_DIR", str(tmp_path / "match"))
    items = C.load_set("val")[:3]
    refs = [a["refcode"] for a in items]
    R0 = torch.stack([a["R0"].double() for a in items])
    cand = str(tmp_path / "pytest-cand_s0_val.pt")
    torch.save({"arm": "pytest-cand", "seed": 0, "set": "val", "refcodes": refs, "kind": "rasym",
                "R": torch.stack([R0, torch.stack([_rz(90.0) @ r for r in R0])], 1), "valid": torch.ones(3, 2, dtype=torch.bool),
                "sel": {"lj": torch.tensor([[0.0, 1.0]] * 3), "lj_pre": torch.tensor([[1.0, 0.0]] * 3)}, "meta": {}}, cand)
    ref = str(tmp_path / "p1ref_s-1_val.pt")

    def save_ref(valid):
        torch.save({"arm": "p1ref", "seed": -1, "set": "val", "refcodes": refs, "kind": "rasym", "R": R0[:, None].clone(),
                    "valid": torch.tensor(valid)[:, None], "sel": {"lj": torch.zeros(3, 1)},
                    "meta": {"converged": torch.tensor([[True], [True], [False]])}}, ref)
    save_ref([True, False, True])
    SC.run_match([cand], "primary", workers=1, reference=ref, quiet=True)
    info = SC.ref_info(ref)
    assert info["order"] == "cand_ref"
    rows = SC.load_match(cand, SC.mkey("primary", info))
    assert rows[(1, 0)]["status"] == rows[(1, 1)]["status"] == "ref_invalid" and not rows[(1, 0)]["ok"]
    assert rows[(0, 0)]["ok"] and rows[(2, 0)]["ok"] and rows[(0, 0)]["status"] == "ok"
    meta = json.load(open(SC.match_path(cand, SC.mkey("primary", info))[:-6] + ".meta.json"))
    assert meta["reference_order"] == "cand_ref" and meta["reference_sha256"] == info["sha256"]
    d = SC.basin_outcomes([cand], ref, "lj")
    assert d["h"] == [[1.0], [0.0], [1.0]] and d["ref_invalid_refcodes"] == [refs[1]]
    assert d["ref_unconverged_refcodes"] == [refs[2]] and d["match_rule"] == "StructureMatcher() fit(candidate, reference)"
    assert d["draw1"][1] == [0.0]                                     # rule (iii) for every draw of the crystal
    cfg = {"name": "p1-test", "matcher": "primary", "reference": ref, "menu": SC.P1_MENU,
           "stage_a": {"cells": [{"label": "p1", "bundles": [cand]}], "tie": ["selector"]}}
    json.dump(cfg, open(tmp_path / "cfg.json", "w"))
    rec = SC.cmd_select(type("A", (), {"config": str(tmp_path / "cfg.json")})())
    hits = {r["selector"]: r["hits"] for r in rec["stage_a"]["table"]}
    assert rec["reference"]["arm"] == "p1ref" and hits["lj"] == 2.0 and rec["stage_a"]["chosen"]["selector"] == "lj"
    save_ref([True, True, True])                                      # another reference: its old rows are refused
    with pytest.raises(SystemExit, match="another reference"):
        SC.run_match([cand], "primary", workers=1, reference=ref, quiet=True)
    with pytest.raises(SystemExit, match="another reference"):
        SC.basin_outcomes([cand], ref, "lj")
    with pytest.raises(SystemExit, match="not a reference arm"):
        SC.ref_info(cand)


def test_report_basin_jsons_refuse_mixed_or_wrong_references(tmp_path):
    refs = ["X0", "X1"]
    mk = lambda name, arm, sha: (json.dump({"refcodes": refs, "h": [[1.0], [0.0]], "reference": f"{arm}.pt",
                                            "reference_sha256": sha, "reference_arm": arm}, open(tmp_path / name, "w"))
                                 or str(tmp_path / name))
    a, b = mk("a.json", "fftruth", "1" * 64), mk("b.json", "fftruth", "2" * 64)
    arr, ref = SC._basin_jsons({"OURS": a}, refs, ("fftruth",), "h2_basin")
    assert arr["OURS"].shape == (2, 1) and ref["arm"] == "fftruth"
    with pytest.raises(SystemExit, match="different references"):
        SC._basin_jsons({"OURS": a, "iii-s": b}, refs, ("fftruth",), "h2_basin")
    with pytest.raises(SystemExit, match="reference arm"):
        SC._basin_jsons({"OURS": mk("c.json", "p1ref", "3" * 64)}, refs, ("fftruth",), "h2_basin")
    with pytest.raises(SystemExit, match="frozen order"):
        SC._basin_jsons({"OURS": a}, ["X1", "X0"], ("fftruth",), "h2_basin")


# ------------------------------------------------------------------ statistics
def test_stats_selftest():
    ST._selftest()


def _lj_flags(path):
    rows = {json.loads(l)["refcode"]: json.loads(l) for l in open(path)}
    keys = sorted(rows)
    arms = [k for k in rows[keys[0]] if k not in ("refcode", "sg")]
    lj = lambda d: d[min(range(len(d)), key=lambda i: (d[i]["e_lj"], i))]["exact"]
    return keys, np.array([[float(lj(rows[k][a])) for a in arms] for k in keys])


@pytest.mark.skipif(not (os.path.exists(N1_TEST) and os.path.exists(N2_TEST)), reason="stored N1/N2 draws absent")
def test_g2_vs_n2_devtest_from_stored_flags():
    """Protocol §1.4c: G2 7.5% vs N2 5.0% per-seed mean, crystal-level sign-flip p = 0.023; review BCa CI.
    Uses the STORED flags only (a statistics regression test, not a reported number; no new match)."""
    k1, X = _lj_flags(N1_TEST)
    k2, Y = _lj_flags(N2_TEST)
    assert k1 == k2 and X.shape == Y.shape == (200, 3)
    assert X.mean(0).tolist() == [0.085, 0.075, 0.065] and round(X.mean(), 4) == 0.075
    assert round(Y.mean(), 4) == 0.05
    c = SC.contrast(X, Y, n_boot=10000)
    assert c["delta_pp"] == pytest.approx(2.5)
    assert c["p"] == pytest.approx(0.0232, abs=5e-5)
    assert c["bca_pp"][0] == pytest.approx(0.67, abs=0.005) and c["bca_pp"][1] == pytest.approx(4.67, abs=0.005)
    # the §2.4 unit: every d_i is a multiple of 1/48
    ST.to_units(X.mean(1) - Y.mean(1))


def test_descriptive_contrasts_carry_no_p(tmp_path):
    """§4 ii-S terms and the ANCHORED/DRIFTED/UNRESOLVED subsets: estimates + CIs, never a p-value."""
    rng = np.random.default_rng(5)
    n = 120
    h = {k: (rng.random((n, 3)) < q).astype(float) for k, q in
         (("OURS", 0.15), ("MCF-R", 0.03), ("MCF-R+G", 0.04), ("ii-S", 0.08), ("iv-P", 0.05))}
    h2 = {"OURS": h["OURS"], "iii-s": (rng.random((n, 1)) < 0.03).astype(float)}
    refs = [f"X{i}" for i in range(n)]
    ff = tmp_path / "ff.json"
    json.dump([{"refcode": r, "ff_label": ("ANCHORED", "DRIFTED", "UNRESOLVED")[i % 3]} for i, r in enumerate(refs)],
              open(ff, "w"))
    assert "p" not in SC.contrast(h["OURS"], h["MCF-R"], 200, test=False)
    spec = {"H1": ["MCF-R", "MCF-R+G"], "H2": "iii-s", "H3": "iv-P"}
    rep = {"family_primary": SC.family(h, "OURS", spec, h2, 200), "family_n2": {}, "h2_form": "EXACT"}
    assert rep["family_primary"]["H1"]["supported"]
    w = SC.wording(rep, {"ff_labels": str(ff)}, h, h2, None, refs, spec, "OURS", 200)
    assert set(w["H2_by_ff_label"]) == {"ANCHORED", "DRIFTED", "UNRESOLVED"}
    assert all("p" not in c for c in w["H2_by_ff_label"].values())
    assert all("p" not in v for k, v in w["ii_S_decomposition"].items() if isinstance(v, dict))
    assert "p" in rep["family_primary"]["H2"]                         # the test itself keeps its p


def test_family_holm_and_iut_verdicts():
    rng = np.random.default_rng(3)
    n = 400
    ours = (rng.random((n, 3)) < 0.12).astype(float)
    weak = (rng.random((n, 3)) < 0.02).astype(float)
    same = ours.copy()
    h = {"OURS": ours, "MCF-R": weak, "MCF-R+G": weak[:, ::-1].copy(), "iv-P": same}
    fam = SC.family(h, "OURS", {"H1": ["MCF-R", "MCF-R+G"], "H2": "iii-s", "H3": "iv-P"},
                    {"OURS": ours, "iii-s": weak[:, :1].copy()}, n_boot=500)
    assert fam["complete"]
    assert fam["H1"]["verdict"] == "OURS better" and fam["H1"]["p"] == max(
        c["p"] for c in fam["H1"]["components"].values())
    assert fam["H2"]["verdict"] == "OURS better"
    assert fam["H3"]["verdict"].startswith("difference not resolved at n = 400")
    assert fam["H3"]["p"] == 1.0
    split = SC.family({"OURS": ours, "MCF-R": weak, "MCF-R+G": np.ones((n, 3)), "iv-P": same}, "OURS",
                      {"H1": ["MCF-R", "MCF-R+G"], "H2": "iii-s", "H3": "iv-P"},
                      {"OURS": ours, "iii-s": weak[:, :1].copy()}, n_boot=500)
    assert split["H1"]["verdict"] == "split: OURS better than MCF-R; MCF-R+G better than OURS"
    inc = SC.family({"OURS": ours, "iv-P": same}, "OURS", {"H1": ["MCF-R", "MCF-R+G"], "H2": "iii-s",
                                                           "H3": "iv-P"}, {}, n_boot=200)
    assert not inc["complete"] and "incomplete" in inc["H3"]["verdict"]


# ------------------------------------------------------------------ harness
def _probe(task):
    kind, x = task
    if kind == "exc":
        raise ValueError("boom")
    if kind == "crash":
        os._exit(3)
    return x + 1


def test_harness_outcomes():
    import n3_harness as H
    tasks = [(0, ("ok", 1)), (1, ("exc", 0)), (2, ("crash", 0)), (3, ("ok", 5))]
    ev = []
    out = {k: (st, r) for k, st, r, info in H.run(_probe, tasks, workers=2, log=ev.append)}
    assert out == {0: ("ok", 2), 1: ("det_fail", None), 2: ("det_fail", None), 3: ("ok", 6)}
    assert sum(e["event"] == "deterministic_failure" for e in ev) == 2
    assert sum(e["event"] == "rerun_isolated" for e in ev) == 2
    # a prior failure recorded with err_text() recurs identically -> deterministic after ONE isolated run
    ev = []
    got = list(H.run(_probe, [(9, ("exc", 0))], workers=1, log=ev.append,
                     prior={9: [H.err_text(ValueError("boom"))]}))
    assert got[0][1] == "det_fail" and got[0][3]["attempts"] == 2
    assert len(H.err_text(ValueError("x" * 1000))) == H.ERR_LEN


def test_jsonl_resume_after_kill(tmp_path):
    import n3_harness as H
    p = str(tmp_path / "rows.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"i": 0}) + "\n" + json.dumps({"i": 1}) + "\n" + '{"i": 2, "ok": tr')   # killed mid-row
    assert [r["i"] for r in H.read_jsonl(p)] == [0, 1]
    assert H.repair_jsonl(p) > 0 and H.repair_jsonl(p) == 0
    with open(p, "a") as f:
        f.write(json.dumps({"i": 2}) + "\n")
    assert [r["i"] for r in H.read_jsonl(p)] == [0, 1, 2]
    with open(p, "a") as f:
        f.write("garbage\n" + json.dumps({"i": 3}) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        H.read_jsonl(p)


def test_sampler_refuses_resume_with_other_checkpoints(tmp_path):
    import n3_ours as O
    part = str(tmp_path / "p.jsonl")
    head = {"arm": "ours", "set": "sel", "seed_base": 90000, "ckpt_sha256": ["a", "b", "c"], "S": 16, "steps": 40}
    with open(part + ".events.jsonl", "w") as f:
        f.write(json.dumps({"event": "start", **head}) + "\n")
    O._check_resume(part, {0: {}}, head)
    with pytest.raises(SystemExit, match="ckpt_sha256"):
        O._check_resume(part, {0: {}}, {**head, "ckpt_sha256": ["a", "b", "X"]})
    with pytest.raises(SystemExit, match="seed_base"):
        O._check_resume(part, {0: {}}, {**head, "seed_base": 80000})
    os.remove(part + ".events.jsonl")
    with pytest.raises(SystemExit, match="no 'start'"):
        O._check_resume(part, {0: {}}, head)
    O._check_resume(part, {}, head)                                    # a fresh part file needs no history


# ------------------------------------------------------------------ MCF oracles and Gate C-a (input-side)
def _export(set_name):
    import n3_mcf_export as X
    P = X.paths(set_name)
    if not os.path.exists(P["pickle"]):
        pytest.skip(f"no {set_name} MCF export")
    return X, X.load_pkl_gz(P["pickle"])


def test_f6_content_check_refuses_non_oracle_predictions():
    """gate-f6 matches only files whose every draw is the export pickle's own pose (§1.3 blinding): noise on
    the coordinates, a changed rotation (an MCF-R draw keeps lattice and centroids) or a changed lattice is
    refused before anything is matched."""
    X, entries = _export("val")
    entries = entries[:6]
    pred = X.oracle_predictions(entries, 3)
    assert SC.check_oracle_predictions(pred, entries) is None
    bad = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in pred.items()}
    bad["cart_coords"][2] += 0.7
    assert "cart_coords" in SC.check_oracle_predictions(bad, entries)
    rot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in pred.items()}
    rot["pred_rotmats"][1, 0] = torch.eye(3)
    assert "pred_rotmats" in SC.check_oracle_predictions(rot, entries)
    lat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in pred.items()}
    lat["lattices"][0, 3] *= 1.01
    assert "lattices" in SC.check_oracle_predictions(lat, entries)


def test_gate_f6_cli_refuses_arm_like_file(tmp_path):
    X, entries = _export("val")
    pred = X.oracle_predictions(entries, 2)
    pred["cart_coords"][1] += 0.5
    p = str(tmp_path / "predictions_2.pt")
    torch.save(pred, p)
    before = sorted(os.listdir(SC.ORACLE_DIR)) if os.path.isdir(SC.ORACLE_DIR) else []
    args = type("A", (), {"set": "val", "pickle": None, "predictions": p, "sidecar": None, "draws": 2,
                          "matchers": ["primary"], "workers": 1, "slow": 60.0})()
    with pytest.raises(SystemExit, match="refuses"):
        SC.cmd_gate_f6(args)
    assert (sorted(os.listdir(SC.ORACLE_DIR)) if os.path.isdir(SC.ORACLE_DIR) else []) == before


def _truths(entries):
    return [{"lattice": e["lattice_1"].double().numpy(), "cart": e["gt_coords"].double().numpy(),
             "species": np.array([SC.MCF_IDX_TO_Z[int(t)] for t in e["atom_types"]])} for e in entries]


def test_gate_ca_guard_refuses_n3_blinded_export():
    X, dev = _export("devtest")
    with pytest.raises(SystemExit, match="devtest"):
        SC.gateca_guard(_truths(dev))                                  # by content, no pickle path given
    with pytest.raises(SystemExit, match="devtest"):
        SC.gateca_guard(_truths(dev[:10]), X.paths("devtest")["pickle"])   # by the sidecar SHA-256
    _, val = _export("val")
    g = SC.gateca_guard(_truths(val))                                  # VAL is not blinded
    assert g["overlap_counts"]["devtest"] == 0 and "devtest" in g["sets_checked"]


# ------------------------------------------------------------------ sampler bundle writer
@pytest.mark.skipif(not os.path.exists(N1_VAL), reason="stored N1 VAL draws absent")
def test_sampler_part_rows_to_bundles_match_convert():
    """n3_ours.write_bundles on part-file rows (built here from the stored N1 VAL draws, seed 40000 + i) gives
    the same R / valid / selector inputs as n3_ours.convert on the same draws."""
    import n3_common as C
    import n3_ours as O
    conv = [C.bundle_path("ours", s, "val") for s in range(3)]
    if not all(os.path.exists(p) for p in conv):
        pytest.skip("run n3_ours.py convert --arm ours --set val first")
    stored = {json.loads(l)["refcode"]: json.loads(l) for l in open(N1_VAL)}
    items = C.load_set("val")
    keys = ["eval_s0.pt", "eval_s1.pt", "eval_s2.pt"]
    rows = {i: {"i": i, "refcode": a["refcode"], "seed": 40000 + i,
                "models": [{"R": [d["R"] for d in stored[a["refcode"]][k]], "e_lj": [d["e_lj"] for d in stored[a["refcode"]][k]],
                            "torque_end": [d["torque_end"] for d in stored[a["refcode"]][k]], "valid": [True] * 16,
                            "e_err": {}, "sec": 1.0, "arm_fail": None} for k in keys]}
            for i, a in enumerate(items)}
    rows[3]["models"][1]["valid"][5] = False                          # an arm failure survives into the bundle
    head = {"ckpts": O.arm_ckpts("ours"), "ckpt_sha256": ["x"] * 3, "seed_base": 40000}
    O.write_bundles("pytest_ours", "val", rows, head)
    try:
        for s in range(3):
            a, b = C.load_bundle(C.bundle_path("pytest_ours", s, "val")), C.load_bundle(conv[s])
            assert torch.equal(a["R"], b["R"]) and a["kind"] == "rasym" and a["refcodes"] == b["refcodes"]
            for k in ("lj", "resid"):
                assert torch.equal(a["sel"][k], b["sel"][k])
            exp = b["valid"].clone()
            if s == 1:
                exp[3, 5] = False
            assert torch.equal(a["valid"], exp)
            assert a["meta"]["seed_index"] == s and a["meta"]["n_lj_per_crystal"] == 16
    finally:
        for s in range(3):
            p = C.bundle_path("pytest_ours", s, "val")
            if os.path.exists(p):
                os.remove(p)


# ------------------------------------------------------------------ heavy: real VAL match results
def _val_bundles():
    import n3_common as C
    paths = [C.bundle_path("ours", s, "val") for s in range(3)]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("run n3_ours.py convert --arm ours --set val first")
    return paths, [C.load_bundle(p) for p in paths]


def test_picks_export_is_the_scored_pick(tmp_path):
    """'picks' (truth-free, for the FF* unit's BASIN minimisations) gives exactly the draw arm_outcomes scores:
    argmin of the stored e_lj, non-finite last, ties to the lowest index."""
    import n3_common as C
    paths, bundles = _val_bundles()
    out = tmp_path / "picks.json"
    SC.cmd_picks(type("A", (), {"bundles": paths, "selector": "lj", "mode_tiebreak": None, "out": str(out)})())
    d = json.load(open(out))
    assert len(d["per_seed"]) == 300 and len(d["ensemble"]) == 100 and d["set"] == "val"
    items = C.load_set("val")
    cr = [SC.Crystal(bundles, items, i) for i in range(100)]
    for p in d["per_seed"]:
        b = bundles[p["bundle"]]
        assert p["j"] == SC.argmin_pick(b["valid"][p["i"]].tolist(), b["sel"]["lj"][p["i"]].tolist())
        assert p["j"] == cr[p["i"]].pick("lj", p["bundle"])
    for e in d["ensemble"]:
        assert (e["bundle"], e["j"]) == cr[e["i"]].pick("lj")


@pytest.mark.skipif(not HEAVY, reason="N3_HEAVY=1 only (needs VAL match results)")
def test_rematch_reproduces_n1_val_lj_hits():
    """Re-matched N1 VAL LJ picks give N1's per-seed hits 10/4/7 (results/n1_final.json val lj)."""
    import n3_common as C
    paths, bundles = _val_bundles()
    items = C.load_set("val")
    res = [SC.load_match(p, "primary") for p in paths]
    o = SC.arm_outcomes(bundles, res, items, "lj", need_draws=False)
    assert o["picked"].sum(0).tolist() == [10.0, 4.0, 7.0]
    final = json.load(open(os.path.join(REPO, "results", "n1_final.json")))["val"]["lj"]["per_seed"]
    assert final == [10.0, 4.0, 7.0]


@pytest.mark.skipif(not HEAVY, reason="N3_HEAVY=1 only (needs every VAL draw matched)")
def test_selection_engine_on_n1_val_picks_lj(tmp_path):
    """§2.3: re-applying the selector rule to the N1 VAL draws gives LJ 21 vs RANDOM 2.9 vs RESID 2 seed-summed
    hits, so OURS keeps LJ."""
    paths, _ = _val_bundles()
    cfg = {"name": "test", "matcher": "primary", "out": str(tmp_path / "sel.json"),
           "stage_a": {"cells": [{"label": "N1 VAL", "bundles": paths}], "tie": ["selector"]}}
    json.dump(cfg, open(tmp_path / "cfg.json", "w"))
    rec = SC.cmd_select(type("A", (), {"config": str(tmp_path / "cfg.json")})())
    hits = {r["selector"]: r["hits"] for r in rec["stage_a"]["table"]}
    assert rec["stage_a"]["chosen"]["selector"] == "lj"
    assert hits["lj"] == 21.0 and hits["resid"] == 2.0 and round(hits["random"], 1) == 2.9
    assert max(v for k, v in hits.items() if k != "lj") < 21.0


@pytest.mark.skipif(not HEAVY, reason="N3_HEAVY=1 only (needs VAL match results)")
def test_rematch_agrees_with_stored_flags_on_matched_draws():
    """Every re-matched VAL draw agrees with g2_rank's stored flag (test-only read of the stored flags);
    when every draw is matched, draw@1 per seed equals the stored one."""
    paths, bundles = _val_bundles()
    rows = {json.loads(l)["refcode"]: json.loads(l) for l in open(N1_VAL)}
    keys = ["eval_s0.pt", "eval_s1.pt", "eval_s2.pt"]
    n_cmp = 0
    for s, (p, b) in enumerate(zip(paths, bundles)):
        res = SC.load_match(p, "primary")
        for (i, j), r in res.items():
            assert r["ok"] == rows[b["refcodes"][i]][keys[s]][j]["exact"], (s, i, j)
            n_cmp += 1
        if len(res) == 100 * 16:
            stored = sum(d["exact"] for ref in b["refcodes"] for d in rows[ref][keys[s]]) / 1600
            assert sum(r["ok"] for r in res.values()) / 1600 == stored
    assert n_cmp > 0
