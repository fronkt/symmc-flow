"""The step-9 diagnostic (c) summary (scripts/n3_step9_summary.py): bundle list from the real freeze, then the
per-z_MCF / per-matcher aggregation on synthetic outcomes (the scorer's functions are replaced; nothing is matched
and no set is loaded)."""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

SEL_PATH = os.path.join(REPO, "results", "n3", "n3_selection.json")
pytestmark = pytest.mark.skipif(not os.path.exists(SEL_PATH), reason="no published freeze")

import n3_step9_summary as SM  # noqa: E402


def test_bundles_from_the_freeze():
    sel = json.load(open(SEL_PATH, encoding="utf-8"))
    for st in ("testB", "devtest"):
        paths = SM.oal_bundles(sel, st)
        assert len(paths) == 3 and all(p.endswith(f"_{st}.pt") and "/mcfRoal_" in p for p in paths)
        assert [p.split("_s")[-1][0] for p in paths] == ["0", "1", "2"]
        assert sel["report_configs"][st]["arms"]["MCF-R"]["selector"] == "lj"


def test_aggregation(tmp_path, monkeypatch):
    n, z = 6, [1, 2, 2, 3, 4, 1]
    refs = [f"R{i}" for i in range(n)]
    bundles = {p: {"set": "testB", "refcodes": refs, "seed": k, "meta": {"z_MCF": z}}
               for k, p in enumerate(["a_s0_testB.pt", "a_s1_testB.pt", "a_s2_testB.pt"])}
    sel = {"runs": {"step9": [{"id": f"diag-c/testB/s{k}", "set": "testB",
                               "outputs": [f"results/n3/private/draws/mcfRoal_x_s{k}_testB.pt"]} for k in range(3)]},
           "report_configs": {"testB": {"arms": {"MCF-R": {"selector": "lj"}}}}}
    sp = tmp_path / "sel.json"
    sp.write_text(json.dumps(sel))
    seen = {}
    monkeypatch.setattr(SM.S, "guard", lambda st, step9, s, sha: {"sha256": sha} if step9 else pytest.fail("no step9"))
    monkeypatch.setattr(SM.S, "rel", lambda p: os.path.basename(p).replace("mcfRoal_x", "a"))
    monkeypatch.setattr(SM.C, "load_bundle", lambda p: bundles[p])
    monkeypatch.setattr(SM.S, "validate_bundles", lambda b, seed_ids: None)
    monkeypatch.setattr(SM.C, "load_set", lambda st: [{"refcode": r} for r in refs])
    monkeypatch.setattr(SM.C, "sha256", lambda p: "h")
    monkeypatch.setattr(SM.S, "rows_current", lambda p, m, selection, bundle: {"ok": 1})
    monkeypatch.setattr(SM.S, "load_match", lambda p, m: m)
    monkeypatch.setattr(SM.S, "scoring_identity", lambda st: {"id": 1})

    def outcomes(bs, results, items, selector, mtb, ensemble):
        assert selector == "lj" and ensemble is False and len(set(results)) == 1
        seen[results[0]] = True
        # crystal i, seed k: draw1 = i odd; any = 1; picked = (k == 0)
        d = np.array([[float(i % 2)] * 3 for i in range(n)])
        return {"draw1": d, "any": np.ones((n, 3)), "picked": np.array([[1.0, 0.0, 0.0]] * n)}
    monkeypatch.setattr(SM.S, "arm_outcomes", outcomes)
    out = tmp_path / "diag_c.json"
    res = SM.summarize("testB", str(sp), "f" * 64, True, str(out))
    assert set(seen) == set(SM.MATCHERS)
    assert res["z_MCF = 1"]["n"] == 2
    for m in SM.MATCHERS:
        z2, z3 = res["matchers"][m]["z_MCF = 2"], res["matchers"][m]["z_MCF >= 3"]
        assert z2["n"] == 2 and z3["n"] == 2
        assert z2["draw1_mean_pct"] == 50.0 and z3["draw1_mean_pct"] == 50.0      # crystals 1, 2 and 3, 4
        assert z2["any16_per_seed_pct"] == [100.0, 100.0, 100.0]
        assert z2["picked1_per_seed_pct"] == [100.0, 0.0, 0.0] and round(z2["picked1_mean_pct"], 2) == 33.33
    assert json.load(open(out))["selection"] == {"sha256": "f" * 64}
