"""§7 step 6 assembler (scripts/n3_select.py) on SYNTHETIC selection records and bundles (no CSD content, no
match, nothing written outside pytest's tmp_path). The OURS / OURS-N2 checkpoints are the real committed-path
files (hashed only)."""
import json
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_select as NS  # noqa: E402

SEEDS = (0, 1, 2)
R_LAB = {0: "ep10_st140", 1: "ep12_st168", 2: "last"}          # MCF-R Stage-A checkpoint per seed
R_CHOSEN = {0: "ep20_st280", 1: "ep12_st168", 2: "last"}       # MCF-R Stage-B choice (at s_uR 2)
G_CHOSEN = {0: "ep20_st280", 1: "last", 2: "ep30_st420"}       # +G Stage-B choice (at s_uR 3)
A_CHOSEN = {0: "ep40_st560", 1: "last", 2: "ep41_st574"}


class World:
    """A complete, consistent set of synthetic step-6 inputs under tmp; .inputs = {role: path}."""

    def __init__(self, tmp, degrade=()):
        self.tmp, self.d = tmp, tmp / "draws"
        self.d.mkdir(exist_ok=True)
        self.degrade = list(degrade)
        self.inputs = {}
        self.ck = {}
        self.build()

    def bundle(self, name, run=None):
        p = self.d / name
        if not p.exists():
            torch.save({"arm": name, "meta": {"run": run} if run else {}}, p)
        return str(p)

    def ckpt(self, arm, s, lab):
        key = (arm, s, lab)
        self.ck.setdefault(key, f"{arm}{s}{lab}".encode().hex().ljust(64, "0")[:64])
        return {"ckpt": f"n3_mcf/runs/n3_mcf{arm}_s{s}/ckpt/{lab}.ckpt", "ckpt_sha256": self.ck[key],
                "ckpt_label": lab, "train_seed": s}

    def record(self, role, cfg, stage_a=None, stage_b=None, menu="default", reference=None):
        paths = []
        for c in (cfg.get("stage_a") or {}).get("cells", []):
            paths += c["bundles"]
        for cands in ((cfg.get("stage_b") or {}).get("seeds") or {}).values():
            paths += [c["bundle"] for c in cands]
        rec = {"name": cfg.get("name", role), "matcher": "primary", "config": cfg,
               "bundles_sha256": {p: NS.C.sha256(p) for p in paths}}
        if stage_a is not None:
            rec["menu"] = NS.MENU if menu == "default" else menu
            rec["stage_a"] = stage_a
        if stage_b is not None:
            rec["stage_b"] = stage_b
        if reference:
            rec["reference"] = reference
        self.write(role, rec)
        return rec

    def write(self, role, obj, name=None):
        p = self.tmp / (name or f"{role.replace(':', '_')}.json")
        json.dump(obj, open(p, "w"), indent=1)
        self.inputs[role] = str(p)

    def sa(self, cell, sel, tie=("hits", "selector")):
        return {"tie": list(tie), "table": [], "chosen": {"cell": cell, "selector": sel}, "decided_by": "hits"}

    def build(self):
        W = self
        W.write("degrade", {"items": W.degrade, "amendment": "A3 (2026-10-01): step-4 degrade decision"})
        W.write("anchor", {"summary": {"set": "valsel", "n": 400, "anchoring": 0.42, "anchoring_val_only": 0.45,
                                       "form": "BASIN", "partial": False, "fftruth_sha256": "0" * 64},
                           "rows": [{"refcode": "NOT_COPIED"}]})
        W.write("gate_c", {"C-a": {"i_pass": True, "ii_pass_test": False, "ii_pass_val": False, "pass": False},
                           "C-b": {"pass": True}, "C-c": {"pass": True}})
        for role, arm in (("iii-u", "iii-u"), ("iii-p", "iii-p")):
            cells = [{"label": role, "bundles": [W.bundle(f"{arm}_s{s}_valsel.pt") for s in SEEDS]}]
            W.record(role, {"name": role, "mode_tiebreak": "lj", "stage_a": {"cells": cells}},
                     stage_a=W.sa(role, "lj" if role == "iii-p" else "random"), menu=NS.MENU_CLASSICAL)
        ref = {"path": "results/n3/private/draws/p1ref_s-1_valsel.pt", "sha256": "1" * 64, "arm": "p1ref",
               "order": "cand_ref"}
        for role, arm in (("p1-ours", "p1-ours"), ("p1-iii-p", "iii-p")):
            cells = [{"label": role, "bundles": [W.bundle(f"{arm}_s{s}_valsel.pt") for s in SEEDS]}]
            W.record(role, {"name": role, "reference": ref["path"], "stage_a": {"cells": cells}},
                     stage_a=W.sa(role, "lj_pre"), menu=NS.P1_MENU, reference=ref)
        # MCF-R and +G: Stage A over 5 s_uR cells (seed's lowest-loss checkpoint), Stage B at the chosen s_uR
        for pre, u_ch, sel_ch, chosen in (("mcfR", 2, "lj", R_CHOSEN), ("mcfRG", 3, "resid", G_CHOSEN)):
            cells = [{"label": f"u{u}", "bundles": [W.bundle(f"{pre}_{R_LAB[s]}_u{u}_s{s}_valsel.pt",
                                                            run={**W.ckpt("R", s, R_LAB[s]), "s_uR": u})
                                                   for s in SEEDS]}
                     for u in (1, 2, 3, 5, 10)]
            W.record(f"{pre}_A", {"name": f"{pre}_A", "stage_a": {"cells": cells, "prefer_cell": "u3"}},
                     stage_a=W.sa(f"u{u_ch}", sel_ch, ("hits", "draw1", "any16", "prefer_cell", "selector")))
            seeds, sb = {}, {"selector": sel_ch, "tie": ["picked", "draw1", "any16", "val_loss"], "seeds": {}}
            for s in SEEDS:
                labs = sorted({R_LAB[s], chosen[s], "ep5_st70", "last"})
                seeds[str(s)] = [{"label": lab, "val_loss": 0.1 * k + 1,
                                  "bundle": W.bundle(f"{pre}_{lab}_u{u_ch}_s{s}_valsel.pt",
                                                     run={**W.ckpt("R", s, lab), "s_uR": u_ch})}
                                 for k, lab in enumerate(labs)]
                sb["seeds"][str(s)] = {"table": [], "chosen": chosen[s], "decided_by": "picked"}
            W.record(f"{pre}_B", {"name": f"{pre}_B", "stage_b": {"selector": "from_stage_a", "seeds": seeds}},
                     stage_b=sb)
        cells = [{"label": "iiS", "bundles": [W.bundle(f"mcfRS_{G_CHOSEN[s]}_u3_s{s}_valsel.pt") for s in SEEDS]}]
        W.record("iiS", {"name": "iiS", "stage_a": {"cells": cells}}, stage_a=W.sa("iiS", "mode@0.5"))
        # MCF-A: stage 1 checkpoint per seed at (9, 3); stage 2 knob grid; stage 3 selector
        a_seeds = [0] if 3 in W.degrade else list(SEEDS)
        seeds, sb = {}, {"selector": None, "tie": ["any10@mcf10", "any10@mcf08", "val_loss"], "seeds": {}}
        for s in a_seeds:
            seeds[str(s)] = [{"label": lab, "val_loss": 50.0 + k,
                              "bundle": W.bundle(f"mcfA_{lab}_u3_f9_s{s}_valsel.pt",
                                                 run={**W.ckpt("A", s, lab), "s_uR": 3, "s_uF": 9})}
                             for k, lab in enumerate(sorted({A_CHOSEN[s], "last", "ep2_st28"}))]
            sb["seeds"][str(s)] = {"chosen": A_CHOSEN[s], "decided_by": "any10@mcf10"}
        W.record("mcfA_1", {"name": "mcfA_1", "stage_b": {"selector": None, "seeds": seeds}}, stage_b=sb)
        knobs = (9, 3) if 4 in W.degrade else (13, 2)
        if 4 not in W.degrade:
            cells = [{"label": f"({f},{u})", "bundles": [W.bundle(f"mcfA_{A_CHOSEN[s]}_u{u}_f{f}_s{s}_valsel.pt")
                                                         for s in a_seeds]}
                     for f in (5, 9, 13) for u in (1, 2, 3)]
            W.record("mcfA_2", {"name": "mcfA_2", "stage_a": {"cells": cells, "menu": None, "prefer_cell": "(9,3)"}},
                     stage_a=W.sa("(13,2)", None, ("any10@mcf10", "prefer_cell")), menu=None)
        cells = [{"label": f"({knobs[0]},{knobs[1]})",
                  "bundles": [W.bundle(f"mcfA_{A_CHOSEN[s]}_u{knobs[1]}_f{knobs[0]}_s{s}_valsel.pt") for s in a_seeds]}]
        W.record("mcfA_3", {"name": "mcfA_3", "stage_a": {"cells": cells, "menu": NS.MCFA_MENU}},
                 stage_a=W.sa(cells[0]["label"], "resid", ("picked@mcf10", "selector")), menu=NS.MCFA_MENU)
        # iv-P / iv-S
        ep = {0: 12, 1: 30, 2: 7}
        cand = {str(s): {"best": ep[s] if s != 1 else 9, "last": 30} for s in SEEDS}
        W.write("ivP_window", {"lr": 0.0003, "window": 30, "late_extension": False, "candidates": cand,
                               "scorer_config": "results/n3/private/select/ivP.json"})
        seeds = {str(s): [{"label": f"ep{e:02d}", "bundle": W.bundle(f"ivP_lr3e-4_ep{e:02d}_s{s}_valsel.pt"),
                           "val_loss": 1.0} for e in sorted({cand[str(s)]["best"], 30})] for s in SEEDS}
        W.record("ivP", {"name": "ivP", "mode_tiebreak": "resid",
                         "stage_a": {"cells": [{"label": "best", "bundles": [seeds[str(s)][0]["bundle"] for s in SEEDS]}]},
                         "stage_b": {"selector": "from_stage_a", "seeds": seeds}},
                 stage_a=W.sa("best", "mode@0.5"),
                 stage_b={"selector": "mode@0.5", "seeds": {str(s): {"chosen": f"ep{ep[s]:02d}"} for s in SEEDS}})
        runs = [f"python scripts/n3_old_symmc.py sample --run lr3e-4_s{s} --epoch {ep[s]} --set {st} --arm ivP "
                f"--crystals-per-batch 8 --device cuda --pack" for st in ("testB", "devtest") for s in SEEDS] + \
               [f"python scripts/n3_old_symmc.py ivs --bundle results/n3/private/draws/ivP_s{s}_{st}.pt"
                for st in ("testB", "devtest") for s in SEEDS]
        W.write("ivP_frozen", {"arm": "iv-P", "lr": 0.0003, "window": 30, "selector": "mode@0.5",
                               "epoch_per_seed": {str(s): e for s, e in ep.items()},
                               "checkpoints": {str(s): f"results/n3/private/old/runs/lr3e-4_s{s}/ep{e:02d}.pt"
                                               for s, e in ep.items()},
                               "checkpoint_sha256": {str(s): "2" * 64 for s in SEEDS},
                               "sampling": {"encoder": "packed", "crystals_per_batch": 8, "device_type": "cuda"},
                               "step7_runs": runs})
        cells = [{"label": "ivS", "bundles": [W.bundle(f"ivS_lr3e-4_ep{ep[s]:02d}_s{s}_valsel.pt") for s in SEEDS]}]
        W.record("ivS", {"name": "ivS", "mode_tiebreak": "resid", "stage_a": {"cells": cells}},
                 stage_a=W.sa("ivS", "lj"))
        # diagnostic (a) on every selected checkpoint (+G differs from MCF-R on seeds 1 and 2) and the provenance
        for lab, arm, sha, u, f in ([(f"mcfR_s{s}", "R", W.ck[("R", s, R_CHOSEN[s])], 2, None) for s in SEEDS]
                                    + [(f"mcfRG_s{s}", "R", W.ck[("R", s, G_CHOSEN[s])], 3, None) for s in (1, 2)]
                                    + [(f"mcfA_s{s}", "A", W.ck[("A", s, A_CHOSEN[s])], knobs[1], knobs[0])
                                       for s in a_seeds]
                                    + [("thurlemann", "A", "3" * 64, 3, 9)]):
            W.diag(lab, arm, sha, u, f)

    def diag(self, lab, arm, sha, u, f, verdict="PASS", dev=1.1e-5):
        self.write(f"diag_a:{lab}", {"diagnostic": "a (equivariance)", "label": lab, "arm": arm, "ckpt_sha256": sha,
                                     "reportable": True, "provenance_repeat": lab == "thurlemann",
                                     "knobs": {"s_uR": u, "s_uF": 9.0 if f is None else f, "steps": 50},
                                     "statistics": {"q": 1e-15, "lat": 1e-14, "tr": 2e-15, "R": 6.6e-7},
                                     "verdict": verdict, "sampler_deviation": {"float32_native": {"R": dev}},
                                     "sampler_deviation_threshold_1e-3": {"float32_native": dev <= 1e-3}})


def _run(W, **kw):
    sel, P, I = NS.assemble({**NS.default_inputs(), **W.inputs}, **kw)
    return sel, P


def test_complete_world_assembles_every_frozen_choice(tmp_path):
    W = World(tmp_path)
    sel, P = _run(W)
    assert P.rows == []
    F = sel["frozen"]
    assert F["MCF-R"]["s_uR"] == 2.0 and F["MCF-R"]["selector"] == "lj"
    assert {s: c["ckpt_label"] for s, c in F["MCF-R"]["checkpoints"].items()} == {str(k): v for k, v in R_CHOSEN.items()}
    assert F["MCF-R+G"]["s_uR"] == 3.0 and F["MCF-R+G"]["separate_sampling"] == {"0": True, "1": True, "2": True}
    assert F["ii-S"]["selector"] == "mode@0.5" and F["iii-u"]["selector"] == "random" and F["iii-p"]["selector"] == "lj"
    assert F["P1"]["p1-ours"]["selector"] == "lj_pre" and F["P1"]["p1-iii-p"]["selector"] == "lj_pre"
    assert (F["MCF-A"]["s_uF"], F["MCF-A"]["s_uR"], F["MCF-A"]["selector"]) == (13.0, 2.0, "resid")
    assert F["MCF-A"]["native_stols"] == [0.5, 0.8, 1.0]
    assert F["iv-P"]["selector"] == "mode@0.5" and F["iv-P"]["epoch_per_seed"] == {"0": 12, "1": 30, "2": 7}
    assert F["iv-S"]["selector"] == "lj"
    assert F["OURS"]["selector"] == "lj" and len(F["OURS"]["checkpoints"]) == 3
    assert all(len(c["sha256"]) == 64 for c in F["OURS-N2"]["checkpoints"].values())
    assert sel["h2_form"]["form"] == "BASIN" and "rows" not in json.dumps(sel["h2_form"])
    assert sel["gate_c"]["gate_ca_ii_failed"] is True
    assert sel["report_configs"]["testB"]["gate_ca_failed"] is True
    a = sel["mcf_diagnostic_a"]
    assert a["verdict"] == "PASS" and set(a["per_checkpoint"]) == {"mcfR_s0", "mcfR_s1", "mcfR_s2", "mcfRG_s1",
                                                                   "mcfRG_s2", "mcfA_s0", "mcfA_s1", "mcfA_s2",
                                                                   "thurlemann"}
    # SHA-256 of every input record, and every VALSEL cell score (the scorer records, whole)
    for role in ("degrade", "anchor", "gate_c", "mcfR_A", "mcfR_B", "iiS", "mcfA_2", "ivP_frozen", "diag_a:mcfA_s2"):
        assert sel["inputs"][role]["sha256"] == NS.C.sha256(W.inputs[role])
    assert "stage_a" in sel["valsel_cell_scores"]["mcfR_A"] and "stage_b" in sel["valsel_cell_scores"]["mcfRG_B"]
    assert "NOT_COPIED" not in json.dumps(sel)                        # anchor rows (refcodes) are never copied


def test_run_list_has_every_arm_with_seeds(tmp_path):
    W = World(tmp_path)
    sel, P = _run(W)
    runs = {r["id"]: r for r in sel["runs"]["step7"]}
    for st, sp in (("testB", 1), ("devtest", 2)):
        for b in SEEDS:
            assert f"manual_seed({600000 + 10000 * b + 1000 * sp} + i)" in runs[f"iii-u/{st}/b{b}"]["seeds"]
        for s in SEEDS:
            r = runs[f"MCF-R/{st}/s{s}"]
            assert f"inference.seed = {100000 + 1000 * s + sp}" in r["seeds"]
            assert f"--ckpt n3_mcf/runs/n3_mcfR_s{s}/ckpt/{R_CHOSEN[s]}.ckpt --set {st} --s_uR 2 " in r["commands"][0]
            g = runs[f"MCF-R+G/{st}/s{s}"]                           # +G selected (ckpt, s_uR) of its own
            assert g["commands"][0].startswith("python scripts/n3_mcf_run.py infer") and "--s_uR 3" in g["commands"][0]
            assert g["commands"][-1].startswith("python scripts/n3_mcf_gauge.py")
            assert runs[f"ii-S/{st}/s{s}"]["outputs"] == [f"results/n3/private/draws/mcfRS_{G_CHOSEN[s]}_u3_s{s}_{st}.pt"]
            a = runs[f"MCF-A/{st}/s{s}"]
            assert "--s_uR 2 --s_uF 13" in a["commands"][0] and f"{200000 + 1000 * s + sp}" in a["seeds"]
            assert f"{300000 + 1000 * s + sp}" in runs[f"MCF-A-floor/{st}/s{s}"]["seeds"]
            assert "--from-ie" in runs[f"i-E+basin-ours/{st}/s{s}"]["commands"][1]
            assert runs[f"basin-ours_n2/{st}/s{s}"]["commands"] == [
                f"python scripts/n3_classical.py basin --draws results/n3/private/draws/ours_n2_s{s}_{st}.pt --workers {{G}}"]
            assert f"p1-ours/{st}/s{s}" in runs and f"iv-P/{st}/s{s}" in runs and f"iv-S/{st}/s{s}" in runs
            assert f"{700000 + 10000 * s + 1000 * sp} + i" in runs[f"iv-P/{st}/s{s}"]["seeds"]
        for rid in ("iii-s", "fftruth", "p1ref"):
            assert f"{rid}/{st}" in runs
    assert "80000 + i" in runs["ours/testB"]["seeds"] and "convert" in runs["ours_n2/devtest"]["commands"][0]
    nine = {r["id"] for r in sel["runs"]["step9"]}
    assert {"diag-b/testB", "diag-b/devtest", "diag-c/testB/s0"} <= nine
    cfg = sel["report_configs"]["testB"]
    assert cfg["h2_form"] == "BASIN" and set(cfg["h2_basin"]) == {"OURS", "OURS-N2", "iii-s", "i-E"}
    assert cfg["arms"]["MCF-R+G"]["selector"] == "resid" and cfg["arms"]["floor"]["selector"] == "random"
    sc = " ".join(sel["runs"]["step9_scoring"]["testB"])
    assert "--reference results/n3/private/draws/fftruth_s-1_testB.pt" in sc and "--selector lj_pre" in sc
    assert all("--step9" in c for c in sel["runs"]["step9_scoring"]["testB"] if " match " in c or " basin " in c)


def test_missing_record_refuses_to_write_and_dry_run_lists_it(tmp_path, monkeypatch, capsys):
    W = World(tmp_path)
    os.remove(W.inputs["mcfR_B"])
    inp = tmp_path / "inputs.json"
    json.dump(W.inputs, open(inp, "w"))
    out = tmp_path / "n3_selection.json"
    monkeypatch.setattr(sys, "argv", ["n3_select.py", "--inputs", str(inp), "--out", str(out)])
    with pytest.raises(SystemExit, match="nothing written"):
        NS.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", ["n3_select.py", "--inputs", str(inp), "--dry-run"])
    with pytest.raises(SystemExit) as e:
        NS.main()
    assert e.value.code == 0
    txt = capsys.readouterr().out
    assert "mcfR_B" in txt and "missing" in txt and not out.exists()
    # with the record back the file is written, and never overwritten without --force
    World(tmp_path)                                                  # rebuilds the same (deterministic) records
    monkeypatch.setattr(sys, "argv", ["n3_select.py", "--inputs", str(inp), "--out", str(out)])
    NS.main()
    d = json.load(open(out))
    assert d["frozen"]["MCF-R"]["s_uR"] == 2.0
    with pytest.raises(SystemExit, match="exists"):
        NS.main()


def test_inconsistent_records_are_refused(tmp_path):
    W = World(tmp_path)
    rec = json.load(open(W.inputs["mcfR_B"]))
    rec["stage_b"]["selector"] = "resid"                               # Stage B must use Stage A's selector
    json.dump(rec, open(W.inputs["mcfR_B"], "w"))
    torch.save({"arm": "changed"}, next(iter(json.load(open(W.inputs["iii-u"]))["bundles_sha256"])))
    anc = json.load(open(W.inputs["anchor"]))
    anc["summary"]["anchoring"] = 0.61                                 # FORM BASIN at >= 50% anchoring
    json.dump(anc, open(W.inputs["anchor"], "w"))
    ivf = json.load(open(W.inputs["ivP_frozen"]))
    ivf["epoch_per_seed"]["2"] = 8
    json.dump(ivf, open(W.inputs["ivP_frozen"], "w"))
    _, P = _run(W)
    msgs = {r: " ".join(P.of(r)) for r, _ in P.rows}
    assert "Stage A's lj" in msgs["mcfR_B"]
    assert "changed since it was scored" in msgs["iii-u"]
    assert "50% rule" in msgs["anchor"]
    assert "epochs" in msgs["ivP_frozen"]


def test_degrade_items_shape_selection_and_runs(tmp_path):
    W = World(tmp_path, degrade=[1, 2, 3, 4, 5, 6])
    sel, P = _run(W)
    assert P.rows == [], P.rows
    A = sel["frozen"]["MCF-A"]
    assert A["seeds"] == [0] and (A["s_uF"], A["s_uR"]) == (9.0, 3.0) and A["native_stols"] == [0.8, 1.0]
    ids = [r["id"] for r in sel["runs"]["step7"]]
    assert not [i for i in ids if i.endswith("/devtest") or "/devtest/" in i if not i.startswith("ours")]
    assert "ours/devtest" in ids and "ours_n2/devtest" in ids
    runs = {r["id"]: r for r in sel["runs"]["step7"]}
    assert "--from-ie" in " ".join(runs["i-E+basin-ours/testB/s0"]["commands"])
    for s in (1, 2):                                                   # degrade 6: direct BASIN of the LJ pick
        assert runs[f"i-E+basin-ours/testB/s{s}"]["commands"] == [
            f"python scripts/n3_classical.py basin --draws results/n3/private/draws/ours_s{s}_testB.pt --workers {{G}}"]
        assert f"p1-ours/testB/s{s}" not in runs
    assert "MCF-A/testB/s1" not in runs and "MCF-A/testB/s0" in runs
    mcfa = [c for c in sel["runs"]["step9_scoring"]["testB"] if "mcfA_" in c or "mcfAfloor" in c]
    assert mcfa and not [c for c in mcfa if "--matcher mcf05" in c]  # degrade 2: native sweep {0.8, 1.0}
    W7 = World(tmp_path / "w7", degrade=list(range(1, 8))) if (tmp_path / "w7").mkdir() is None else None
    sel7, P7 = _run(W7)
    assert not [r for r in sel7["runs"]["step7"] if " ie " in " ".join(r["commands"])]
    assert "i-E" not in sel7["report_configs"]["testB"]["arms"]
    # a knob-grid record after degrade item 4, and a non-prefix degrade set, are refused
    W.record("mcfA_2", {"name": "mcfA_2", "stage_a": {"cells": []}}, stage_a=W.sa("x", None), menu=None)
    _, P = _run(W)
    assert any("degrade item 4" in m for m in P.of("mcfA_2"))
    W.write("degrade", {"items": [2], "amendment": "A3"})
    _, P = _run(W)
    assert any("prefix" in m for m in P.of("degrade"))


def test_diagnostic_a_unresolved_or_failed_is_recorded_failed(tmp_path):
    W = World(tmp_path)
    os.remove(W.inputs["diag_a:mcfA_s1"])
    _, P = _run(W)
    assert P.of("diag_a:mcfA_s1")                                      # silently missing: refused
    sel, P = _run(W, diag_a_unresolved=True)
    assert P.rows == [] and sel["mcf_diagnostic_a"]["verdict"] == "FAILED"
    assert sel["mcf_diagnostic_a"]["per_checkpoint"]["mcfA_s1"]["verdict"] == "FAILED"
    assert not [r for r in sel["runs"]["step9"] if r["id"].startswith("diag-")]   # (b)-(d) not reported
    W = World(tmp_path / "h") if (tmp_path / "h").mkdir() is None else None
    W.diag("mcfR_s0", "R", W.ck[("R", 0, R_CHOSEN[0])], 2, None, verdict="HARNESS_CHECK")
    sel, P = _run(W)
    assert sel["mcf_diagnostic_a"]["verdict"] == "FAILED"
    W.diag("mcfR_s0", "R", W.ck[("R", 0, R_CHOSEN[0])], 2, None, dev=2e-3)
    sel, P = _run(W)
    assert sel["mcf_diagnostic_a"]["verdict"] == "PASS" and sel["mcf_diagnostic_a"]["approximate_checkpoints"] == ["mcfR_s0"]
    W.diag("mcfR_s0", "R", "f" * 64, 2, None)                          # (a) on a checkpoint that was not selected
    _, P = _run(W)
    assert any("not the selected one" in m for m in P.of("diag_a:mcfR_s0"))


def test_classical_select_configs(tmp_path):
    cfg = NS.classical_configs(NS.default_inputs())
    assert cfg["iii-u"]["menu"] == NS.MENU_CLASSICAL and cfg["iii-u"]["mode_tiebreak"] == "lj"
    assert cfg["p1-iii-p"]["reference"] == "results/n3/private/draws/p1ref_s-1_valsel.pt"
    assert cfg["p1-iii-p"]["stage_a"]["cells"][0]["bundles"][2] == "results/n3/private/draws/iii-p_s2_valsel.pt"
    assert cfg["p1-ours"]["menu"] == ["lj", "lj_pre", "random"] and "resid" not in cfg["iii-p"]["menu"]
