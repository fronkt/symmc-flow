"""Step-5b driver (scripts/n3_step5b.py): the checkpoint designation on the real training records, and the VR / VA /
VC / VS jobs run end to end against a fake executor in tmp_path whose select records are then checked by
n3_select's own validators. No sampling, matching or CSD content; nothing written outside tmp_path."""
import json
import os
import re
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_score as SC  # noqa: E402
import n3_select as NS  # noqa: E402
import n3_step5b as S5  # noqa: E402

SEEDS = (0, 1, 2)


# ------------------------------------------------------------------------------------------- real designation
def test_designation_of_the_real_training_records():
    if not os.path.exists(os.path.join(REPO, S5.MCF, "train_report", "n3_mcfR_s0.json")):
        pytest.skip("train_report not pulled to this machine")
    d = S5.build_designation(REPO)
    assert {s: d["runs"][f"n3_mcfR_s{s}"]["stage_a"] for s in SEEDS} == \
        {0: "ep1471_st20608", 1: "ep715_st10024", 2: "ep1067_st14952"}
    assert d["runs"]["n3_mcfA_s0"]["stage_a"] == "last" and d["runs"]["n3_mcfA_s2"]["stage_a"] == "last"
    for run, r in d["runs"].items():
        labels = [c["label"] for c in r["candidates"]]
        assert len(labels) == len(set(labels)) == 3 and "last" in labels            # set reading
        assert r["last_identical_to"] not in labels                               # the twin is listed once, as last
        assert len({c["ckpt_sha256"] for c in r["candidates"]}) == 3
        logged = [c["logged_valid_loss"] for c in r["candidates"]]
        assert logged == sorted(logged)
    text = json.dumps(d)
    assert "/anvil" not in text and ":\\\\" not in text                           # public: no box / local paths
    committed = os.path.join(REPO, S5.DESIGNATION)
    if os.path.exists(committed):
        assert json.load(open(committed)) == d


# ------------------------------------------------------------------------------------------- synthetic world
def _w(root, rel, obj):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1)


def synthetic_training(root):
    """Three top-3 checkpoints + last per run; last is byte-identical to the latest epoch (as on Anvil)."""
    fixed = []
    for arm in ("R", "A"):
        for s in SEEDS:
            run = f"n3_mcf{arm}_s{s}"
            ck = os.path.join(root, "n3_mcf", "runs", run, "ckpt")
            os.makedirs(ck, exist_ok=True)
            open(os.path.join(ck, "config.yaml"), "w").write(f"experiment:\n  seed: {s}\n")
            rows = []
            # MCF-A seed 0: the lowest-loss checkpoint is the latest (last's twin), as on the box
            losses = {100: 0.50 + s / 100, 200: 0.40 + s / 100, 300: 0.45 + s / 100}
            if arm == "A" and s == 0:
                losses = {100: 0.5, 200: 0.45, 300: 0.40}
            for e, loss in losses.items():
                f = f"epoch_{e}-step_{e * 14}-loss_{loss:.4f}.ckpt"
                open(os.path.join(ck, f), "wb").write(f"{arm}{s}{e}".encode())
                rows.append({"ckpt": f, "epoch": e, "global_step": e * 14, "logged_valid_loss": loss,
                             "ckpt_sha256": S5.sha256(os.path.join(ck, f))})
            open(os.path.join(ck, "last.ckpt"), "wb").write(f"{arm}{s}300".encode())
            rows.append({"ckpt": "last.ckpt", "epoch": 300, "global_step": 4200, "logged_valid_loss": rows[-1]["logged_valid_loss"],
                         "ckpt_sha256": S5.sha256(os.path.join(ck, "last.ckpt"))})
            _w(root, f"{S5.MCF}/train_report/{run}.json", {"run": run, "ckpt_dir": f"n3_mcf/runs/{run}/ckpt",
                                                           "checkpoints": rows})
            for r in rows:
                lab = "last" if r["ckpt"] == "last.ckpt" else f"ep{r['epoch']}_st{r['global_step']}"
                fixed.append({"run_name": run, "label": lab, "ckpt_sha256": r["ckpt_sha256"],
                              "fixed_val_loss": 1.0 + r["logged_valid_loss"]})
    p = os.path.join(root, S5.MCF, "fixed_val_loss.jsonl")
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in fixed))


def label_of(ckpt):
    b = os.path.basename(ckpt)
    m = S5.CKPT_RE.match(b)
    return "last" if b == "last.ckpt" else f"ep{int(m[1])}_st{int(m[2])}"


class Fake:
    """Executes the driver's commands by writing their outputs; select records are synthesised from each config
    (the n3_score.cmd_select schema) with the winners in `win`."""

    def __init__(self, root, win):
        self.root, self.win, self.calls, self.matched = root, win, [], []

    def p(self, rel):
        return os.path.join(self.root, rel)

    def bundle(self, rel, run=None):
        os.makedirs(os.path.dirname(self.p(rel)), exist_ok=True)
        assert not os.path.exists(self.p(rel)), f"bundle rewritten: {rel}"
        torch.save({"arm": os.path.basename(rel), "set": "valsel", "meta": {"run": run} if run else {}}, self.p(rel))

    def __call__(self, argv, fatal=True):
        argv = [str(a) for a in argv]
        script, sub = argv[1], argv[2] if len(argv) > 2 else None
        self.calls.append(argv)
        opt = lambda k: argv[argv.index(k) + 1] if k in argv else None
        cpu_parallel = {("scripts/n3_mcf_run.py", "to_bundle"), ("scripts/n3_score.py", "match"),
                        ("scripts/n3_old_symmc.py", "sample"), ("scripts/n3_old_symmc.py", "ivs"),
                        ("scripts/n3_ours.py", "sample"), ("scripts/n3_classical.py", "haar"),
                        ("scripts/n3_classical.py", "press"), ("scripts/n3_classical.py", "p1basin"),
                        ("scripts/n3_classical.py", "screen"), ("scripts/n3_classical.py", "search")}
        if (script, sub) in cpu_parallel or script in ("scripts/n3_mcf_gauge.py", "scripts/n3_mcf_sg.py"):
            assert "--workers" in argv, argv
        if script.startswith("scripts/n3_mcf"):
            assert argv[0] == "PYMCF", argv
        else:
            assert argv[0] == "PYSYMMC", argv
        if script == "scripts/n3_mcf_run.py":
            if sub == "check":
                return 0 if os.path.exists(self.p(os.path.join(opt("--run_dir"), "done.json"))) else 1
            if sub == "infer":
                ckpt, u, f = opt("--ckpt"), float(opt("--s_uR")), opt("--s_uF")
                arm = "R" if "/n3_mcfR_" in ckpt else "A"
                s = int(re.search(r"_s(\d)/", ckpt)[1])
                rd = S5.mcf_run_dir(arm, label_of(ckpt), u, s, None if f is None else float(f))
                _w(self.root, f"{rd}/done.json", {"ckpt": ckpt, "u": u, "f": f, "arm": arm, "s": s})
                return 0
            if sub == "to_bundle":
                d = json.load(open(self.p(opt("--run_dir") + "/done.json")))
                f = None if d["f"] is None else float(d["f"])
                lab = label_of(d["ckpt"])
                run = {"ckpt": self.p(d["ckpt"]), "ckpt_sha256": S5.sha256(self.p(d["ckpt"])), "ckpt_label": lab,
                       "train_seed": d["s"], "s_uR": d["u"], "s_uF": 9.0 if f is None else f}
                self.bundle(S5.mcf_bundle("mcfR" if d["arm"] == "R" else "mcfA", lab, d["u"], d["s"], f), run)
                return 0
        if script == "scripts/n3_mcf_gauge.py":
            d = json.load(open(self.p(opt("--run_dir") + "/done.json")))
            src = S5.mcf_bundle("mcfR", label_of(d["ckpt"]), d["u"], d["s"])
            self.bundle(src.replace("/mcfR_", "/mcfRG_"), torch.load(self.p(src), weights_only=False)["meta"]["run"])
            return 0
        if script == "scripts/n3_mcf_sg.py":
            src = opt("--bundle")
            self.bundle(src.replace("/mcfRG_", "/mcfRS_"), torch.load(self.p(src), weights_only=False)["meta"]["run"])
            return 0
        if script == "scripts/n3_mcf_diag.py":
            lab = opt("--label")
            ckpt = opt("--ckpt")
            prov = "--provenance" in argv
            f = opt("--s_uF")
            _w(self.root, f"{S5.MCF}/diag_a_{lab}.json", {
                "diagnostic": "a (equivariance)", "label": lab, "arm": "A" if prov or "/n3_mcfA_" in ckpt else "R",
                "ckpt_sha256": "3" * 64 if prov else S5.sha256(self.p(ckpt)), "reportable": True,
                "provenance_repeat": prov, "set": "x.pkl.gz" if prov else "valsel", "dtype": "float64",
                "device": "cpu", "n_crystals": 10, "G_trials": 3,
                "knobs": {"s_uR": float(opt("--s_uR")), "s_uF": 9.0 if f is None else float(f), "steps": 50},
                "statistics": {"q": 1e-15, "lat": 1e-14, "tr": 2e-15, "R": 6.6e-7},
                "tolerances": {"q": 1e-6, "lat": 1e-6, "tr": 1e-6, "R": 1e-5},
                "requirements_met": {"q": True, "lat": True, "tr": True, "R": True},
                "states": {"generic": {"controls_min": {"q": 0.01, "lat": 0.01, "tr": 0.01, "R": 0.01}}},
                "verdict": "PASS", "sampler_deviation_threshold_1e-3": {"float32_native": True}})
            return 0
        if script == "scripts/n3_score.py" and sub == "select":
            self.select(json.load(open(self.p(opt("--config")))))
            return 0
        if script == "scripts/n3_score.py" and sub == "match":
            bundles = argv[argv.index("--bundles") + 1:]
            parsed = [S5.BUNDLE_RE.match(os.path.basename(b)) for b in bundles]
            for b, m in zip(bundles, parsed):
                assert os.path.exists(self.p(b)), f"matching a bundle that does not exist: {b}"
                assert m is not None, b
            seeds = [int(m[2]) for m in parsed]                   # n3_score.validate_bundles (:218-219, :234)
            assert len(set(seeds)) == len(seeds), f"bundle seeds must be unique integers: {bundles}"
            assert len({SC.arm_identity(m[1]) for m in parsed}) == 1, f"one arm per call: {bundles}"
            self.matched += bundles
            return 0
        return self.other(script, sub, argv, opt)

    def other(self, script, sub, argv, opt):
        D = S5.DRAWS
        if script == "scripts/n3_old_symmc.py" and sub == "sample":
            tag, s = re.fullmatch(r"(lr.+)_s(\d)", opt("--run")).groups()
            self.bundle(f"{D}/ivP_{tag}_ep{int(opt('--epoch')):02d}_s{s}_valsel.pt")
        elif script == "scripts/n3_old_symmc.py" and sub == "select":
            rec = json.load(open(self.p(opt("--record"))))
            ch = {s: v["chosen"] for s, v in rec["stage_b"]["seeds"].items()}
            _w(self.root, f"{S5.OLD}/ivP_frozen.json", {"chosen": ch})
            _w(self.root, S5.cfg_path("ivS"), {"name": "ivS", "matcher": "primary", "mode_tiebreak": "resid",
                                               "stage_a": {"cells": [{"label": "ivS", "bundles": [
                                                   f"{D}/ivS_lr1e-4_{ch[str(s)]}_s{s}_valsel.pt" for s in SEEDS]}],
                                                   "tie": ["selector"]},
                                               "out": S5.rec_path("ivS")})
        elif script == "scripts/n3_old_symmc.py" and sub == "ivs":
            self.bundle(opt("--bundle").replace("/ivP_", "/ivS_"))
        elif script == "scripts/n3_ours.py":
            arm = opt("--arm")
            for s in SEEDS:
                self.bundle(f"{D}/{arm}_s{s}_{'sel' if sub == 'sample' else 'valsel'}.pt")
        elif script == "scripts/n3_classical.py" and sub == "haar":
            self.bundle(f"{D}/iii-u_s{opt('--base')}_valsel.pt")
        elif script == "scripts/n3_classical.py" and sub == "press":
            src = opt("--draws")
            self.bundle(f"{D}/iii-p_s{opt('--base')}_valsel.pt" if src is None else src.replace("/ours_", "/p1-ours_"))
        elif script == "scripts/n3_classical.py" and sub == "p1basin":
            self.bundle(f"{D}/p1ref_s-1_valsel.pt")
        elif script == "scripts/n3_classical.py" and sub == "screen":
            pass
        elif script == "scripts/n3_classical.py" and sub == "search":
            assert not any(a.startswith("--ff-") or a == "--device" for a in argv), argv
            self.bundle(f"{D}/iii-s_s-1_val.pt")
        else:
            raise AssertionError(f"unexpected command {argv}")
        return 0

    def select(self, cfg):
        """n3_score.cmd_select's record schema; the winner of each stage is the configured one."""
        name = cfg["name"]
        rec = {"name": name, "matcher": cfg.get("matcher", "primary"), "config": cfg, "bundles_sha256": {}}
        if cfg.get("reference"):
            rec["reference"] = {"path": cfg["reference"], "sha256": S5.sha256(self.p(cfg["reference"])),
                                "arm": "p1ref", "order": "cand_ref"}
        if "stage_a" in cfg:
            A = cfg["stage_a"]
            menu = A["menu"] if "menu" in A else cfg.get("menu", S5.MENU)
            tie = [A.get("rank", "hits")] + A.get("tie", ["selector"] if menu else [])
            want = self.win[name]["a"]
            rows = []
            for cell in A["cells"]:
                for b in cell["bundles"]:
                    rec["bundles_sha256"][b] = S5.sha256(self.p(b))
                for sel in (menu or [None]):
                    win = (cell["label"], sel) == want
                    row = {"cell": cell["label"], "selector": sel, "n": 400, "order": len(rows),
                           "preferred": cell["label"] == A.get("prefer_cell"), "draw1": 1.0, "any16": 30.0}
                    for k in tie:
                        if k in ("selector", "prefer_cell"):
                            continue
                        row[k] = (12.0 if win else 8.0) if k == tie[0] else 2.0
                        if k.startswith(("hits", "picked")):
                            row[k + "_per_seed"] = [row[k] / len(cell["bundles"])] * len(cell["bundles"])
                    rows.append(row)
            best, decider = NS.ranked(rows, tie)
            assert (best["cell"], best["selector"]) == want
            rec["menu"] = menu
            rec["stage_a"] = {"tie": tie, "table": rows, "chosen": {"cell": best["cell"], "selector": best["selector"]},
                              "decided_by": decider}
        if "stage_b" in cfg:
            B = cfg["stage_b"]
            tie = [B.get("rank", "picked")] + B.get("tie", ["draw1", "any16", "val_loss"])
            sb = {"selector": B.get("selector"), "tie": tie, "seeds": {}}
            for seed, cands in B["seeds"].items():
                rows = []
                for c in cands:
                    rec["bundles_sha256"][c["bundle"]] = S5.sha256(self.p(c["bundle"]))
                    win = c["label"] == self.win[name]["b"][int(seed)]
                    row = {"label": c["label"], "bundle": c["bundle"], "val_loss": float(c["val_loss"]),
                           "order": len(rows), "draw1": 1.0, "any16": 30.0}
                    for k in tie:
                        if k == "val_loss":
                            continue
                        row[k] = (12.0 if win else 8.0) if k == tie[0] else 2.0
                        if k.startswith(("hits", "picked")):
                            row[k + "_per_seed"] = [row[k]]
                    rows.append(row)
                best, decider = NS.ranked(rows, tie)
                assert best["label"] == self.win[name]["b"][int(seed)]
                sb["seeds"][seed] = {"table": rows, "chosen": best["label"], "decided_by": decider}
            rec["stage_b"] = sb
        _w(self.root, cfg["out"], rec)


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = str(tmp_path)
    synthetic_training(root)
    d = S5.build_designation(root)
    _w(root, S5.DESIGNATION, d)
    monkeypatch.setattr(NS.C, "REPO", root)           # n3_select resolves repo-relative paths against the world
    return root, d


def labels(d, arm, s):
    return [c["label"] for c in S5.cands(d, arm, s)]


WIN = {"mcfR_A": {"a": ("u5", "lj")}, "mcfRG_A": {"a": ("u3", "resid")},
       "mcfR_B": {"b": {0: "last", 1: "ep200_st2800", 2: "ep100_st1400"}},
       "mcfRG_B": {"b": {0: "last", 1: "last", 2: "ep100_st1400"}},
       "iiS": {"a": ("iiS", "mode@0.5")},
       "mcfA_1": {"b": {0: "last", 1: "ep100_st1400", 2: "ep200_st2800"}},
       "mcfA_2": {"a": ("f13_u2", None)}, "mcfA_3": {"a": ("f13_u2", "resid")}}


def runner(root, fake):
    X = S5.Runner(root, False, "28", "1", "PYSYMMC", "PYMCF")
    X.call = fake
    return X


def test_designation_set_reading(world):
    root, d = world
    for s in SEEDS:
        r = d["runs"][f"n3_mcfR_s{s}"]
        assert r["last_identical_to"] == "ep300_st4200" and "ep300_st4200" not in labels(d, "R", s)
        assert labels(d, "R", s)[0] == r["stage_a"] == "ep200_st2800"
    assert d["runs"]["n3_mcfA_s0"]["stage_a"] == "last"            # lowest logged loss is last's twin
    assert S5.load_designation(root) == d
    tr = os.path.join(root, S5.MCF, "train_report", "n3_mcfR_s1.json")
    rep = json.load(open(tr))
    rep["checkpoints"][-1]["epoch"] = 299                          # last's SHA names another epoch -> refused
    json.dump(rep, open(tr, "w"))
    with pytest.raises(SystemExit):
        S5.build_designation(root)


def test_stage_b_samples_depend_on_both_stage_a_choices(world):
    _, d = world
    same = S5.stage_b_samples(d, 3.0, 3.0)
    assert len(same) == 6 and {u for _, u, _ in same} == {3.0}
    diff = S5.stage_b_samples(d, 5.0, 3.0)
    assert len(diff) == 12 and {u for _, u, _ in diff} == {3.0, 5.0}
    assert all(lab != S5.stage_a_label(d, s) for lab, _, s in diff)   # Stage-A draws exist at every s_uR


def validate(root, roles):
    P = NS.Problems()
    I = NS.Inputs({r: os.path.join(root, S5.rec_path(r)) for r in roles}, P)
    return P, I


def test_vr_end_to_end_records_pass_n3_select(world):
    root, d = world
    fake = Fake(root, WIN)
    S5.job_vr(runner(root, fake))
    infers = [c for c in fake.calls if c[1:3] == ["scripts/n3_mcf_run.py", "infer"]]
    assert len(infers) == 15 + 12                                  # +G chose another s_uR: candidates at both
    rb = json.load(open(os.path.join(root, S5.rec_path("mcfR_B"))))
    assert rb["config"]["stage_b"]["selector"] == "lj"
    assert all("_u5_" in c["bundle"] for v in rb["config"]["stage_b"]["seeds"].values() for c in v)
    assert [c["val_loss"] for c in rb["config"]["stage_b"]["seeds"]["0"]] == \
        [c["fixed_val_loss"] for c in S5.cands(d, "R", 0)]
    P, I = validate(root, ["mcfR_A", "mcfR_B", "mcfRG_A", "mcfRG_B", "iiS"])
    R = NS.mcf_r(P, I, "mcfR", False)
    G = NS.mcf_r(P, I, "mcfRG", False, same_cells_as=R["_stage_a_keys"])
    assert (R["s_uR"], R["selector"], G["s_uR"], G["selector"]) == (5.0, "lj", 3.0, "resid")
    assert {s: c["ckpt_label"] for s, c in R["checkpoints"].items()} == {"0": "last", "1": "ep200_st2800",
                                                                          "2": "ep100_st1400"}
    iis = NS.single_cell(P, I, "iiS", r"^mcfRS_.+_s{s}_valsel\.pt$", NS.MENU)
    got = {str(x["seed"]): f"{x['label']}_u{NS.g(x['u'])}" for x in map(NS.parse_mcf, iis["bundles"])}
    assert got == {str(s): f"{c['ckpt_label']}_u3" for s, c in G["checkpoints"].items()}
    for lab in ["mcfR_s0", "mcfR_s1", "mcfR_s2", "mcfRG_s1", "thurlemann"]:
        I.paths[f"diag_a:{lab}"] = os.path.join(root, S5.MCF, f"diag_a_{lab}.json")
    a = NS.diag_a(P, I, {"MCF-R": R, "MCF-R+G": G, "MCF-A": {}}, set(), False)
    assert P.rows == [], P.rows
    assert set(a["per_checkpoint"]) == {"mcfR_s0", "mcfR_s1", "mcfR_s2", "mcfRG_s1", "thurlemann"}  # +G s0/s2 = MCF-R's
    assert not os.path.exists(os.path.join(root, S5.MCF, "diag_a_mcfRG_s0.json"))
    n = len(fake.calls)                                           # resubmission: nothing re-sampled or re-scored
    S5.job_vr(runner(root, fake))
    again = fake.calls[n:]
    assert not [c for c in again if c[2:3] in (["infer"], ["to_bundle"], ["select"])
                or c[1] in ("scripts/n3_mcf_gauge.py", "scripts/n3_mcf_sg.py", "scripts/n3_mcf_diag.py")]


def test_vr_same_s_ur_needs_no_extra_draws(world):
    root, _ = world
    win = dict(WIN, mcfRG_A={"a": ("u5", "mode@1.0")})
    fake = Fake(root, win)
    S5.job_vr(runner(root, fake))
    assert len([c for c in fake.calls if c[2:3] == ["infer"]]) == 15 + 6


def test_va_end_to_end_records_pass_n3_select(world):
    root, d = world
    fake = Fake(root, WIN)
    S5.job_va(runner(root, fake))
    infers = [c for c in fake.calls if c[2:3] == ["infer"]]
    assert len(infers) == 9 + 24
    assert {c[c.index("--matcher") + 1] for c in fake.calls if c[2:3] == ["match"]} == {"mcf10", "mcf08"}
    r1 = json.load(open(os.path.join(root, S5.rec_path("mcfA_1"))))
    assert [c["val_loss"] for c in r1["config"]["stage_b"]["seeds"]["0"]] == \
        [c["logged_valid_loss"] for c in S5.cands(d, "A", 0)]
    P, I = validate(root, ["mcfA_1", "mcfA_2", "mcfA_3"])
    A = NS.mcf_a(P, I, set(), False)
    assert (A["s_uF"], A["s_uR"], A["selector"]) == (13.0, 2.0, "resid")
    for s in SEEDS:
        I.paths[f"diag_a:mcfA_s{s}"] = os.path.join(root, S5.MCF, f"diag_a_mcfA_s{s}.json")
    I.paths["diag_a:thurlemann"] = os.path.join(root, "absent.json")
    NS.diag_a(P, I, {"MCF-A": A}, set(), True)                    # thurlemann is VR's
    assert P.rows == [], P.rows


def test_select_refuses_a_changed_config(world):
    root, d = world
    X = runner(root, Fake(root, WIN))
    os.makedirs(os.path.join(root, S5.SEL), exist_ok=True)
    _w(root, S5.cfg_path("mcfR_A"), {"name": "mcfR_A", "other": True})
    with pytest.raises(SystemExit):
        X.select("mcfR_A", S5.cfg_mcfr_a(d, "mcfR"))


def test_ivp_runs_follow_the_printed_window_commands():
    win_p = os.path.join(REPO, S5.OLD, "selection_window.json")
    if not os.path.exists(win_p):
        pytest.skip("selection_window.json not pulled to this machine")
    win = json.load(open(win_p))
    runs = S5.ivp_runs(win)
    assert sorted((s, e) for _, e, s, _ in runs) == [(0, 9), (0, 30), (1, 23), (1, 30), (2, 7), (2, 30)]
    bad = dict(win, next_commands=[c.replace("--crystals-per-batch 8", "--crystals-per-batch 4")
                                   for c in win["next_commands"]])
    with pytest.raises(SystemExit):
        S5.ivp_runs(bad)


def test_vc_and_vs_commands(tmp_path):
    root = str(tmp_path)
    win = {"lr": 0.0001, "window": 30, "scorer_config": S5.cfg_path("ivP"),
           "candidates": {"0": {"best": 9, "last": 30}, "1": {"best": 23, "last": 30}, "2": {"best": 7, "last": 30}},
           "next_commands": [f"python scripts/n3_old_symmc.py sample --run lr1e-4_s{s} --epoch {e} --set valsel "
                             "--pack --crystals-per-batch 8 --device cuda" for s, es in ((0, (9, 30)), (1, (23, 30)),
                                                                                          (2, (7, 30))) for e in es]}
    _w(root, f"{S5.OLD}/selection_window.json", win)
    seeds = {str(s): [{"label": f"ep{e:02d}", "bundle": f"{S5.DRAWS}/ivP_lr1e-4_ep{e:02d}_s{s}_valsel.pt",
                       "val_loss": 5.3} for e in es] for s, es in ((0, (9, 30)), (1, (23, 30)), (2, (7, 30)))}
    _w(root, S5.cfg_path("ivP"), {"name": "ivP", "matcher": "primary", "mode_tiebreak": "resid",
                                  "stage_a": {"cells": [{"label": "best", "bundles": [v[0]["bundle"] for v in seeds.values()]}],
                                              "tie": ["selector"]},
                                  "stage_b": {"selector": "from_stage_a", "tie": ["draw1", "any16", "val_loss"],
                                              "seeds": seeds},
                                  "out": S5.rec_path("ivP")})
    for role, cfg in NS.classical_configs(NS.default_inputs()).items():   # repo-relative paths
        _w(root, S5.cfg_path(role), cfg)
    for arm in ("ours", "ours_n2"):
        for s in SEEDS:
            p = os.path.join(root, S5.DRAWS, f"{arm}_s{s}_val.pt")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            torch.save({"arm": arm}, p)
    _w(root, f"{S5.CLS}/anchor_valsel.json", {"summary": {"form": "EXACT"}})
    win_sel = {"ivP": {"a": ("best", "lj"), "b": {0: "ep09", 1: "ep30", 2: "ep07"}}, "ivS": {"a": ("ivS", "resid")},
               "iii-u": {"a": ("iii-u", "random")}, "iii-p": {"a": ("iii-p", "lj")},
               "p1-ours": {"a": ("p1-ours", "lj_pre")}, "p1-iii-p": {"a": ("p1-iii-p", "lj")}}
    fake = Fake(root, win_sel)
    S5.job_vc(runner(root, fake))
    S5.job_vs(runner(root, fake))
    seq = [(c[1], c[2]) for c in fake.calls]
    assert seq.index(("scripts/n3_ours.py", "concat")) < seq.index(("scripts/n3_classical.py", "press"))
    assert [c for c in fake.calls if c[1:3] == ["scripts/n3_ours.py", "sample"]][-1][4] == "ours_n2"   # N2 last
    samples = [c for c in fake.calls if c[1:3] == ["scripts/n3_old_symmc.py", "sample"]]
    assert len(samples) == 6 and all(c[c.index("--set") + 1:c.index("--set") + 7] ==
                                     ["valsel"] + S5.IVP_FLAGS for c in samples)
    ref_matches = [c for c in fake.calls if c[2:3] == ["match"] and "--reference" in c]
    assert len(ref_matches) == 2 and all(c[c.index("--reference") + 1].endswith("p1ref_s-1_valsel.pt") for c in ref_matches)
    for role in ("ivP", "ivS") + S5.CLASSICAL_ROLES:
        cfg = json.load(open(os.path.join(root, S5.cfg_path(role))))
        assert os.path.exists(os.path.join(root, cfg["out"])), role
    search = [c for c in fake.calls if c[1:3] == ["scripts/n3_classical.py", "search"]]
    assert len(search) == 1 and search[0][search[0].index("--workers") + 1] == "1"
    n = len(fake.calls)
    S5.job_vs(runner(root, fake))
    assert len(fake.calls) == n                                   # the VAL row exists: nothing re-run


def test_validators_are_live_on_the_emitted_records(world):
    """Negative control: the same records with Stage B's selector changed are refused by n3_select."""
    root, _ = world
    S5.job_vr(runner(root, Fake(root, WIN)))
    p = os.path.join(root, S5.rec_path("mcfR_B"))
    rec = json.load(open(p))
    rec["stage_b"]["selector"] = "random"
    json.dump(rec, open(p, "w"))
    P, I = validate(root, ["mcfR_A", "mcfR_B"])
    NS.mcf_r(P, I, "mcfR", False)
    assert any("Stage B selector" in m for _, m in P.rows), P.rows


def test_match_groups_mirror_validate_bundles():
    d = S5.DRAWS
    ivp = [f"{d}/ivP_lr1e-4_ep{e:02d}_s{s}_valsel.pt" for s, es in ((0, (9, 30)), (1, (23, 30)), (2, (7, 30)))
           for e in es]
    groups = S5.match_groups(ivp)
    assert groups == [ivp[0::2], ivp[1::2]]                       # seeds 0, 1, 2 each; one ivP_lr1e-4 identity
    per_seed = [S5.mcf_bundle("mcfA", lab, 3, 0, 9) for lab in ("last", "ep648_st9086", "ep752_st10542")]
    assert S5.match_groups(per_seed) == [[b] for b in per_seed]
    cell = [S5.mcf_bundle("mcfR", lab, 5, s) for s, lab in enumerate(("ep1", "ep2", "last"))]
    assert S5.match_groups(cell) == [cell]
    mixed = [S5.mcf_bundle("mcfR", "a", 3, 0), S5.mcf_bundle("mcfR", "a", 5, 1), "odd.pt"]
    assert S5.match_groups(mixed) == [[mixed[0]], [mixed[1]], ["odd.pt"]]
    for name in ("mcfR_ep1_st2_u3", "mcfA_last_u2_f13", "ivP_lr1e-4_ep09", "ivS_lr1e-4_ep30", "iii-p", "p1-ours"):
        assert S5.arm_identity(name) == SC.arm_identity(name)


def test_every_match_call_is_valid_and_complete(world):
    """Regression (review 2026-11-06): per-seed Stage-B / stage-1 matches carried one seed several times."""
    root, d = world
    fake = Fake(root, WIN)
    S5.job_vr(runner(root, fake))
    S5.job_va(runner(root, fake))
    for s in SEEDS:                                               # every candidate bundle was matched
        for c in S5.cands(d, "A", s):
            assert fake.matched.count(S5.mcf_bundle("mcfA", c["label"], 3, s, 9)) >= 2      # mcf10 + mcf08 (+ the (9,3) cell)
        for c in S5.cands(d, "R", s):
            assert S5.mcf_bundle("mcfR", c["label"], 5, s) in fake.matched


class FailingInfer(Fake):
    def __call__(self, argv, fatal=True):
        argv = [str(a) for a in argv]
        if argv[2:3] == ["infer"]:
            self.calls.append(argv)
            os.makedirs(self.p(S5.mcf_run_dir("R", "ep200_st2800", 1.0, 0)), exist_ok=True)
            return 1
        return super().__call__(argv, fatal)


def test_mcf_sampling_regenerates_once_per_submission(world):
    root, d = world
    c = S5.cand(d, "R", 0, "ep200_st2800")
    fake = FailingInfer(root, WIN)
    with pytest.raises(SystemExit):                               # fresh run fails -> one regeneration -> stop
        runner(root, fake).mcf_sample("R", 0, c, 1.0)
    infers = [a for a in fake.calls if a[2:3] == ["infer"]]
    assert len(infers) == 2 and "--discard" not in infers[0] and "--discard" in infers[1]
    fake.calls.clear()                                            # resubmission over the failed run dir:
    with pytest.raises(SystemExit):                               # exactly one regeneration, then stop
        runner(root, fake).mcf_sample("R", 0, c, 1.0)
    infers = [a for a in fake.calls if a[2:3] == ["infer"]]
    assert len(infers) == 1 and "--discard" in infers[0]


def test_truncated_record_is_named_not_a_traceback(world, capsys):
    root, d = world
    X = runner(root, Fake(root, WIN))
    cfg = S5.cfg_mcfr_a(d, "mcfR")
    os.makedirs(os.path.join(root, S5.SEL), exist_ok=True)
    open(os.path.join(root, S5.cfg_path("mcfR_A")), "w").write('{"name": "mcfR_A", "matc')
    with pytest.raises(SystemExit, match="truncated by a wall-time kill"):
        X.select("mcfR_A", cfg)
    os.remove(os.path.join(root, S5.cfg_path("mcfR_A")))
    open(os.path.join(root, S5.rec_path("mcfR_A")), "w").write("{")
    with pytest.raises(SystemExit, match="truncated by a wall-time kill"):
        X.score("mcfR_A", S5.cfg_path("mcfR_A"), cfg)
