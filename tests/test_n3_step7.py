"""Step-7 driver (scripts/n3_step7.py) on the REAL published freeze (results/n3/n3_selection.json, public): job
membership, verbatim commands, refusals, and the 2026-11-08 resumption rule, against a fake executor in tmp_path.
No sampling, FF* or CSD content; nothing is written outside tmp_path."""
import copy
import json
import os
import shlex
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_step7 as S7  # noqa: E402

SEL_PATH = os.path.join(REPO, S7.SELECTION)
pytestmark = pytest.mark.skipif(not os.path.exists(SEL_PATH), reason="no published freeze")
SUB = {"{W}": "28", "{G}": "1", "{GPU}": "0"}


def real_sel():
    return json.load(open(SEL_PATH, encoding="utf-8"))


def is_dir_output(rel):
    return "." not in os.path.basename(rel)


class Fake:
    """Writes each command's outputs and side outputs; infer refuses a run dir that already holds predictions (as
    n3_mcf_run does). fail[(rid, i)] = n makes that command fail n times (after its partial writes)."""

    def __init__(self, X):
        self.X, self.calls, self.fail, self.check_pass, self.gauge_into = X, [], {}, True, set()
        self.table = {}
        for rid in X.order:
            if X.mode(rid) == "uploaded":
                continue
            for i, c in enumerate(X.runs[rid]["commands"]):
                key = tuple(X.argv(c)[1:])
                assert key not in self.table, key
                self.table[key] = (rid, i)

    def p(self, rel):
        return os.path.join(self.X.root, rel)

    def write(self, rel, data):
        os.makedirs(os.path.dirname(self.p(rel)), exist_ok=True)
        open(self.p(rel), "w").write(data)

    def __call__(self, argv):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        script = argv[1]
        assert argv[0] == ("PYMCF" if script in S7.MCF_SCRIPTS else "PYSYMMC"), argv
        rid, i = self.table[tuple(argv[1:])]
        cmds, outs = self.X.runs[rid]["commands"], self.X.runs[rid]["outputs"]
        t, _, sub = S7.parse(cmds[i])
        if (script, sub) == ("scripts/n3_mcf_run.py", "infer"):
            d = S7.opt(shlex.split(cmds[i + 1]), "--run_dir")
            if os.path.exists(self.p(f"{d}/out/predictions_16.pt")):
                return 2                                            # "already holds a run"
            self.write(f"{d}/out/predictions_16.pt", f"pred {rid}")
            self.write(f"{d}/check.json", json.dumps({"pass": self.check_pass, "failed": [] if self.check_pass
                                                      else ["finite_R"]}))
            if rid in self.gauge_into:
                self.write(f"{d}/gauge_16_step5.jsonl", f"stray rows {rid}")
        if script == "scripts/n3_mcf_gauge.py":
            d = S7.opt(argv, "--run_dir")
            assert os.path.isdir(self.p(d)), d
            self.write(f"{d}/gauge_16_step5.jsonl", f"rows {rid}")
        if self.fail.get((rid, i), 0) > 0:
            self.fail[(rid, i)] -= 1
            return 1
        if (script, sub) == ("scripts/n3_mcf_run.py", "to_bundle"):
            self.write(S7.mcf_bundle_of(S7.opt(argv, "--run_dir")), f"bundle {rid}")
        if (script, sub) == ("scripts/n3_old_symmc.py", "sample"):
            for o in outs:
                self.write(o[:-3] + ".rot.pt", f"rot {rid}")
        if script == "scripts/n3_mcf_gauge.py":
            for o in outs:
                self.write(f"{S7.GAUGE_COUNTS}/{os.path.basename(o)[:-3]}.json", f"counts {rid}")
        if i == len(cmds) - 1:
            for o in outs:
                if is_dir_output(o):
                    os.makedirs(self.p(o), exist_ok=True)
                    if not os.listdir(self.p(o)):
                        self.write(f"{o}/n3_run.json", f"run {rid}")
                else:
                    self.write(o, f"out {rid}")
        return 0

    def executed(self):
        return [self.table[tuple(a[1:])] for a in self.calls]


@pytest.fixture
def world(tmp_path):
    sel = real_sel()
    root = str(tmp_path)
    for row in sel["step7_inputs"]:
        p = os.path.join(root, row["path"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(f"input {row['role']}")
        row["sha256"] = S7.sha256(p)
    return root, sel


def runner(root, sel, dry=False):
    X = S7.Runner(root, sel, "f" * 64, 28, 1, 0, "PYSYMMC", "PYMCF", dry=dry, driver_sha="d" * 64)
    fake = Fake(X)
    X.call = fake
    return X, fake


def events(root):
    p = os.path.join(root, S7.STATE, "events.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


# ------------------------------------------------------------------------------------------- membership
def test_every_freeze_run_is_in_exactly_one_job():
    sel = real_sel()
    runs = sel["runs"]["step7"]
    assert len(runs) == 76
    table = S7.jobs_table(runs)
    assert set(table) == set(S7.JOBS)
    ids = [rid for j in S7.JOBS for rid in table[j]]
    assert sorted(ids) == sorted(r["id"] for r in runs) and len(ids) == len(set(ids))
    assert {j: len(v) for j, v in table.items()} == {"7O": 5, "7C": 27, "7M": 30, "7S": 1, "7Sd": 1, "7E0": 2,
                                                     "7E1": 2, "7E2": 2, "7Ed": 6}
    assert table["7S"] == ["iii-s/testB"] and table["7Sd"] == ["iii-s/devtest"]
    assert set(table["7E1"]) == {"i-E+basin-ours/testB/s1", "basin-ours_n2/testB/s1"}


def test_inputs_come_from_earlier_runs_of_the_job_or_from_7O_or_uploads(world):
    root, sel = world
    X, _ = runner(root, sel)
    pos = {rid: k for k, rid in enumerate(X.order)}
    for rid in X.order:
        if X.mode(rid) == "uploaded":
            continue
        made = X.made_in_run(rid)
        for c in X.runs[rid]["commands"]:
            t = S7.parse(c)[0]
            for flag in S7.INPUT_FLAGS:
                rel = S7.opt(t, flag)
                prod = X.producer.get(rel)
                if rel is None or rel in made or prod is None:
                    continue
                if X.mode(prod) == "uploaded":
                    continue
                if X.job[prod] == X.job[rid]:
                    assert pos[prod] < pos[rid], (rid, prod)
                else:
                    assert X.job[prod] in S7.AFTEROK.get(X.job[rid], ()), (rid, prod)


def test_modes_follow_the_clarification(world):
    root, sel = world
    X, _ = runner(root, sel)
    modes = {rid: X.mode(rid) for rid in X.order}
    assert modes["ours/devtest"] == modes["ours_n2/devtest"] == "uploaded"
    for rid in ("iii-u/testB/b0", "iii-s/testB", "fftruth/devtest", "i-E+basin-ours/testB/s2", "ours/testB",
                "p1ref/testB", "basin-ours_n2/devtest/s0", "p1-ours/testB/s1"):
        assert modes[rid] == "resume", rid
    for rid in ("MCF-R/testB/s0", "MCF-R+G/testB/s0", "MCF-R+G/devtest/s1", "ii-S/testB/s2", "MCF-A/testB/s1",
                "MCF-A-floor/devtest/s0", "iv-P/testB/s0", "iv-S/devtest/s2"):
        assert modes[rid] == "restart", rid


# ------------------------------------------------------------------------------------------- full runs
def test_all_jobs_run_every_command_once_verbatim(world):
    root, sel = world
    X, fake = runner(root, sel)
    for j in S7.JOBS:
        assert X.run_job(j)
    done = fake.executed()
    want = [(rid, i) for rid in X.order if X.mode(rid) != "uploaded" for i in range(len(X.runs[rid]["commands"]))]
    assert sorted(done) == sorted(want) and len(done) == len(set(done))
    for argv, (rid, i) in zip(fake.calls, done):
        tmpl = shlex.split(X.runs[rid]["commands"][i])
        exp = [SUB.get(x, x) for x in tmpl]
        exp[0] = "PYMCF" if tmpl[1] in S7.MCF_SCRIPTS else "PYSYMMC"
        assert argv == exp
        assert not any(x in ("match", "anchor", "report") for x in argv)
        assert argv[1] not in ("scripts/n3_score.py", "scripts/n3_mcf_diag.py", "scripts/n3_select.py")
        assert not any(x.endswith(tuple(S7.PLACEHOLDERS)) for x in argv)
    for rid in X.order:
        rec = json.load(open(X.rec_path(rid)))
        assert rec["commands"] == X.runs[rid]["commands"] and set(rec["outputs"]) == set(X.runs[rid]["outputs"])
    for j in S7.JOBS:
        assert os.path.exists(os.path.join(root, S7.STATE, f"{j}_checksums.tsv"))
    assert not any(a[1:3] == ["scripts/n3_ours.py", "convert"] for a in fake.calls)


def test_a_done_run_is_skipped_and_a_changed_output_stops(world):
    root, sel = world
    X, fake = runner(root, sel)
    X.run_job("7O")
    n = len(fake.calls)
    X.run_job("7O")
    assert len(fake.calls) == n
    open(os.path.join(root, "results/n3/private/draws/ours_s0_testB.pt"), "w").write("tampered")
    with pytest.raises(SystemExit, match="outputs changed after the run finished"):
        X.run_job("7O")


def test_plan_executes_nothing(world):
    root, sel = world
    X, fake = runner(root, sel, dry=True)
    for j in S7.JOBS:
        X.run_job(j)
    assert fake.calls == [] and not os.path.exists(os.path.join(root, S7.STATE))


# ------------------------------------------------------------------------------------------- resumption
def ev_of(root, rid):
    return [e["event"] for e in events(root) if e["run"] == rid]


def test_resumable_run_reinvokes_its_commands_after_a_stop(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.fail[("iii-u/testB/b0", 1)] = 1                             # press stops after haar finished
    with pytest.raises(SystemExit, match="iii-u/testB/b0: rc 1"):    # no in-job retry for a resumable run
        X.run_job("7C")
    assert fake.executed().count(("iii-u/testB/b0", 1)) == 1
    X2, fake2 = runner(root, sel)
    X2.run_job("7C")
    assert fake2.executed()[:2] == [("iii-u/testB/b0", 0), ("iii-u/testB/b0", 1)]   # haar re-invoked (resumes)
    assert ev_of(root, "iii-u/testB/b0") == ["start", "command_failed", "attempt_failed", "resume", "done"]
    assert X2.record("iii-u/testB/b0") is not None


def test_single_call_run_gets_one_in_job_retry(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.fail[("MCF-R/testB/s0", 1)] = 1                             # to_bundle fails once after infer wrote the dir
    X.run_job("7M")                                                  # infer would refuse (rc 2) without the delete
    assert ev_of(root, "MCF-R/testB/s0") == ["start", "command_failed", "attempt_failed", "deleted_partial",
                                             "retry_in_job", "done"]
    assert fake.executed()[:4] == [("MCF-R/testB/s0", 0), ("MCF-R/testB/s0", 1)] * 2


def test_single_call_run_failing_twice_stops_then_restarts_on_resubmission(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.fail[("MCF-R/testB/s0", 1)] = 2
    with pytest.raises(SystemExit, match=r"MCF-R/testB/s0: rc 1 .* \(after its one in-job retry\)"):
        X.run_job("7M")
    assert os.path.exists(os.path.join(root, "n3_mcf/infer/mcfR/testB/s0_last_u5/out/predictions_16.pt"))
    X2, fake2 = runner(root, sel)
    X2.run_job("7M")
    assert ev_of(root, "MCF-R/testB/s0")[-3:] == ["deleted_partial", "restart", "done"]
    assert fake2.executed()[:2] == [("MCF-R/testB/s0", 0), ("MCF-R/testB/s0", 1)]


def test_gauge_restart_keeps_the_scanned_mcf_run(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.fail[("MCF-R+G/testB/s1", 0)] = 2                           # gauge rows written, then the scan stops twice
    with pytest.raises(SystemExit, match=r"MCF-R\+G/testB/s1: rc 1"):
        X.run_job("7M")
    src = os.path.join(root, "n3_mcf/infer/mcfR/testB/s1_last_u5")
    pred = open(os.path.join(src, "out/predictions_16.pt")).read()
    X2, fake2 = runner(root, sel)
    gone = X2.delete_partial("MCF-R+G/testB/s1", "7M")
    assert gone == ["n3_mcf/infer/mcfR/testB/s1_last_u5/gauge_16_step5.jsonl"]
    X2.run_job("7M")
    assert open(os.path.join(src, "out/predictions_16.pt")).read() == pred
    assert X2.record("MCF-R/testB/s1") is not None                   # the MCF-R record still holds
    assert ("MCF-R/testB/s1", 0) not in fake2.executed()


def test_restart_targets_recorded_by_another_finished_run_are_kept(world):
    root, sel = world
    # a restart target (the sibling .jsonl of MCF-A-floor's bundle) that a finished earlier run recorded
    s = _tampered(sel, "MCF-R/testB/s0", outputs=["n3_mcf/infer/mcfR/testB/s0_last_u5",
                                                  "results/n3/private/draws/mcfR_last_u5_s0_testB.pt",
                                                  "results/n3/private/draws/mcfAfloor_s0_testB.jsonl"])
    X, fake = runner(root, s)
    sib = os.path.join(root, "results/n3/private/draws/mcfAfloor_s0_testB.jsonl")
    with pytest.raises(SystemExit, match="MCF-A-floor/testB/s0: results/n3/private/draws/mcfAfloor_s0_testB.jsonl "
                                         "holds a finished run's recorded output"):
        X.run_job("7M")
    assert open(sib).read() == "out MCF-R/testB/s0"
    assert X.record("MCF-R/testB/s0") is not None


def test_gauge_glob_target_recorded_by_the_mcf_run_is_kept(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.gauge_into.add("MCF-R/testB/s1")                            # MCF-R s1 finished with a gauge_* file inside
    with pytest.raises(SystemExit, match=r"MCF-R\+G/testB/s1: .*gauge_16_step5.jsonl holds a finished run's "
                                         "recorded output"):
        X.run_job("7M")
    rows = os.path.join(root, "n3_mcf/infer/mcfR/testB/s1_last_u5/gauge_16_step5.jsonl")
    assert open(rows).read() == "stray rows MCF-R/testB/s1"
    assert X.record("MCF-R/testB/s1") is not None


def test_one_output_listed_by_two_runs_is_refused(world):
    root, sel = world
    s = copy.deepcopy(sel)
    for r in s["runs"]["step7"]:
        if r["id"] == "MCF-R+G/testB/s1":
            r["outputs"].append("results/n3/private/draws/mcfR_last_u5_s1_testB.pt")
    with pytest.raises(SystemExit, match="is listed by both"):
        runner(root, s)


def test_cut_off_shard_line_is_trimmed_before_a_resume(world):
    root, sel = world
    shard = os.path.join(root, S7.CLS, "iii-u_s0_testB.jsonl")
    whole = os.path.join(root, S7.CLS, "iii-u_s1_testB.jsonl")
    other = os.path.join(root, S7.CLS, "screen_testB.jsonl")         # job 7S's shard: never touched by 7C
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    open(shard, "wb").write(b'{"refcode": "A"}\n{"refc')
    open(whole, "wb").write(b'{"refcode": "B"}\n')
    open(other, "wb").write(b'{"refcode": "C"}\n{"ref')
    X, fake = runner(root, sel)
    X.run_job("7C")
    assert open(shard, "rb").read() == b'{"refcode": "A"}\n'
    assert open(whole, "rb").read() == b'{"refcode": "B"}\n'
    assert open(other, "rb").read() == b'{"refcode": "C"}\n{"ref'
    rep = [e for e in events(root) if e["event"] == "repaired_shard"]
    assert len(rep) == 1 and rep[0]["run"] == "iii-u/testB/b0"
    d = rep[0]["detail"]
    assert d["path"] == f"{S7.CLS}/iii-u_s0_testB.jsonl" and d["bytes_removed"] == 6
    assert d["sha256_after"] == S7.sha256(shard) and d["sha256_before"] != d["sha256_after"]


def test_a_cut_off_file_recorded_by_a_finished_run_is_not_trimmed(world):
    root, sel = world
    rel = f"{S7.CLS}/iii-u_s0_testB.jsonl"
    outs = next(r for r in sel["runs"]["step7"] if r["id"] == "ours/testB")["outputs"]
    X, fake = runner(root, _tampered(sel, "ours/testB", outputs=list(outs) + [rel]))
    X.run_job("7O")                                                  # the fake writes it without a final newline
    before = open(os.path.join(root, rel), "rb").read()
    with pytest.raises(SystemExit, match="recorded by a finished run; not touched"):
        X.run_job("7C")
    assert open(os.path.join(root, rel), "rb").read() == before


def test_classical_shard_names_follow_n3_classical(world):
    root, sel = world
    X, _ = runner(root, sel)
    C = S7.CLS
    assert X.classical_shards("iii-u/testB/b0") == [f"{C}/iii-u_s0_testB.jsonl", f"{C}/iii-p_s0_testB.jsonl"]
    assert X.classical_shards("p1-ours/devtest/s1") == [f"{C}/p1-ours_s1_devtest.jsonl"]
    assert X.classical_shards("i-E+basin-ours/testB/s2") == [f"{C}/i-E-ours_s2_testB.jsonl"]   # basin --from-ie: none
    assert X.classical_shards("basin-ours_n2/testB/s0") == [f"{C}/basin-ours_n2_s0_testB.jsonl"]
    assert X.classical_shards("iii-s/testB") == [f"{C}/screen_testB.jsonl", f"{C}/iii-s_testB.jsonl"]
    assert X.classical_shards("fftruth/devtest") == [f"{C}/fftruth_devtest.jsonl"]
    assert X.classical_shards("p1ref/testB") == [f"{C}/p1ref_testB.jsonl"]
    assert X.classical_shards("MCF-R/testB/s0") == [] and X.classical_shards("ours/testB") == []


def test_mcf_bundle_naming_matches_the_freeze(world):
    root, sel = world
    X, _ = runner(root, sel)
    side = {}
    for rid in X.order:
        if X.mode(rid) == "uploaded":
            continue
        for c in X.runs[rid]["commands"]:
            t, script, sub = S7.parse(c)
            if (script, sub) == ("scripts/n3_mcf_run.py", "to_bundle"):
                b = S7.mcf_bundle_of(S7.opt(t, "--run_dir"))
                if not rid.startswith("MCF-R+G/"):
                    assert b in X.runs[rid]["outputs"], (rid, b)        # the naming rule reproduces the freeze
                else:
                    side[rid] = b
    assert side == {"MCF-R+G/testB/s0": "results/n3/private/draws/mcfR_ep1331_st18648_u5_s0_testB.pt",
                    "MCF-R+G/devtest/s0": "results/n3/private/draws/mcfR_ep1331_st18648_u5_s0_devtest.pt"}


def test_side_outputs_are_checksummed(world):
    root, sel = world
    X, fake = runner(root, sel)
    X.run_job("7M")
    X.run_job("7C")
    g = json.load(open(X.rec_path("MCF-R+G/testB/s0")))["side_outputs"]
    assert set(g) == {"results/n3/private/draws/mcfR_ep1331_st18648_u5_s0_testB.pt",
                      f"{S7.GAUGE_COUNTS}/mcfRG_ep1331_st18648_u5_s0_testB.json"}
    assert set(json.load(open(X.rec_path("MCF-R+G/testB/s1")))["side_outputs"]) == \
        {f"{S7.GAUGE_COUNTS}/mcfRG_last_u5_s1_testB.json"}
    assert set(json.load(open(X.rec_path("iv-P/devtest/s2")))["side_outputs"]) == \
        {"results/n3/private/draws/ivP_s2_devtest.rot.pt"}
    assert json.load(open(X.rec_path("MCF-R/testB/s0")))["side_outputs"] == {}
    tsv = open(os.path.join(root, S7.STATE, "7C_checksums.tsv")).read()
    assert "results/n3/private/draws/ivP_s0_testB.rot.pt" in tsv
    open(os.path.join(root, "results/n3/private/draws/ivP_s0_testB.rot.pt"), "w").write("changed")
    with pytest.raises(SystemExit, match="iv-P/testB/s0: outputs changed after the run finished"):
        X.run_job("7C")


def test_cross_job_input_must_be_done(world):
    root, sel = world
    X, fake = runner(root, sel)
    with pytest.raises(SystemExit, match=r"comes from ours/testB \(job 7O\), which is not done"):
        X.run_job("7E0")
    assert fake.calls == []
    X.run_job("7O")
    assert X.run_job("7E0")


def test_uploaded_bundles_must_match_the_freeze(world):
    root, sel = world
    X, fake = runner(root, sel)
    open(os.path.join(root, "results/n3/private/draws/ours_s1_devtest.pt"), "w").write("other draws")
    with pytest.raises(SystemExit, match="ours/devtest: uploaded output .* differs from its freeze SHA-256"):
        X.run_job("7C")


def test_a_failed_mcf_check_stops_after_one_retry_and_is_kept_in_the_log(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.check_pass = False
    with pytest.raises(SystemExit, match=r"check.json is missing or did not pass \(after its one in-job retry\)"):
        X.run_job("7M")
    dele = [e for e in events(root) if e["run"] == "MCF-R/testB/s0" and e["event"] == "deleted_partial"]
    assert len(dele) == 1
    fc = dele[0]["detail"]["failed_check"]
    assert fc == {"n3_mcf/infer/mcfR/testB/s0_last_u5": {"pass": False, "failed": ["finite_R"]}}


# ------------------------------------------------------------------------------------------- refusals
def _tampered(sel, rid, i=None, cmd=None, **fields):
    s = copy.deepcopy(sel)
    for r in s["runs"]["step7"]:
        if r["id"] == rid:
            if cmd is not None:
                r["commands"][i] = cmd
            r.update(fields)
    return s


@pytest.mark.parametrize("rid,i,cmd,msg", [
    ("iii-u/testB/b0", 1, "python scripts/n3_classical.py anchor --set testB", "subcommand not permitted"),
    ("MCF-R/testB/s0", 1, "python scripts/n3_score.py match --bundles x.pt --matcher primary", "script not permitted"),
    ("MCF-R/testB/s0", 0, "python scripts/n3_mcf_run.py check --run_dir x", "subcommand not permitted"),
    ("iii-s/testB", 1, "python scripts/n3_classical.py search --set testB --workers {G} --limit 5", "forbidden token"),
    ("ii-S/testB/s0", 0, "python scripts/n3_mcf_sg.py --oracle --set testB --workers {W}", "forbidden token"),
    ("iv-P/testB/s0", 0, "python scripts/n3_old_symmc.py sample --run lr1e-4_s0 --epoch {E} --set testB",
     "unknown placeholder"),
    ("ours/devtest", 0, "python scripts/n3_ours.py sample --arm ours --set devtest --workers {W}",
     "uploaded run must be the local OURS conversion"),
    ("MCF-A/testB/s0", 0, "bash -c 'rm -rf /'", "not a `python scripts"),
])
def test_unlisted_or_forbidden_commands_are_refused_at_load(world, rid, i, cmd, msg):
    root, sel = world
    with pytest.raises(SystemExit, match=msg):
        runner(root, _tampered(sel, rid, i, cmd))


def test_unknown_runs_and_jobs_are_refused(world):
    root, sel = world
    s = copy.deepcopy(sel)
    s["runs"]["step7"].append({"id": "zz/testB", "arm": "zz", "set": "testB", "where": "box", "seeds": "",
                               "commands": ["python scripts/n3_classical.py haar --set testB --base 9 --workers {W}"],
                               "outputs": ["results/n3/private/draws/zz.pt"]})
    with pytest.raises(SystemExit, match="belongs to no step-7 job"):
        runner(root, s)
    s = copy.deepcopy(sel)
    s["runs"]["step7"].append(copy.deepcopy(s["runs"]["step7"][0]))
    with pytest.raises(SystemExit, match="duplicate run id"):
        runner(root, s)
    X, _ = runner(root, sel)
    with pytest.raises(SystemExit, match="unknown job"):
        X.run_job("7X")


def test_a_restart_deletes_only_inside_the_sampling_trees(world):
    root, sel = world
    s = _tampered(sel, "MCF-R/testB/s0", outputs=["n3_mcf/infer/mcfR/testB/s0_last_u5", "results/n3/n3_results.json"])
    X, _ = runner(root, s)
    with pytest.raises(SystemExit, match="refusing to delete results/n3/n3_results.json"):
        X.run_job("7M")


def test_a_run_needs_the_published_freeze(world):
    root, sel = world
    os.makedirs(os.path.join(root, "results/n3"), exist_ok=True)
    json.dump(sel, open(os.path.join(root, S7.SELECTION), "w"))
    with pytest.raises(SystemExit, match="is not the published freeze"):
        S7.load_selection(root, S7.FREEZE_SHA256)
    got, sha = S7.load_selection(root, S7.sha256(os.path.join(root, S7.SELECTION)))
    assert got["complete"] is True
    assert S7.sha256(SEL_PATH) == S7.FREEZE_SHA256                   # the committed freeze is the published one
