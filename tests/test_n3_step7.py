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
    """Writes each command's outputs; infer refuses a run dir that already holds predictions (as n3_mcf_run does)."""

    def __init__(self, X):
        self.X, self.calls, self.fail_once, self.check_pass = X, [], set(), True
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
        cmds = self.X.runs[rid]["commands"]
        t, _, sub = S7.parse(cmds[i])
        if (script, sub) == ("scripts/n3_mcf_run.py", "infer"):
            d = S7.opt(shlex.split(cmds[i + 1]), "--run_dir")
            if os.path.exists(self.p(f"{d}/out/predictions_16.pt")):
                return 2                                            # "already holds a run"
            self.write(f"{d}/out/predictions_16.pt", f"pred {rid}")
            self.write(f"{d}/check.json", json.dumps({"pass": self.check_pass}))
        if script == "scripts/n3_mcf_gauge.py":
            d = S7.opt(argv, "--run_dir")
            assert os.path.isdir(self.p(d)), d
            self.write(f"{d}/gauge_16_step5.jsonl", f"rows {rid}")
        if (rid, i) in self.fail_once:
            self.fail_once.discard((rid, i))
            return 1
        if i == len(cmds) - 1:
            for o in self.X.runs[rid]["outputs"]:
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
def test_resumable_run_reinvokes_its_commands_after_a_stop(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.fail_once.add(("iii-u/testB/b0", 1))                        # press stops after haar finished
    with pytest.raises(SystemExit, match="iii-u/testB/b0: rc 1"):
        X.run_job("7C")
    X2, fake2 = runner(root, sel)
    X2.run_job("7C")
    assert fake2.executed()[:2] == [("iii-u/testB/b0", 0), ("iii-u/testB/b0", 1)]   # haar re-invoked (resumes)
    ev = [e for e in events(root) if e["run"] == "iii-u/testB/b0"]
    assert [e["event"] for e in ev] == ["start", "command_failed", "resume", "done"]
    assert X2.record("iii-u/testB/b0") is not None


def test_single_call_run_deletes_its_partial_outputs_and_restarts(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.fail_once.add(("MCF-R/testB/s0", 1))                        # to_bundle fails after infer wrote the run dir
    with pytest.raises(SystemExit, match="MCF-R/testB/s0: rc 1"):
        X.run_job("7M")
    rd = os.path.join(root, "n3_mcf/infer/mcfR/testB/s0_last_u5/out/predictions_16.pt")
    assert os.path.exists(rd)
    X2, fake2 = runner(root, sel)
    X2.run_job("7M")                                                 # infer would refuse (rc 2) without the delete
    ev = [e["event"] for e in events(root) if e["run"] == "MCF-R/testB/s0"]
    assert ev == ["start", "command_failed", "deleted_partial", "restart", "done"]
    assert fake2.executed()[:2] == [("MCF-R/testB/s0", 0), ("MCF-R/testB/s0", 1)]


def test_gauge_restart_keeps_the_scanned_mcf_run(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.fail_once.add(("MCF-R+G/testB/s1", 0))                      # gauge rows written, then the scan stops
    with pytest.raises(SystemExit, match="MCF-R\\+G/testB/s1: rc 1"):
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


def test_a_finished_runs_recorded_file_is_never_deleted(world):
    root, sel = world
    X, fake = runner(root, sel)
    X.run_job("7M")
    sel2 = copy.deepcopy(sel)
    for r in sel2["runs"]["step7"]:
        if r["id"] == "MCF-R+G/testB/s1":                            # pretend the scan also re-made MCF-R's bundle
            r["outputs"].append("results/n3/private/draws/mcfR_last_u5_s1_testB.pt")
            r["commands"] = list(r["commands"])
    with pytest.raises(SystemExit):
        S7.Runner(root, sel2, "f" * 64, 28, 1, 0, "PYSYMMC", "PYMCF")   # one output, two runs: refused at load
    X3, _ = runner(root, sel)
    keep = X3.recorded_files()
    assert os.path.normpath("n3_mcf/infer/mcfR/testB/s1_last_u5/out/predictions_16.pt") in keep
    with pytest.raises(SystemExit, match="holds a finished run's recorded output"):
        X3.delete_partial("MCF-R/testB/s1", "7M")                    # its run dir is recorded: never deleted
    os.remove(X3.rec_path("MCF-R+G/testB/s1"))
    X3.delete_partial("MCF-R+G/testB/s1", "7M")                      # the scan's own rows + bundle only
    assert os.path.exists(os.path.join(root, "n3_mcf/infer/mcfR/testB/s1_last_u5/out/predictions_16.pt"))
    assert not os.path.exists(os.path.join(root, "n3_mcf/infer/mcfR/testB/s1_last_u5/gauge_16_step5.jsonl"))


def test_cross_job_input_must_be_done(world):
    root, sel = world
    X, fake = runner(root, sel)
    with pytest.raises(SystemExit, match="comes from ours/testB \\(job 7O\\), which is not done"):
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


def test_mcf_check_must_pass(world):
    root, sel = world
    X, fake = runner(root, sel)
    fake.check_pass = False
    with pytest.raises(SystemExit, match="check.json is missing or did not pass"):
        X.run_job("7M")


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
