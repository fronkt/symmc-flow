"""Step-9 driver (scripts/n3_step9.py) on the REAL published freeze (results/n3/n3_selection.json, public): the
frozen command list, the three permitted rewrites, refusals, the blinding preflight and the 2026-11-15 resumption
rule, against a fake executor in tmp_path. Nothing is matched; no CSD content is read; inputs are empty
placeholder files in tmp_path."""
import json
import os
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_step9 as S9  # noqa: E402

SEL_PATH = os.path.join(REPO, S9.SELECTION)
pytestmark = pytest.mark.skipif(not os.path.exists(SEL_PATH), reason="no published freeze")
ENV = {"python": "3.12.10", "executable": "/x/python", "implementation": "CPython",
       "libraries": {"torch": "2.11.0+cu128", "pymatgen": "2026.5.4", "scipy": "1.17.1", "numpy": "2.3.5"}}


def real_sel():
    return json.load(open(SEL_PATH, encoding="utf-8"))


def make_root(tmp_path):
    root = str(tmp_path / "box")
    for rel in (S9.SELECTION, S9.PROTOCOL):
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), os.path.join(root, rel))
    sel = real_sel()
    for st in S9.SETS:
        X = S9.Runner(root, sel, S9.FREEZE_SHA256, st, 28, "PY", dry=True, env_of=lambda py: ENV)
        made = set(X.producer)
        for _, cmds, _ in X.units:
            for c in cmds:
                t = X.parse_c(c)[0]
                for flag in S9.PATH_FLAGS:
                    for rel in S9.values(t, flag):
                        if flag == "--out" or rel in made:
                            continue
                        p = os.path.join(root, rel)
                        if flag == "--run_dir":
                            os.makedirs(p, exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            open(p, "w").write(f"placeholder {rel}")
    return root, sel


class Fake:
    """Writes what each command would write (match rows, the unit's listed outputs). fail[(uid, k)] = n fails that
    command n times after a partial write."""

    def __init__(self, X):
        self.X, self.calls, self.fail = X, [], {}
        self.by_log = {}
        for uid, cmds, outs in X.units:
            for k in range(len(cmds)):
                self.by_log[os.path.basename(X.log_path(uid, k))] = (uid, k, cmds, outs)

    def write(self, rel, data, mode="w"):
        p = os.path.join(self.X.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, mode) as fh:
            fh.write(data)

    def __call__(self, argv, log):
        argv = [str(a) for a in argv]
        uid, k, cmds, outs = self.by_log[os.path.basename(log)]
        self.calls.append((uid, k))
        assert argv[0] == "PY"
        script, sub = argv[1], argv[2]
        if script == "scripts/n3_score.py" and sub == "match":
            for b in S9.values(argv, "--bundles"):
                m = S9.values(argv, "--matcher")[0]
                self.write(f"results/n3/private/match/{os.path.basename(b)[:-3]}__{m}.jsonl", f"row {uid}\n", "a")
        if self.fail.get((uid, k), 0) > 0:
            self.fail[(uid, k)] -= 1
            self.write(f"{S9.STATE}/logs/{self.X.set}/{os.path.basename(log)}",
                       "Traceback (most recent call last)\n  boom\nSystemExit: bad\n", "a")
            return 1
        if script == "scripts/n3_score.py" and sub == "haar-floor":
            self.write(S9.values(argv, "--out")[0], "{}")
        elif script == "scripts/n3_score.py" and sub == "basin":
            self.write(f"results/n3/private/basin/{os.path.basename(S9.values(argv, '--bundles')[0])[:-3]}__x.json", "{}")
        elif script == "scripts/n3_score.py" and sub == "report":
            self.write(outs[0], "{}")
        elif script == "scripts/n3_step9_summary.py":
            self.write(outs[0], "{}")
        elif script == "scripts/n3_classical.py":
            self.write(f"results/n3/private/classical/anchor_{self.X.set}.json", "{}")
        elif script == "scripts/n3_mcf_diag.py" and sub in ("b", "c", "d"):
            if k == 0 or sub == "d":
                for o in outs:
                    self.write(o, f"{sub} {uid}")
        return 0


def runner(root, sel, st="testB", **kw):
    X = S9.Runner(root, sel, S9.FREEZE_SHA256, st, 28, "PY", env_of=kw.pop("env_of", lambda py: ENV), **kw)
    fake = Fake(X)
    X._call = fake
    return X, fake


# ------------------------------------------------------------------ the frozen list
def test_frozen_list_and_rewrites():
    sel = real_sel()
    for st in S9.SETS:
        X = S9.Runner(REPO, sel, S9.FREEZE_SHA256, st, 28, "PY", dry=True)
        cmds = [(u, c) for u, cs, _ in X.units for c in cs]
        assert len(X.units) == 8 + 69 + 1 and len(cmds) == 23 + 69 + 1
        assert [u for u, _, _ in X.units][:8] == [r["id"] for r in sel["runs"]["step9"] if r["set"] == st]
        for _, c in cmds:
            t, script, sub = X.parse_c(c)
            a = X.argv(c)
            assert a[0] == "PY" and "{W}" not in a and ("28" in a) == ("{W}" in t)
            tail = a[len(t):]
            assert tail == (["--selection-sha256", S9.FREEZE_SHA256] if (script, sub) in S9.TAKES_SEL_SHA else [])
            assert [x for x in a[1:len(t)] if x != "28"] == [x for x in t[1:] if x != "{W}"]
        assert X.units[-2][1][0].endswith(f"report --config results/n3/report_{st}.json --step9")
        assert X.units[-2][2] == [sel["report_configs"][st]["out"]]
        assert X.units[-1] == (f"post/{st}/diag-c-summary", [f"python scripts/n3_step9_summary.py c --set {st} --step9"],
                               [f"results/n3/private/mcf/diag_c_{st}.json"])


@pytest.mark.parametrize("cmd", [
    "python scripts/n3_ours.py sample --set testB",
    "python scripts/n3_classical.py search --set testB",
    "python scripts/n3_mcf_diag.py a --set testB",
    "python scripts/n3_score.py match --bundles x_testB.pt --matcher primary",
    "python scripts/n3_score.py match --bundles x_testB.pt --matcher primary --step9 --force",
    "python scripts/n3_score.py match --bundles x_testB.pt --step9 --selection-sha256 00",
    "python scripts/n3_mcf_diag.py c --run_dir n3_mcf/infer/mcfR/testB/s0 --shims",
    "python scripts/n3_score.py match --bundles x_testB.pt --step9 --workers {G}",
    "bash -c 'rm -rf /'",
    "python scripts/n3_step9_summary.py c --set testB --step9",
])
def test_refusals(cmd):
    with pytest.raises(S9.Stop):
        S9.parse(cmd)


def test_mixed_set_refused(tmp_path):
    sel = real_sel()
    sel["runs"]["step9_scoring"]["testB"][0] = sel["runs"]["step9_scoring"]["testB"][0].replace(
        "ours_s1_testB.pt", "ours_s1_devtest.pt")
    with pytest.raises(S9.Stop, match="names set"):
        S9.Runner(REPO, sel, S9.FREEZE_SHA256, "testB", 28, "PY", dry=True)


# ------------------------------------------------------------------ preflight
def test_plan_on_the_laptop_tree():
    """The real repo: every input exists and no TEST-B / DEV-TEST step-9 output exists (blinding)."""
    sel = real_sel()
    for st in S9.SETS:
        X = S9.Runner(REPO, sel, S9.FREEZE_SHA256, st, "$W", "$PY_SYMMC", dry=True)
        if os.path.exists(X.started_path()):
            pytest.skip("step 9 has started in this tree")
        first, env = X.preflight()
        assert first and env is None


def test_changed_freeze_or_protocol(tmp_path):
    root, sel = make_root(tmp_path)
    with open(os.path.join(root, S9.SELECTION), "a") as fh:
        fh.write(" ")
    with pytest.raises(S9.Stop, match="published freeze"):
        S9.load_selection(root, S9.FREEZE_SHA256)
    root2, sel2 = make_root(tmp_path / "b")
    with open(os.path.join(root2, S9.PROTOCOL), "a") as fh:
        fh.write("\n")
    X, _ = runner(root2, sel2)
    with pytest.raises(S9.Stop, match="protocol_sha256"):
        X.run()


@pytest.mark.parametrize("planted", ["results/n3/private/match/zz_testB__primary.jsonl",
                                     "results/n3/private/basin/iii-s_testB__fftruth_s-1_testB_ff.json",
                                     "results/n3/private/classical/anchor_testB.json",
                                     "results/n3/private/mcf/diag_b_testB.json",
                                     "results/n3/private/draws/mcfRoal_last_u5_s0_testB.pt",
                                     "results/n3/report_testB.out.json"])
def test_blinding_planted_controls(tmp_path, planted):
    root, sel = make_root(tmp_path)
    p = os.path.join(root, planted)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write("x")
    X, fake = runner(root, sel)
    with pytest.raises(S9.Stop, match="BLINDING"):
        X.run()
    assert fake.calls == []
    # the other set is not blocked by a testB file
    Y, _ = runner(root, sel, "devtest")
    assert Y.preflight()[0] is True


def test_missing_input(tmp_path):
    root, sel = make_root(tmp_path)
    os.remove(os.path.join(root, "results/n3/private/draws/ours_s0_testB.pt"))
    X, fake = runner(root, sel)
    with pytest.raises(S9.Stop, match="missing"):
        X.run()
    assert fake.calls == []


def test_report_config_written_and_checked(tmp_path):
    root, sel = make_root(tmp_path)
    cp = os.path.join(root, "results/n3/report_testB.json")
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    json.dump({**sel["report_configs"]["testB"], "n_boot": 100}, open(cp, "w"))
    X, fake = runner(root, sel)
    with pytest.raises(S9.Stop, match="differs from report_configs"):
        X.run()
    os.remove(cp)
    X, fake = runner(root, sel)
    X.run()
    assert json.load(open(cp)) == sel["report_configs"]["testB"]


def manifest_for(root, paths, step=9):
    files = [{"group": "g", "csd": False, "path": r, "bytes": os.path.getsize(os.path.join(root, r)),
              "sha256": S9.sha256(os.path.join(root, r))} for r in paths]
    p = os.path.join(root, S9.MANIFEST)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"step": step, "files": files}, open(p, "w"))


def all_files(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return out


def test_manifest_checks(tmp_path):
    root, sel = make_root(tmp_path)
    drv = os.path.join(root, "scripts/n3_step9.py")
    os.makedirs(os.path.dirname(drv), exist_ok=True)
    shutil.copyfile(S9.__file__, drv)
    files = all_files(root)
    manifest_for(root, files)
    X, _ = runner(root, sel, manifest=S9.MANIFEST)
    assert X.preflight()[0] is True
    with open(os.path.join(root, "results/n3/private/draws/ours_s0_testB.pt"), "a") as fh:
        fh.write("changed")
    with pytest.raises(S9.Stop, match="changed since the step-9 upload"):
        X.preflight()
    root2, sel2 = make_root(tmp_path / "b")
    os.makedirs(os.path.join(root2, "scripts"), exist_ok=True)
    shutil.copyfile(S9.__file__, os.path.join(root2, "scripts/n3_step9.py"))
    manifest_for(root2, [f for f in all_files(root2) if not f.endswith("ours_s0_testB.pt")])
    X2, _ = runner(root2, sel2, manifest=S9.MANIFEST)
    with pytest.raises(S9.Stop, match="not in the step-9 manifest"):
        X2.preflight()


# ------------------------------------------------------------------ runs and resumption
def events(root, st="testB"):
    return [json.loads(x) for x in open(os.path.join(root, S9.STATE, f"events_{st}.jsonl"))]


def test_full_run(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    X.run()
    n = sum(len(c) for _, c, _ in X.units)
    assert len(fake.calls) == n
    ev = events(root)
    assert sum(e["event"] == "start" for e in ev) == n and sum(e["event"] == "done" for e in ev) == n
    assert ev[0]["event"] == "first_start" and ev[-1]["event"] == "set_done"
    tsv = open(os.path.join(root, S9.STATE, "testB_checksums.tsv")).read()
    assert "results/n3/report_testB.out.json" in tsv and "mcfRoal_last_u5_s0_testB.pt" in tsv
    assert "results/n3/private/mcf/diag_c_testB.json" in tsv
    started = json.load(open(X.started_path()))
    assert started["environment"] == ENV and started["selection_sha256"] == S9.FREEZE_SHA256
    # a second run is a no-op resume
    Y, fake2 = runner(root, sel)
    Y.run()
    assert fake2.calls == []


def test_stop_and_resume(tmp_path, capsys):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    stop_at = ("scoring/testB/14", 0)
    fake.fail[stop_at] = 1
    with pytest.raises(S9.Stop) as e:
        X.run()
    assert "rc 1" in str(e.value) and "Traceback" in str(e.value) and "boom" in str(e.value)
    done_before = list(fake.calls)
    assert done_before[-1] == stop_at
    Y, fake2 = runner(root, sel)
    Y.run()
    assert fake2.calls[0] == stop_at and stop_at not in fake2.calls[1:]
    assert not set(fake2.calls) & set(done_before[:-1])
    ev = events(root)
    assert [e["event"] for e in ev if (e["unit"], e["command"]) == stop_at] == ["start", "command_failed", "resume",
                                                                                "done"]
    assert any(e["event"] == "restart" for e in ev)


def test_diag_c_remakes_its_bundle(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    fake.fail[("diag-c/testB/s0", 0)] = 1
    out = "results/n3/private/draws/mcfRoal_last_u5_s0_testB.pt"
    with pytest.raises(S9.Stop):
        X.run()
    fake.write(out, "half")                               # an unrecorded bundle from the stopped attempt
    Y, fake2 = runner(root, sel)
    Y.run()
    ev = events(root)
    dele = [e for e in ev if e["event"] == "deleted_partial"]
    assert dele and dele[0]["detail"]["deleted"] == [out]
    assert open(os.path.join(root, out)).read() == "c diag-c/testB/s0"


def test_recorded_file_never_deleted(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    fake.fail[("diag-c/testB/s0", 1)] = 1                  # c done and recorded; its first match fails
    with pytest.raises(S9.Stop):
        X.run()
    Y, fake2 = runner(root, sel)
    Y.run()
    assert ("diag-c/testB/s0", 0) not in fake2.calls       # c is not re-made
    assert not any(e["event"] == "deleted_partial" for e in events(root))


def test_environment_change_stops_a_resume(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    fake.fail[("scoring/testB/00", 0)] = 1
    with pytest.raises(S9.Stop):
        X.run()
    other = {**ENV, "libraries": {**ENV["libraries"], "pymatgen": "2026.6.1"}}
    Y, fake2 = runner(root, sel, env_of=lambda py: other)
    with pytest.raises(S9.Stop, match="environment changed"):
        Y.run()
    assert fake2.calls == []


def test_listed_output_missing_stops(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    orig = fake.__call__

    def no_haar(argv, log):
        if argv[2] == "haar-floor":
            fake.calls.append("haar")
            return 0
        return orig(argv, log)
    X._call = no_haar
    with pytest.raises(S9.Stop, match="without its listed output"):
        X.run()


def test_done_record_for_another_command(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    X.run()
    p = X.rec_path("scoring/testB/00", 0)
    rec = json.load(open(p))
    rec["command"] = rec["command"].replace("primary", "mcf05")
    json.dump(rec, open(p, "w"))
    Y, _ = runner(root, sel)
    with pytest.raises(S9.Stop, match="another command"):
        Y.run()


def test_plan_writes_nothing(tmp_path):
    root, sel = make_root(tmp_path)
    before = sorted(all_files(root))
    X = S9.Runner(root, sel, S9.FREEZE_SHA256, "testB", "$W", "$PY_SYMMC", dry=True)
    X._call = lambda a, log: pytest.fail("plan executed a command")
    X.run()
    assert sorted(all_files(root)) == before


def test_git_work_tree_refused_on_the_box(tmp_path):
    import subprocess
    root, sel = make_root(tmp_path)
    os.makedirs(os.path.join(root, "scripts"), exist_ok=True)
    shutil.copyfile(S9.__file__, os.path.join(root, "scripts/n3_step9.py"))
    manifest_for(root, all_files(root))
    if subprocess.run(["git", "init", "-q", root]).returncode != 0:
        pytest.skip("no git")
    X, _ = runner(root, sel, manifest=S9.MANIFEST)
    with pytest.raises(S9.Stop, match="git work tree"):
        X.preflight()


def test_output_dirs_created(tmp_path):
    root, sel = make_root(tmp_path)
    X, _ = runner(root, sel)
    X.run()
    assert all(os.path.isdir(os.path.join(root, d)) for d in S9.WATCH)


# ------------------------------------------------------------------ review findings (11-15)
def test_other_set_files_never_enter_records(tmp_path):
    """9T and 9D run at once: the other job's files (and its vanishing temporaries) are not this set's."""
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    orig = fake.__call__

    def busy_neighbour(argv, log):
        rc = orig(argv, log)
        fake.write("results/n3/private/match/ours_s0_devtest__primary.jsonl", "row devtest\n", "a")
        fake.write("results/n3/private/draws/mcfRoal_last_u5_s0_devtest.pt.tmp", "tmp")
        fake.write("results/n3/private/draws/mcfRoal_last_u5_s0_testB.pt.tmp", "own tmp")
        return rc
    X._call = busy_neighbour
    real_hash = X.hashes

    def vanishing(rels):                     # a temporary disappears between the walk and the hash
        for r in rels:
            if r.endswith(".tmp") and os.path.exists(os.path.join(root, r)):
                os.remove(os.path.join(root, r))
        return real_hash(rels)
    X.hashes = vanishing
    X.run()
    tsv = open(os.path.join(root, S9.STATE, "testB_checksums.tsv")).read()
    assert "devtest" not in tsv and ".tmp" not in tsv
    for f in os.listdir(os.path.join(root, S9.STATE, "done")):
        rec = json.load(open(os.path.join(root, S9.STATE, "done", f)))
        assert not any("devtest" in r or r.endswith(".tmp") for r in rec["files"])


def test_files_of_a_stopped_attempt_stay_recorded(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)
    orig = fake.__call__
    meta = "results/n3/private/match/ours_s0_testB__primary.meta.json"

    def meta_once(argv, log):
        if "ours_s0_testB.pt" in " ".join(argv) and "primary" in argv and not os.path.exists(os.path.join(root, meta)):
            fake.write(meta, "{}")           # n3_score writes a rows file's meta once, on the first attempt
        return orig(argv, log)
    X._call = meta_once
    fake.fail[("scoring/testB/00", 0)] = 1
    with pytest.raises(S9.Stop):
        X.run()
    Y, fake2 = runner(root, sel)
    orig2 = fake2.__call__
    Y._call = lambda a, log: orig2(a, log)
    Y.run()
    rec = json.load(open(Y.rec_path("scoring/testB/00", 0)))
    assert meta in rec["files"]
    assert meta in open(os.path.join(root, S9.STATE, "testB_checksums.tsv")).read()


def test_error_tail_shows_only_the_last_attempts_traceback(tmp_path):
    root, sel = make_root(tmp_path)
    X, _ = runner(root, sel)
    log = str(tmp_path / "x.log")
    open(log, "w").write("### attempt 1\nOURS: picked@1 42.00%\nTraceback (most recent call last)\n  old\n"
                         "### attempt 2\nOURS: picked@1 42.00%\n  delta 3.1 pp\n")
    assert X.error_tail(log) == []
    open(log, "a").write("Traceback (most recent call last)\n  File x\nValueError: boom\n")
    assert X.error_tail(log) == ["Traceback (most recent call last)", "  File x", "ValueError: boom"]


def test_killed_command_message_has_no_output(tmp_path):
    root, sel = make_root(tmp_path)
    X, fake = runner(root, sel)

    def killed(argv, log):
        os.makedirs(os.path.dirname(log), exist_ok=True)
        open(log, "a").write("### attempt\nOURS: picked@1 42.00%\n")
        return -9
    X._call = killed
    with pytest.raises(S9.Stop) as e:
        X.run()
    assert "42" not in str(e.value) and "no traceback" in str(e.value) and "rc -9" in str(e.value)


def test_stale_records_refused_on_first_start(tmp_path):
    root, sel = make_root(tmp_path)
    X, _ = runner(root, sel)
    stale = X.rec_path("scoring/testB/00", 0)
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    open(stale, "w").write("{}")
    with pytest.raises(S9.Stop, match="before the first start"):
        X.run()
    Y, _ = runner(root, sel, "devtest")     # the other set's first start is not blocked by a testB record
    assert Y.preflight()[0] is True
