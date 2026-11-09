"""N3 §7 step 7 on the box (protocol §7 step 7; runbook "Step 7"; tasks/n3_step7_clarification_2026-11-08.md):
exactly the frozen runs of results/n3/n3_selection.json (`runs.step7`, 76 runs on TEST-B / DEV-TEST), each with its
committed command and seed. Nothing here matches, anchors, reports or runs a truth-based diagnostic.

    python scripts/n3_step7.py jobs                                   # job -> run ids (nothing is executed)
    python scripts/n3_step7.py plan --job 7O                          # what `run` would do now (nothing is executed)
    python scripts/n3_step7.py run --job 7O --workers W --ff-workers G [--gpu 0]

Every command is read from the freeze and run verbatim. Only the scheduling placeholders {W}, {G}, {GPU} are
replaced, and `python` becomes $PY_MCF (n3_mcf_run / n3_mcf_gauge / n3_mcf_sg) or $PY_SYMMC (everything else; FF*
starts through $N3_FF_PYTHON). A command whose script / subcommand is not a step-7 sampling or FF* command (e.g.
n3_score match / report, n3_classical anchor, n3_mcf_diag) or that carries a smoke / oracle / force flag stops the
driver before anything runs. The freeze's SHA-256 must be the published one (772a36e: 553f1dc9...).

Jobs (Frank's OK 2026-11-08: the plan in tasks/todo.md); inside a job the runs keep the freeze's order, so iii-u comes
before its press, MCF-R before the +G gauge scan of its run dir, +G before ii-S, iv-P before iv-S:
  7O   OURS + OURS-N2 TEST-B sampling, P1 press of OURS TEST-B (s0-s2)
  7C   iii-u (+ press) x6, p1ref x2, P1 press of OURS DEV-TEST x3, iv-P x6, iv-S x6, FF* truths x2, and the two
       DEV-TEST OURS / OURS-N2 conversions done locally before the freeze (their bundles arrive in the step-7 pack;
       they are never re-run here, only checked against step7_inputs)
  7M   MCF-R x6, MCF-R+G x6, ii-S x6, MCF-A x6, MCF-A prior floor x6
  7S   iii-s TEST-B          7Sd  iii-s DEV-TEST
  7E0-7E2  i-E + basin-ours and basin-ours_n2 of TEST-B seed k (after 7O: they read 7O's bundles)
  7Ed  i-E + basin-ours and basin-ours_n2 of DEV-TEST s0-s2
An input made by a run of another job must be done (its record checks) before a run that reads it starts.

Resumption (the 2026-11-08 clarification: one Slurm job ending early is not a box death):
  * a run with a done record whose listed outputs still have their recorded SHA-256 is skipped; a done record whose
    outputs changed stops the driver (that is box-root corruption, not a resumable stop);
  * per-crystal resumable runs (n3_classical haar / press / screen / search / truths / ie / basin / p1basin and the
    n3_ours sample part file) re-invoke their commands: each continues from its complete, configuration-checked
    records;
  * single-call runs (MCF infer / to_bundle / prior_floor, the gauge scan, ii-S, iv-P sample, iv-S) delete their
    partial outputs (listed outputs, part files, the run dir an infer creates, gauge rows of a scanned run dir)
    and run again from the start with the committed seed. A file recorded by a finished run is never deleted.
Every start, continuation and finish is appended to results/n3/private/step7/events.jsonl (for the run record).
After a run: every listed output exists, each infer run dir's check.json passes (R:234; to_bundle also refuses a
failed check, and save_bundle refuses a bundle without the full frozen refcode order), and a done record with the
SHA-256 of every output file is written. At the end of a job, <job>_checksums.tsv lists them (R:231).
Stdlib only: it runs under $PY_SYMMC.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION = "results/n3/n3_selection.json"
FREEZE_SHA256 = "553f1dc9efc0ec2c6d2d92b6d61bd5c76bbdc46c1d8a8ef9c814843222e1f491"   # 772a36e, published 11-08
STATE = "results/n3/private/step7"
JOBS = ("7O", "7C", "7M", "7S", "7Sd", "7E0", "7E1", "7E2", "7Ed")
AFTEROK = {"7E0": ("7O",), "7E1": ("7O",), "7E2": ("7O",)}           # sbatch --dependency=afterok (informational)

# the step-7 commands of the freeze, by (script, subcommand); None = a script without subcommands
ALLOWED = {
    "scripts/n3_ours.py": {"sample"},
    "scripts/n3_classical.py": {"haar", "press", "screen", "search", "truths", "ie", "basin", "p1basin"},
    "scripts/n3_mcf_run.py": {"infer", "to_bundle", "prior_floor"},
    "scripts/n3_mcf_gauge.py": {None},
    "scripts/n3_mcf_sg.py": {None},
    "scripts/n3_old_symmc.py": {"sample", "ivs"},
}
UPLOADED_CMD = ("scripts/n3_ours.py", "convert")                     # local conversion, done before the freeze
MCF_SCRIPTS = {"scripts/n3_mcf_run.py", "scripts/n3_mcf_gauge.py", "scripts/n3_mcf_sg.py"}
RESUMABLE = {("scripts/n3_classical.py", s) for s in ALLOWED["scripts/n3_classical.py"]} | {("scripts/n3_ours.py",
                                                                                              "sample")}
FORBIDDEN_TOKENS = {"match", "anchor", "report", "check", "--oracle", "--after_sampling", "--step9", "--force",
                    "--discard", "--smoke", "--limit", "--only", "--shims", "--cpu_emulate", "--haar_floor"}
PLACEHOLDERS = ("{W}", "{G}", "{GPU}")
INPUT_FLAGS = ("--draws", "--bundle", "--run_dir", "--ckpt")
DELETABLE = ("n3_mcf/infer/", "results/n3/private/draws/")           # the only trees a restart may delete in


class Stop(SystemExit):
    pass


def die(msg):
    raise Stop(f"n3_step7: {msg}")


def stage(msg):
    print(f"=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def opt(tokens, flag):
    return tokens[tokens.index(flag) + 1] if flag in tokens and tokens.index(flag) + 1 < len(tokens) else None


def parse(cmd, uploaded=False):
    """-> (tokens, script, subcommand); refuses anything that is not a step-7 sampling / FF* command."""
    if not isinstance(cmd, str):
        die(f"command is not a string: {cmd!r}")
    t = shlex.split(cmd)
    if len(t) < 2 or t[0] != "python":
        die(f"not a `python scripts/...` command: {cmd}")
    script = t[1]
    if uploaded:
        if tuple(t[1:3]) != UPLOADED_CMD:
            die(f"an uploaded run must be the local OURS conversion: {cmd}")
        return t, script, t[2]
    if script not in ALLOWED:
        die(f"script not permitted in step 7: {cmd}")
    sub = None if ALLOWED[script] == {None} else (t[2] if len(t) > 2 else None)
    if sub not in ALLOWED[script]:
        die(f"subcommand not permitted in step 7: {cmd}")
    bad = [x for x in t if x in FORBIDDEN_TOKENS]
    if bad:
        die(f"forbidden token(s) {bad} in: {cmd}")
    left = [x for x in t if ("{" in x or "}" in x) and x not in PLACEHOLDERS]
    if left:
        die(f"unknown placeholder(s) {left} in: {cmd}")
    return t, script, sub


def job_of(run):
    """The plan's job of one freeze run (None = unassigned, which stops the driver)."""
    head, st = run["id"].split("/")[0], run["set"]
    if st not in ("testB", "devtest"):
        return None
    if head in ("ours", "ours_n2", "p1-ours"):
        return "7O" if st == "testB" else "7C"
    if head in ("iii-u", "p1ref", "iv-P", "iv-S", "fftruth"):
        return "7C"
    if head in ("MCF-R", "MCF-R+G", "ii-S", "MCF-A", "MCF-A-floor"):
        return "7M"
    if head == "iii-s":
        return "7S" if st == "testB" else "7Sd"
    if head in ("i-E+basin-ours", "basin-ours_n2"):
        if st == "devtest":
            return "7Ed"
        tail = run["id"].rsplit("/", 1)[-1]
        return f"7E{tail[1]}" if len(tail) == 2 and tail[0] == "s" and tail[1] in "012" else None
    return None


def load_selection(root, want_sha):
    p = os.path.join(root, SELECTION)
    if not os.path.isfile(p):
        die(f"no {SELECTION} under {root}")
    got = sha256(p)
    if got != want_sha:
        die(f"{SELECTION} sha256 {got[:16]}... is not the published freeze {want_sha[:16]}...")
    sel = json.load(open(p, encoding="utf-8"))
    if sel.get("complete") is not True:
        die("the freeze is not complete")
    return sel, got


class Runner:
    def __init__(self, root, sel, sel_sha, workers, ff_workers, gpu, py_symmc, py_mcf, dry=False, driver_sha=None):
        self.root, self.sel_sha, self.dry = root, sel_sha, dry
        self.sub = {"{W}": str(workers), "{G}": str(ff_workers), "{GPU}": str(gpu)}
        self.py, self.pym = py_symmc, py_mcf
        self.driver_sha = driver_sha or sha256(os.path.abspath(__file__))
        runs = (sel.get("runs") or {}).get("step7")
        if not isinstance(runs, list) or not runs:
            die("the freeze has no runs.step7")
        self.runs, self.order, self.job, self.producer = {}, [], {}, {}
        for r in runs:
            rid = r.get("id")
            if not isinstance(rid, str) or rid in self.runs:
                die(f"missing or duplicate run id {rid!r}")
            if not isinstance(r.get("commands"), list) or not r["commands"] or not isinstance(r.get("outputs"), list):
                die(f"run {rid}: commands / outputs missing")
            j = job_of(r)
            if j is None:
                die(f"run {rid} belongs to no step-7 job")
            self.runs[rid], self.job[rid] = r, j
            self.order.append(rid)
            for c in r["commands"]:
                parse(c, uploaded=self.mode(rid) == "uploaded")
            for o in r["outputs"]:
                if o in self.producer:
                    die(f"output {o} is listed by both {self.producer[o]} and {rid}")
                self.producer[o] = rid
        self.inputs7 = {row["path"]: row["sha256"] for row in sel.get("step7_inputs") or []}
        if not self.inputs7:
            die("the freeze has no step7_inputs")
        self.planned = set()             # dry run: runs this plan would complete
        self._sha = {}

    # ------------------------------------------------------------------ helpers
    def p(self, rel):
        return os.path.join(self.root, rel)

    def call(self, argv):
        return subprocess.run([str(a) for a in argv], cwd=self.root).returncode

    def mode(self, rid):
        r = self.runs[rid]
        if str(r.get("where", "")).startswith("local conversion (done)"):
            return "uploaded"
        kinds = [parse(c)[1:] for c in r["commands"]]
        return "resume" if all(k in RESUMABLE for k in kinds) else "restart"

    def argv(self, cmd):
        t, script, _ = parse(cmd)
        out = [self.sub.get(x, x) for x in t]
        out[0] = self.pym if script in MCF_SCRIPTS else self.py
        return out

    def safe(self, rid):
        return rid.replace("/", "__")

    def rec_path(self, rid, kind="done"):
        return self.p(f"{STATE}/{kind}/{self.safe(rid)}.json")

    def event(self, job, rid, what, detail=None):
        if self.dry:
            return
        os.makedirs(self.p(STATE), exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "slurm_job": os.environ.get("SLURM_JOB_ID"), "job": job,
               "run": rid, "event": what}
        if detail:
            row["detail"] = detail
        with open(self.p(f"{STATE}/events.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def atomic_json(self, obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(path + ".tmp", path)

    def checksum(self, rel):
        p = self.p(rel)
        if os.path.isdir(p):
            files = {}
            for d, dirs, fs in os.walk(p):
                dirs.sort()
                for f in sorted(fs):
                    q = os.path.join(d, f)
                    files[os.path.relpath(q, p).replace(os.sep, "/")] = sha256(q)
            return {"dir": True, "files": files}
        return {"sha256": sha256(p), "bytes": os.path.getsize(p)}

    def outputs_changed(self, rec):
        """Recorded output files that are missing or changed (files added later inside a recorded dir are allowed:
        the +G gauge scan writes its rows into the MCF-R run dir it scans)."""
        bad = []
        for rel, c in rec["outputs"].items():
            if c.get("dir"):
                for f, h in c["files"].items():
                    q = os.path.join(self.p(rel), f)
                    if not os.path.isfile(q) or sha256(q) != h:
                        bad.append(f"{rel}/{f}")
            elif not os.path.isfile(self.p(rel)) or sha256(self.p(rel)) != c["sha256"]:
                bad.append(rel)
        return bad

    def record(self, rid):
        """The valid done record of rid, or None; a record that no longer holds stops the driver."""
        path = self.rec_path(rid)
        if not os.path.exists(path):
            return None
        try:
            rec = json.load(open(path))
        except json.JSONDecodeError as e:
            die(f"{path} is not valid JSON ({e}); inspect it")
        if rec.get("run") != rid or rec.get("commands") != self.runs[rid]["commands"]:
            die(f"{rid}: its done record was made for other commands")
        if rec.get("selection_sha256") != self.sel_sha:
            die(f"{rid}: its done record was made under another freeze")
        bad = self.outputs_changed(rec)
        if bad:
            die(f"{rid}: outputs changed after the run finished {bad[:5]}: box-root corruption, not a resumable "
                "stop (clarification 2026-11-08 item 4)")
        return rec

    def recorded_files(self):
        out = set()
        d = self.p(f"{STATE}/done")
        if not os.path.isdir(d):
            return out
        for f in os.listdir(d):
            if not f.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(d, f)))
            for rel, c in rec.get("outputs", {}).items():
                if c.get("dir"):
                    out |= {os.path.normpath(os.path.join(rel, x)) for x in c["files"]}
                else:
                    out.add(os.path.normpath(rel))
        return out

    def sha_of(self, rel):
        if rel not in self._sha:
            self._sha[rel] = sha256(self.p(rel))
        return self._sha[rel]

    # ------------------------------------------------------------------ inputs and partial outputs
    def made_in_run(self, rid):
        """Paths this run makes itself: its listed outputs and the run dir of each MCF infer (the --run_dir of the
        run's next command)."""
        r = self.runs[rid]
        made = set(r["outputs"])
        toks = [parse(c) for c in r["commands"]]
        for i, (t, script, sub) in enumerate(toks):
            if (script, sub) == ("scripts/n3_mcf_run.py", "infer"):
                later = [opt(t2, "--run_dir") for t2, _, _ in toks[i + 1:] if opt(t2, "--run_dir")]
                if not later:
                    die(f"{rid}: an infer without a following --run_dir command; its run dir cannot be named")
                made.add(later[0])
        return made

    def inputs_ready(self, rid):
        """True when every input of rid's commands is present (and, for frozen inputs, has its freeze SHA-256)."""
        made = self.made_in_run(rid)
        for c in self.runs[rid]["commands"]:
            t = parse(c)[0]
            for flag in INPUT_FLAGS:
                rel = opt(t, flag)
                if rel is None or rel in made:
                    continue
                prod = self.producer.get(rel)
                if prod is None or self.mode(prod) == "uploaded":
                    if not os.path.exists(self.p(rel)):
                        die(f"{rid}: input {rel} is missing")
                    if rel in self.inputs7 and self.sha_of(rel) != self.inputs7[rel]:
                        die(f"{rid}: input {rel} differs from its freeze step7_inputs SHA-256")
                    continue
                if prod in self.planned or self.record(prod) is not None:
                    continue
                if self.dry:
                    print(f"  [waits] {rid}: needs {rel} from {prod} (job {self.job[prod]}), not done", flush=True)
                    return False
                die(f"{rid}: input {rel} comes from {prod} (job {self.job[prod]}), which is not done")
        return True

    def partial_targets(self, rid):
        """What a single-call run deletes before it runs again from the start."""
        r = self.runs[rid]
        toks = [parse(c) for c in r["commands"]]
        made = self.made_in_run(rid)
        targets = []
        for o in sorted(made):
            targets.append(o)
            if o.endswith(".pt"):
                targets += [o + ".part.pt", o[:-3] + ".jsonl", o[:-3] + ".events.jsonl"]
        for t, script, _ in toks:
            if script == "scripts/n3_mcf_gauge.py":
                d = opt(t, "--run_dir")
                if d not in made and os.path.isdir(self.p(d)):
                    targets += [f"{d}/{f}" for f in sorted(os.listdir(self.p(d))) if fnmatch.fnmatch(f, "gauge_*")]
        return targets

    def delete_partial(self, rid, job):
        keep = self.recorded_files()
        gone = []
        for rel in self.partial_targets(rid):
            norm = os.path.normpath(rel).replace(os.sep, "/")
            if not norm.startswith(DELETABLE) or ".." in norm.split("/"):
                die(f"{rid}: refusing to delete {rel} (outside {DELETABLE})")
            p = self.p(norm)
            if not os.path.lexists(p):
                continue
            inside = {os.path.normpath(norm)} if not os.path.isdir(p) else \
                {k for k in keep if k == os.path.normpath(norm) or k.startswith(os.path.normpath(norm) + os.sep)}
            if inside & keep:
                die(f"{rid}: {rel} holds a finished run's recorded output; not deleted")
            gone.append(norm)
            if self.dry:
                continue
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        if gone:
            print(f"  {'would delete' if self.dry else 'deleted'} partial outputs of {rid}: {gone}", flush=True)
            self.event(job, rid, "deleted_partial", gone)
        return gone

    # ------------------------------------------------------------------ runs and jobs
    def run_one(self, rid):
        job, r = self.job[rid], self.runs[rid]
        if self.record(rid) is not None:
            print(f"  [done] {rid}", flush=True)
            return True
        mode = self.mode(rid)
        if mode == "uploaded":
            for o in r["outputs"]:
                if o not in self.inputs7:
                    die(f"{rid}: uploaded output {o} is not in step7_inputs")
                if not os.path.isfile(self.p(o)) or self.sha_of(o) != self.inputs7[o]:
                    die(f"{rid}: uploaded output {o} is missing or differs from its freeze SHA-256")
            print(f"  [uploaded] {rid}: {len(r['outputs'])} bundles match step7_inputs", flush=True)
            if not self.dry:
                self.write_record(rid, mode, [])
            self.planned.add(rid)
            return True
        if not self.inputs_ready(rid):
            return False
        started = os.path.exists(self.rec_path(rid, "started"))
        stage(f"{rid} [{mode}{', continuing after a stop' if started else ''}]")
        if mode == "restart":
            self.delete_partial(rid, job)
        if not self.dry:
            self.atomic_json({"run": rid, "job": job, "first_start": time.strftime("%Y-%m-%d %H:%M:%S")}
                             if not started else json.load(open(self.rec_path(rid, "started"))),
                             self.rec_path(rid, "started"))
            self.event(job, rid, ("resume" if mode == "resume" else "restart") if started else "start")
        executed = []
        for c in r["commands"]:
            a = self.argv(c)
            print("$ " + " ".join(shlex.quote(str(x)) for x in a), flush=True)
            executed.append(a)
            if self.dry:
                continue
            rc = self.call(a)
            if rc:
                self.event(job, rid, "command_failed", {"rc": rc, "command": c})
                die(f"{rid}: rc {rc} from `{c}`")
        if self.dry:
            self.planned.add(rid)
            return True
        missing = [o for o in r["outputs"] if not os.path.exists(self.p(o))]
        if missing:
            die(f"{rid}: did not write {missing}")
        for d in sorted(self.made_in_run(rid)):
            if any(parse(c)[1:] == ("scripts/n3_mcf_run.py", "infer") for c in r["commands"]) and \
                    os.path.isdir(self.p(d)) and d.startswith("n3_mcf/infer/"):
                chk = self.p(f"{d}/check.json")
                if not os.path.isfile(chk) or json.load(open(chk)).get("pass") is not True:
                    die(f"{rid}: {d}/check.json is missing or did not pass")
        self.write_record(rid, mode, executed)
        self.event(job, rid, "done")
        return True

    def write_record(self, rid, mode, executed):
        r = self.runs[rid]
        rec = {"run": rid, "job": self.job[rid], "mode": mode, "selection_sha256": self.sel_sha,
               "driver_sha256": self.driver_sha, "commands": r["commands"], "seeds": r.get("seeds"),
               "executed": [[str(x) for x in a] for a in executed],
               "outputs": {o: self.checksum(o) for o in r["outputs"]},
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.atomic_json(rec, self.rec_path(rid))

    def run_job(self, job):
        if job not in JOBS:
            die(f"unknown job {job}")
        ids = [rid for rid in self.order if self.job[rid] == job]
        stage(f"step 7 job {job}: {len(ids)} runs" + (" (plan: nothing is executed)" if self.dry else ""))
        complete = True
        for rid in ids:
            complete = self.run_one(rid) and complete
        if self.dry:
            return complete
        rows = []
        for rid in ids:
            rec = self.record(rid)
            for o, c in rec["outputs"].items():
                if c.get("dir"):
                    rows += [(h, f"{o}/{f}") for f, h in c["files"].items()]
                else:
                    rows.append((c["sha256"], o))
        with open(self.p(f"{STATE}/{job}_checksums.tsv"), "w") as fh:
            fh.write("".join(f"{h}\t{p}\n" for h, p in sorted(rows, key=lambda x: x[1])))
        stage(f"job {job} complete: {len(ids)} runs, {len(rows)} output files checksummed")
        return True


def jobs_table(sel_runs):
    out = {j: [] for j in JOBS}
    for r in sel_runs:
        out.setdefault(job_of(r), []).append(r["id"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--job", required=True, choices=JOBS)
        p.add_argument("--workers", default=None)
        p.add_argument("--ff-workers", default=None, help="FF* GPU worker processes (A3: 1, as timed)")
        p.add_argument("--gpu", default="0")
        p.add_argument("--py-symmc", default=None)
        p.add_argument("--py-mcf", default=None)
    for p in sub.choices.values():
        p.add_argument("--root", default=REPO, help=argparse.SUPPRESS)
        p.add_argument("--selection-sha256", default=FREEZE_SHA256, help=argparse.SUPPRESS)
    args = ap.parse_args()
    sel, sel_sha = load_selection(args.root, args.selection_sha256)
    if args.cmd == "jobs":
        for j, ids in jobs_table(sel["runs"]["step7"]).items():
            print(f"{j} ({len(ids)}){' afterok ' + ','.join(AFTEROK[j]) if j in AFTEROK else ''}: {', '.join(ids)}")
        return
    dry = args.cmd == "plan"
    py = args.py_symmc or os.environ.get("PY_SYMMC") or ("$PY_SYMMC" if dry else None)
    pym = args.py_mcf or os.environ.get("PY_MCF") or ("$PY_MCF" if dry else None)
    if py is None or pym is None:
        die("PY_SYMMC / PY_MCF unset (source .n3_box/env.sh, or pass --py-symmc / --py-mcf)")
    if not dry and (args.workers is None or args.ff_workers is None):
        die("run needs --workers and --ff-workers")
    X = Runner(args.root, sel, sel_sha, args.workers or "$W", args.ff_workers or "$G", args.gpu, py, pym, dry=dry)
    X.run_job(args.job)


if __name__ == "__main__":
    main()
