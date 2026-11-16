"""N3 §7 step 9 on the box (protocol §7 step 9; runbook "Step 9"; tasks/n3_step9_clarification_2026-11-15.md): the
frozen scoring of results/n3/n3_selection.json, `runs.step9` (MCF diagnostics (b)-(d) and the Haar floor) then
`runs.step9_scoring[set]` (matches, basins, anchor, report), for one set at a time, in the freeze's order.

    python scripts/n3_step9.py plan --set testB                    # what `run` would do now (nothing is executed)
    python scripts/n3_step9.py run  --set testB --workers 28

Every command is read from the freeze and run verbatim, with three non-scientific changes: {W} becomes --workers,
`python` becomes $PY_SYMMC, and `--selection-sha256 <published freeze>` is appended to every subcommand that takes it
(n3_score match / haar-floor / basin / report, n3_mcf_diag b / d; runbook "Step 9": a box without git). A command
whose script / subcommand is not a step-9 scoring command (sampling, FF*, diagnostic (a)), that lacks --step9 where
n3_score needs it, or that carries a smoke / oracle / force flag stops the driver before anything runs. The report
config is written from `report_configs[set]` to results/n3/report_<set>.json (the frozen report command's path);
n3_score's `frozen_report` refuses any scientific difference. The freeze's SHA-256 must be the published one
(772a36e: 553f1dc9...), and tasks/n3_protocol.md must have the freeze's protocol_sha256.

Blinding: on the first start of a set, no match, basin, anchor, report or diagnostic (b)-(d) output of that set may
exist under the root. Each command's stdout / stderr goes to its own private log under
results/n3/private/step9/logs/ (report, basin, anchor and the diagnostics print outcomes); the job log gets only
start / done lines and, on a failure, the error text. Progress is followed from the events file, never from the
command logs, until both reports exist. CUDA_VISIBLE_DEVICES is empty for every command (no step-9 command uses the
GPU, and no FF* evaluation runs after step 8).

Resumption (the 2026-11-15 clarification): a command with a done record is skipped. Otherwise it runs again:
  * n3_score match and n3_mcf_diag b continue from their per-fit / per-(crystal, matcher) rows;
  * n3_mcf_diag c first deletes its own unrecorded output bundle (written atomically, but it is re-made whole);
  * haar-floor, basin, anchor, report and diagnostic d re-run whole and overwrite their outputs (deterministic).
Nothing recorded is deleted. On a resume the environment (Python, torch, pymatgen, scipy, numpy) must be the
recorded one (runbook: one CPU environment for the whole of step 9). A failure stops the driver (no retry).
Every start, resume, deletion, failure and finish is appended to results/n3/private/step9/events_<set>.jsonl. A done
record holds the SHA-256 of every file the command created or changed under the watched trees; at the end,
<set>_checksums.tsv lists them. Stdlib only: it runs under $PY_SYMMC.
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION = "results/n3/n3_selection.json"
PROTOCOL = "tasks/n3_protocol.md"
FREEZE_SHA256 = "553f1dc9efc0ec2c6d2d92b6d61bd5c76bbdc46c1d8a8ef9c814843222e1f491"   # 772a36e, published 11-08
MANIFEST = ".n3_box/manifests/step9.manifest.json"
STATE = "results/n3/private/step9"
SETS = ("testB", "devtest")
LIBS = ("torch", "pymatgen", "scipy", "numpy")

ALLOWED = {
    "scripts/n3_score.py": {"match", "haar-floor", "basin", "report"},
    "scripts/n3_mcf_diag.py": {"b", "c", "d"},
    "scripts/n3_classical.py": {"anchor"},
    "scripts/n3_step9_summary.py": {"c"},
}
# the P §3.2 (c) summary is not a freeze command: the driver appends it as each set's last unit (clarification
# 2026-11-15); a freeze command naming it stops the driver
POST_ONLY = {"scripts/n3_step9_summary.py"}
POST_ID = "post/{set}/diag-c-summary"
POST_CMD = "python scripts/n3_step9_summary.py c --set {set} --step9"
POST_OUT = "results/n3/private/mcf/diag_c_{set}.json"
NEEDS_STEP9 = {("scripts/n3_score.py", s) for s in ALLOWED["scripts/n3_score.py"]} | {("scripts/n3_step9_summary.py",
                                                                                         "c")}
TAKES_SEL_SHA = NEEDS_STEP9 | {("scripts/n3_mcf_diag.py", "b"), ("scripts/n3_mcf_diag.py", "d")}
CONTINUES = {("scripts/n3_score.py", "match"), ("scripts/n3_mcf_diag.py", "b")}
REMAKES = {("scripts/n3_mcf_diag.py", "c")}
FORBIDDEN_TOKENS = {"--force", "--discard", "--smoke", "--limit", "--only", "--shims", "--oracle", "--cpu_emulate",
                    "--selection", "--selection-sha256", "--mcf-reading", "--provenance", "--pickle", "--label",
                    "--before-form", "--allow-blind", "--allow_blind", "--slow", "--out-dir"}
PLACEHOLDERS = ("{W}",)
PATH_FLAGS = ("--bundles", "--reference", "--run_dir", "--ceiling", "--haar_floor", "--config", "--out")
WATCH = ("results/n3/private/match", "results/n3/private/basin", "results/n3/private/mcf",
         "results/n3/private/classical", "results/n3/private/draws", "n3_mcf/infer")   # (c) writes LJ rows there
WATCH_TOP = "results/n3"                                             # its own files only (the report outputs)
DELETABLE = "results/n3/private/draws/mcfRoal_"                      # diagnostic (c) bundles, the only deletions
OWN_SET = {s: re.compile(rf"(?:^|[/_~]){s}(?:[._/~]|$)") for s in ("testB", "devtest")}
SET_TOKEN = re.compile(r"_(testB|devtest)\.(?:pt|json)$|/(testB|devtest)/|^(testB|devtest)$|report_(testB|devtest)\.json$")


def blinded_globs(set_name):
    """Step-9 outputs of one set (any of them before the set's first start means something was scored early)."""
    P = "results/n3/private"
    return [f"{P}/match/*_{set_name}__*", f"{P}/match/*_{set_name}.partial__*", f"{P}/basin/*_{set_name}__*",
            f"{P}/classical/anchor_{set_name}*", f"{P}/classical/ff_labels_{set_name}*",
            f"{P}/mcf/haar_floor_{set_name}*", f"{P}/mcf/diag_b_{set_name}*", f"{P}/mcf/diag_c_{set_name}*",
            f"{P}/mcf/diag_d_{set_name}*", f"{P}/draws/mcfRoal_*_{set_name}.pt*",
            f"results/n3/report_{set_name}.out.json", f"n3_mcf/infer/*/{set_name}/*/oracle_aligned_*"]


class Stop(SystemExit):
    pass


def die(msg):
    raise Stop(f"n3_step9: {msg}")


def stage(msg):
    print(f"=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def values(tokens, flag):
    """The values after a flag, up to the next --flag (n3_score's --bundles takes several)."""
    if flag not in tokens:
        return []
    out = []
    for x in tokens[tokens.index(flag) + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


def parse(cmd, post=False):
    """-> (tokens, script, subcommand); refuses anything that is not a step-9 scoring command (post: the driver's own
    (c) summary unit, the only place its script may appear)."""
    if not isinstance(cmd, str):
        die(f"command is not a string: {cmd!r}")
    t = shlex.split(cmd)
    if len(t) < 3 or t[0] != "python":
        die(f"not a `python scripts/... <subcommand>` command: {cmd}")
    script, sub = t[1], t[2]
    if script not in ALLOWED:
        die(f"script not permitted in step 9: {cmd}")
    if (script in POST_ONLY) != post:
        die(f"{'only' if post else 'not'} the driver's (c) summary may run {script}: {cmd}")
    if sub not in ALLOWED[script]:
        die(f"subcommand not permitted in step 9: {cmd}")
    bad = [x for x in t if x in FORBIDDEN_TOKENS]
    if bad:
        die(f"forbidden token(s) {bad} in: {cmd}")
    if (script, sub) in NEEDS_STEP9 and "--step9" not in t:
        die(f"an n3_score command without --step9: {cmd}")
    left = [x for x in t if ("{" in x or "}" in x) and x not in PLACEHOLDERS]
    if left:
        die(f"unknown placeholder(s) {left} in: {cmd}")
    return t, script, sub


def sets_named(tokens):
    found = set()
    for x in tokens:
        for m in SET_TOKEN.finditer(x):
            found |= {g for g in m.groups() if g}
    return found


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


def environment(py):
    """The scoring environment of the child processes (versions read from package metadata, nothing imported)."""
    code = ("import importlib.metadata as m, json, sys, platform\n"
            "v = {}\n"
            f"for n in {LIBS!r}:\n"
            "    try:\n        v[n] = m.version(n)\n"
            "    except m.PackageNotFoundError:\n        v[n] = None\n"
            "print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable, "
            "'implementation': platform.python_implementation(), 'libraries': v}))\n")
    r = subprocess.run([py, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"cannot read the scoring environment of {py}: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


class Runner:
    def __init__(self, root, sel, sel_sha, set_name, workers, py, dry=False, manifest=None, call=None, env_of=None):
        if set_name not in SETS:
            die(f"unknown set {set_name}")
        self.root, self.sel, self.sel_sha, self.set, self.dry, self.py = root, sel, sel_sha, set_name, dry, py
        self.workers = str(workers)
        self.driver_sha = sha256(os.path.abspath(__file__))
        self.manifest = manifest
        self._call = call
        self._env_of = env_of or environment
        runs = (sel.get("runs") or {}).get("step9")
        scoring = ((sel.get("runs") or {}).get("step9_scoring") or {}).get(set_name)
        if not isinstance(runs, list) or not runs or not isinstance(scoring, list) or not scoring:
            die("the freeze has no runs.step9 / runs.step9_scoring")
        cfg = (sel.get("report_configs") or {}).get(set_name)
        if not isinstance(cfg, dict) or cfg.get("set") != set_name or not cfg.get("out"):
            die(f"the freeze has no report_configs.{set_name}")
        self.report_cfg = cfg
        self.units = []                      # [(unit id, [commands], [listed outputs])], the freeze's order
        for r in runs:
            if r.get("set") == set_name:
                if not isinstance(r.get("commands"), list) or not r["commands"] or not isinstance(r.get("outputs"), list):
                    die(f"run {r.get('id')}: commands / outputs missing")
                self.units.append((r["id"], list(r["commands"]), list(r["outputs"])))
        if not self.units:
            die(f"runs.step9 has no {set_name} run")
        for i, c in enumerate(scoring):
            t, script, sub = parse(c)
            outs = [cfg["out"]] if (script, sub) == ("scripts/n3_score.py", "report") else []
            self.units.append((f"scoring/{set_name}/{i:02d}", [c], outs))
        self.units.append((POST_ID.format(set=set_name), [POST_CMD.format(set=set_name)],
                           [POST_OUT.format(set=set_name)]))
        ids = [u[0] for u in self.units]
        if len(set(ids)) != len(ids):
            die("duplicate unit ids")
        self.config_path = f"results/n3/report_{set_name}.json"
        self.producer = {self.config_path: "(driver: report_configs)"}
        for uid, cmds, outs in self.units:
            for c in cmds:
                t, script, sub = self.parse_c(c)
                named = sets_named(t)
                if named != {set_name}:
                    die(f"{uid}: command names set(s) {sorted(named) or 'none'}, not only {set_name}: {c}")
                if (script, sub) == ("scripts/n3_score.py", "report") and values(t, "--config") != [self.config_path]:
                    die(f"{uid}: the report command does not read {self.config_path}")
            for o in outs:
                if o in self.producer:
                    die(f"output {o} is listed twice")
                self.producer[o] = uid

    # ------------------------------------------------------------------ helpers
    def parse_c(self, cmd):
        return parse(cmd, post=cmd == POST_CMD.format(set=self.set))

    def p(self, rel):
        return os.path.join(self.root, rel)

    def safe(self, uid, k):
        return f"{uid.replace('/', '__')}__{k}"

    def rec_path(self, uid, k):
        return self.p(f"{STATE}/done/{self.safe(uid, k)}.json")

    def started_path(self):
        return self.p(f"{STATE}/started_{self.set}.json")

    def log_path(self, uid, k):
        return self.p(f"{STATE}/logs/{self.set}/{self.safe(uid, k)}.log")

    def event(self, uid, k, what, detail=None):
        if self.dry:
            return
        os.makedirs(self.p(STATE), exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "slurm_job": os.environ.get("SLURM_JOB_ID"),
               "set": self.set, "unit": uid, "command": k, "event": what}
        if detail:
            row["detail"] = detail
        with open(self.p(f"{STATE}/events_{self.set}.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def atomic_json(self, obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as fh:
            json.dump(obj, fh, indent=1)
        for i in range(40):                                 # a poller holding the target open (WinError 5)
            try:
                os.replace(path + ".tmp", path)
                return
            except PermissionError:
                time.sleep(0.25)
        os.replace(path + ".tmp", path)

    def argv(self, cmd):
        t, script, sub = self.parse_c(cmd)
        out = [self.workers if x == "{W}" else x for x in t]
        out[0] = self.py
        if (script, sub) in TAKES_SEL_SHA:
            out += ["--selection-sha256", self.sel_sha]
        return out

    def call(self, argv, log):
        if self._call is not None:
            return self._call(argv, log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as fh:
            fh.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} job {os.environ.get('SLURM_JOB_ID')} "
                     f"$ {' '.join(shlex.quote(str(a)) for a in argv)}\n")
            fh.flush()
            return subprocess.run([str(a) for a in argv], cwd=self.root, stdout=fh, stderr=subprocess.STDOUT,
                                  env=env).returncode

    def mine(self, rel):
        """A file of this set: every step-9 output path names its set (so 9T and 9D, running at once, never see each
        other's files); temporaries of an atomic write are not outputs."""
        return OWN_SET[self.set].search(rel) is not None and not rel.endswith(".tmp")

    def snapshot(self):
        """{path: [size, mtime_ns]} of this set's files under the watched trees; a file that vanishes mid-walk (another
        process's temporary) is skipped."""
        snap = {}
        cands = []
        for top in WATCH:
            for d, dirs, fs in os.walk(self.p(top)):
                dirs.sort()
                cands += [os.path.join(d, f) for f in fs]
        top = self.p(WATCH_TOP)
        if os.path.isdir(top):
            cands += [os.path.join(top, f) for f in os.listdir(top)]
        for q in cands:
            rel = os.path.relpath(q, self.root).replace(os.sep, "/")
            if not self.mine(rel):
                continue
            try:
                st = os.stat(q)
            except FileNotFoundError:
                continue
            if os.path.isfile(q):
                snap[rel] = [st.st_size, st.st_mtime_ns]
        return snap

    def hashes(self, rels):
        out = {}
        for r in rels:
            try:
                out[r] = sha256(self.p(r))
            except FileNotFoundError:
                continue
        return out

    def record(self, uid, k):
        path = self.rec_path(uid, k)
        if not os.path.exists(path):
            return None
        try:
            rec = json.load(open(path))
        except json.JSONDecodeError as e:
            die(f"{path} is not valid JSON ({e}); inspect it")
        cmds = dict((u, c) for u, c, _ in self.units)[uid]
        if rec.get("unit") != uid or rec.get("k") != k or rec.get("command") != cmds[k]:
            die(f"{uid} #{k}: its done record was made for another command")
        if rec.get("selection_sha256") != self.sel_sha:
            die(f"{uid} #{k}: its done record was made under another freeze")
        return rec

    def recorded_files(self):
        out = set()
        d = self.p(f"{STATE}/done")
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".json"):
                    out |= {r for r in json.load(open(os.path.join(d, f))).get("files", {}) if self.mine(r)}
        return out

    # ------------------------------------------------------------------ preflight
    def check_manifest(self):
        """Box: every file of the step-9 manifest (code, freeze, protocol, inputs) still has its upload SHA-256."""
        if self.manifest is None:
            return None
        mp = self.p(self.manifest)
        if not os.path.isfile(mp):
            die(f"no step-9 manifest {self.manifest} under the root")
        man = json.load(open(mp))
        if man.get("step") != 9 or not isinstance(man.get("files"), list):
            die(f"{self.manifest} is not a step-9 manifest")
        bad = [f["path"] for f in man["files"]
               if not os.path.isfile(self.p(f["path"])) or os.path.getsize(self.p(f["path"])) != f["bytes"]
               or sha256(self.p(f["path"])) != f["sha256"]]
        if bad:
            die(f"{len(bad)} file(s) missing or changed since the step-9 upload: {bad[:5]}")
        need = {"scripts/n3_step9.py", SELECTION, PROTOCOL}
        listed = {f["path"] for f in man["files"]}
        if need - listed:
            die(f"the step-9 manifest does not list {sorted(need - listed)}")
        return {f["path"]: f["sha256"] for f in man["files"]}

    def in_git_tree(self):
        try:
            r = subprocess.run(["git", "-C", self.root, "rev-parse", "--is-inside-work-tree"], capture_output=True,
                               text=True)
        except (FileNotFoundError, OSError):
            return False
        return r.returncode == 0 and r.stdout.strip() == "true"

    def preflight(self):
        if self.manifest is not None and self.in_git_tree():
            die("the box root is inside a git work tree: n3_score's guard would take its git route, not "
                "--selection-sha256 (runbook Step 9)")
        proto = self.p(PROTOCOL)
        if not os.path.isfile(proto) or sha256(proto) != self.sel.get("protocol_sha256"):
            die(f"{PROTOCOL} is missing or differs from the freeze's protocol_sha256")
        listed = self.check_manifest()
        made = set(self.producer)
        missing = []
        for uid, cmds, _ in self.units:
            for c in cmds:
                t = self.parse_c(c)[0]
                for flag in PATH_FLAGS:
                    if flag == "--out":
                        continue
                    for rel in values(t, flag):
                        if rel in made:
                            continue
                        if not os.path.exists(self.p(rel)):
                            missing.append(f"{uid}: {rel}")
                        elif listed is not None and os.path.isfile(self.p(rel)) and rel not in listed:
                            die(f"{uid}: input {rel} is not in the step-9 manifest")
        if missing:
            die(f"{len(missing)} input(s) missing: {missing[:5]}")
        first = not os.path.exists(self.started_path())
        if first:
            stale = [q for q in glob.glob(self.p(f"{STATE}/done/*")) + glob.glob(self.p(f"{STATE}/snap/*"))
                     if OWN_SET[self.set].search(os.path.basename(q).replace("__", "/"))]
            stale += glob.glob(self.p(f"{STATE}/logs/{self.set}/*"))
            if stale:
                die(f"{len(stale)} {self.set} step-9 record(s) / log(s) exist before the first start: "
                    f"{[os.path.relpath(q, self.root) for q in stale[:5]]}")
            early = sorted({m for g in blinded_globs(self.set) for m in glob.glob(self.p(g))})
            if early:
                rels = [os.path.relpath(m, self.root).replace(os.sep, "/") for m in early]
                die(f"BLINDING: {len(early)} {self.set} step-9 output(s) exist before the first start: {rels[:5]}")
        cp = self.p(self.config_path)
        if os.path.exists(cp) and json.load(open(cp, encoding="utf-8")) != self.report_cfg:
            die(f"{self.config_path} exists and differs from report_configs.{self.set}")
        env = self._env_of(self.py) if not (self.dry and self.py.startswith("$")) else None
        if env is not None and not first:
            was = json.load(open(self.started_path())).get("environment")
            if was != env:
                die(f"the scoring environment changed since the first start: {was} -> {env}")
        return first, env

    # ------------------------------------------------------------------ runs
    def delete_remade(self, uid, k, t):
        keep = self.recorded_files()
        gone = []
        for rel in self.producer_outputs(uid):
            if not rel.startswith(DELETABLE) or ".." in rel.split("/"):
                die(f"{uid}: refusing to delete {rel} (outside {DELETABLE}*)")
            for q in (rel, rel + ".tmp"):
                if os.path.lexists(self.p(q)):
                    if q in keep:
                        die(f"{uid}: {q} is recorded by a finished command; not deleted")
                    gone.append(q)
        for q in gone:
            if not self.dry:
                os.remove(self.p(q))
        if gone:
            print(f"  {'would delete' if self.dry else 'deleted'} the unrecorded output of {uid}: {gone}", flush=True)
            self.event(uid, k, "deleted_partial", {"deleted": gone})

    def producer_outputs(self, uid):
        return [o for o, u in self.producer.items() if u == uid]

    def error_tail(self, log):
        """The traceback of the LAST attempt only (lines after its `### ` header, from `Traceback` on): a command may
        print outcomes before it fails, and an earlier attempt may have printed them before a stop. No traceback
        (e.g. a kill) -> nothing; the message then names the log, which is read only after unblinding."""
        try:
            lines = open(log, errors="replace").read().splitlines()
        except OSError:
            return []
        head = max((i for i, x in enumerate(lines) if x.startswith("### ")), default=-1)
        lines = lines[head + 1:]
        tb = max((i for i, x in enumerate(lines) if x.startswith("Traceback (most recent call last)")), default=None)
        return lines[tb:][-25:] if tb is not None else []

    def snap_path(self, uid, k):
        return self.p(f"{STATE}/snap/{self.safe(uid, k)}.json")

    def baseline_path(self):
        return self.p(f"{STATE}/baseline_{self.set}.json")

    def run_command(self, uid, k, cmd):
        if self.record(uid, k) is not None:
            print(f"  [done] {uid} #{k}", flush=True)
            return
        t, script, sub = self.parse_c(cmd)
        a = self.argv(cmd)
        log = self.log_path(uid, k)
        tried = os.path.exists(log)
        stage(f"{uid} #{k} {script.split('/')[-1]} {sub}" + (" (again, after a stop)" if tried else ""))
        print("$ " + " ".join(shlex.quote(str(x)) for x in a), flush=True)
        if (script, sub) in REMAKES:
            self.delete_remade(uid, k, t)
        if self.dry:
            return
        # the snapshot before the FIRST attempt is kept, so files a stopped attempt wrote (and the resumed one leaves
        # alone, e.g. a match .meta.json or rows finished before the stop) still enter this command's record
        sp = self.snap_path(uid, k)
        if os.path.exists(sp):
            before = json.load(open(sp))
        else:
            before = self.snapshot()
            self.atomic_json(before, sp)
        self.event(uid, k, "resume" if tried else "start",
                   {"mode": "continue" if (script, sub) in CONTINUES else "whole"})
        t0 = time.time()
        rc = self.call(a, log)
        wall = round(time.time() - t0, 1)
        if rc:
            self.event(uid, k, "command_failed", {"rc": rc, "wall_s": wall})
            tail = self.error_tail(log)
            rl = os.path.relpath(log, self.root).replace(os.sep, "/")
            die(f"{uid} #{k}: rc {rc} from `{cmd}`; its private log is {rl}"
                + ("\n" + "\n".join(tail) if tail else " (no traceback: not shown)"))
        after = self.snapshot()
        changed = sorted(r for r, v in after.items() if before.get(r) != v)
        files = self.hashes(changed)
        rec = {"unit": uid, "k": k, "command": cmd, "executed": [str(x) for x in a], "rc": rc, "wall_s": wall,
               "selection_sha256": self.sel_sha, "driver_sha256": self.driver_sha, "files": files,
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.atomic_json(rec, self.rec_path(uid, k))
        self.event(uid, k, "done", {"wall_s": wall, "files_changed": len(files)})

    def run(self):
        stage(f"step 9 {self.set}: {len(self.units)} units, {sum(len(c) for _, c, _ in self.units)} commands"
              + (" (plan: nothing is executed)" if self.dry else ""))
        first, env = self.preflight()
        stage(f"preflight OK ({'first start' if first else 'resume'}); environment {env}")
        if not self.dry:
            if first:
                self.atomic_json(self.snapshot(), self.baseline_path())     # this set's files before any command
                self.atomic_json({"set": self.set, "first_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                                  "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
                                  "platform": platform.platform(), "environment": env,
                                  "selection_sha256": self.sel_sha, "driver_sha256": self.driver_sha},
                                 self.started_path())
                self.event("-", None, "first_start", {"environment": env})
            else:
                self.event("-", None, "restart", {"host": socket.gethostname()})
            for d in WATCH:                    # n3_classical anchor writes into classical/ without creating it
                os.makedirs(self.p(d), exist_ok=True)
            cp = self.p(self.config_path)
            if not os.path.exists(cp):
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                with open(cp, "w", encoding="utf-8") as fh:
                    json.dump(self.report_cfg, fh, indent=1)
                self.event("-", None, "wrote_report_config", {"path": self.config_path, "sha256": sha256(cp)})
        for uid, cmds, outs in self.units:
            for k, c in enumerate(cmds):
                self.run_command(uid, k, c)
            if self.dry:
                continue
            lost = [o for o in outs if not os.path.exists(self.p(o))]
            if lost:
                self.event(uid, None, "outputs_missing", {"missing": lost})
                die(f"{uid}: finished without its listed output(s) {lost}")
        if self.dry:
            return
        # every file of this set that is new or changed since the set's first start (independent of the per-command
        # records, so no stop can drop one); each recorded file must be among them
        base = json.load(open(self.baseline_path()))
        now = self.snapshot()
        final = self.hashes(sorted(r for r, v in now.items() if base.get(r) != v))
        recorded = {r for uid, cmds, _ in self.units for k in range(len(cmds)) for r in self.record(uid, k)["files"]}
        lost = sorted(recorded - set(final))
        if lost:
            die(f"recorded output(s) vanished before the end of the set: {lost[:5]}")
        with open(self.p(f"{STATE}/{self.set}_checksums.tsv"), "w") as fh:
            fh.write("".join(f"{h}\t{r}\n" for r, h in sorted(final.items())))
        self.event("-", None, "set_done", {"files": len(final)})
        stage(f"step 9 {self.set} complete: {len(self.units)} units, {len(final)} output files checksummed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--set", required=True, choices=SETS)
        p.add_argument("--workers", default=None)
        p.add_argument("--py-symmc", default=None)
        p.add_argument("--manifest", default=None, help=f"the step-9 upload manifest (box: {MANIFEST})")
        p.add_argument("--root", default=REPO, help=argparse.SUPPRESS)
        p.add_argument("--selection-sha256", default=FREEZE_SHA256, help=argparse.SUPPRESS)
    args = ap.parse_args()
    sel, sel_sha = load_selection(args.root, args.selection_sha256)
    dry = args.cmd == "plan"
    py = args.py_symmc or os.environ.get("PY_SYMMC") or ("$PY_SYMMC" if dry else None)
    if py is None:
        die("PY_SYMMC unset (source .n3_box/env.sh, or pass --py-symmc)")
    if not dry and args.workers is None:
        die("run needs --workers")
    X = Runner(args.root, sel, sel_sha, args.set, args.workers or "$W", py, dry=dry, manifest=args.manifest)
    X.run()


if __name__ == "__main__":
    main()
