"""N3 step 5b on the box (protocol §2.3, §3, §7 step 5; runbook "Step 5"): sample every VALSEL candidate, score and
select each arm, run diagnostic (a). Frank, 2026-11-06: MCF "top-3 + last" is read as a set (a last.ckpt that is
byte-identical to a top-3 checkpoint is listed once, as `last`); the iii-s VAL sanity row runs here (job VS).

    python scripts/n3_step5b.py designate [--check]      # MCF checkpoint designation from the raw training records
    python scripts/n3_step5b.py plan --job VR|VA|VC|VS     # what `run` would do now (nothing is executed)
    python scripts/n3_step5b.py run --job VR|VA|VC|VS --workers W [--ff-workers G]

Jobs. Each is resumable: a step whose outputs exist is skipped, a bundle is never re-sampled or rewritten, matching
only adds missing rows, and a select record is reused only when it was made from the identical config.
  VR  MCF-R Stage A (5 s_uR x 3 seeds at each seed's lowest logged valid/loss checkpoint) -> to_bundle -> +G gauge
      scan -> primary match -> select mcfR_A, mcfRG_A -> Stage-B candidates at each arm's chosen s_uR (MCF-R draws
      at +G's s_uR are sampled only when it differs; Stage-A draws are reused) -> gauge -> match -> select mcfR_B,
      mcfRG_B -> ii-S of +G's frozen draws -> match -> select iiS -> diagnostic (a) of the selected MCF-R (and +G,
      when it selected another checkpoint) checkpoints at their frozen s_uR, then the Thurlemann provenance repeat
  VA  MCF-A stage 1 (every candidate at s_uF 9, s_uR 3) -> mcf10 + mcf08 match -> select mcfA_1 -> stage 2 (the
      other 8 knob cells at the stage-1 checkpoints) -> mcf10 match -> select mcfA_2 -> select mcfA_3 ->
      diagnostic (a) at the frozen (s_uF, s_uR)
  VC  iv-P: the six `sample` commands `select window` printed -> match -> select ivP -> select ivs -> iv-S ->
      match -> select ivS; OURS SEL sampling and VAL+SEL concat; iii-u, iii-p, p1ref, P1 press of OURS -> match
      (P1 against the pressed truth) -> select iii-u, iii-p, p1-ours, p1-iii-p (configs from
      `n3_select.py --write-configs`); last, OURS-N2 SEL sampling and concat (nothing selects on it)
  VS  iii-s VAL sanity row (§3.3): screen, then search under the fixed FF* configuration
Stdlib only: it runs under $PY_SYMMC and calls $PY_MCF for the n3_mcf_* scripts (paths from the environment that
.n3_box/env.sh exports, or --py-symmc / --py-mcf).
"""
import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIV = "results/n3/private"
DRAWS, SEL, MCF, OLD, CLS = (f"{PRIV}/{d}" for d in ("draws", "select", "mcf", "old", "classical"))
DESIGNATION = "results/n3/mcf_designation_valsel.json"
SEEDS = (0, 1, 2)
MCFR_U = (1.0, 2.0, 3.0, 5.0, 10.0)                                   # = n3_select.MCFR_U (§3.2 ii-R)
MCFA_GRID = tuple((f, u) for f in (5.0, 9.0, 13.0) for u in (1.0, 2.0, 3.0))   # (s_uF, s_uR) = n3_select.MCFA_KNOBS
MCFA_STAGE1 = (9.0, 3.0)
MENU = ["lj", "resid"] + [f"mode@{r}" for r in (0.25, 0.5, 0.75, 1.0, 1.5)] + ["random"]   # = n3_score.MENU
MCFA_MENU = ["random", "lj", "resid"]                                 # = n3_select.MCFA_MENU
CLASSICAL_ROLES = ("iii-u", "iii-p", "p1-ours", "p1-iii-p")           # configs: n3_select.py --write-configs
THURLEMANN_CKPT = "external/mcf_release/model-checkpoints/thurlemann23/best.ckpt"
THURLEMANN_PICKLE = "external/mcf_release/thurlemann23/preprocessed/normalized/test_molcrystal_normalized.pkl.gz"
IVP_FLAGS = ["--pack", "--crystals-per-batch", "8", "--device", "cuda"]   # A3: iv-P pack, 8 crystals per batch
CKPT_RE = re.compile(r"^epoch_(\d+)-step_(\d+)")
RULES = {
    "candidates": "MCF ModelCheckpoint top-3 valid/loss plus last (protocol §3.2: P:517-518, 553, 579, 1234), read "
                  "as a set: a last.ckpt byte-identical (SHA-256) to a top-3 checkpoint is listed once, under the "
                  "label `last`; candidates are listed by ascending logged valid/loss (the listed-order fallback)",
    "decision": "Frank 2026-11-06: set reading",
    "stage_a": "MCF-R Stage A samples each seed's lowest logged valid/loss checkpoint at all five s_uR (P:549)",
    "val_loss": "MCF-R / +G Stage B ties: fixed-draw VAL loss (P:554); MCF-A stage-1 ties: MCF's logged "
                "valid/loss (P:606)",
    "logged_valid_loss": "train_report checkpoints[].logged_valid_loss: MCF's logged valid/loss at the checkpoint",
    "fixed_val_loss": "n3_mcf_run.py fixed_val_loss (fixed-draw VAL loss, results/n3/private/mcf/fixed_val_loss.jsonl)",
}


class Pending(Exception):
    """plan: the rest of the job is decided by a record that does not exist yet."""


def g(x):
    return f"{float(x):g}"


def die(msg):
    raise SystemExit(f"n3_step5b: {msg}")


def stage(msg):
    print(f"=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mcf_bundle(pre, label, u, s, f=None):
    """Repo-relative VALSEL bundle of an MCF arm (n3_mcf_run.bundle_arm / n3_mcf_gauge / n3_mcf_sg naming)."""
    return f"{DRAWS}/{pre}_{label}_u{g(u)}" + (f"_f{g(f)}" if f is not None else "") + f"_s{s}_valsel.pt"


def mcf_run_dir(arm, label, u, s, f=None):
    """n3_mcf_lib.INFER_ROOT/<arm_name>/valsel/<run_label> of one sampling call."""
    return (f"n3_mcf/infer/{'mcfR' if arm == 'R' else 'mcfA'}/valsel/s{s}_{label}_u{g(u)}"
            + (f"_f{g(f)}" if arm == "A" else ""))


def rec_path(role):
    return f"{SEL}/{role}_record.json"


def cfg_path(role):
    return f"{SEL}/{role}.json"


def knob_label(f, u):
    return f"f{g(f)}_u{g(u)}"


# ================================================================================================ designation
def build_designation(root=REPO):
    """The MCF checkpoint designation, from the raw training records only (runbook: before any VALSEL score)."""
    fpath = f"{MCF}/fixed_val_loss.jsonl"
    fixed = {}
    for line in open(os.path.join(root, fpath)):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["run_name"], r["label"])
        if key in fixed and fixed[key]["ckpt_sha256"] != r["ckpt_sha256"]:
            die(f"{fpath}: two fixed-draw losses of {key} with different checkpoint SHA-256")
        fixed[key] = r
    inputs = {fpath: sha256(os.path.join(root, fpath))}
    runs = {}
    for arm in ("R", "A"):
        for s in SEEDS:
            run = f"n3_mcf{arm}_s{s}"
            tp = f"{MCF}/train_report/{run}.json"
            rep = json.load(open(os.path.join(root, tp)))
            inputs[tp] = sha256(os.path.join(root, tp))
            ck_dir = f"n3_mcf/runs/{run}/ckpt"
            if rep.get("run") != run or str(rep.get("ckpt_dir", "")).replace("\\", "/").rstrip("/") != ck_dir:
                die(f"{tp}: run / ckpt_dir {rep.get('run')} / {rep.get('ckpt_dir')} is not {run} / {ck_dir}")
            epochs, last = [], None
            for c in rep["checkpoints"]:
                if c["ckpt"] == "last.ckpt":
                    last = c
                    continue
                m = CKPT_RE.match(c["ckpt"])
                if m is None or (int(m[1]), int(m[2])) != (c["epoch"], c["global_step"]):
                    die(f"{tp}: {c['ckpt']} does not name its epoch / step {c['epoch']} / {c['global_step']}")
                epochs.append(c)
            if len(epochs) != 3 or last is None:
                die(f"{tp}: {len(epochs)} top-3 checkpoints and last {'present' if last else 'absent'}")
            twins = [c for c in epochs if c["ckpt_sha256"] == last["ckpt_sha256"]]
            if len(twins) > 1 or len({c["ckpt_sha256"] for c in epochs}) != 3:
                die(f"{tp}: top-3 checkpoints are not three distinct files")
            if twins and (twins[0]["epoch"], twins[0]["global_step"]) != (last["epoch"], last["global_step"]):
                die(f"{tp}: last.ckpt has the SHA-256 of {twins[0]['ckpt']} but another epoch / step")

            def entry(c):
                lab = "last" if c["ckpt"] == "last.ckpt" else f"ep{c['epoch']}_st{c['global_step']}"
                fr = fixed.get((run, lab))
                if fr is None:
                    die(f"{fpath}: no fixed-draw VAL loss for {run} {lab}")
                if fr["ckpt_sha256"] != c["ckpt_sha256"]:
                    die(f"{run} {lab}: fixed_val_loss SHA-256 differs from train_report")
                for k, v in (("logged valid/loss", c.get("logged_valid_loss")), ("fixed-draw VAL loss",
                                                                                 fr.get("fixed_val_loss"))):
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                        die(f"{run} {lab}: {k} {v} is not finite")
                return {"label": lab, "ckpt": f"{ck_dir}/{c['ckpt']}", "ckpt_sha256": c["ckpt_sha256"],
                        "epoch": c["epoch"], "global_step": c["global_step"],
                        "logged_valid_loss": c["logged_valid_loss"], "fixed_val_loss": fr["fixed_val_loss"]}
            ranked = sorted(epochs, key=lambda c: c["logged_valid_loss"])
            if ranked[0]["logged_valid_loss"] == ranked[1]["logged_valid_loss"]:
                die(f"{run}: exact tie for the lowest logged valid/loss (no rule; ask Frank)")
            twin = twins[0] if twins else None
            cands = [entry(last) if c is twin else entry(c) for c in ranked] + ([] if twin else [entry(last)])
            runs[run] = {"arm": arm, "seed": s, "checkpoints": [entry(c) for c in epochs] + [entry(last)],
                         "last_identical_to": None if twin is None else entry(twin)["label"],
                         "lowest_logged_valid_loss": entry(ranked[0])["label"],
                         "stage_a": cands[0]["label"], "candidates": cands}
    return {"what": "N3 step-5b MCF checkpoint designation (§3.2), made from the raw training records before any "
                    "VALSEL score (runbook step 5)",
            "rules": RULES, "inputs": inputs, "runs": runs}


def load_designation(root=REPO):
    p = os.path.join(root, DESIGNATION)
    if not os.path.exists(p):
        die(f"{DESIGNATION} missing: run `designate` on the laptop, commit and upload it")
    d = json.load(open(p))
    if d != build_designation(root):
        die(f"{DESIGNATION} differs from the training records under {root}")
    return d


def cmd_designate(args):
    d = build_designation(args.root)
    p = os.path.join(args.root, DESIGNATION)
    if args.check:
        load_designation(args.root)
        print(f"{DESIGNATION}: matches the training records")
        return
    if os.path.exists(p) and json.load(open(p)) != d and not args.force:
        die(f"{DESIGNATION} exists and differs (--force replaces it before any VALSEL score only)")
    with open(p + ".tmp", "w") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    os.replace(p + ".tmp", p)
    for run, r in d["runs"].items():
        print(f"{run}: stage A {r['stage_a']}; candidates {[c['label'] for c in r['candidates']]}; "
              f"last = {r['last_identical_to']}")
    print("wrote", DESIGNATION)


def cands(D, arm, s):
    return D["runs"][f"n3_mcf{arm}_s{s}"]["candidates"]


def cand(D, arm, s, label):
    return next(c for c in cands(D, arm, s) if c["label"] == label)


def stage_a_label(D, s):
    return D["runs"][f"n3_mcfR_s{s}"]["stage_a"]


# ================================================================================================ select configs
def cfg_mcfr_a(D, pre):
    """mcfR_A / mcfRG_A: five s_uR cells of the seeds' Stage-A draws x the §2.3 menu, pooled over seeds; ties
    draw@1 -> any@16 -> s_uR 3 -> selector (P:546-551)."""
    cells = [{"label": f"u{g(u)}", "bundles": [mcf_bundle(pre, stage_a_label(D, s), u, s) for s in SEEDS]}
             for u in MCFR_U]
    return {"name": f"{pre}_A", "matcher": "primary", "menu": MENU, "mode_tiebreak": "resid",
            "stage_a": {"cells": cells, "rank": "hits", "tie": ["draw1", "any16", "prefer_cell", "selector"],
                        "prefer_cell": "u3"},
            "out": rec_path(f"{pre}_A")}


def cfg_mcfr_b(D, pre, u, selector):
    """mcfR_B / mcfRG_B: each seed's candidates at the Stage-A s_uR and selector; ties draw@1 -> any@16 ->
    fixed-draw VAL loss (P:552-554)."""
    seeds = {str(s): [{"label": c["label"], "bundle": mcf_bundle(pre, c["label"], u, s), "val_loss": c["fixed_val_loss"]}
                      for c in cands(D, "R", s)] for s in SEEDS}
    return {"name": f"{pre}_B", "matcher": "primary", "mode_tiebreak": "resid",
            "stage_b": {"selector": selector, "rank": "picked", "tie": ["draw1", "any16", "val_loss"], "seeds": seeds},
            "out": rec_path(f"{pre}_B")}


def cfg_iis(chosen, u):
    """iiS: the ii-S expansion of +G's frozen draws, its own §2.3 selector (P:595)."""
    return {"name": "iiS", "matcher": "primary", "menu": MENU, "mode_tiebreak": "resid",
            "stage_a": {"cells": [{"label": "iiS", "bundles": [mcf_bundle("mcfRS", chosen[s], u, s) for s in SEEDS]}],
                        "rank": "hits", "tie": ["selector"]},
            "out": rec_path("iiS")}


def cfg_mcfa_1(D):
    """mcfA_1: each seed's candidates at (s_uF 9, s_uR 3), any-of-10 at stol 1.0 -> 0.8 -> logged valid/loss
    (P:604-606; A2 item 11)."""
    f, u = MCFA_STAGE1
    seeds = {str(s): [{"label": c["label"], "bundle": mcf_bundle("mcfA", c["label"], u, s, f),
                       "val_loss": c["logged_valid_loss"]} for c in cands(D, "A", s)] for s in SEEDS}
    return {"name": "mcfA_1", "matcher": "primary",
            "stage_b": {"selector": None, "rank": "any10@mcf10", "tie": ["any10@mcf08", "val_loss"], "seeds": seeds},
            "out": rec_path("mcfA_1")}


def cfg_mcfa_2(chosen):
    """mcfA_2: the 3 x 3 (s_uF, s_uR) grid at the stage-1 checkpoints, seed-summed any-of-10 at stol 1.0, ties
    (9, 3) (P:608)."""
    cells = [{"label": knob_label(f, u), "bundles": [mcf_bundle("mcfA", chosen[s], u, s, f) for s in SEEDS]}
             for f, u in MCFA_GRID]
    return {"name": "mcfA_2", "matcher": "primary",
            "stage_a": {"cells": cells, "menu": None, "rank": "any10@mcf10", "tie": ["prefer_cell"],
                        "prefer_cell": knob_label(*MCFA_STAGE1)},
            "out": rec_path("mcfA_2")}


def cfg_mcfa_3(chosen, f, u):
    """mcfA_3: {RANDOM, LJ, RESID} on the chosen cell by picked@1 at stol 1.0 (P:609)."""
    cell = {"label": knob_label(f, u), "bundles": [mcf_bundle("mcfA", chosen[s], u, s, f) for s in SEEDS]}
    return {"name": "mcfA_3", "matcher": "primary",
            "stage_a": {"cells": [cell], "menu": MCFA_MENU, "rank": "picked@mcf10", "tie": ["selector"]},
            "out": rec_path("mcfA_3")}


def chosen_u(rec):
    """(s_uR, selector) chosen by an mcfR_A / mcfRG_A record."""
    by_label = {f"u{g(u)}": u for u in MCFR_U}
    ch = rec["stage_a"]["chosen"]
    if ch["cell"] not in by_label:
        die(f"{rec.get('name')}: chosen cell {ch['cell']} is not an s_uR cell")
    return by_label[ch["cell"]], ch["selector"]


def chosen_b(rec):
    return {int(s): v["chosen"] for s, v in rec["stage_b"]["seeds"].items()}


def chosen_knobs(rec):
    by_label = {knob_label(f, u): (f, u) for f, u in MCFA_GRID}
    cell = rec["stage_a"]["chosen"]["cell"]
    if cell not in by_label:
        die(f"mcfA_2: chosen cell {cell} is not a knob cell")
    return by_label[cell]


def stage_b_samples(D, uR, uG):
    """(label, s_uR, seed) MCF-R draws Stage B needs beyond Stage A: every candidate at MCF-R's s_uR, and at +G's
    s_uR only when it differs (P:579-580); each seed's Stage-A checkpoint already has draws at every s_uR."""
    out = []
    for s in SEEDS:
        for c in cands(D, "R", s):
            if c["label"] == stage_a_label(D, s):
                continue
            for u in sorted({uR, uG}):
                out.append((c["label"], u, s))
    return out


BUNDLE_RE = re.compile(r"^(.+)_s(-?\d+)_valsel\.pt$")


def arm_identity(arm):
    """= n3_score.arm_identity: checkpoint labels may differ across the seeds of one arm / knob cell."""
    m = re.fullmatch(r"(mcfR|mcfRG|mcfRS|mcfRoal|mcfA)_.+(_u[^_]+(?:_f[^_]+)?)", arm)
    if m:
        return m[1] + m[2]
    m = re.fullmatch(r"((?:ivP|ivS)_lr[^_]+)_ep\d+", arm)
    return m[1] if m else arm


def match_groups(bundles):
    """Split bundles (order kept) into groups of one arm identity with unique seeds (n3_score.validate_bundles);
    a name that does not parse goes alone."""
    groups = []
    for b in bundles:
        m = BUNDLE_RE.match(os.path.basename(b))
        if m is None:
            groups.append(([b], None, None))
            continue
        ident, seed = arm_identity(m[1]), int(m[2])
        for grp, gid, seeds in groups:
            if gid == ident and seed not in seeds:
                grp.append(b)
                seeds.add(seed)
                break
        else:
            groups.append(([b], ident, {seed}))
    return [grp for grp, _, _ in groups]


def ivp_runs(win):
    """The (run, epoch, seed) of the `sample` commands `select window` printed, checked against the A3 flags."""
    out = []
    for c in win["next_commands"]:
        t = shlex.split(c)
        if t[2:3] != ["sample"]:
            continue
        if t[:2] != ["python", "scripts/n3_old_symmc.py"] or t[3] != "--run" or t[5] != "--epoch" or \
                t[7:9] != ["--set", "valsel"] or t[9:] != IVP_FLAGS:
            die(f"unexpected iv-P sample command: {c}")
        m = re.fullmatch(r"(lr[0-9e.+-]+)_s(\d)", t[4])
        if m is None:
            die(f"iv-P run name {t[4]}")
        out.append((t[4], int(t[6]), int(m[2]), m[1]))
    want = {(s, e) for s in SEEDS for e in (win["candidates"][str(s)]["best"], win["candidates"][str(s)]["last"])}
    if {(s, e) for _, e, s, _ in out} != want or len(out) != len(want):
        die(f"the printed iv-P sample commands {[(s, e) for _, e, s, _ in out]} are not the window candidates {want}")
    return out


# ================================================================================================ runner
class Runner:
    def __init__(self, root, dry, workers, ff_workers, py_symmc, py_mcf, gpu=0):
        self.root, self.dry, self.W, self.G, self.gpu = root, dry, workers, ff_workers, gpu
        self.py, self.pym = py_symmc, py_mcf

    def p(self, rel):
        return os.path.join(self.root, rel)

    def exists(self, rel):
        return os.path.exists(self.p(rel))

    def load(self, rel):
        try:
            with open(self.p(rel)) as fh:
                return json.load(fh)
        except json.JSONDecodeError as e:
            die(f"{rel} is not valid JSON ({e}): truncated by a wall-time kill? inspect and remove it")

    def call(self, argv, fatal=True):
        print("$ " + " ".join(shlex.quote(str(a)) for a in argv), flush=True)
        if self.dry:
            return 0
        rc = subprocess.run([str(a) for a in argv], cwd=self.root).returncode
        if rc and fatal:
            die(f"rc {rc}: {' '.join(map(str, argv[1:4]))} ...")
        return rc

    def step(self, name, argv, outputs=()):
        """Run argv unless every output exists."""
        if outputs and all(self.exists(o) for o in outputs):
            print(f"  [have] {name}", flush=True)
            return
        stage(name)
        self.call(argv)
        missing = [o for o in outputs if not self.exists(o)]
        if missing and not self.dry:
            die(f"{name}: did not write {missing}")

    def need(self, rel):
        if not self.exists(rel):
            if self.dry:
                raise Pending(rel)
            die(f"{rel} missing")

    # ------------------------------------------------------------------ MCF
    def verify_ckpts(self, D, arm):
        absent = 0
        for s in SEEDS:
            for c in cands(D, arm, s):
                if not self.exists(c["ckpt"]):
                    if not self.dry:
                        die(f"{c['ckpt']} missing")
                    absent += 1
                elif sha256(self.p(c["ckpt"])) != c["ckpt_sha256"]:
                    die(f"{c['ckpt']}: SHA-256 is not the designated one")
        stage(f"MCF-{arm} candidate checkpoints match {DESIGNATION}"
              + (f" ({absent} not present here: plan only)" if absent else ""))

    def mcf_sample(self, arm, s, c, u, f=None):
        """One MCF sampling call (with its RESID pass and check) and its bundle. A complete run whose bundle is
        missing is only bundled; an interrupted or failing run is discarded and regenerated once (§3.2: never
        scored)."""
        out = mcf_bundle("mcfR" if arm == "R" else "mcfA", c["label"], u, s, f)
        if self.exists(out):
            print(f"  [have] {out}", flush=True)
            return out
        rd = mcf_run_dir(arm, c["label"], u, s, f)
        infer = [self.pym, "scripts/n3_mcf_run.py", "infer", "--ckpt", c["ckpt"], "--set", "valsel", "--s_uR", g(u)] \
            + (["--s_uF", g(f)] if arm == "A" else []) + ["--gpu", str(self.gpu), "--run"]
        stage(f"MCF-{arm} s{s} {c['label']} s_uR {g(u)}" + (f" s_uF {g(f)}" if arm == "A" else ""))
        complete = self.exists(rd) and not self.dry and self.call(
            [self.pym, "scripts/n3_mcf_run.py", "check", "--run_dir", rd], fatal=False) == 0
        regenerated = False                           # §3.2 / P:544: at most ONE regeneration per submission
        while not complete:
            discard = self.exists(rd) and not self.dry
            if discard:
                if regenerated:
                    die(f"MCF-{arm} s{s} {c['label']} u{g(u)}: the regenerated run failed too (inspect {rd})")
                regenerated = True
                stage(f"discard {rd} and regenerate once (§3.2: a failing or interrupted run is never scored)")
            if self.call(infer + (["--discard"] if discard else []), fatal=False) == 0:
                break
            if not self.exists(rd):
                die(f"MCF-{arm} s{s} {c['label']} u{g(u)}: infer failed before writing {rd}")
        self.call([self.pym, "scripts/n3_mcf_run.py", "to_bundle", "--run_dir", rd, "--workers", self.W])
        if not self.dry and not self.exists(out):
            die(f"to_bundle did not write {out}")
        return out

    def gauge(self, label, u, s):
        rd = mcf_run_dir("R", label, u, s)
        self.step(f"+G gauge scan s{s} {label} s_uR {g(u)}",
                  [self.pym, "scripts/n3_mcf_gauge.py", "--run_dir", rd, "--workers", self.W],
                  [mcf_bundle("mcfRG", label, u, s)])

    def match(self, bundles, matcher, reference=None):
        """n3_score.validate_bundles takes one arm identity and unique seeds per call; match rows and metadata are
        per bundle, so one call per such group is equivalent."""
        for group in match_groups(bundles):
            stage(f"match {matcher}" + (f" vs {os.path.basename(reference)}" if reference else "")
                  + f": {len(group)} bundles")
            self.call([self.py, "scripts/n3_score.py", "match", "--matcher", matcher, "--draws", "all",
                       "--workers", self.W] + (["--reference", reference] if reference else []) + ["--bundles"] + group)

    def select(self, role, cfg):
        """Write role's config (an existing one must be identical) and score it, unless its record already exists
        from that config."""
        cp = cfg_path(role)
        if self.exists(cp):
            if self.load(cp) != cfg:
                die(f"{cp} exists and differs from the config this job derives")
        elif self.dry:
            print(f"  (would write {cp})")
        else:
            os.makedirs(os.path.dirname(self.p(cp)), exist_ok=True)
            with open(self.p(cp) + ".tmp", "w") as fh:
                json.dump(cfg, fh, indent=1)
            os.replace(self.p(cp) + ".tmp", self.p(cp))
        self.score(role, cp, cfg)

    def score(self, role, cp, cfg):
        out = cfg["out"]
        if self.exists(out):
            if self.load(out).get("config") != cfg:
                die(f"{out} exists but was not made from {cp}")
            print(f"  [have] {out}", flush=True)
            return
        stage(f"select {role}")
        self.call([self.py, "scripts/n3_score.py", "select", "--config", cp])
        if not self.dry and not self.exists(out):
            die(f"select {role} did not write {out}")

    def score_existing(self, role, cp):
        self.need(cp)
        self.score(role, cp, self.load(cp))

    def record(self, role, cfg=None):
        rp_ = rec_path(role)
        self.need(rp_)
        rec = self.load(rp_)
        if cfg is not None and rec.get("config") != cfg:
            die(f"{rp_} was not made from the config this job derives")
        return rec

    def diag(self, label, ckpt, sha, u, f=None, provenance=False, pickle=None):
        out = f"{MCF}/diag_a_{label}.json"
        if self.exists(out):
            d = self.load(out)
            k = d.get("knobs") or {}
            same = (d.get("label") == label and d.get("reportable") is True and bool(d.get("provenance_repeat")) is provenance
                    and (sha is None or d.get("ckpt_sha256") == sha) and float(k.get("s_uR", -1)) == float(u)
                    and (f is None or float(k.get("s_uF", -1)) == float(f)))
            if not same:
                die(f"{out} exists for another checkpoint / knobs (diagnostic (a) is never re-run over a record)")
            print(f"  [have] {out}: {d.get('verdict')}", flush=True)
            return
        argv = [self.pym, "scripts/n3_mcf_diag.py", "a", "--ckpt", ckpt, "--label", label, "--s_uR", g(u)]
        argv += (["--s_uF", g(f)] if f is not None else [])
        argv += (["--provenance", "--pickle", pickle] if provenance else ["--set", "valsel"])
        self.step(f"diagnostic (a) {label}", argv, [out])


# ================================================================================================ jobs
def job_vr(X):
    D = load_designation(X.root)
    X.verify_ckpts(D, "R")
    sa = {s: cand(D, "R", s, stage_a_label(D, s)) for s in SEEDS}
    stage("MCF-R Stage A: 5 s_uR x 3 seeds, each seed's lowest logged valid/loss checkpoint")
    for u in MCFR_U:
        for s in SEEDS:
            X.mcf_sample("R", s, sa[s], u)
    for u in MCFR_U:
        for s in SEEDS:
            X.gauge(sa[s]["label"], u, s)
    for pre in ("mcfR", "mcfRG"):
        for u in MCFR_U:
            X.match([mcf_bundle(pre, sa[s]["label"], u, s) for s in SEEDS], "primary")
    for pre in ("mcfR", "mcfRG"):
        X.select(f"{pre}_A", cfg_mcfr_a(D, pre))
    uR, selR = chosen_u(X.record("mcfR_A", cfg_mcfr_a(D, "mcfR")))
    uG, selG = chosen_u(X.record("mcfRG_A", cfg_mcfr_a(D, "mcfRG")))
    stage(f"Stage B: MCF-R at s_uR {g(uR)} ({selR}), +G at s_uR {g(uG)} ({selG})")
    for label, u, s in stage_b_samples(D, uR, uG):
        X.mcf_sample("R", s, cand(D, "R", s, label), u)
    for s in SEEDS:
        for c in cands(D, "R", s):
            X.gauge(c["label"], uG, s)
    for s in SEEDS:
        X.match([mcf_bundle("mcfR", c["label"], uR, s) for c in cands(D, "R", s)], "primary")
        X.match([mcf_bundle("mcfRG", c["label"], uG, s) for c in cands(D, "R", s)], "primary")
    X.select("mcfR_B", cfg_mcfr_b(D, "mcfR", uR, selR))
    X.select("mcfRG_B", cfg_mcfr_b(D, "mcfRG", uG, selG))
    rch = chosen_b(X.record("mcfR_B", cfg_mcfr_b(D, "mcfR", uR, selR)))
    gch = chosen_b(X.record("mcfRG_B", cfg_mcfr_b(D, "mcfRG", uG, selG)))
    for s in SEEDS:
        X.step(f"ii-S s{s} of +G {gch[s]}", [X.pym, "scripts/n3_mcf_sg.py", "--bundle",
                                             mcf_bundle("mcfRG", gch[s], uG, s), "--workers", X.W],
               [mcf_bundle("mcfRS", gch[s], uG, s)])
    X.match([mcf_bundle("mcfRS", gch[s], uG, s) for s in SEEDS], "primary")
    X.select("iiS", cfg_iis(gch, uG))
    for s in SEEDS:
        c = cand(D, "R", s, rch[s])
        X.diag(f"mcfR_s{s}", c["ckpt"], c["ckpt_sha256"], uR)
    for s in SEEDS:                                   # n3_select.diag_a: +G only where it chose another checkpoint
        c = cand(D, "R", s, gch[s])
        if c["ckpt_sha256"] != cand(D, "R", s, rch[s])["ckpt_sha256"]:
            X.diag(f"mcfRG_s{s}", c["ckpt"], c["ckpt_sha256"], uG)
    X.diag("thurlemann", THURLEMANN_CKPT, None, 3.0, None, provenance=True, pickle=THURLEMANN_PICKLE)


def job_va(X):
    D = load_designation(X.root)
    X.verify_ckpts(D, "A")
    f1, u1 = MCFA_STAGE1
    stage("MCF-A stage 1: every candidate at (s_uF 9, s_uR 3)")
    for s in SEEDS:
        for c in cands(D, "A", s):
            X.mcf_sample("A", s, c, u1, f1)
    for s in SEEDS:
        for m in ("mcf10", "mcf08"):
            X.match([mcf_bundle("mcfA", c["label"], u1, s, f1) for c in cands(D, "A", s)], m)
    X.select("mcfA_1", cfg_mcfa_1(D))
    ach = chosen_b(X.record("mcfA_1", cfg_mcfa_1(D)))
    stage(f"MCF-A stage 2 at the stage-1 checkpoints {ach}")
    for f, u in MCFA_GRID:
        if (f, u) != MCFA_STAGE1:
            for s in SEEDS:
                X.mcf_sample("A", s, cand(D, "A", s, ach[s]), u, f)
    for f, u in MCFA_GRID:
        X.match([mcf_bundle("mcfA", ach[s], u, s, f) for s in SEEDS], "mcf10")
    X.select("mcfA_2", cfg_mcfa_2(ach))
    f, u = chosen_knobs(X.record("mcfA_2", cfg_mcfa_2(ach)))
    X.select("mcfA_3", cfg_mcfa_3(ach, f, u))
    X.record("mcfA_3", cfg_mcfa_3(ach, f, u))
    for s in SEEDS:
        c = cand(D, "A", s, ach[s])
        X.diag(f"mcfA_s{s}", c["ckpt"], c["ckpt_sha256"], u, f)


def ours(X, arm):
    for s in SEEDS:
        X.need(f"{DRAWS}/{arm}_s{s}_val.pt")
    X.step(f"{arm} SEL sampling (seed 90000 + i)", [X.py, "scripts/n3_ours.py", "sample", "--arm", arm, "--set", "sel",
                                                    "--workers", X.W],
           [f"{DRAWS}/{arm}_s{s}_sel.pt" for s in SEEDS])
    X.step(f"{arm} VAL + SEL concat", [X.py, "scripts/n3_ours.py", "concat", "--arm", arm],
           [f"{DRAWS}/{arm}_s{s}_valsel.pt" for s in SEEDS])


def job_vc(X):
    for role in CLASSICAL_ROLES:
        X.need(cfg_path(role))
    X.need(f"{OLD}/selection_window.json")
    X.need(cfg_path("ivP"))
    win = X.load(f"{OLD}/selection_window.json")
    if win.get("scorer_config") != cfg_path("ivP"):
        die(f"selection_window.json names scorer config {win.get('scorer_config')}")
    ivp_cfg = X.load(cfg_path("ivP"))
    runs = ivp_runs(win)
    bundles = [f"{DRAWS}/ivP_{tag}_ep{e:02d}_s{s}_valsel.pt" for _, e, s, tag in runs]
    configured = {c["bundle"] for v in ivp_cfg["stage_b"]["seeds"].values() for c in v}
    configured |= {b for c in ivp_cfg["stage_a"]["cells"] for b in c["bundles"]}
    if set(bundles) != configured:
        die(f"iv-P sample bundles {sorted(bundles)} are not ivP.json's {sorted(configured)}")
    stage(f"iv-P: {len(runs)} VALSEL sampling runs (lr {win['lr']}, window {win['window']})")
    for (run, e, s, _), b in zip(runs, bundles):
        X.step(f"iv-P sample {run} epoch {e}", [X.py, "scripts/n3_old_symmc.py", "sample", "--run", run, "--epoch", e,
                                                "--set", "valsel"] + IVP_FLAGS + ["--workers", X.W], [b])
    X.match(bundles, "primary")
    X.score_existing("ivP", cfg_path("ivP"))
    X.step("iv-P freeze (select ivs)", [X.py, "scripts/n3_old_symmc.py", "select", "ivs", "--record", rec_path("ivP")],
           [f"{OLD}/ivP_frozen.json", cfg_path("ivS")])
    X.need(cfg_path("ivS"))
    ivs = X.load(cfg_path("ivS"))["stage_a"]["cells"][0]["bundles"]
    for b in ivs:
        src = os.path.dirname(b) + "/" + re.sub(r"^ivS_", "ivP_", os.path.basename(b))
        X.step(f"iv-S of {os.path.basename(src)}", [X.py, "scripts/n3_old_symmc.py", "ivs", "--bundle", src,
                                                    "--workers", X.W], [b])
    X.match(ivs, "primary")
    X.score_existing("ivS", cfg_path("ivS"))
    ours(X, "ours")
    for b in SEEDS:
        X.step(f"iii-u base {b}", [X.py, "scripts/n3_classical.py", "haar", "--set", "valsel", "--base", b,
                                   "--workers", X.W], [f"{DRAWS}/iii-u_s{b}_valsel.pt"])
    for b in SEEDS:
        X.step(f"iii-p base {b}", [X.py, "scripts/n3_classical.py", "press", "--set", "valsel", "--base", b,
                                   "--workers", X.W], [f"{DRAWS}/iii-p_s{b}_valsel.pt"])
    ref = f"{DRAWS}/p1ref_s-1_valsel.pt"
    X.step("p1ref: press(true R0)", [X.py, "scripts/n3_classical.py", "p1basin", "--set", "valsel", "--workers", X.W],
           [ref])
    for s in SEEDS:
        X.step(f"P1 press of OURS s{s}", [X.py, "scripts/n3_classical.py", "press", "--set", "valsel", "--draws",
                                          f"{DRAWS}/ours_s{s}_valsel.pt", "--workers", X.W],
               [f"{DRAWS}/p1-ours_s{s}_valsel.pt"])
    X.match([f"{DRAWS}/iii-u_s{b}_valsel.pt" for b in SEEDS], "primary")
    X.match([f"{DRAWS}/iii-p_s{b}_valsel.pt" for b in SEEDS], "primary")
    X.match([f"{DRAWS}/p1-ours_s{s}_valsel.pt" for s in SEEDS], "primary", reference=ref)
    X.match([f"{DRAWS}/iii-p_s{b}_valsel.pt" for b in SEEDS], "primary", reference=ref)
    for role in CLASSICAL_ROLES:
        X.score_existing(role, cfg_path(role))
    ours(X, "ours_n2")


def job_vs(X):
    X.need(f"{CLS}/anchor_valsel.json")                # §3.3: the H2 FORM before any iii-s minimisation
    out = f"{DRAWS}/iii-s_s-1_val.pt"
    if X.exists(out):
        print(f"  [have] {out}", flush=True)
        return
    X.step("iii-s VAL screen (resumes its shard)",
           [X.py, "scripts/n3_classical.py", "screen", "--set", "val", "--workers", X.W])
    X.step("iii-s VAL search (fixed FF* configuration, no --ff-* flags)",
           [X.py, "scripts/n3_classical.py", "search", "--set", "val", "--workers", X.G], [out])


JOBS = {"VR": job_vr, "VA": job_va, "VC": job_vc, "VS": job_vs}


def cmd_run(args, dry):
    py = args.py_symmc or os.environ.get("PY_SYMMC") or ("$PY_SYMMC" if dry else None)
    pym = args.py_mcf or os.environ.get("PY_MCF") or ("$PY_MCF" if dry else None)
    if py is None or pym is None:
        die("PY_SYMMC / PY_MCF unset (source .n3_box/env.sh, or pass --py-symmc / --py-mcf)")
    X = Runner(args.root, dry, str(args.workers), str(args.ff_workers), py, pym, args.gpu)
    stage(f"step 5b job {args.job}" + (" (plan: nothing is executed)" if dry else ""))
    try:
        JOBS[args.job](X)
    except Pending as e:
        print(f"  ... the rest of {args.job} is decided by {e.args[0]}, which does not exist yet", flush=True)
        return
    stage(f"job {args.job} complete")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("designate")
    d.add_argument("--check", action="store_true", help="exit 1 unless the committed designation matches the records")
    d.add_argument("--force", action="store_true")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--job", required=True, choices=sorted(JOBS))
        p.add_argument("--workers", type=int, default=28 if name == "run" else 0)
        p.add_argument("--ff-workers", type=int, default=1, help="FF* GPU worker processes (A3: 1, as timed)")
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--py-symmc", default=None)
        p.add_argument("--py-mcf", default=None)
    for p in sub.choices.values():
        p.add_argument("--root", default=REPO, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.cmd == "designate":
        cmd_designate(args)
    else:
        if args.cmd == "plan":
            args.workers = args.workers or "$W"
            args.ff_workers = "$G"
        cmd_run(args, args.cmd == "plan")


if __name__ == "__main__":
    main()
