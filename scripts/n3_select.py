"""N3 / G3 §7 step 6: assemble results/n3/n3_selection.json from the per-arm VALSEL selection records.

The file holds (protocol §7 step 6): every VALSEL cell score (the scorer's select records, whole), every frozen
choice, the H2 FORM (anchor_valsel.json), the MCF diagnostic-(a) statistics and verdict, the degrade record, the
complete list of permitted TEST-B / DEV-TEST runs with exact commands and seeds (§7 step 7) plus the step-9
scoring commands and report configs, and the SHA-256 of every input record. It refuses to write while any
required record is missing or inconsistent (--dry-run lists what is present, missing or invalid). Nothing here
matches anything or reads a truth; bundles are only hashed (and the chosen MCF bundles' meta read for their
checkpoints).

Inputs (role -> default path; --inputs <json> overrides any of them):
  degrade        results/n3/private/select/degrade.json   the §7 step-4 degrade decision (input, from the dated
                 amendment): {"items": [k, ...] (a prefix 1..k of the §7 degrade order; [] = none),
                 "amendment": "<the dated amendment that fixed it>"}
  anchor         results/n3/private/classical/anchor_valsel.json   H2 FORM (n3_classical.py anchor --set valsel)
  iii-u, iii-p   select/iiiu_record.json, iiip_record.json   (n3_score select; menu without RESID, MODE ties LJ)
  p1-ours, p1-iii-p   select/p1_ours_record.json, p1_iiip_record.json   (select with "reference" p1ref, P1 menu)
  mcfR_A, mcfR_B, mcfRG_A, mcfRG_B, iiS   select/<role>_record.json   (§3.2 ii-R, ii-R+G, ii-S)
  mcfA_1, mcfA_2, mcfA_3   select/<role>_record.json   (§3.2 ii-A stage 1 checkpoint, stage 2 knobs, selector;
                 mcfA_2 is not required when degrade item 4 is taken)
  ivP_window, ivP_frozen   results/n3/private/old/selection_window.json, ivP_frozen.json (n3_old_symmc.py)
  ivP, ivS       select/ivP_record.json, ivS_record.json
  diag_a:<label> results/n3/private/mcf/diag_a_<label>.json for every selected MCF checkpoint (mcfR_s<s>,
                 mcfRG_s<s> when +G selected another checkpoint, mcfA_s<s>) and the Thurlemann provenance repeat
                 (thurlemann). §3.2: "If (a) is unresolved at step 6, it is recorded FAILED" -- a missing file is
                 an error unless --diag-a-unresolved, which records it FAILED.
  gate_c         results/n3/gateC.json (§3.2 Gate C: C-a (i), C-b and C-c must have passed; a C-a (ii) failure
                 only sets the H1 harness clause, gate_ca_failed in the report configs)
  (OURS / OURS-N2 have no VALSEL knob: their checkpoints are hashed from n3_ours.arm_ckpts.)

    python scripts/n3_select.py --dry-run                 # what is present / missing / invalid
    python scripts/n3_select.py                           # validate everything and write results/n3/n3_selection.json
    python scripts/n3_select.py --inputs my_paths.json --out <path> [--allow-absent-bundles] [--diag-a-unresolved]
    python scripts/n3_select.py --write-configs           # select configs of iii-u, iii-p, P1 (OURS, iii-p)

Degrade items (§7, fixed at step 4) shape the run list: 2 -> MCF-A native stols {0.8, 1.0}; 3 -> MCF-A seed 0;
4 -> no MCF-A knob grid (9, 3); 5 -> no DEV-TEST run for any arm but the stored OURS / OURS-N2 draws; 6 -> i-E and
the steric BASIN P1 on seed 0 (OURS seeds 1-2 get their H2 BASIN candidate by a direct `basin` of the LJ pick);
7 -> no i-E (every OURS BASIN candidate by a direct `basin`). If diagnostic (a) is FAILED, (b)-(d) are not listed.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

SEL_DIR = os.path.join(C.PRIVATE, "select")
OLD_DIR = os.path.join(C.PRIVATE, "old")
CLS_DIR = os.path.join(C.PRIVATE, "classical")
MCF_DIR = os.path.join(C.PRIVATE, "mcf")
BASIN_DIR = os.path.join(C.PRIVATE, "basin")
OUT = os.path.join(C.REPO, "results", "n3", "n3_selection.json")
PROTOCOL = os.path.join(C.REPO, "tasks", "n3_protocol.md")

MENU = ["lj", "resid"] + [f"mode@{r}" for r in (0.25, 0.5, 0.75, 1.0, 1.5)] + ["random"]   # = n3_score.MENU
MENU_CLASSICAL = [s for s in MENU if s != "resid"]                                         # no RESID (§2.3)
P1_MENU = ["lj", "lj_pre", "random"]                                                       # = n3_score.P1_MENU
MCFA_MENU = ["random", "lj", "resid"]
MCFR_U = {1.0, 2.0, 3.0, 5.0, 10.0}
MCFA_KNOBS = {(f, u) for f in (5.0, 9.0, 13.0) for u in (1.0, 2.0, 3.0)}
SEEDS = (0, 1, 2)
SPLIT = {"testB": 1, "devtest": 2}
DEGRADE = {1: "drop the §8 tier-2 diagnostics", 2: "MCF-A native sweep reduced to {0.8, 1.0}",
           3: "MCF-A seed 0 only", 4: "MCF-A without its knob grid", 5: "drop the DEV-TEST rows for new arms",
           6: "i-E and the steric BASIN on seed 0 only", 7: "drop i-E (the H2 BASIN minimisations of the OURS LJ "
                                                           "picks still run)"}
PLACEHOLDERS = {"{W}": "CPU worker processes: scheduling only (every CPU job is worker-count invariant)",
                "{G}": "FF* GPU worker processes: scheduling only",
                "{GPU}": "GPU index of the box: scheduling only"}

MCF_RE = re.compile(r"^(mcfRG|mcfRS|mcfR|mcfA)_(.+?)_u([0-9.]+)(?:_f([0-9.]+))?_s(\d+)_valsel\.pt$")


def rel(p):
    return p if os.path.isabs(p) else os.path.join(C.REPO, p)


def rp(p):
    """Repo-relative '/' path (absolute when outside the repo)."""
    p = os.path.abspath(rel(p))
    try:
        r = os.path.relpath(p, C.REPO)
    except ValueError:
        return p.replace("\\", "/")
    return (p if r.startswith("..") else r).replace("\\", "/")


def g(x):
    return f"{float(x):g}"


def default_inputs():
    s = lambda name: os.path.join(SEL_DIR, name)
    return {"degrade": s("degrade.json"), "anchor": os.path.join(CLS_DIR, "anchor_valsel.json"),
            "iii-u": s("iiiu_record.json"), "iii-p": s("iiip_record.json"),
            "p1-ours": s("p1_ours_record.json"), "p1-iii-p": s("p1_iiip_record.json"),
            "mcfR_A": s("mcfR_A_record.json"), "mcfR_B": s("mcfR_B_record.json"),
            "mcfRG_A": s("mcfRG_A_record.json"), "mcfRG_B": s("mcfRG_B_record.json"), "iiS": s("iiS_record.json"),
            "mcfA_1": s("mcfA_1_record.json"), "mcfA_2": s("mcfA_2_record.json"), "mcfA_3": s("mcfA_3_record.json"),
            "ivP_window": os.path.join(OLD_DIR, "selection_window.json"), "ivP": s("ivP_record.json"),
            "ivP_frozen": os.path.join(OLD_DIR, "ivP_frozen.json"), "ivS": s("ivS_record.json"),
            "gate_c": os.path.join(C.REPO, "results", "n3", "gateC.json")}


class Problems:
    def __init__(self):
        self.rows = []

    def err(self, role, msg):
        self.rows.append((role, msg))

    def of(self, role):
        return [m for r, m in self.rows if r == role]


# ------------------------------------------------------------------------------------------------ records
class Inputs:
    """Input records by role: existence, JSON, SHA-256."""

    def __init__(self, paths, P):
        self.paths, self.P, self.data, self.sha, self.status = dict(paths), P, {}, {}, {}

    def get(self, role, required=True):
        if role in self.data:
            return self.data[role]
        path = self.paths.get(role)
        if path is None or not os.path.exists(rel(path)):
            self.status[role] = "missing" if required else "absent (optional)"
            if required:
                self.P.err(role, f"missing: {rp(path) if path else '(no path)'}")
            return None
        try:
            d = json.load(open(rel(path)))
        except ValueError as e:
            self.status[role] = "invalid"
            self.P.err(role, f"not JSON: {e}")
            return None
        self.data[role], self.sha[role], self.status[role] = d, C.sha256(rel(path)), "present"
        return d


def check_bundles(P, role, rec, allow_absent):
    """Every bundle a select record scored is a VALSEL bundle and unchanged since (SHA-256 of the record)."""
    for p, h in (rec.get("bundles_sha256") or {}).items():
        if not os.path.basename(p).endswith("_valsel.pt"):
            P.err(role, f"{p}: not a VALSEL bundle (selection is on VALSEL only, §1.1)")
        if not os.path.exists(rel(p)):
            if not allow_absent:
                P.err(role, f"{p}: scored bundle absent (pass --allow-absent-bundles to record it unverified)")
        elif C.sha256(rel(p)) != h:
            P.err(role, f"{p}: changed since it was scored")


def is_record(P, role, rec, stages):
    if not isinstance(rec, dict) or not all(k in rec for k in ("name", "matcher", "config", "bundles_sha256")):
        P.err(role, "not an n3_score.py select record")
        return False
    for st in stages:
        if st not in rec:
            P.err(role, f"record has no {st}")
            return False
    if rec.get("matcher") != "primary":
        P.err(role, f"matcher {rec.get('matcher')} (every VALSEL selection is under the primary matcher)")
    return True


def cells_of(rec):
    return rec["config"]["stage_a"]["cells"]


def chosen_a(rec):
    return rec["stage_a"]["chosen"]


def parse_mcf(p):
    m = MCF_RE.match(os.path.basename(p))
    if m is None:
        return None
    return {"arm": m[1], "label": m[2], "u": float(m[3]), "f": float(m[4]) if m[4] else None, "seed": int(m[5])}


def bundle_meta(P, role, path):
    if not os.path.exists(rel(path)):
        P.err(role, f"{path}: the chosen bundle is needed for its checkpoint (meta.run) but is absent")
        return None
    return C.load_bundle(rel(path))["meta"]


def seeded_cell(P, role, cell, prefix, n_seeds=3):
    """The bundles of one cell: one per seed 0..n-1, all of arm `prefix`, one (label?, u, f) -> list of parses."""
    ps = [parse_mcf(b) for b in cell["bundles"]]
    if any(x is None or x["arm"] != prefix for x in ps):
        P.err(role, f"cell {cell['label']}: bundles are not {prefix}_<ckpt>_u<u>[_f<f>]_s<s>_valsel.pt")
        return None
    if [x["seed"] for x in ps] != list(range(n_seeds)):
        P.err(role, f"cell {cell['label']}: seeds {[x['seed'] for x in ps]} != {list(range(n_seeds))}")
        return None
    if len({(x["u"], x["f"]) for x in ps}) != 1:
        P.err(role, f"cell {cell['label']}: mixed sampler knobs")
        return None
    return ps


# ------------------------------------------------------------------------------------------------ arms
def single_cell(P, I, role, prefix_re, menu, mtb=None, ref_arm=None, allow_absent=False, n_seeds=3):
    """Stage-A selector choice on one cell of seed bundles (iii-u, iii-p, P1, iv-S, ii-S)."""
    rec = I.get(role)
    if rec is None or not is_record(P, role, rec, ["stage_a"]):
        return None
    check_bundles(P, role, rec, allow_absent)
    cells = cells_of(rec)
    if len(cells) != 1:
        P.err(role, f"{len(cells)} cells; one cell of seed bundles expected")
        return None
    names = [os.path.basename(b) for b in cells[0]["bundles"]]
    if len(names) != n_seeds or not all(re.match(prefix_re.format(s=s), n) for s, n in zip(range(n_seeds), names)):
        P.err(role, f"bundles {names} are not {prefix_re} for seeds 0..{n_seeds - 1}")
    if rec.get("menu") != menu:
        P.err(role, f"menu {rec.get('menu')} != {menu}")
    if mtb is not None and rec["config"].get("mode_tiebreak") != mtb:
        P.err(role, f"mode_tiebreak {rec['config'].get('mode_tiebreak')} != {mtb}")
    tie = rec["stage_a"].get("tie") or []
    if not tie or tie[0] != "hits" or tie[-1] != "selector":
        P.err(role, f"tie {tie}: rank by seed-summed hits, §2.3 selector preference last")
    ref = rec.get("reference")
    if ref_arm is not None and (not ref or ref.get("arm") != ref_arm
                                or os.path.basename(ref.get("path", "")) != f"{ref_arm}_s-1_valsel.pt"):
        P.err(role, f"reference {ref} is not the VALSEL {ref_arm} bundle")
    if ref_arm is None and ref:
        P.err(role, f"unexpected reference {ref}")
    sel = chosen_a(rec)["selector"]
    if sel not in menu:
        P.err(role, f"chosen selector {sel} not in the menu")
    return {"selector": sel, "decided_by": rec["stage_a"].get("decided_by"), "bundles": cells[0]["bundles"]}


def mcf_r(P, I, pre, allow_absent, same_cells_as=None):
    """MCF-R (pre 'mcfR') or MCF-R+G (pre 'mcfRG'): Stage A s_uR x menu pooled over seeds, Stage B checkpoint per
    seed at that s_uR (§3.2)."""
    ra, rb = I.get(f"{pre}_A"), I.get(f"{pre}_B")
    out = {}
    if ra is not None and is_record(P, f"{pre}_A", ra, ["stage_a"]):
        role = f"{pre}_A"
        check_bundles(P, role, ra, allow_absent)
        cells, us, keys = cells_of(ra), {}, set()
        for c in cells:
            ps = seeded_cell(P, role, c, pre)
            if ps:
                us[c["label"]] = ps[0]["u"]
                keys |= {(x["label"], x["u"], x["seed"]) for x in ps}
        if set(us.values()) != MCFR_U or len(us) != len(cells):
            P.err(role, f"s_uR cells {sorted(set(us.values()))} != {sorted(MCFR_U)} (one cell each)")
        if ra.get("menu") != MENU:
            P.err(role, f"menu {ra.get('menu')} != the §2.3 menu")
        if ra["stage_a"].get("tie") != ["hits", "draw1", "any16", "prefer_cell", "selector"]:
            P.err(role, f"tie {ra['stage_a'].get('tie')} != [hits, draw1, any16, prefer_cell, selector]")
        pc = ra["config"]["stage_a"].get("prefer_cell")
        if us.get(pc) != 3.0:
            P.err(role, f"prefer_cell {pc} is not the s_uR = 3 cell")
        if same_cells_as is not None and keys != same_cells_as:
            P.err(role, "+G Stage A must use the gauge-scanned draws of MCF-R's own Stage-A cells")
        ch = chosen_a(ra)
        out.update(s_uR=us.get(ch["cell"]), selector=ch["selector"], stage_a_decided_by=ra["stage_a"].get("decided_by"),
                   _stage_a_keys=keys)
        if ch["selector"] not in MENU:
            P.err(role, f"chosen selector {ch['selector']} not in the menu")
    if rb is not None and is_record(P, f"{pre}_B", rb, ["stage_b"]) and "s_uR" in out:
        role = f"{pre}_B"
        check_bundles(P, role, rb, allow_absent)
        B = rb["stage_b"]
        if B.get("selector") != out["selector"]:
            P.err(role, f"Stage B selector {B.get('selector')} != Stage A's {out['selector']}")
        if B.get("tie") != ["picked", "draw1", "any16", "val_loss"]:
            P.err(role, f"tie {B.get('tie')} != [picked, draw1, any16, val_loss]")
        seeds = rb["config"]["stage_b"]["seeds"]
        if sorted(seeds) != ["0", "1", "2"]:
            P.err(role, f"seeds {sorted(seeds)} != 0, 1, 2")
        ck = {}
        for s, cands in sorted(seeds.items()):
            if not 1 <= len(cands) <= 4:
                P.err(role, f"seed {s}: {len(cands)} candidates (top-3 valid/loss + last)")
            for c in cands:
                x = parse_mcf(c["bundle"])
                if x is None or x["arm"] != pre or x["seed"] != int(s) or x["u"] != out["s_uR"]:
                    P.err(role, f"seed {s}: {c['bundle']} is not a {pre} seed-{s} bundle at s_uR {g(out['s_uR'])}")
                if not isinstance(c.get("val_loss"), (int, float)) or not math.isfinite(c["val_loss"]):
                    P.err(role, f"seed {s}: {c['label']} has no finite fixed-draw VAL loss")
            lab = B["seeds"].get(s, {}).get("chosen")
            bpath = next((c["bundle"] for c in cands if c["label"] == lab), None)
            if bpath is None:
                P.err(role, f"seed {s}: chosen {lab} is not a candidate")
                continue
            m = bundle_meta(P, role, bpath)
            if m is None:
                continue
            run = m.get("run") or {}
            if int(run.get("train_seed", -1)) != int(s) or float(run.get("s_uR", -1)) != out["s_uR"]:
                P.err(role, f"seed {s}: {bpath} meta.run does not match (train_seed {run.get('train_seed')}, "
                            f"s_uR {run.get('s_uR')})")
            ck[s] = {"ckpt": rp(run.get("ckpt", "")), "ckpt_sha256": run.get("ckpt_sha256"),
                     "ckpt_label": run.get("ckpt_label", parse_mcf(bpath)["label"]), "valsel_bundle": bpath,
                     "decided_by": B["seeds"][s].get("decided_by")}
        out["checkpoints"] = ck
    return out


def mcf_a(P, I, degrade, allow_absent):
    """MCF-A (§3.2 ii-A; A2 item 11): stage 1 checkpoint per seed at (s_uF 9, s_uR 3) by any-of-10 at stol 1.0 ->
    stol 0.8 -> valid/loss; stage 2 (s_uF, s_uR) grid pooled over seeds, ties -> (9, 3); stage 3 selector from
    {RANDOM, LJ, RESID} by picked@1 at stol 1.0. Degrade 3: seed 0 only; degrade 4: no grid, (9, 3)."""
    seeds = [0] if 3 in degrade else list(SEEDS)
    out = {"seeds": seeds, "native_stols": [0.8, 1.0] if 2 in degrade else [0.5, 0.8, 1.0]}
    r1 = I.get("mcfA_1")
    ck = {}
    if r1 is not None and is_record(P, "mcfA_1", r1, ["stage_b"]):
        check_bundles(P, "mcfA_1", r1, allow_absent)
        B = r1["stage_b"]
        if B.get("selector") is not None:
            P.err("mcfA_1", "stage 1 ranks checkpoints without a selector")
        if B.get("tie") != ["any10@mcf10", "any10@mcf08", "val_loss"]:
            P.err("mcfA_1", f"tie {B.get('tie')} != [any10@mcf10, any10@mcf08, val_loss]")
        sd = r1["config"]["stage_b"]["seeds"]
        if sorted(sd) != [str(s) for s in seeds]:
            P.err("mcfA_1", f"seeds {sorted(sd)} != {seeds}")
        for s, cands in sorted(sd.items()):
            for c in cands:
                x = parse_mcf(c["bundle"])
                if x is None or x["arm"] != "mcfA" or x["seed"] != int(s) or (x["f"], x["u"]) != (9.0, 3.0):
                    P.err("mcfA_1", f"seed {s}: {c['bundle']} is not an MCF-A seed-{s} bundle at (s_uF 9, s_uR 3)")
            lab = B["seeds"].get(s, {}).get("chosen")
            bpath = next((c["bundle"] for c in cands if c["label"] == lab), None)
            if bpath is None:
                P.err("mcfA_1", f"seed {s}: chosen {lab} is not a candidate")
                continue
            m = bundle_meta(P, "mcfA_1", bpath)
            if m is not None:
                run = m.get("run") or {}
                ck[s] = {"ckpt": rp(run.get("ckpt", "")), "ckpt_sha256": run.get("ckpt_sha256"),
                         "ckpt_label": run.get("ckpt_label", parse_mcf(bpath)["label"]), "valsel_bundle": bpath,
                         "decided_by": B["seeds"][s].get("decided_by")}
    out["checkpoints"] = ck
    labels = {int(s): v["ckpt_label"] for s, v in ck.items()}
    knobs = (9.0, 3.0)
    if 4 in degrade:
        out["stage2"] = "not run (degrade item 4): (s_uF, s_uR) = (9, 3)"
        if I.paths.get("mcfA_2") and os.path.isfile(rel(I.paths["mcfA_2"])):
            P.err("mcfA_2", "degrade item 4 was taken: an MCF-A knob-grid record must not exist")
    else:
        r2 = I.get("mcfA_2")
        if r2 is not None and is_record(P, "mcfA_2", r2, ["stage_a"]):
            check_bundles(P, "mcfA_2", r2, allow_absent)
            if r2.get("menu"):
                P.err("mcfA_2", "stage 2 ranks knob cells without a selector (menu null)")
            if r2["stage_a"].get("tie") != ["any10@mcf10", "prefer_cell"]:
                P.err("mcfA_2", f"tie {r2['stage_a'].get('tie')} != [any10@mcf10, prefer_cell]")
            cellk = {}
            for c in cells_of(r2):
                ps = seeded_cell(P, "mcfA_2", c, "mcfA", len(seeds))
                if ps:
                    cellk[c["label"]] = (ps[0]["f"], ps[0]["u"])
                    if labels and [x["label"] for x in ps] != [labels.get(s) for s in seeds]:
                        P.err("mcfA_2", f"cell {c['label']}: not the stage-1 checkpoints {labels}")
            if set(cellk.values()) != MCFA_KNOBS or len(cellk) != 9:
                P.err("mcfA_2", f"knob cells {sorted(cellk.values())} != the 3 x 3 (s_uF, s_uR) grid")
            if cellk.get(r2["config"]["stage_a"].get("prefer_cell")) != (9.0, 3.0):
                P.err("mcfA_2", "prefer_cell is not (9, 3)")
            knobs = cellk.get(chosen_a(r2)["cell"], knobs)
            out["stage2_decided_by"] = r2["stage_a"].get("decided_by")
    out["s_uF"], out["s_uR"] = knobs
    r3 = I.get("mcfA_3")
    if r3 is not None and is_record(P, "mcfA_3", r3, ["stage_a"]):
        check_bundles(P, "mcfA_3", r3, allow_absent)
        cells = cells_of(r3)
        ps = seeded_cell(P, "mcfA_3", cells[0], "mcfA", len(seeds)) if len(cells) == 1 else None
        if len(cells) != 1:
            P.err("mcfA_3", "stage 3 runs on the one chosen cell")
        elif ps and ((ps[0]["f"], ps[0]["u"]) != knobs or (labels and [x["label"] for x in ps] != [labels.get(s) for s in seeds])):
            P.err("mcfA_3", f"cell is not the chosen knobs {knobs} at the stage-1 checkpoints")
        if r3.get("menu") != MCFA_MENU:
            P.err("mcfA_3", f"menu {r3.get('menu')} != {MCFA_MENU}")
        tie = r3["stage_a"].get("tie") or []
        if not tie or tie[0] != "picked@mcf10" or tie[-1] != "selector":
            P.err("mcfA_3", f"tie {tie}: picked@1 at stol 1.0 (picked@mcf10), selector preference last")
        out["selector"] = chosen_a(r3)["selector"]
        out["stage3_decided_by"] = r3["stage_a"].get("decided_by")
    return out


def iv_p(P, I, allow_absent):
    """iv-P (§3.4): window / lr (selection_window.json), Stage A selector + Stage B epochs (ivP_record.json),
    frozen by `n3_old_symmc.py select ivs` (ivP_frozen.json, with the step-7 runs); iv-S selector."""
    win, rec, fz = I.get("ivP_window"), I.get("ivP"), I.get("ivP_frozen")
    if win is None or rec is None or fz is None:
        return None
    if not is_record(P, "ivP", rec, ["stage_a", "stage_b"]):
        return None
    check_bundles(P, "ivP", rec, allow_absent)
    if rec["config"].get("name") != "ivP":
        P.err("ivP", "record is not the iv-P selection (config name ivP)")
    if float(fz.get("lr", -1)) != float(win.get("lr", -2)) or fz.get("window") != win.get("window"):
        P.err("ivP_frozen", f"lr / window {fz.get('lr')} / {fz.get('window')} != selection_window "
                            f"{win.get('lr')} / {win.get('window')}")
    if fz.get("selector") != chosen_a(rec)["selector"]:
        P.err("ivP_frozen", f"selector {fz.get('selector')} != the record's Stage A {chosen_a(rec)['selector']}")
    ep = {s: int(v["chosen"][2:]) for s, v in rec["stage_b"]["seeds"].items()}
    if {str(k): v for k, v in (fz.get("epoch_per_seed") or {}).items()} != {k: v for k, v in ep.items()}:
        P.err("ivP_frozen", f"epochs {fz.get('epoch_per_seed')} != the record's Stage B {ep}")
    for s in ("0", "1", "2"):
        c = (win.get("candidates") or {}).get(s)
        if c is None or ep.get(s) not in (c.get("best"), c.get("last")):
            P.err("ivP", f"seed {s}: epoch {ep.get(s)} is not a window candidate {c}")
    if sorted((fz.get("checkpoint_sha256") or {})) != ["0", "1", "2"] or not fz.get("step7_runs"):
        P.err("ivP_frozen", "needs the checkpoint SHA-256 of seeds 0-2 and the step-7 run list")
    return {"lr": fz.get("lr"), "window": fz.get("window"), "late_extension": win.get("late_extension"),
            "selector": fz.get("selector"), "epoch_per_seed": fz.get("epoch_per_seed"),
            "checkpoints": fz.get("checkpoints"), "checkpoint_sha256": fz.get("checkpoint_sha256"),
            "sampling": fz.get("sampling"), "train_config": fz.get("train_config"),
            "stage_a_decided_by": rec["stage_a"].get("decided_by"),
            "stage_b_decided_by": {s: v.get("decided_by") for s, v in rec["stage_b"]["seeds"].items()},
            "_step7_runs": fz.get("step7_runs")}


def anchor(P, I):
    d = I.get("anchor")
    if d is None:
        return None
    s = d.get("summary") or {}
    if s.get("set") != "valsel" or s.get("n") != 400 or s.get("partial"):
        P.err("anchor", f"not a full VALSEL anchor (set {s.get('set')}, n {s.get('n')}, partial {s.get('partial')})")
    form = s.get("form")
    if form not in ("EXACT", "BASIN") or form != ("EXACT" if (s.get("anchoring") or 0) >= 0.5 else "BASIN"):
        P.err("anchor", f"FORM {form} does not follow the 50% rule at anchoring {s.get('anchoring')}")
    b = C.bundle_path("fftruth", -1, "valsel")
    if os.path.exists(b) and C.sha256(b) != s.get("fftruth_sha256"):
        P.err("anchor", f"{rp(b)} changed since the anchor was computed")
    keep = ("anchoring", "anchoring_val_only", "form", "form_rule", "labels", "no_minimum", "match_det_fail",
            "median_drift_deg", "fftruth", "fftruth_sha256", "match", "ffident", "fairchem_core")
    return {k: s.get(k) for k in keep}


def ours_frozen(P):
    import n3_ours as O
    out = {}
    for arm, name in (("ours", "OURS"), ("ours_n2", "OURS-N2")):
        try:
            ck = O.arm_ckpts(arm)
        except Exception as e:  # noqa: BLE001
            P.err(name, f"checkpoints: {e}")
            continue
        miss = [p for p in ck if not os.path.exists(p)]
        if miss:
            P.err(name, f"missing checkpoints {miss}")
            continue
        dev = {s: C.bundle_path(arm, s, "devtest") for s in SEEDS}
        out[name] = {"frozen": "protocol §3.1 (no VALSEL knob)", "selector": "lj", "S": O.S_DRAWS, "euler_steps": O.STEPS,
                     "checkpoints": {str(s): {"ckpt": rp(p), "sha256": C.sha256(p)} for s, p in zip(SEEDS, ck)},
                     "seeds": {"testB": f"per crystal i: {O.SEED_BASE['testB']} + i, set once; checkpoints sampled in "
                                        "seed order from that stream", "sel": f"{O.SEED_BASE['sel']} + i (i within SEL)",
                               "devtest": f"stored {'N1' if arm == 'ours' else 'N2'} draws (g2_rank base 50000 + i)"},
                     "devtest_bundles": {str(s): {"bundle": rp(p), "sha256": C.sha256(p) if os.path.exists(p) else None}
                                         for s, p in dev.items()}}
    return out


def diag_a(P, I, frozen, degrade, unresolved):
    """§3.2 diagnostic (a) on every selected MCF-R (and +G, when it selected another checkpoint) and MCF-A
    checkpoint, plus the Thurlemann provenance repeat. Unresolved at step 6 -> recorded FAILED."""
    want = {}
    for s, c in (frozen.get("MCF-R", {}).get("checkpoints") or {}).items():
        want[f"mcfR_s{s}"] = (c["ckpt_sha256"], "R", frozen["MCF-R"].get("s_uR"), None)
    for s, c in (frozen.get("MCF-R+G", {}).get("checkpoints") or {}).items():
        r = (frozen.get("MCF-R", {}).get("checkpoints") or {}).get(s)
        if r is None or r["ckpt_sha256"] != c["ckpt_sha256"]:
            want[f"mcfRG_s{s}"] = (c["ckpt_sha256"], "R", frozen["MCF-R+G"].get("s_uR"), None)
    A = frozen.get("MCF-A", {})
    for s, c in (A.get("checkpoints") or {}).items():
        want[f"mcfA_s{s}"] = (c["ckpt_sha256"], "A", A.get("s_uR"), A.get("s_uF"))
    out, verdicts, approx = {}, {}, []
    for lab in list(want) + ["thurlemann"]:
        role = f"diag_a:{lab}"
        I.paths.setdefault(role, os.path.join(MCF_DIR, f"diag_a_{lab}.json"))
        d = I.get(role, required=not unresolved)
        if d is None:
            out[lab] = {"verdict": "FAILED", "reason": "unresolved at §7 step 6 (§3.2: recorded FAILED, never re-run)"}
            verdicts[lab] = "FAILED"
            continue
        if d.get("diagnostic") != "a (equivariance)" or not d.get("reportable", False):
            P.err(role, "not a reportable (un-shimmed) diagnostic-(a) record")
        if lab == "thurlemann":
            if not d.get("provenance_repeat"):
                P.err(role, "the Thurlemann repeat must be a --provenance run")
        else:
            sha, arm, u, f = want[lab]
            if d.get("ckpt_sha256") != sha:
                P.err(role, f"checkpoint {d.get('ckpt_sha256')} is not the selected one {sha}")
            if d.get("arm") != arm or float((d.get("knobs") or {}).get("s_uR", -1)) != float(u if u is not None else -2):
                P.err(role, f"arm / s_uR {d.get('arm')} / {(d.get('knobs') or {}).get('s_uR')} != {arm} / {u}")
            if f is not None and float((d.get("knobs") or {}).get("s_uF", -1)) != float(f):
                P.err(role, f"s_uF {(d.get('knobs') or {}).get('s_uF')} != {f}")
            thr = d.get("sampler_deviation_threshold_1e-3") or {}
            if thr.get("float32_native") is False:
                approx.append(lab)
        v = d.get("verdict")
        verdicts[lab] = "PASS" if v == "PASS" else "FAILED"
        out[lab] = {"verdict_recorded": v, "verdict": verdicts[lab], "ckpt_sha256": d.get("ckpt_sha256"),
                    "statistics": d.get("statistics"), "requirements_met": d.get("requirements_met"),
                    "tolerances": d.get("tolerances"), "sampler_deviation": d.get("sampler_deviation"),
                    "sampler_deviation_threshold_1e-3": d.get("sampler_deviation_threshold_1e-3"),
                    "torch_threads": d.get("torch_threads"), "knobs": d.get("knobs"),
                    "controls_min": (d.get("states") or {}).get("generic", {}).get("controls_min"),
                    "corrupt_batch_state": {k: (d.get("states") or {}).get("corrupt_batch", {}).get(k)
                                            for k in ("verdict", "statistics", "branch_cut_edges")}}
    sel = [lab for lab in want]
    overall = "PASS" if sel and all(verdicts[lab] == "PASS" for lab in sel) else "FAILED"
    return {"per_checkpoint": out, "verdict": overall,
            "rule": "PASS iff (a) PASSES on every selected MCF-R / +G / MCF-A checkpoint; HARNESS_CHECK, FAIL or "
                    "unresolved at step 6 = FAILED (§3.2); the Thurlemann repeat is provenance, not gated",
            "consequence": None if overall == "PASS" else
            "invariance statement withdrawn; (b)-(d) not reported; Table 2 omits the ceiling column; the §4 z_MCF "
            "bullet drops its invariance clause and its (b) citation",
            "approximate": bool(approx), "approximate_checkpoints": approx,
            "approximate_rule": "A2 item 3: (b)-(d) labelled 'approximate' if a selected checkpoint's 50-step "
                                "sampler deviation (float32 native) exceeds 1e-3"}


def degrade_record(P, I):
    d = I.get("degrade")
    if d is None:
        return None, set()
    items = d.get("items")
    if not isinstance(items, list) or not all(isinstance(k, int) and k in DEGRADE for k in items):
        P.err("degrade", f"items {items}: a list of degrade item numbers 1-7")
        return d, set()
    if sorted(items) != list(range(1, len(items) + 1)):
        P.err("degrade", f"items {items}: the §7 order takes the fewest items in order, i.e. a prefix 1..k")
    if not str(d.get("amendment", "")).strip():
        P.err("degrade", "names no dated amendment (§7 step 4: committed and pushed before step 5)")
    return {**d, "taken": {str(k): DEGRADE[k] for k in sorted(items)}}, set(items)


# ------------------------------------------------------------------------------------------------ run list
def _dp(arm, seed, set_name):
    return rp(C.bundle_path(arm, seed, set_name))


def build_runs(F, degrade, a_pass):
    """§7 step 7 run list (box / TEST-B and DEV-TEST inputs; each with its seeds) and the step-9 local runs."""
    step7, step9 = [], []
    sets = ["testB"] + ([] if 5 in degrade else ["devtest"])
    ie_seeds = [] if 7 in degrade else ([0] if 6 in degrade else list(SEEDS))
    p1_seeds = [0] if 6 in degrade else list(SEEDS)
    PY_C = "python scripts/n3_classical.py"

    def add(lst, rid, arm, st, cmds, seeds, outputs, where):
        lst.append({"id": rid, "arm": arm, "set": st, "where": where, "commands": cmds, "seeds": seeds,
                    "outputs": outputs})
    for st in ["testB", "devtest"]:
        sp = SPLIT[st]
        for arm, name in (("ours", "OURS"), ("ours_n2", "OURS-N2")):
            if st == "testB":
                add(step7, f"{arm}/testB", name, st, [f"python scripts/n3_ours.py sample --arm {arm} --set testB --workers {{W}}"],
                    "per crystal i (position in testB_refcodes.txt): torch seed 80000 + i, set once; the arm's three "
                    "checkpoints sampled in seed order from that stream", [_dp(arm, s, st) for s in SEEDS], "box, step 7")
            else:
                add(step7, f"{arm}/devtest", name, st, [f"python scripts/n3_ours.py convert --arm {arm} --set devtest"],
                    f"stored {'N1' if arm == 'ours' else 'N2'} DEV-TEST draws (g2_rank base 50000 + i); no new sampling",
                    [_dp(arm, s, st) for s in SEEDS], "local conversion (done); the bundles are uploaded at step 7")
        if st not in sets:
            continue
        for b in SEEDS:
            add(step7, f"iii-u/{st}/b{b}", "iii-u", st,
                [f"{PY_C} haar --set {st} --base {b} --workers {{W}}", f"{PY_C} press --set {st} --base {b} --workers {{W}}"],
                f"torch.Generator().manual_seed({600000 + 10000 * b + 1000 * sp} + i); press deterministic",
                [_dp("iii-u", b, st), _dp("iii-p", b, st)], "box or CPU box, step 7")
        add(step7, f"iii-s/{st}", "iii-s", st, [f"{PY_C} screen --set {st} --workers {{W}}",
                                                f"{PY_C} search --set {st} --workers {{G}}"],
            "deterministic (grid pre-rotation torch.manual_seed(20260927)); 64 lowest minimised, <= 100 evaluations",
            [_dp("iii-s", -1, st)], "box, step 7 (FF*)")
        add(step7, f"fftruth/{st}", "fftruth", st, [f"{PY_C} truths --set {st} --workers {{G}}"],
            "deterministic; <= 1,000 evaluations", [_dp("fftruth", -1, st)], "box, step 7 (FF*)")
        for s in SEEDS:
            cmds, outs = [], []
            if s in ie_seeds:
                cmds += [f"{PY_C} ie --draws {_dp('ours', s, st)} --workers {{G}}",
                         f"{PY_C} basin --draws {_dp('i-E-ours', s, st)} --from-ie"]
                outs += [_dp("i-E-ours", s, st), _dp("basin-ours", s, st)]
            else:                                     # §7 degrade 6/7: the H2 BASIN minimisation still runs
                cmds += [f"{PY_C} basin --draws {_dp('ours', s, st)} --workers {{G}}"]
                outs += [_dp("basin-ours", s, st)]
            add(step7, f"i-E+basin-ours/{st}/s{s}", "i-E / basin-ours", st, cmds,
                "deterministic (FF*); BASIN candidate = OURS' LJ pick", outs, "box, step 7 (FF*)")
            add(step7, f"basin-ours_n2/{st}/s{s}", "basin-ours_n2", st,
                [f"{PY_C} basin --draws {_dp('ours_n2', s, st)} --workers {{G}}"],
                "deterministic (FF*); the OURS-N2 LJ pick", [_dp("basin-ours_n2", s, st)], "box, step 7 (FF*)")
        add(step7, f"p1ref/{st}", "p1ref", st, [f"{PY_C} p1basin --set {st} --workers {{W}}"], "deterministic",
            [_dp("p1ref", -1, st)], "box or CPU box, step 7")
        for s in p1_seeds:
            add(step7, f"p1-ours/{st}/s{s}", "p1-ours", st,
                [f"{PY_C} press --draws {_dp('ours', s, st)} --arm p1-ours --workers {{W}}"], "deterministic",
                [_dp("p1-ours", s, st)], "box or CPU box, step 7")
        R, G, A = F.get("MCF-R", {}), F.get("MCF-R+G", {}), F.get("MCF-A", {})
        for s in SEEDS:
            c = (R.get("checkpoints") or {}).get(str(s))
            if not c:
                continue
            u = g(R["s_uR"])
            rdir = f"n3_mcf/infer/mcfR/{st}/s{s}_{c['ckpt_label']}_u{u}"
            seed = 100000 + 1000 * s + sp
            add(step7, f"MCF-R/{st}/s{s}", "MCF-R", st,
                [f"python scripts/n3_mcf_run.py infer --ckpt {c['ckpt']} --set {st} --s_uR {u} --run --gpu {{GPU}}",
                 f"python scripts/n3_mcf_run.py to_bundle --run_dir {rdir} --workers {{W}}"],
                f"inference.seed = {seed} (100000 + 1000 s + split)", [rdir, _dp(f"mcfR_{c['ckpt_label']}_u{u}", s, st)],
                "box, step 7 (+ RESID pass and check inside infer --run)")
            cg = (G.get("checkpoints") or {}).get(str(s))
            if not cg:
                continue
            ug = g(G["s_uR"])
            gdir = f"n3_mcf/infer/mcfR/{st}/s{s}_{cg['ckpt_label']}_u{ug}"
            cmds = []
            if (G.get("separate_sampling") or {}).get(str(s)):
                cmds += [f"python scripts/n3_mcf_run.py infer --ckpt {cg['ckpt']} --set {st} --s_uR {ug} --run --gpu {{GPU}}",
                         f"python scripts/n3_mcf_run.py to_bundle --run_dir {gdir} --workers {{W}}"]
            cmds.append(f"python scripts/n3_mcf_gauge.py --run_dir {gdir} --workers {{W}}")
            gb = _dp(f"mcfRG_{cg['ckpt_label']}_u{ug}", s, st)
            add(step7, f"MCF-R+G/{st}/s{s}", "MCF-R+G", st, cmds,
                f"inference.seed = {seed} when separately sampled; the gauge scan is deterministic", [gb],
                "box, step 7 (sampling); the gauge scan on CPU (box or local after step 8)")
            add(step7, f"ii-S/{st}/s{s}", "ii-S", st, [f"python scripts/n3_mcf_sg.py --bundle {gb} --workers {{W}}"],
                "deterministic", [_dp(f"mcfRS_{cg['ckpt_label']}_u{ug}", s, st)], "CPU (box or local after step 8)")
            if a_pass:
                add(step9, f"diag-c/{st}/s{s}", "MCF diagnostic (c)", st,
                    [f"python scripts/n3_mcf_diag.py c --run_dir {rdir} --workers {{W}}"], "deterministic (oracle-aligned)",
                    [_dp(f"mcfRoal_{c['ckpt_label']}_u{u}", s, st)], "local, step 9")
        for s in A.get("seeds") or []:
            c = (A.get("checkpoints") or {}).get(str(s))
            if not c:
                continue
            u, f = g(A["s_uR"]), g(A["s_uF"])
            adir = f"n3_mcf/infer/mcfA/{st}/s{s}_{c['ckpt_label']}_u{u}_f{f}"
            fdir = f"n3_mcf/infer/mcfA_floor/{st}/s{s}"
            add(step7, f"MCF-A/{st}/s{s}", "MCF-A (context)", st,
                [f"python scripts/n3_mcf_run.py infer --ckpt {c['ckpt']} --set {st} --s_uR {u} --s_uF {f} --run --gpu {{GPU}}",
                 f"python scripts/n3_mcf_run.py to_bundle --run_dir {adir} --workers {{W}}"],
                f"inference.seed = {200000 + 1000 * s + sp} (200000 + 1000 s + split)",
                [adir, _dp(f"mcfA_{c['ckpt_label']}_u{u}_f{f}", s, st)], "box, step 7")
            add(step7, f"MCF-A-floor/{st}/s{s}", "MCF-A prior floor", st,
                [f"python scripts/n3_mcf_run.py prior_floor --seed {s} --set {st}",
                 f"python scripts/n3_mcf_run.py to_bundle --run_dir {fdir} --workers {{W}}"],
                f"torch.manual_seed / np.random.seed({300000 + 1000 * s + sp})", [fdir, _dp("mcfAfloor", s, st)],
                "box or CPU box, step 7")
        iv = F.get("iv-P") or {}
        for cmd in iv.get("_step7_runs") or []:
            if f"--set {st}" in cmd or f"_{st}.pt" in cmd:
                is_s = " ivs " in f" {cmd} "
                s = re.search(r"_s(\d)_" if is_s else r"_s(\d)\b", cmd)
                sd = int(s.group(1)) if s else None
                add(step7, f"{'iv-S' if is_s else 'iv-P'}/{st}/s{sd}", "iv-S" if is_s else "iv-P", st, [cmd],
                    "iv-S: deterministic (identity-copy expansion)" if is_s else
                    f"torch.manual_seed({700000 + 10000 * (sd or 0) + 1000 * sp} + i) (700000 + 10000 s + 1000 split + i)",
                    [_dp("ivS" if is_s else "ivP", sd, st)], "box, step 7" if not is_s else "CPU (box or local)")
        if a_pass:
            add(step9, f"diag-b/{st}", "MCF diagnostic (b)", st,
                [f"python scripts/n3_mcf_diag.py b --set {st} --workers {{W}} --after_sampling "
                 f"--haar_floor results/n3/private/mcf/haar_floor_{st}.json"],
                "deterministic (oracle ceiling from truth)", [f"results/n3/private/mcf/diag_b_{st}.json"], "local, step 9")
    return step7, step9


def report_configs(F, degrade, form, gate_c_failed):
    """The step-9 report config per set (arms -> frozen bundles and selectors), and its scoring commands."""
    out, cmds = {}, {}
    ie_seeds = [] if 7 in degrade else ([0] if 6 in degrade else list(SEEDS))
    p1_seeds = [0] if 6 in degrade else list(SEEDS)
    sets = ["testB"] + ([] if 5 in degrade else ["devtest"])
    for st in ["testB", "devtest"]:
        arms = {"OURS": ([_dp("ours", s, st) for s in SEEDS], "lj", "rasym"),
                "OURS-N2": ([_dp("ours_n2", s, st) for s in SEEDS], "lj", "rasym")}
        if st in sets:
            arms.update({
                "floor": ([_dp("iii-u", b, st) for b in SEEDS], "random", "rasym"),
                "iii-u": ([_dp("iii-u", b, st) for b in SEEDS], (F.get("iii-u") or {}).get("selector"), "rasym"),
                "iii-p": ([_dp("iii-p", b, st) for b in SEEDS], (F.get("iii-p") or {}).get("selector"), "rasym"),
                "iii-s": ([_dp("iii-s", -1, st)], "ff", "rasym")})
            if ie_seeds:
                arms["i-E"] = ([_dp("i-E-ours", s, st) for s in ie_seeds], "ff", "rasym")
            R, G, A, iv = F.get("MCF-R") or {}, F.get("MCF-R+G") or {}, F.get("MCF-A") or {}, F.get("iv-P") or {}
            full = lambda X, seeds: bool(X.get("checkpoints")) and all(str(s) in X["checkpoints"] for s in seeds)
            if full(R, SEEDS):
                arms["MCF-R"] = ([_dp(f"mcfR_{R['checkpoints'][str(s)]['ckpt_label']}_u{g(R['s_uR'])}", s, st)
                                  for s in SEEDS], R.get("selector"), "cells")
            if full(G, SEEDS):
                lab = lambda s: f"{G['checkpoints'][str(s)]['ckpt_label']}_u{g(G['s_uR'])}"
                arms["MCF-R+G"] = ([_dp(f"mcfRG_{lab(s)}", s, st) for s in SEEDS], G.get("selector"), "cells")
                arms["ii-S"] = ([_dp(f"mcfRS_{lab(s)}", s, st) for s in SEEDS], (F.get("ii-S") or {}).get("selector"),
                                "rasym")
            if iv.get("selector"):
                arms["iv-P"] = ([_dp("ivP", s, st) for s in SEEDS], iv.get("selector"), "cells")
                arms["iv-S"] = ([_dp("ivS", s, st) for s in SEEDS], (F.get("iv-S") or {}).get("selector"), "rasym")
            if full(A, A.get("seeds") or []):
                arms["MCF-A"] = ([_dp(f"mcfA_{A['checkpoints'][str(s)]['ckpt_label']}_u{g(A['s_uR'])}_f{g(A['s_uF'])}",
                                      s, st) for s in A["seeds"]], A.get("selector"), "cells")
                arms["MCF-A-floor"] = ([_dp("mcfAfloor", s, st) for s in A["seeds"]], "random", "cells")
        ref_ff, ref_p1 = _dp("fftruth", -1, st), _dp("p1ref", -1, st)
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        bj = lambda arm, ref, sel: rp(os.path.join(BASIN_DIR, f"{arm}_{st}__{stem(ref)}_{sel}.json"))
        h2b, p1b, c = {}, {}, []
        S9 = "python scripts/n3_score.py"
        for name, (bs, sel, _k) in arms.items():
            bl = " ".join(bs)
            if name == "floor":
                for m in ("mcf05", "mcf08"):
                    c.append(f"{S9} match --bundles {bl} --matcher {m} --draws all --step9 --workers {{W}}")
                continue
            c.append(f"{S9} match --bundles {bl} --matcher primary --draws all --step9 --workers {{W}}")
            if name.startswith("MCF-A"):
                for m in ("mcf05", "mcf08", "mcf10") if 2 not in degrade else ("mcf08", "mcf10"):
                    c.append(f"{S9} match --bundles {bl} --matcher {m} --draws all --step9 --workers {{W}}")
            elif sel and sel != "random":
                for m in ("mcf05", "mcf08", "mcf10"):
                    c.append(f"{S9} match --bundles {bl} --matcher {m} --draws picks --selector {sel} --step9 "
                             f"--workers {{W}}")
            elif sel == "random":
                for m in ("mcf05", "mcf08", "mcf10"):
                    c.append(f"{S9} match --bundles {bl} --matcher {m} --draws all --step9 --workers {{W}}")
        if st in sets:                                 # the H2 pair in BASIN form (both forms run, §3.3)
            for name, arm, seeds, sel, draws in (("OURS", "basin-ours", SEEDS, "ff", "all"),
                                                 ("OURS-N2", "basin-ours_n2", SEEDS, "ff", "all"),
                                                 ("iii-s", "iii-s", [-1], "ff", "picks"),
                                                 ("i-E", "i-E-ours", ie_seeds, "ff", "picks")):
                if not seeds:
                    continue
                bl = " ".join(_dp(arm, s, st) for s in seeds)
                c.append(f"{S9} match --bundles {bl} --matcher primary --reference {ref_ff} --draws {draws}"
                         + (f" --selector {sel}" if draws == "picks" else "") + " --step9 --workers {W}")
                c.append(f"{S9} basin --bundles {bl} --reference {ref_ff} --selector {sel} --step9")
                h2b[name] = bj(arm, ref_ff, sel)
            for name, arm, seeds, key in (("p1-OURS", "p1-ours", p1_seeds, "p1-ours"),
                                          ("p1-iii-p", "iii-p", p1_seeds, "p1-iii-p")):
                sel = ((F.get("P1") or {}).get(key) or {}).get("selector")
                bl = " ".join(_dp(arm, s, st) for s in seeds)
                c.append(f"{S9} match --bundles {bl} --matcher primary --reference {ref_p1} --draws all --step9 "
                         "--workers {W}")
                c.append(f"{S9} basin --bundles {bl} --reference {ref_p1} --selector {sel} --step9")
                p1b[name] = bj(arm, ref_p1, sel)
            c.append(f"{S9} match --bundles {ref_ff} --matcher primary --step9 --workers {{W}}")
            c.append(f"python scripts/n3_classical.py anchor --set {st}")
        cfg = {"set": st, "matcher": "primary",
               "arms": {k: {"bundles": b, "selector": s} for k, (b, s, _kd) in arms.items()},
               "ours": "OURS", "ours_n2": "OURS-N2", "H1": ["MCF-R", "MCF-R+G"], "H2": "iii-s", "H3": "iv-P",
               "ii_s": "ii-S", "i_e": "i-E", "h2_form": form, "h2_basin": h2b, "steric_basin": p1b,
               "steric_basin_ours": "p1-OURS",
               "ff_labels": rp(os.path.join(CLS_DIR, f"ff_labels_{st}.json")) if st in sets else None,
               "labels": [rp(os.path.join(C.PRIVATE, f"labels_{st}.json"))],
               "secondary": ["mcf05", "mcf08"], "rms_matcher": "mcf10", "gate_ca_failed": gate_c_failed,
               "n_boot": 10000, "out": f"results/n3/report_{st}.out.json"}
        c.append(f"{S9} report --config results/n3/report_{st}.json --step9")
        out[st], cmds[st] = cfg, c
    return out, cmds


# ------------------------------------------------------------------------------------------------ assemble
def assemble(paths, allow_absent=False, diag_a_unresolved=False):
    P = Problems()
    I = Inputs(paths, P)
    deg, degrade = degrade_record(P, I)
    F = {}
    F.update(ours_frozen(P))
    F["iii-u"] = single_cell(P, I, "iii-u", r"^iii-u_s{s}_valsel\.pt$", MENU_CLASSICAL, "lj", allow_absent=allow_absent)
    F["iii-p"] = single_cell(P, I, "iii-p", r"^iii-p_s{s}_valsel\.pt$", MENU_CLASSICAL, "lj", allow_absent=allow_absent)
    F["iii-s"] = {"selector": "ff (lowest FF* minimum; fixed physics selector, §2.3 / §3.3)", "m": 1}
    F["P1"] = {"p1-ours": single_cell(P, I, "p1-ours", r"^p1-ours_s{s}_valsel\.pt$", P1_MENU, ref_arm="p1ref",
                                      allow_absent=allow_absent),
               "p1-iii-p": single_cell(P, I, "p1-iii-p", r"^iii-p_s{s}_valsel\.pt$", P1_MENU, ref_arm="p1ref",
                                       allow_absent=allow_absent),
               "reference": "p1ref (press(true R0)); hit = StructureMatcher() fit(pressed draw, pressed truth)",
               "seeds": [0] if 6 in degrade else list(SEEDS)}
    R = mcf_r(P, I, "mcfR", allow_absent)
    G = mcf_r(P, I, "mcfRG", allow_absent, same_cells_as=R.get("_stage_a_keys"))
    for s, c in (G.get("checkpoints") or {}).items():
        r = (R.get("checkpoints") or {}).get(s) or {}
        G.setdefault("separate_sampling", {})[s] = (c.get("ckpt_sha256"), G.get("s_uR")) != (r.get("ckpt_sha256"),
                                                                                             R.get("s_uR"))
    F["MCF-R"], F["MCF-R+G"] = R, G
    if G.get("checkpoints"):
        lab = {s: f"{c['ckpt_label']}_u{g(G['s_uR'])}" for s, c in G["checkpoints"].items()}
        F["ii-S"] = single_cell(P, I, "iiS", r"^mcfRS_.+_s{s}_valsel\.pt$", MENU, allow_absent=allow_absent)
        if F["ii-S"]:
            got = {str(x["seed"]): f"{x['label']}_u{g(x['u'])}" for x in map(parse_mcf, F["ii-S"]["bundles"]) if x}
            if got != lab:
                P.err("iiS", f"ii-S candidates {got} are not the +G frozen draws {lab}")
    else:
        I.get("iiS")
    F["MCF-A"] = mcf_a(P, I, degrade, allow_absent)
    iv = iv_p(P, I, allow_absent)
    F["iv-P"] = iv
    F["iv-S"] = single_cell(P, I, "ivS", r"^ivS_.+_ep\d+_s{s}_valsel\.pt$", MENU, "resid", allow_absent=allow_absent)
    if F["iv-S"] and iv:
        eps = [int(re.search(r"_ep(\d+)_s", b).group(1)) for b in F["iv-S"]["bundles"]]
        if [str(e) for e in eps] != [str((iv.get("epoch_per_seed") or {}).get(str(s))) for s in SEEDS]:
            P.err("ivS", f"iv-S candidates at epochs {eps} are not iv-P's frozen checkpoints")
    form = anchor(P, I)
    a = diag_a(P, I, F, degrade, diag_a_unresolved)
    gate_ca_failed, gate = None, None
    gc = I.get("gate_c")
    if gc is not None:                                # §3.2 Gate C: C-a (i), C-b, C-c pass before MCF training;
        ca = gc.get("C-a") or {}                      # C-a (ii) failing only adds the H1 harness clause
        for part, ok in (("C-a (i)", ca.get("i_pass")), ("C-b", (gc.get("C-b") or {}).get("pass")),
                         ("C-c", (gc.get("C-c") or {}).get("pass"))):
            if ok is not True:
                P.err("gate_c", f"{part} has not passed (no MCF arm trains before it does)")
        if "ii_pass_test" in ca or "ii_pass_val" in ca:
            gate_ca_failed = not (ca.get("ii_pass_test") or ca.get("ii_pass_val"))
        gate = {"C-a": {k: ca.get(k) for k in ("i_pass", "ii_pass_test", "ii_pass_val", "pass",
                                               "disclose_only_val_agrees")},
                "C-b": (gc.get("C-b") or {}).get("pass"), "C-c": (gc.get("C-c") or {}).get("pass"),
                "gate_ca_ii_failed": gate_ca_failed,
                "consequence": "every H1 sentence carries the §3.2 Gate C-a harness clause" if gate_ca_failed else None}
    step7, step9 = build_runs(F, degrade, a["verdict"] == "PASS")
    rcfg, scmd = report_configs(F, degrade, (form or {}).get("form"), gate_ca_failed)
    for k in list(F):
        if isinstance(F[k], dict):
            F[k] = {kk: v for kk, v in F[k].items() if not kk.startswith("_")}
    records = {role: I.data[role] for role in I.data if role not in ("anchor",) and not role.startswith("diag_a:")}
    try:
        head = subprocess.run(["git", "-C", C.REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, OSError):
        head = None
    sel = {"protocol": "tasks/n3_protocol.md §7 step 6", "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "git_head": head, "protocol_sha256": C.sha256(PROTOCOL) if os.path.exists(PROTOCOL) else None,
           "inputs": {role: {"path": rp(I.paths[role]), "sha256": I.sha[role]} for role in sorted(I.sha)},
           "degrade": deg, "h2_form": form, "gate_c": gate, "frozen": F, "mcf_diagnostic_a": a,
           "valsel_cell_scores": records,
           "runs": {"placeholders": PLACEHOLDERS,
                    "rule": "§7 step 7: exactly these runs, with these seeds, on TEST-B / DEV-TEST; a run not listed "
                            "is disclosed and that arm's run is invalid. Step 9: local scoring after step 8.",
                    "step7": step7, "step9": step9, "step9_scoring": scmd},
           "report_configs": rcfg}
    return sel, P, I


def classical_configs(paths):
    """n3_score.py select configs of the single-cell classical selections (§3.3): iii-u and iii-p (§2.3 menu
    without RESID, MODE ties by LJ) and the steric-BASIN P1 selectors of OURS and iii-p (P1 menu, rows matched
    against the VALSEL p1ref bundle). Their 'out' is the role's default record path."""
    ref = _dp("p1ref", -1, "valsel")
    cfgs = {}
    for role, arm, menu, reference, mtb in (("iii-u", "iii-u", MENU_CLASSICAL, None, "lj"),
                                            ("iii-p", "iii-p", MENU_CLASSICAL, None, "lj"),
                                            ("p1-ours", "p1-ours", P1_MENU, ref, None),
                                            ("p1-iii-p", "iii-p", P1_MENU, ref, None)):
        cfg = {"name": role, "matcher": "primary", "menu": menu,
               "stage_a": {"cells": [{"label": role, "bundles": [_dp(arm, s, "valsel") for s in SEEDS]}],
                           "tie": ["selector"]},
               "out": rp(paths[role])}
        if mtb:
            cfg["mode_tiebreak"] = mtb
        if reference:
            cfg["reference"] = reference
        cfgs[role] = cfg
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-configs", action="store_true",
                    help="write the select configs of iii-u, iii-p and the two P1 selections to "
                         "results/n3/private/select/<role>.json (refuses to overwrite) and stop")
    ap.add_argument("--inputs", default=None, help="JSON {role: path} overriding the default input paths")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true", help="list present / missing / invalid records; write nothing")
    ap.add_argument("--allow-absent-bundles", action="store_true",
                    help="record scored bundles that are not on this machine as unverified instead of refusing")
    ap.add_argument("--diag-a-unresolved", action="store_true",
                    help="record missing diagnostic-(a) files as FAILED (§3.2: unresolved at step 6)")
    ap.add_argument("--force", action="store_true", help="replace an existing output file")
    args = ap.parse_args()
    paths = default_inputs()
    if args.inputs:
        paths.update(json.load(open(rel(args.inputs))))
    if args.write_configs:
        for role, cfg in classical_configs(paths).items():
            p = os.path.join(SEL_DIR, f"{role}.json")
            if os.path.exists(p):
                print(f"exists, kept: {rp(p)}")
                continue
            os.makedirs(SEL_DIR, exist_ok=True)
            json.dump(cfg, open(p, "w"), indent=1)
            print(f"wrote {rp(p)}  ->  python scripts/n3_score.py select --config {rp(p)}")
        return
    sel, P, I = assemble(paths, args.allow_absent_bundles, args.diag_a_unresolved)
    roles = sorted(set(I.status) | {r for r, _ in P.rows})
    if args.dry_run:
        for r in roles:
            st = I.status.get(r, "checked")
            errs = P.of(r)
            print(f"{r:>18s}  {st:<18s} {rp(I.paths[r]) if r in I.paths else '':<58s}"
                  + ("" if not errs else "  PROBLEM: " + " | ".join(errs)))
        print(f"\n{len(P.rows)} problem(s); diagnostic (a): {sel['mcf_diagnostic_a']['verdict']}; "
              f"H2 FORM: {(sel['h2_form'] or {}).get('form')}; step-7 runs listed: {len(sel['runs']['step7'])}")
        sys.exit(0)
    if P.rows:
        for r, m in P.rows:
            print(f"  {r}: {m}")
        raise SystemExit(f"n3_select: {len(P.rows)} problem(s); nothing written (run --dry-run for the full list)")
    out = rel(args.out)
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists (it is committed at step 6); --force replaces it")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(sel, f, indent=1)
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {len(sel['inputs'])} input records; H2 FORM {sel['h2_form']['form']}; diagnostic (a) "
          f"{sel['mcf_diagnostic_a']['verdict']}; {len(sel['runs']['step7'])} step-7 runs")


if __name__ == "__main__":
    main()
