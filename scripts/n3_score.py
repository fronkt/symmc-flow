"""N3 unified scorer (protocol §2.1-§2.6, §3 selection, §4, §6): every match and every reported number.

match     match draws of n3_common bundles against the truth (to_structure(orig)); one matcher per run:
            primary  StructureMatcher() defaults, fit(truth, candidate)                         every draw
            mcfXX    StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=XX/10)
                     .get_rms_dist(truth, candidate) is not None (and its rms)                  picks, iii-u
                     (mcf05 / mcf08 are the §2.1 secondaries; mcf10 is the §2.2 continuous-RMS matcher,
                     no match = +inf; mcf08 / mcf09 / mcf10 are Gate C-a's stols and MCF-A's native sweep)
          adapters: kind 'rasym' -> n3_common.rasym_structure(a, R); kind 'cells' -> n3_common.cell_structure.
          --draws all = every draw; --draws picks = the --selector pick of every bundle (seed) plus the
          ensemble pick over all bundles given (one arm, one set). Each fit is an n3_harness task: a crash,
          exception or slow fit (> --slow s) is a HARNESS failure re-run alone in a fresh process with no
          wall-clock limit; identical recurrence = DETERMINISTIC = non-match; all logged (§1.3). An invalid or
          non-finite draw is an ARM failure: non-match, no fit. Identical candidates of one crystal are fitted
          once. Rows are appended per fit to results/n3/private/match/<bundle>__<matcher>.jsonl (resumable;
          refused if the bundle changed since; an unfinished final line from a kill is cut off first).
          BLINDING (§1.3): bundles of SEL, TEST-B or DEV-TEST are refused unless --step9, which also needs the
          step-6 file (--selection, default results/n3/n3_selection.json) committed, unchanged since its
          commit and pushed (git), or, without git, equal to --selection-sha256; its SHA-256 and commit go into
          the match meta. VAL, VALSEL and TRAIN bundles are allowed.
          --reference <bundle> (§3.3 BASIN forms): match against the reference bundle's draw 0 instead of
          to_structure(orig): 'fftruth' (the FF*-minimised truth; H2 FORM = BASIN; primary = fit(reference,
          candidate)) or 'p1ref' (the pressed truth; steric BASIN P1; G2's fit(pressed draw, pressed truth)).
          Rows go to <bundle>__<matcher>~<reference stem>.jsonl, refused if the bundle or the reference changed.
          Rule (iii): a reference without a minimum (invalid draw 0) gives 'ref_invalid' rows (non-match, no fit).
basin     per-crystal, per-seed picked@1 of one arm against a reference bundle from those rows -> the h2_basin
          JSON {refcodes, h [n][m], ...} (report 'h2_basin' / 'steric_basin'); rule (iii) sets h = 0 (d_i = 0)
          for every crystal whose reference has no minimum (not anchored) and lists unconverged references.
          H2 BASIN: basin-ours_s{0,1,2}, basin-ours_n2_s{0,1,2} (selector ff; one draw), iii-s_s-1 (selector ff =
          its draw 0, the lowest minimum), i-E-ours (ff) vs fftruth_s-1. P1: p1-ours_s{0,1,2}, iii-p_s{0,1,2}
          (selector from lj / lj_pre / random, chosen on VALSEL by `select` with "reference") vs p1ref_s-1.
select    Stage A / Stage B selection engine (§2.3, §3.2, §3.4) on VALSEL match results -> JSON record.
report    Table 1 / Table 2 inputs and the confirmatory statistics (§2.4, §4, §6 abstract inputs) -> JSON.
picks     truth-free export of each seed's (and the ensemble's) selector pick {refcode, seed, j} -> JSON, e.g.
          the LJ picks the FF* unit minimises for H2 BASIN (§3.3); allowed on every set (no truth is read).
oracle    rasym-adapter oracle: expand(a, R0) x 16 per crystal must score 100% under every matcher.
gate-f6   scorer oracle of §3.2 Gate F6: draws built from an MCF export pickle's own poses must score 100%
          picked@1, draw@1 and any@16 under every matcher (input-side; allowed on every set). The predictions
          file is checked draw by draw against the export pickle (lattice, pose tensors, species, Cartesian
          coordinates) BEFORE anything is matched; anything else (e.g. an arm's predictions) is refused.
gate-ca   §3.2 Gate C-a (i)/(ii): any-of-10 of released-checkpoint predictions_10.pt runs against their own
          pickle's ground truth (gt_coords, lattice_1, atom_types: what MCF's scorer reads) at MCF's settings,
          stol 0.8 / 0.9 / 1.0 (mcf08 / mcf09 / mcf10), through the same harness; per-(run, crystal, stol)
          outcomes, 5-run medians vs Fig. 3a, and, given MCF's run_structure_matching.py output dirs, the
          scorer agreement with every (crystal, stol) and per-draw disagreement listed. --mcf-reading
          (diagnostic, own --label) scores MCF's xyz reading instead: 6-dp round trip and the lattice rebuilt
          from its six parameters around unchanged Cartesian coordinates, which is NOT the same crystal when
          the cell is not in pymatgen's parameter orientation (68/100 VAL, 115/200 DEV-TEST exported cells);
          agreement in that mode attributes the default mode's disagreements to MCF's reader. Refuses a pickle
          that is an N3 SEL / TEST-B / DEV-TEST (VALSEL) export (SHA-256, or >= 50% of its crystals by reduced
          composition + volume per atom).

    python scripts/n3_score.py match --bundles results/n3/private/draws/ours_s{0,1,2}_valsel.pt --matcher primary --workers 8
    python scripts/n3_score.py match --bundles <3 seed bundles> --matcher mcf05 --draws picks --selector lj
    python scripts/n3_score.py select --config results/n3/private/select/mcfR_A.json
    python scripts/n3_score.py picks --bundles results/n3/private/draws/ours_s{0,1,2}_testB.pt --selector lj
    python scripts/n3_score.py report --config results/n3/report_testB.json --step9
    python scripts/n3_score.py gate-f6 --set devtest --workers 8        # export's oracle predictions_16.pt
    python scripts/n3_score.py gate-f6 --set devtest --pickle n3_mcf/devtest/test_molcrystal_normalized.pkl.gz
    python scripts/n3_score.py oracle --set val                          # rasym adapter: expand(a, R0)
    python scripts/n3_score.py oracle --set val --reference              # + reference-bundle matching oracle
    python scripts/n3_score.py match --bundles results/n3/private/draws/basin-ours_s{0,1,2}_testB.pt --matcher primary \
        --reference results/n3/private/draws/fftruth_s-1_testB.pt --step9
    python scripts/n3_score.py basin --bundles results/n3/private/draws/basin-ours_s{0,1,2}_testB.pt --selector ff \
        --reference results/n3/private/draws/fftruth_s-1_testB.pt --step9
    python scripts/n3_score.py gate-ca --label test --pickle <thurlemann23/.../test_molcrystal_normalized.pkl.gz> \
        --predictions <run0>/predictions_10.pt ... <run4>/predictions_10.pt --mcf-results <run0> ... <run4> --workers 8

Select config (JSON; bundle paths relative to the repo). Metric keys: 'hits' (= 'picked', seed-summed picked@1
of the row's selector), 'draw1', 'any16' (any draw), 'any<K>' (any of the FIRST K draws, e.g. MCF-A's any10),
each with an optional '@<matcher>' (default: the config matcher; the match rows must exist). Higher is better;
'val_loss' lower is better; 'prefer_cell' and 'selector' (LJ > RESID > MODE small r > RANDOM) are tie keys.
    {"name": str, "matcher": "primary", "menu": [selectors] (default §2.3 menu), "mode_tiebreak": "resid"|"lj",
     "reference": path (optional; rows of `match --reference` against it, e.g. P1 on p1ref with menu P1_MENU),
     "stage_a": {"cells": [{"label": str, "bundles": [seed0, seed1, seed2]}, ...],
                 "menu": [selectors] | null  (overrides the top-level menu; null = cells only, no selector),
                 "rank": "hits" (default) | metric key,
                 "tie": ["draw1", "any16", "prefer_cell", "selector"] (default ["selector"]),
                 "prefer_cell": str},
     "stage_b": {"selector": name | "from_stage_a" | null,
                 "seeds": {"0": [{"label": str, "bundle": path, "val_loss": float}, ...], ...},
                 "rank": "picked" (default) | metric key,
                 "tie": ["draw1", "any16", "val_loss"]},
     "out": path}
  MCF-R / +G / iv-P / classical: stage_a (menu) with tie [draw1, any16, prefer_cell, selector], stage_b default.
  MCF-A (§3.2 ii-A), three runs: (1) stage_b at (9, 3), selector null, rank "any10@mcf10", tie ["any10@mcf08",
  "val_loss"]; (2) stage_a over the 9 knob cells, menu null, rank "any10@mcf10", tie ["prefer_cell"],
  prefer_cell "(9,3)"; (3) stage_a on the chosen cell, menu ["random", "lj", "resid"], rank "picked@mcf10".
Report config (JSON):
    {"set": "testB"|"devtest", "matcher": "primary",
     "arms": {name: {"bundles": [...], "selector": name, "kind": "rasym"|"cells"}},  (OURS, OURS-N2, MCF-R,
             MCF-R+G, iii-s, iv-P, iii-u, floor, iii-p, ii-S, ...; any subset while arms are missing)
     "ours": "OURS", "ours_n2": "OURS-N2", "H1": ["MCF-R", "MCF-R+G"], "H2": "iii-s", "H3": "iv-P",
     "ii_s": "ii-S", "h2_form": "EXACT"|"BASIN",
     "h2_basin": {arm: `basin` json vs fftruth {"refcodes": [...], "h": [[per-seed 0/1], ...]}} (OURS, OURS-N2,
             iii-s, i-E; BASIN form: the H2 contrast; EXACT form: reported as the secondary form, SI),
     "steric_basin": {arm: `basin` json vs p1ref}, "steric_basin_ours": name (descriptive P1 contrasts),
     "i_e": "i-E" (i-E vs iii-s with a CI in the H2 FORM; EXACT form: an 'arms' entry with selector ff),
     "dry_run": false (true: VAL / VALSEL / TRAIN only -- run the family / wording code paths as a check),
     "ff_labels": json [{refcode, ff_label}] or {refcode: label}, "labels": [label files],
     "strata": [keys] (default K, z_mcf, class, mixed, improper_achiral, legacy_unrep, formula_disjoint,
     omc25_nonoverlap when present), "secondary": ["mcf05", "mcf08"], "rms_matcher": "mcf10",
     "gate_ca_failed": false, "n_boot": 10000, "out": path}
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

import numpy as np
import torch

MATCH_DIR = os.path.join(C.PRIVATE, "match")
ORACLE_DIR = os.path.join(C.PRIVATE, "oracle")
BASIN_DIR = os.path.join(C.PRIVATE, "basin")
SELECTION = os.path.join(C.REPO, "results", "n3", "n3_selection.json")
BLIND = ("sel", "testB", "devtest")
MODE_R = (0.25, 0.5, 0.75, 1.0, 1.5)
MENU = ["lj", "resid"] + [f"mode@{r}" for r in MODE_R] + ["random"]
P1_MENU = ["lj", "lj_pre", "random"]      # §3.3 steric BASIN: post-press LJ, pre-press LJ then press, random
# Reference bundles (§3.3): which structure goes first in StructureMatcher.fit / get_rms_dist.
#   fftruth  H2 BASIN: the primary matcher, fit(reference, candidate) -- the FF*-minimised truth takes the truth's
#            place in §2.1's fit(truth, candidate)
#   p1ref    steric BASIN P1: "fit(pressed draw, pressed truth)" (§3.3; G2's g2_eval / g2_asym_baselines order)
#   oracle*  truth-as-bundle test references: the order is the bundle's meta['ref_order']
REF_ORDER = {"fftruth": "ref_cand", "p1ref": "cand_ref"}
# MCF compact atom-type index -> atomic number: MCF's 12-element map plus protocol P1 (Mg, As, Se, Te -> 12-15).
# The ONE inverse map of the benchmark (Gate F2 decodes species through it).
MCF_IDX_TO_Z = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9, 5: 16, 6: 17, 7: 15, 8: 35, 9: 53, 10: 5, 11: 14,
                12: 12, 13: 33, 14: 34, 15: 52}


def rel(p):
    return p if os.path.isabs(p) else os.path.join(C.REPO, p)


# ============================================================================== blinding guard
def _git(*a):
    try:
        return subprocess.run(["git", "-C", C.REPO, *a], capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        return None


def selection_record(selection, expect_sha=None):
    """§7 step 6: the selection file is committed, unchanged since that commit and pushed (a remote-tracking
    branch contains the commit). Without git its SHA-256 must equal expect_sha (--selection-sha256, the
    committed value). Returns {sha256, commit, checked}."""
    if not os.path.exists(selection):
        raise SystemExit(f"--step9 needs the committed step-6 file {selection}")
    sha = C.sha256(selection)
    if expect_sha is not None and sha != expect_sha:
        raise SystemExit(f"{selection}: SHA-256 {sha} != --selection-sha256 {expect_sha}")
    r = _git("rev-parse", "--is-inside-work-tree")
    if r is None or r.returncode != 0 or r.stdout.strip() != "true":
        if expect_sha is None:
            raise SystemExit("--step9 without git: pass --selection-sha256 <SHA-256 of the committed file>")
        return {"sha256": sha, "commit": None, "checked": "sha256 only (no git)"}
    rp = os.path.relpath(os.path.abspath(selection), C.REPO)
    if rp.startswith("..") or _git("ls-files", "--error-unmatch", "--", rp).returncode != 0:
        raise SystemExit(f"--step9: {selection} is not committed in {C.REPO}")
    if _git("diff", "--quiet", "HEAD", "--", rp).returncode != 0:
        raise SystemExit(f"--step9: {selection} differs from its committed version")
    commit = _git("log", "-1", "--format=%H", "--", rp).stdout.strip()
    if not _git("branch", "-r", "--contains", commit).stdout.strip():
        raise SystemExit(f"--step9: commit {commit[:10]} of {selection} is on no remote-tracking branch (push it)")
    return {"sha256": sha, "commit": commit, "checked": "git: committed, unchanged, pushed"}


def guard(set_name, step9, selection, expect_sha=None):
    """Hard blinding guard (§1.3): arm output of SEL / TEST-B / DEV-TEST is matched only at step 9."""
    if set_name in BLIND and not step9:
        raise SystemExit(f"BLINDING: refusing to score a '{set_name}' bundle. SEL is scored only inside VALSEL "
                         "bundles (step 5); TEST-B and DEV-TEST only at step 9 (--step9).")
    if step9:
        if set_name not in ("testB", "devtest"):
            raise SystemExit("--step9 is for TEST-B / DEV-TEST bundles only")
        return selection_record(selection, expect_sha)
    return None


# ============================================================================== adapters
def payload(b, i, j):
    """Picklable description of draw (i, j) for a worker (numpy, no torch shared memory)."""
    if b["kind"] == "rasym":
        return b["R"][i, j].double().numpy()
    c = b["cells"][i][j]
    return {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in c.items()}


def payload_finite(p):
    if isinstance(p, dict):
        return bool(np.isfinite(p["lattice"]).all() and np.isfinite(p["cart"]).all())
    return bool(np.isfinite(p).all())


def payload_hash(p):
    h = hashlib.sha1()
    if isinstance(p, dict):
        for k in ("lattice", "cart", "species"):
            h.update(np.ascontiguousarray(p[k]).tobytes())
    else:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


def structure_from_payload(a, kind, p):
    if kind == "rasym":
        return C.rasym_structure(a, torch.from_numpy(np.asarray(p, dtype=np.float64)))
    return C.cell_structure({k: torch.from_numpy(np.asarray(v)) for k, v in p.items()})


def draw_coords(b, a, i, j):
    """Unwrapped whole-molecule Cartesian atoms of every copy (the adapter's coordinates before pymatgen
    wraps sites) and their atomic numbers -- the MODE@r Chamfer inputs."""
    if b["kind"] == "rasym":
        from g2_asym_baselines import expand
        L, cent, orient, local, Z, am, mm = expand(a, b["R"][i, j].double())
        X = (cent @ L).unsqueeze(1) + torch.einsum("kij,kaj->kai", orient, local.double())
        return X[am].numpy(), Z[am].numpy()
    c = b["cells"][i][j]
    return np.asarray(c["cart"], dtype=np.float64), np.asarray(c["species"])


# ============================================================================== matchers (worker side)
_W = {}


def _winit():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)


def _matcher(name):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    if name not in _W.setdefault("sm", {}):
        if name == "primary":
            _W["sm"][name] = StructureMatcher()
        elif name.startswith("mcf"):
            _W["sm"][name] = StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=int(name[3:]) / 10)
        else:
            raise ValueError(name)
    return _W["sm"][name]


def _truth(set_name, i):
    key = (set_name, i)
    cache = _W.setdefault("truth", {})
    if key not in cache:
        if len(cache) > 256:
            cache.clear()
        cache[key] = C.truth_structure(C.load_set(set_name)[i])
    return cache[key]


def _compare(truth, cand, matcher):
    sm = _matcher(matcher)
    if matcher == "primary":
        return {"ok": bool(sm.fit(truth, cand)), "rms": None}
    r = sm.get_rms_dist(truth, cand)
    return {"ok": r is not None, "rms": None if r is None else float(r[0])}


def _fit_task(task):
    set_name, i, kind, p, matcher = task
    truth = _truth(set_name, i)
    cand = structure_from_payload(C.load_set(set_name)[i], kind, p)
    return _compare(truth, cand, matcher)


def _ref_fit_task(task):
    """Reference-bundle match (§3.3 BASIN forms): the reference draw replaces to_structure(orig)."""
    set_name, i, rkind, rp, kind, p, matcher, order = task
    a = C.load_set(set_name)[i]
    ref = structure_from_payload(a, rkind, rp)
    cand = structure_from_payload(a, kind, p)
    return _compare(ref, cand, matcher) if order == "ref_cand" else _compare(cand, ref, matcher)


def _pair_task(task):
    """Gate C-a: truth and candidate both given as 'cells' payloads (released data, not an N3 set)."""
    t, p, matcher = task
    return _compare(structure_from_payload(None, "cells", t), structure_from_payload(None, "cells", p), matcher)


# ============================================================================== selectors
def _fin(v):
    v = float(v)
    return (0, v) if math.isfinite(v) else (1, 0.0)


def argmin_pick(valid, vals):
    """Lowest value; non-finite last; exact ties -> lowest index; None if no valid draw (arm MISS)."""
    cand = [j for j in range(len(valid)) if valid[j]]
    if not cand:
        return None
    return min(cand, key=lambda j: (_fin(vals[j]), j))


def chamfer(X1, Z1, X2, Z2):
    """Element-matched symmetric Chamfer distance (A), N1's definition (g2_rank._chamfer_matrix)."""
    from scipy.spatial.distance import cdist
    s1 = s2 = 0.0
    for z in np.unique(Z1):
        D = cdist(X1[Z1 == z], X2[Z2 == z])
        s1 += D.min(1).sum()
        s2 += D.min(0).sum()
    return 0.5 * (s1 / len(Z1) + s2 / len(Z2))


def mode_pick(coords, valid, r, tb, D=None):
    """MODE@r: the valid draw with the most valid draws within Chamfer r (itself included); ties -> lowest
    tie-break value (RESID for learned arms, LJ for classical), then lowest index. D = precomputed Chamfer
    matrix over the same draws (optional)."""
    idx = [j for j in range(len(valid)) if valid[j]]
    if not idx:
        return None
    if D is None:
        D = chamfer_matrix(coords, valid)
    cnt = {j: int(sum(D[j, k] < r for k in idx)) for j in idx}
    best = max(cnt.values())
    return min([j for j in idx if cnt[j] == best], key=lambda j: (_fin(tb[j]), j))


def chamfer_matrix(coords, valid, D=None):
    """Symmetric Chamfer matrix over valid draws (0 on the diagonal, inf for invalid); entries already
    finite in D (a cache) are reused."""
    n = len(valid)
    D = np.full((n, n), np.inf) if D is None else D
    for j in range(n):
        if not valid[j]:
            continue
        D[j, j] = 0.0
        for k in range(j + 1, n):
            if valid[k] and not np.isfinite(D[j, k]):
                D[j, k] = D[k, j] = chamfer(*coords[j], *coords[k])
    return D


def sel_pref(name):
    """§2.3 tie preference: LJ > RESID > MODE@r (smaller r first) > RANDOM (other selector keys after RESID)."""
    if name == "lj":
        return (0, 0.0)
    if name == "resid":
        return (1, 0.0)
    if name.startswith("mode@"):
        return (3, float(name[5:]))
    if name == "random":
        return (4, 0.0)
    return (2, 0.0)


class Crystal:
    """The draws of one crystal across the seed bundles of one arm (concatenated for the ensemble). The
    Chamfer matrix is computed lazily, once, and shared by every MODE@r and by the ensemble."""

    def __init__(self, bundles, items, i, mode_tiebreak=None):
        self.bundles, self.i, self.a = bundles, i, items[i]
        self.S = [int(b["valid"].shape[1]) for b in bundles]
        self.off = np.concatenate([[0], np.cumsum(self.S)]).astype(int)
        self.valid = [[bool(v) for v in b["valid"][i]] for b in bundles]
        self.tbkey = mode_tiebreak or ("resid" if "resid" in bundles[0]["sel"] else "lj")
        self._coords = [None] * len(bundles)
        self._D = np.full((self.off[-1], self.off[-1]), np.inf)
        self._blocks = set()

    def vals(self, b, key):
        return [float(v) for v in self.bundles[b]["sel"][key][self.i]]

    def coords(self, b):
        if self._coords[b] is None:
            self._coords[b] = [draw_coords(self.bundles[b], self.a, self.i, j) if self.valid[b][j] else None
                               for j in range(self.S[b])]
        return self._coords[b]

    def matrix(self, b=None):
        """Chamfer matrix of bundle b's draws, or of all draws (b None); finite entries are reused."""
        blocks = [b] if b is not None else list(range(len(self.bundles)))
        key = b if b is not None else "all"
        sl = slice(self.off[blocks[0]], self.off[blocks[-1] + 1])
        if key not in self._blocks:
            if self.bundles[0]["kind"] == "cells":         # Chamfer on unwrapped atoms needs one lattice
                lats = [c["lattice"] for k in blocks for c, v in zip(self.bundles[k]["cells"][self.i],
                                                                      self.valid[k]) if v]
                if any(not torch.allclose(torch.as_tensor(l).double(), torch.as_tensor(lats[0]).double(),
                                          atol=1e-4) for l in lats):
                    raise ValueError("MODE@r needs draws on one lattice (it is not offered to MCF-A, §2.3)")
            coords = [c for k in blocks for c in self.coords(k)]
            valid = [v for k in blocks for v in self.valid[k]]
            self._D[sl, sl] = chamfer_matrix(coords, valid, self._D[sl, sl].copy())
            self._blocks.add(key)
            if b is None:
                self._blocks.update(range(len(self.bundles)))
        return self._D[sl, sl]

    def pick(self, selector, b=None):
        """Draw index in bundle b, or (bundle, draw) for the ensemble (b None); None = no valid draw."""
        if b is not None:
            valid = self.valid[b]
            if selector.startswith("mode@"):
                return mode_pick(None, valid, float(selector[5:]), self.vals(b, self.tbkey), D=self.matrix(b))
            return argmin_pick(valid, self.vals(b, selector))
        flat = [(k, j) for k in range(len(self.bundles)) for j in range(self.S[k])]
        valid = [self.valid[k][j] for k, j in flat]
        if selector.startswith("mode@"):
            tb = [v for k in range(len(self.bundles)) for v in self.vals(k, self.tbkey)]
            u = mode_pick(None, valid, float(selector[5:]), tb, D=self.matrix())
        else:
            u = argmin_pick(valid, [v for k in range(len(self.bundles)) for v in self.vals(k, selector)])
        return None if u is None else flat[u]


# ============================================================================== match results
def match_path(bundle_path, matcher):
    stem = os.path.splitext(os.path.basename(bundle_path))[0]
    in_oracle = os.path.normcase(os.path.dirname(os.path.abspath(bundle_path))) == os.path.normcase(ORACLE_DIR)
    return os.path.join(ORACLE_DIR if in_oracle else MATCH_DIR, f"{stem}__{matcher}.jsonl")


def load_match(bundle_path, matcher):
    """Match rows {(i, j): row} of a bundle under a matcher key (a matcher, or mkey(matcher, reference))."""
    import n3_harness as H
    return {(r["i"], r["j"]): r for r in H.read_jsonl(match_path(bundle_path, matcher))}


def ref_info(ref_path):
    """A reference bundle (§3.3 BASIN forms): its draw 0 per crystal is the structure every candidate is matched
    to, in place of to_structure(orig). Returns {path, rel, stem, sha256, arm, order, bundle, row}; row maps a
    refcode to its row. Rule (iii): an invalid or non-finite draw 0 (no minimum) makes every candidate of that
    crystal a non-match (status 'ref_invalid', no fit)."""
    path = rel(ref_path)
    b = C.load_bundle(path)
    arm = b["arm"]
    order = REF_ORDER.get(arm) or (b["meta"].get("ref_order") if arm.startswith("oracle") else None)
    if order not in ("ref_cand", "cand_ref"):
        raise SystemExit(f"{ref_path}: arm '{arm}' is not a reference arm ({sorted(REF_ORDER)} or an oracle bundle "
                         "with meta['ref_order'])")
    return {"path": path, "rel": os.path.relpath(path, C.REPO).replace("\\", "/"),
            "stem": os.path.splitext(os.path.basename(path))[0], "sha256": C.sha256(path), "arm": arm,
            "order": order, "bundle": b, "row": {r: k for k, r in enumerate(b["refcodes"])}}


def ref_ok(ref, i_ref):
    """Rule (iii) of §3.3: the reference of this crystal exists (valid, finite draw 0)."""
    return bool(ref["bundle"]["valid"][i_ref, 0]) and payload_finite(payload(ref["bundle"], i_ref, 0))


def mkey(matcher, ref=None):
    """Matcher key of the match files: the matcher, or '<matcher>~<reference bundle stem>'."""
    return matcher if ref is None else f"{matcher}~{ref['stem']}"


def base_matcher(key):
    return key.split("~", 1)[0]


def _meta_check(bundle_path, matcher, extra):
    mp = match_path(bundle_path, matcher)[:-len(".jsonl")] + ".meta.json"
    sha = C.sha256(bundle_path)
    if os.path.exists(mp):
        old = json.load(open(mp))
        if old["bundle_sha256"] != sha:
            raise SystemExit(f"{bundle_path} changed since {mp} was written: delete the stale match file first")
        for k in ("reference_sha256", "reference_order"):
            if old.get(k) != extra.get(k):
                raise SystemExit(f"{mp}: {k} {old.get(k)} != {extra.get(k)} (another reference): delete it first")
    else:
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        json.dump({"bundle": os.path.relpath(bundle_path, C.REPO).replace("\\", "/"), "bundle_sha256": sha,
                   "matcher": matcher, **extra}, open(mp, "w"), indent=1)


def rows_current(bundle_path, key, ref=None, sha=None):
    """The match rows of (bundle, key) were made from this bundle (and this reference): else SystemExit."""
    mp = match_path(bundle_path, key)[:-len(".jsonl")] + ".meta.json"
    if not os.path.exists(mp):
        return None
    m = json.load(open(mp))
    if m["bundle_sha256"] != (sha or C.sha256(bundle_path)):
        raise SystemExit(f"{bundle_path} changed since it was matched ({mp})")
    if ref is not None and m.get("reference_sha256") != ref["sha256"]:
        raise SystemExit(f"{mp} was matched against another reference than {ref['rel']}")
    return m


def wanted_draws(bundles, items, draws, selector, crystals, mode_tiebreak=None):
    """-> list per bundle of sorted (i, j) to match."""
    n = len(bundles[0]["refcodes"])
    rng = range(n) if crystals is None else range(*crystals)
    want = [set() for _ in bundles]
    for i in rng:
        if draws == "all":
            for k, b in enumerate(bundles):
                want[k].update((i, j) for j in range(int(b["valid"].shape[1])))
            continue
        cr = Crystal(bundles, items, i, mode_tiebreak)
        for k in range(len(bundles)):
            j = cr.pick(selector, k)
            if j is not None:
                want[k].add((i, j))
        if len(bundles) > 1:
            kj = cr.pick(selector)
            if kj is not None:
                want[kj[0]].add((i, kj[1]))
    return [sorted(w) for w in want]


def _is_oracle_bundle(path, b):
    return (b["arm"] == "oracle" and b["meta"].get("oracle") is True and
            os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(ORACLE_DIR))


def run_match(bundle_paths, matcher, draws="all", selector=None, crystals=None, workers=2, slow=1200.0,
              step9=False, selection=SELECTION, allow_blind=False, mode_tiebreak=None, quiet=False,
              selection_sha=None, reference=None):
    """Match the wanted draws of one arm's seed bundles. reference = a reference bundle path (§3.3 BASIN forms):
    candidates are matched to its draw 0 instead of to_structure(orig), in the order ref_info() fixes, and the
    rows go to <bundle>__<matcher>~<reference stem>.jsonl; an invalid reference gives 'ref_invalid' rows
    (non-matches, no fit; rule iii). The blinding guard is the same (the candidates' set)."""
    import n3_harness as H
    warnings.filterwarnings("ignore")
    bundles = [C.load_bundle(p) for p in bundle_paths]
    sets = {b["set"] for b in bundles}
    assert len(sets) == 1, "one set per run"
    set_name = sets.pop()
    assert all(b["refcodes"] == bundles[0]["refcodes"] for b in bundles)
    if allow_blind and not all(_is_oracle_bundle(p, b) for p, b in zip(bundle_paths, bundles)):
        raise SystemExit("allow_blind is only for truth-derived oracle bundles in results/n3/private/oracle")
    if allow_blind and reference is not None and not _is_oracle_bundle(rel(reference), C.load_bundle(rel(reference))):
        raise SystemExit("allow_blind with a reference needs a truth-derived oracle reference bundle")
    sel_rec = None if allow_blind else guard(set_name, step9, selection, selection_sha)
    if draws == "picks" and (selector is None or selector == "random"):
        raise SystemExit("--draws picks needs a deterministic --selector (RANDOM is an expectation: use all)")
    ref, extra = None, {}
    if reference is not None:
        ref = ref_info(reference)
        rb = ref["bundle"]
        if rb["set"] != set_name or rb["kind"] not in ("rasym", "cells"):
            raise SystemExit(f"reference {ref['rel']}: set {rb['set']} != {set_name}")
        lost = [r for r in bundles[0]["refcodes"] if r not in ref["row"]]
        if lost:
            raise SystemExit(f"reference {ref['rel']} lacks {len(lost)} of the candidates' crystals")
        extra = {"reference": ref["rel"], "reference_sha256": ref["sha256"], "reference_arm": ref["arm"],
                 "reference_order": ref["order"]}
    key = mkey(matcher, ref)
    items = C.load_set(set_name)
    want = wanted_draws(bundles, items, draws, selector, crystals, mode_tiebreak)
    tasks, dups, fhs, logs = [], {}, [], []
    t_start = time.time()
    n_arm = n_ref = 0
    for k, (bp, b) in enumerate(zip(bundle_paths, bundles)):
        _meta_check(bp, key, {"selection": sel_rec, "step9": step9, "slow_s": slow, **extra})
        H.repair_jsonl(match_path(bp, key))
        done = load_match(bp, key)
        fh = open(match_path(bp, key), "a")
        fhs.append(fh)
        logs.append(open(match_path(bp, key)[:-len(".jsonl")] + ".events.jsonl", "a"))
        first = {}
        for i, j in want[k]:
            if (i, j) in done:
                continue
            if ref is not None:
                ir = ref["row"][b["refcodes"][i]]
                if not ref_ok(ref, ir):                # §3.3 rule (iii): no reference minimum -> non-match
                    fh.write(json.dumps({"i": i, "j": j, "ok": False, "rms": None, "status": "ref_invalid"}) + "\n")
                    n_ref += 1
                    continue
            p = payload(b, i, j)
            if not bool(b["valid"][i, j]) or not payload_finite(p):
                fh.write(json.dumps({"i": i, "j": j, "ok": False, "rms": None, "status": "arm_fail"}) + "\n")
                n_arm += 1
                continue
            h = payload_hash(p)
            if (i, h) in first:                       # identical candidate of the same crystal: fit once
                dups.setdefault((k, i, first[(i, h)]), []).append(j)
                continue
            first[(i, h)] = j
            if ref is None:
                tasks.append(((k, i, j), (set_name, i, b["kind"], p, matcher)))
            else:
                tasks.append(((k, i, j), (set_name, i, ref["bundle"]["kind"], payload(ref["bundle"], ir, 0),
                                          b["kind"], p, matcher, ref["order"])))
        fh.flush()
    if not quiet:
        print(f"{key}: {len(tasks)} fits to run ({sum(len(v) for v in dups.values())} duplicates, "
              f"{n_arm} arm failures" + (f", {n_ref} reference-invalid rows" if ref else "")
              + f") over {len(bundles)} bundle(s) of {set_name}", flush=True)

    def log(ev):
        k = ev["key"][0] if isinstance(ev.get("key"), tuple) else 0
        ev = {**ev, "key": list(ev["key"]) if isinstance(ev.get("key"), tuple) else ev.get("key")}
        logs[k].write(json.dumps({"t": round(time.time(), 1), **ev}) + "\n")
        logs[k].flush()

    n_det = 0
    fn = _fit_task if ref is None else _ref_fit_task
    for c, (tkey, status, res, info) in enumerate(H.run(fn, tasks, workers=workers, slow=slow,
                                                        init=_winit, log=log)):
        k, i, j = tkey
        row = {"i": i, "j": j, "sec": round(info.get("sec", 0.0), 3), "attempts": info.get("attempts", 1)}
        if status == "ok":
            row.update(ok=res["ok"], rms=res["rms"], status="ok")
        else:
            row.update(ok=False, rms=None, status="det_fail")   # deterministic harness failure = non-match
            n_det += 1
        if info.get("fails"):
            row["fails"] = info["fails"]
        rows = [row] + [{"i": i, "j": jj, "ok": row["ok"], "rms": row["rms"], "status": "dup", "dup_of": j}
                        for jj in dups.get((k, i, j), [])]
        for r in rows:
            fhs[k].write(json.dumps(r) + "\n")
        fhs[k].flush()
        if not quiet and (c + 1) % 200 == 0:
            print(f"  {c + 1}/{len(tasks)}  {time.time() - t_start:.0f}s", flush=True)
    for k in range(len(bundles)):
        logs[k].write(json.dumps({"t": round(time.time(), 1), "event": "run_done", "fits": len(tasks),
                                  "det_fail": n_det, "sec": round(time.time() - t_start, 1),
                                  "cmd": sys.argv}) + "\n")
        fhs[k].close()
        logs[k].close()
    if not quiet:
        print(f"{key}: done in {time.time() - t_start:.0f}s; deterministic failures {n_det}", flush=True)


# ============================================================================== per-crystal outcomes
class Missing(Exception):
    pass


def _ok(res, i, j):
    r = res.get((i, j))
    if r is None:
        raise Missing((i, j))
    return bool(r["ok"])


def arm_outcomes(bundles, results, items, selector, mode_tiebreak=None, need_draws=True, ensemble=True,
                 cache=None):
    """Per crystal: picked@1 per seed (RANDOM = expectation), draw@1 per seed, any@S per seed, ensemble
    picked@1, any over all draws, and the picked indices. Missing match rows raise Missing. `cache` (dict)
    keeps the per-crystal draw coordinates / Chamfer matrices across selectors of the same bundles."""
    n = len(bundles[0]["refcodes"])
    m = len(bundles)
    out = {k: np.zeros((n, m)) for k in ("picked", "draw1", "any")}
    out["ens"] = np.zeros(n)
    out["any_all"] = np.zeros(n)
    out["pick"] = [[None] * m for _ in range(n)]
    out["ens_pick"] = [None] * n
    miss = []
    out["miss_refcodes_per_seed"] = [[bundles[0]["refcodes"][i] for i in range(n) if not bool(b["valid"][i].any())]
                                     for b in bundles]
    for i in range(n):
        if cache is None:
            cr = Crystal(bundles, items, i, mode_tiebreak)
        else:
            if i not in cache:
                cache[i] = Crystal(bundles, items, i, mode_tiebreak)
            cr = cache[i]
        if need_draws or selector == "random":
            for k, b in enumerate(bundles):
                oks = [_ok(results[k], i, j) for j in range(cr.S[k])]
                out["draw1"][i, k] = sum(oks) / cr.S[k]
                out["any"][i, k] = float(any(oks))
            out["any_all"][i] = float(out["any"][i].max())
        if not any(any(v) for v in cr.valid):
            miss.append(bundles[0]["refcodes"][i])
        if selector == "random":
            out["picked"][i] = out["draw1"][i]
            out["ens"][i] = float(np.sum([out["draw1"][i, k] * cr.S[k] for k in range(m)]) / sum(cr.S))
            continue
        for k in range(m):
            j = cr.pick(selector, k)
            out["pick"][i][k] = j
            out["picked"][i, k] = 0.0 if j is None else float(_ok(results[k], i, j))
        if not ensemble:
            continue
        kj = cr.pick(selector) if m > 1 else (None if out["pick"][i][0] is None else (0, out["pick"][i][0]))
        out["ens_pick"][i] = kj
        out["ens"][i] = 0.0 if kj is None else float(_ok(results[kj[0]], i, kj[1]))
    out["miss_refcodes"] = miss
    return out


# ============================================================================== select (Stage A / B)
METRIC = re.compile(r"^(hits|picked|draw1|any(\d+))(?:@(primary|mcf\d\d))?$")


def metric_spec(key, matcher):
    """Metric key -> (kind 'picked' | 'draw1' | 'any', K or None, matcher), or None for a non-metric key.
    'hits' = 'picked'; 'any16' = any draw (S = 16); 'any<K>' = any of the FIRST K draws."""
    m = METRIC.match(key)
    if m is None:
        return None
    kind = "picked" if m.group(1) in ("hits", "picked") else "draw1" if m.group(1) == "draw1" else "any"
    K = int(m.group(2)) if m.group(2) and m.group(2) != "16" else None
    return kind, K, m.group(3) or matcher


def cell_metrics(bundles, load, items, selector, keys, matcher, mtb=None, cache=None, soft=()):
    """Seed-summed value of every metric key for one cell (its seed bundles) under `selector`; a picked key also
    gives '<key>_per_seed'. load(k, m) -> match results of bundle k under matcher m. A key only in `soft` whose
    match rows are missing is None; a missing row of any other key stops the run."""
    out = {}
    n = len(bundles[0]["refcodes"])
    for key in list(keys) + [k for k in soft if k not in keys]:
        kind, K, m = metric_spec(key, matcher)
        res = [load(k, m) for k in range(len(bundles))]
        try:
            if kind == "picked":
                if selector is None:
                    raise SystemExit(f"metric '{key}' needs a selector (menu or stage_b selector)")
                o = arm_outcomes(bundles, res, items, selector, mtb, need_draws=False, ensemble=False, cache=cache)
                out[key] = float(o["picked"].sum())
                out[key + "_per_seed"] = o["picked"].sum(0).tolist()
                continue
            tot = 0.0
            for k, b in enumerate(bundles):
                S = int(b["valid"].shape[1])
                J = range(S if K is None else min(K, S))
                for i in range(n):
                    oks = [_ok(res[k], i, j) for j in J]
                    tot += sum(oks) / S if kind == "draw1" else float(any(oks))
            out[key] = tot
        except Missing as e:
            if key not in keys:
                out[key] = None
                continue
            raise SystemExit(f"metric '{key}': no {m} match row for (crystal, draw) {e.args[0]} of a bundle of this "
                             f"cell; match every draw the metric needs first")
    return out


def _load_arm(paths, matcher=None, ref=None):
    """Seed bundles of one select cell and a loader load(k, matcher) of their match rows (with a reference
    bundle: the rows matched against it). Rows made from an older bundle or another reference stop the run."""
    paths = [rel(p) for p in paths]
    bundles = [C.load_bundle(p) for p in paths]
    for b in bundles:
        if b["set"] not in ("val", "valsel", "train"):
            raise SystemExit(f"select runs on VALSEL only (got a '{b['set']}' bundle)")
    memo, shas = {}, {}

    def load(k, m):
        key = mkey(m, ref)
        if (k, key) not in memo:
            if k not in shas:
                shas[k] = C.sha256(paths[k])
            rows_current(paths[k], key, ref, shas[k])
            memo[(k, key)] = load_match(paths[k], key)
        return memo[(k, key)]
    return paths, bundles, load


def _rank(rows, tie):
    """Sort candidate rows best-first by the tie list (metric keys higher-better, val_loss lower-better with
    non-finite last, prefer_cell, selector preference; then listed order); returns (rows, deciding key)."""
    def key(r):
        out = []
        for t in tie:
            if t == "val_loss":
                out.append(_fin(r[t]))
            elif t == "prefer_cell":
                out.append(0 if r.get("preferred") else 1)
            elif t == "selector":
                out.append(sel_pref(r["selector"]))
            elif METRIC.match(t):
                out.append(-round(r[t], 9))
            else:
                raise ValueError(t)
        return tuple(out) + (r["order"],)
    srt = sorted(rows, key=key)
    decider = None
    if len(srt) > 1:
        a, b = key(srt[0]), key(srt[1])
        for t, u, v in zip(tie, a, b):
            if u != v:
                decider = t
                break
        decider = decider or "listed order"
    return srt, decider


def _split_keys(tie, matcher):
    picked = [t for t in tie if metric_spec(t, matcher) and metric_spec(t, matcher)[0] == "picked"]
    draws = [t for t in tie if metric_spec(t, matcher) and metric_spec(t, matcher)[0] != "picked"]
    return picked, draws


def stage_a(cells, A, matcher, menu, mtb=None):
    """Stage A: cells x menu, pooled over each cell's seed bundles. cells = [(label, bundles, load, items)].
    menu None/[] = cells only (no selector). Returns the record dict."""
    tie = [A.get("rank", "hits")] + A.get("tie", ["selector"] if menu else [])
    picked, draws = _split_keys(tie, matcher)
    if not menu and (picked or "selector" in tie):
        raise SystemExit(f"stage_a without a menu cannot rank by {picked + ['selector']}")
    rows = []
    for label, bundles, load, items in cells:
        cache = {}
        fixed = cell_metrics(bundles, load, items, None, draws, matcher, mtb, cache, soft=["draw1", "any16"])
        for s in (menu or [None]):
            row = {"cell": label, "selector": s, "n": len(items), "preferred": label == A.get("prefer_cell"),
                   "order": len(rows), **fixed}
            if s is not None:
                row.update(cell_metrics(bundles, load, items, s, picked, matcher, mtb, cache, soft=["hits"]))
            rows.append(row)
    srt, decider = _rank(rows, tie)
    rank = tie[0]
    rec = {"tie": tie, "table": rows, "chosen": {"cell": srt[0]["cell"], "selector": srt[0]["selector"]},
           "decided_by": decider}
    print(f"Stage A: {srt[0]['cell']} / {srt[0]['selector']}  {rank} {srt[0][rank]:.3f}  (decided by {decider})")
    for r in sorted(rows, key=lambda r: -r[rank])[:12]:
        per = r.get(rank + "_per_seed")
        print(f"   {r['cell']:>12s} {str(r['selector']):>9s}  {rank} {r[rank]:7.3f}"
              + (f"  per seed {[round(x, 3) for x in per]}" if per else "")
              + "".join(f"  {t} {r[t]:.3f}" for t in ("draw1", "any16") if r.get(t) is not None))
    return rec


def stage_b(seeds, B, matcher, sel, mtb=None):
    """Stage B: one checkpoint per seed. seeds = {seed: [(label, bundle_path, bundle, load, items, val_loss)]}."""
    tie = [B.get("rank", "picked")] + B.get("tie", ["draw1", "any16", "val_loss"])
    picked, draws = _split_keys(tie, matcher)
    if picked and sel is None:
        raise SystemExit(f"stage_b ranks by {picked}: it needs a selector")
    rec = {"selector": sel, "tie": tie, "seeds": {}}
    for seed, cands in seeds.items():
        rows = []
        for order, (label, path, b, load, items, vl) in enumerate(cands):
            cache = {}
            row = {"label": label, "bundle": path, "val_loss": float(vl), "order": order}
            row.update(cell_metrics([b], load, items, None, draws, matcher, mtb, cache, soft=["draw1", "any16"]))
            if sel is not None:
                row.update(cell_metrics([b], load, items, sel, picked, matcher, mtb, cache, soft=["picked"]))
            rows.append(row)
        srt, decider = _rank(rows, tie)
        rec["seeds"][seed] = {"table": rows, "chosen": srt[0]["label"], "decided_by": decider}
        print(f"Stage B seed {seed}: {srt[0]['label']}  {tie[0]} {srt[0][tie[0]]:.3f}  (decided by {decider})")
    return rec


def cmd_select(args):
    cfg = json.load(open(args.config))
    matcher = cfg.get("matcher", "primary")
    mtb = cfg.get("mode_tiebreak")
    rec = {"name": cfg.get("name"), "matcher": matcher, "config": cfg, "bundles_sha256": {}}
    chosen_sel = None
    ref = None
    if cfg.get("reference"):                          # §3.3 BASIN forms: rows matched against a reference bundle
        ref = ref_info(cfg["reference"])
        rec["reference"] = {"path": ref["rel"], "sha256": ref["sha256"], "arm": ref["arm"], "order": ref["order"]}

    def sha(paths):
        rec["bundles_sha256"].update({os.path.relpath(p, C.REPO).replace("\\", "/"): C.sha256(p) for p in paths})
    if "stage_a" in cfg:
        A = cfg["stage_a"]
        menu = A["menu"] if "menu" in A else cfg.get("menu", MENU)
        rec["menu"] = menu
        cells = []
        for cell in A["cells"]:
            paths, bundles, load = _load_arm(cell["bundles"], ref=ref)
            sha(paths)
            cells.append((cell["label"], bundles, load, C.load_set(bundles[0]["set"])))
        rec["stage_a"] = stage_a(cells, A, matcher, menu, mtb)
        chosen_sel = rec["stage_a"]["chosen"]["selector"]
    if "stage_b" in cfg:
        B = cfg["stage_b"]
        sel = B.get("selector", "from_stage_a")
        sel = chosen_sel if sel == "from_stage_a" else sel
        seeds = {}
        for seed, cands in B["seeds"].items():
            seeds[seed] = []
            for c in cands:
                paths, bundles, load = _load_arm([c["bundle"]], ref=ref)
                sha(paths)
                seeds[seed].append((c["label"], c["bundle"], bundles[0], load, C.load_set(bundles[0]["set"]),
                                    c.get("val_loss", float("nan"))))
        rec["stage_b"] = stage_b(seeds, B, matcher, sel, mtb)
    out = cfg.get("out")
    if out:
        os.makedirs(os.path.dirname(rel(out)), exist_ok=True)
        json.dump(rec, open(rel(out), "w"), indent=1)
        print(f"wrote {out}")
    return rec


# ============================================================================== statistics / report
def _pp(x):
    return None if x is None else round(100.0 * float(x), 3)


def contrast(h_a, h_b, n_boot=10000, test=True):
    """Paired crystal-level contrast of two [n, m] outcome arrays (§2.4). test=False (every descriptive use:
    Table 1 Δ column, subsets, ii-S terms, strata) gives the estimate and CIs without a p-value."""
    import n3_stats as S
    d = np.asarray(h_a, float).mean(1) - np.asarray(h_b, float).mean(1)
    _, lo, hi = S.bca_ci(d, n_boot=n_boot, seed=0)
    ts = S.two_stage_ci(h_a, h_b, n_boot=n_boot, seed=0)
    out = {"n": int(len(d)), "delta_pp": _pp(d.mean()), "bca_pp": [_pp(lo), _pp(hi)],
           "two_stage_pp": [_pp(ts[0]), _pp(ts[1])], "nonzero": int((np.abs(d) > 1e-12).sum())}
    if test:
        out["p"] = float(S.signflip_p(d))
    return out


def _ci_txt(c):
    return f"95% CI [{c['bca_pp'][0]:.2f}, {c['bca_pp'][1]:.2f}] pp"


def family(h, ours, spec, h2_arrays, n_boot):
    """Primary-style family {H1, H2, H3} for `ours` (OURS or OURS-N2): exact sign-flip, Holm, verdicts."""
    import n3_stats as S
    fam = {}
    r_name, g_name = spec["H1"]
    if r_name in h and g_name in h and ours in h:
        cr, cg = contrast(h[ours], h[r_name], n_boot), contrast(h[ours], h[g_name], n_boot)
        comp = cr if cr["delta_pp"] <= cg["delta_pp"] else cg
        fam["H1"] = {"components": {r_name: cr, g_name: cg}, "p": max(cr["p"], cg["p"]),
                     "delta_pp": min(cr["delta_pp"], cg["delta_pp"]), "bca_pp": comp["bca_pp"],
                     "delta_from": r_name if comp is cr else g_name}
    h2 = spec["H2"]
    if ours in h2_arrays and h2 in h2_arrays:
        fam["H2"] = {"comparator": h2, **contrast(h2_arrays[ours], h2_arrays[h2], n_boot)}
    h3 = spec["H3"]
    if ours in h and h3 in h:
        fam["H3"] = {"comparator": h3, **contrast(h[ours], h[h3], n_boot)}
    complete = all(k in fam for k in ("H1", "H2", "H3"))
    keys = ["H1", "H2", "H3"]
    if complete:
        adj = S.holm([fam[k]["p"] for k in keys])
        for k, a in zip(keys, adj):
            fam[k]["p_holm"] = float(a)
    for k in keys:
        if k not in fam:
            continue
        f = fam[k]
        if not complete:
            f["verdict"] = "family incomplete (no Holm verdict)"
            continue
        pa = f["p_holm"]
        n = next(iter(f["components"].values()))["n"] if k == "H1" else f["n"]
        if k == "H1":
            cr, cg = f["components"][r_name], f["components"][g_name]
            v = S.verdict_h1(cr["delta_pp"], cg["delta_pp"], pa)
            if v == "split":
                win, lose = (r_name, g_name) if cr["delta_pp"] > 0 else (g_name, r_name)
                v = f"split: OURS better than {win}; {lose} better than OURS"
            elif v == "not resolved":
                v = (f"not resolved at n = {n} ({r_name}: {cr['delta_pp']:.2f} pp, {_ci_txt(cr)}; "
                     f"{g_name}: {cg['delta_pp']:.2f} pp, {_ci_txt(cg)})")
            f["verdict"] = v
            f["supported"] = v == "OURS better"
            ts_zero = any(c["two_stage_pp"][0] <= 0 <= c["two_stage_pp"][1] for c in (cr, cg))
        else:
            v = S.verdict(f["delta_pp"], pa)
            if v == "ARM better":
                v = f"{f['comparator']} better"
            elif v == "not resolved":
                v = f"difference not resolved at n = {n} ({_ci_txt(f)})"
            f["verdict"] = v
            f["supported"] = v == "OURS better"
            ts_zero = f["two_stage_pp"][0] <= 0 <= f["two_stage_pp"][1]
        if f["supported"] and ts_zero:
            f["seed_caveat"] = "for the trained models; not shown to exceed seed-to-seed training variation"
    fam["complete"] = complete
    return fam


def _read_labels(paths):
    lab = {}
    for p in paths or []:
        d = json.load(open(rel(p)))
        rows = d if isinstance(d, list) else [{"refcode": k, **(v if isinstance(v, dict) else {"label": v})}
                                                for k, v in d.items()]
        for r in rows:
            lab.setdefault(r["refcode"], {}).update({k: v for k, v in r.items() if k != "refcode"})
    return lab


def _group(key, val):
    if key == "K":
        return "<=2" if val <= 2 else "3-4" if val <= 4 else ">=6"
    if key == "z_mcf":
        return "1" if val == 1 else "2" if val == 2 else ">=3"
    return str(val)


def strata_table(h, arms, refcodes, items, labels, keys, h2_pair):
    import n3_stats as S
    rows = {}
    for key in keys:
        groups = {}
        for i, (ref, a) in enumerate(zip(refcodes, items)):
            if key == "K":
                v = int(a["K"])
            else:
                v = labels.get(ref, {})
                if key.startswith("sensitivity."):        # sensitivity.<label, may contain dots>.<field>
                    name, field = key[len("sensitivity."):].rsplit(".", 1)
                    v = ((v.get("sensitivity") or {}).get(name) or {}).get(field)
                else:
                    v = v.get(key)
                if v is None:
                    continue
            groups.setdefault(_group(key, v), []).append(i)
        if not groups:
            continue
        rows[key] = {}
        for g, idx in sorted(groups.items()):
            n = len(idx)
            ent = {"n": n, "too_small": n < 30, "arms": {}}
            for name, o in arms.items():
                if key == "ff_label" and name not in h2_pair:
                    continue
                per = [float(o["picked"][idx, s].sum()) for s in range(o["picked"].shape[1])]
                ens = float(o["ens"][idx].sum())
                ent["arms"][name] = {"k_per_seed": per, "wilson_per_seed": [S.wilson(k, n) for k in per],
                                     "k_ens": ens, "wilson_ens": S.wilson(ens, n)}
            rows[key][g] = ent
    return rows


def _geodesic(items, o, bundles):
    """Picked R_asym vs R0, minimised over the molecule's proper point group (PointGroupAnalyzer tol 0.3)."""
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    out = []
    for i, a in enumerate(items):
        mol = Molecule([int(z) for z in a["Z"]], a["local"].numpy())
        ops = [np.asarray(op.rotation_matrix) for op in
               PointGroupAnalyzer(mol, tolerance=0.3).get_symmetry_operations()]
        ops = [g for g in ops if np.linalg.det(g) > 0] or [np.eye(3)]
        R0 = a["R0"].numpy()
        row = []
        for k, j in enumerate(o["pick"][i]):
            if j is None:
                row.append(float("nan"))
                continue
            R = bundles[k]["R"][i, j].numpy()
            row.append(min(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R.T @ R0 @ g) - 1) / 2))))
                           for g in ops))
        out.append(row)
    return np.array(out)


def _q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {k: round(float(np.quantile(x, q)), 3) for k, q in (("q10", .1), ("q25", .25), ("median", .5),
                                                                 ("q75", .75), ("q90", .9))}


def _basin_jsons(spec, refcodes, ref_arms, what):
    """{name: h [n, m]} from `basin` outputs (config {name: path}); all against ONE reference of an allowed arm.
    Returns (arrays, reference summary incl. the rule-(iii) and unconverged-reference lists)."""
    arr, refs = {}, {}
    for name, p in (spec or {}).items():
        d = json.load(open(rel(p)))
        if d["refcodes"] != refcodes:
            raise SystemExit(f"{what} {name}: refcodes not in the frozen order of the set")
        ra = d.get("reference_arm") or ""
        if ra not in ref_arms and not ("oracle" in ref_arms and ra.startswith("oracle")):
            raise SystemExit(f"{what} {name}: reference arm {d.get('reference_arm')} is not one of {ref_arms}")
        arr[name] = np.asarray(d["h"], float)
        refs[d["reference_sha256"]] = {"reference": d["reference"], "sha256": d["reference_sha256"],
                                       "arm": d["reference_arm"], "order": d.get("reference_order"),
                                       "match_rule": d.get("match_rule"),
                                       "ref_invalid_refcodes": d.get("ref_invalid_refcodes"),
                                       "ref_unconverged_refcodes": d.get("ref_unconverged_refcodes")}
    if len(refs) > 1:
        raise SystemExit(f"{what}: the arms were matched against different references {sorted(refs)}")
    return arr, (next(iter(refs.values())) if refs else None)


def cmd_report(args):
    cfg = json.load(open(args.config))
    set_name = cfg["set"]
    sel_rec = guard(set_name, args.step9, args.selection, getattr(args, "selection_sha256", None))         if set_name in BLIND else None
    matcher = cfg.get("matcher", "primary")
    n_boot = int(cfg.get("n_boot", 10000))
    items = C.load_set(set_name)
    refcodes = [a["refcode"] for a in items]
    arms, h, table1, bundles_of = {}, {}, {}, {}
    missing = {}
    for name, spec in cfg["arms"].items():
        paths = [rel(p) for p in spec["bundles"]]
        bundles = [C.load_bundle(p) for p in paths]
        assert all(b["set"] == set_name and b["refcodes"] == refcodes for b in bundles), name
        bundles_of[name] = (paths, bundles)
        results = [load_match(p, matcher) for p in paths]
        try:
            o = arm_outcomes(bundles, results, items, spec["selector"], spec.get("mode_tiebreak"))
        except Missing as e:
            missing[name] = f"no {matcher} result for (crystal, draw) {e.args[0]} (and possibly more)"
            continue
        arms[name], h[name] = o, o["picked"]
        m = len(bundles)
        per = o["picked"].mean(0)
        det = [[b["refcodes"][i], k, j] for k, (b, r) in enumerate(zip(bundles, results))
               for (i, j), row in sorted(r.items()) if row.get("status") == "det_fail"]
        ent = {"m": m, "selector": spec["selector"], "picked1_per_seed_pct": [_pp(x) for x in per],
               "picked1_mean_pct": _pp(per.mean()), "picked1_sd_pct": _pp(per.std(ddof=1)) if m > 1 else None,
               "draw1_per_seed_pct": [_pp(x) for x in o["draw1"].mean(0)], "draw1_mean_pct": _pp(o["draw1"].mean()),
               "any16_per_seed_pct": [_pp(x) for x in o["any"].mean(0)], "any16_mean_pct": _pp(o["any"].mean()),
               "ensemble_picked1_pct": _pp(o["ens"].mean()), "any_all_pct": _pp(o["any_all"].mean()),
               "ensemble_label": f"{m}-model ensemble, {sum(int(b['valid'].shape[1]) for b in bundles)} draws",
               "arm_miss_refcodes": o["miss_refcodes"], "arm_miss_refcodes_per_seed": o["miss_refcodes_per_seed"],
               "det_fail_fits": len(det), "det_fail_list": det,
               "arm_fail_draws": sum(1 for r in results for row in r.values() if row.get("status") == "arm_fail"),
               "energy_det_fail_refcodes_per_seed": [b["meta"].get("energy_det_fail_refcodes") for b in bundles]}
        sec = [x for b in bundles for x in (b["meta"].get("sec_per_crystal") or []) if x is not None]
        ent["compute"] = {"sec_per_crystal_per_seed_median": round(float(np.median(sec)), 2) if sec else None,
                          "lj_evals_per_crystal": bundles[0]["meta"].get("n_lj_per_crystal"),
                          "ff_evals_per_crystal": bundles[0]["meta"].get("n_ff_per_crystal")}
        sec_blocks = {}
        for sm in cfg.get("secondary", ["mcf05", "mcf08"]) + [cfg.get("rms_matcher", "mcf10")]:
            rs = [load_match(p, sm) for p in paths]
            try:
                so = arm_outcomes(bundles, rs, items, spec["selector"], spec.get("mode_tiebreak"),
                                  need_draws=spec["selector"] == "random")
                sec_blocks[sm] = {"picked1_per_seed_pct": [_pp(x) for x in so["picked"].mean(0)],
                                  "ensemble_picked1_pct": _pp(so["ens"].mean())}
                if spec["selector"] != "random" and sm == cfg.get("rms_matcher", "mcf10"):
                    rms = np.array([[(rs[k].get((i, j)) or {}).get("rms") if j is not None else None
                                     for k, j in enumerate(o["pick"][i])] for i in range(len(items))], dtype=float)
                    rms[~np.isfinite(rms)] = np.inf
                    grid = [round(0.05 * t, 2) for t in range(21)]
                    sec_blocks[sm]["rms_ecdf"] = {"x": grid, "per_seed": [[round(float((rms[:, k] <= x).mean()), 4)
                                                                           for x in grid] for k in range(m)]}
            except Missing:
                sec_blocks[sm] = "missing"
        ent["secondary"] = sec_blocks
        if bundles[0]["kind"] == "rasym" and spec["selector"] != "random":
            geo = _geodesic(items, o, bundles)
            ent["geodesic_deg"] = {"per_seed": [_q(geo[:, k]) for k in range(m)], "pooled": _q(geo)}
        if "representable" in spec:
            ent["representable"] = spec["representable"]
        table1[name] = ent
    rep = {"set": set_name, "n": len(items), "matcher": matcher, "selection": sel_rec,
           "h2_form": cfg.get("h2_form", "EXACT"), "table1": table1, "missing_arms": missing}
    spec = {"H1": cfg.get("H1", ["MCF-R", "MCF-R+G"]), "H2": cfg.get("H2", "iii-s"), "H3": cfg.get("H3", "iv-P")}
    ours, ours_n2 = cfg.get("ours", "OURS"), cfg.get("ours_n2", "OURS-N2")
    dry = bool(cfg.get("dry_run"))
    if dry and set_name not in ("val", "valsel", "train"):
        raise SystemExit("dry_run (families on a non-confirmatory set) is for VAL / VALSEL / TRAIN only")
    if dry:
        rep["dry_run"] = "DRY RUN on a non-confirmatory set: a code-path check, never a result"
    ref_ok_arms = ("fftruth", "oracle") if dry else ("fftruth",)
    basin_arr, basin_ref = _basin_jsons(cfg.get("h2_basin"), refcodes, ref_ok_arms, "h2_basin")
    for name, arr in basin_arr.items():                  # the H2 contrast pair in BASIN form (§3.3), both forms run
        table1.setdefault(name, {}).update(h2_basin_picked1_pct=_pp(arr.mean()),
                                           h2_basin_picked1_per_seed_pct=[_pp(x) for x in arr.mean(0)])
    if basin_ref:
        rep["h2_basin_reference"] = basin_ref
    h2_arr = basin_arr if rep["h2_form"] == "BASIN" else dict(h)
    for a in h:                                           # Table 1 "Delta vs OURS" column: estimate + BCa CI and
        if a != ours and ours in h:                       # the two-stage interval; tests live in the families only
            c = contrast(h[ours], h[a], n_boot, test=False)
            if rep["h2_form"] == "BASIN" and a == spec["H2"]:
                c["form"] = "EXACT (SI); H2 is tested in BASIN form (family H2, h2_basin_delta)"
            table1[a]["delta_ours_minus_arm"] = c
    if ours in basin_arr and spec["H2"] in basin_arr:     # BASIN form: the H2 column; EXACT form: SI (other form)
        table1.setdefault(spec["H2"], {})["h2_basin_delta"] = {
            "form": "BASIN" if rep["h2_form"] == "BASIN" else "BASIN (SI; H2 is tested in EXACT form)",
            **contrast(basin_arr[ours], basin_arr[spec["H2"]], n_boot, test=False)}
    ie = cfg.get("i_e", "i-E")                            # §3.3 i-E: vs iii-s with a CI, in the H2 FORM
    if ie in h2_arr and spec["H2"] in h2_arr:
        rep["i_e_vs_h2"] = {"form": rep["h2_form"], "arm": ie, "comparator": spec["H2"],
                            **contrast(h2_arr[ie], h2_arr[spec["H2"]], n_boot, test=False)}
    p1_arr, p1_ref = _basin_jsons(cfg.get("steric_basin"), refcodes, ("p1ref", "oracle") if dry else ("p1ref",),
                                  "steric_basin")
    if p1_arr:                                            # §3.3 steric BASIN P1 (continuity with G2; descriptive)
        names = list(p1_arr)
        first = cfg.get("steric_basin_ours", names[0])
        rep["steric_basin"] = {"reference": p1_ref, "arms": {k: {
            "picked1_per_seed_pct": [_pp(x) for x in v.mean(0)], "picked1_mean_pct": _pp(v.mean()),
            "selector": json.load(open(rel(cfg["steric_basin"][k]))).get("selector")} for k, v in p1_arr.items()},
            "contrasts": {f"{first} - {k}": contrast(p1_arr[first], p1_arr[k], n_boot, test=False)
                          for k in names if k != first},
            "g2_devtest_context": "G2 DEV-TEST: basin_best 10.1/8.5/7.0% vs Haar 6.5%, p = 0.19/0.54/1.0; "
                                  "basin_any 22.6/19.6/17.6% vs 11.1% (printed beside it, §3.3)"}
    if set_name == "testB" or dry:
        rep["family_primary"] = family(h, ours, spec, h2_arr, n_boot)
        rep["family_n2"] = family(h, ours_n2, spec, h2_arr, n_boot)
        rep["wording"] = wording(rep, cfg, h, h2_arr, items, refcodes, spec, ours, n_boot)
    labels = _read_labels(cfg.get("labels"))
    for k, v in _read_labels([cfg["ff_labels"]] if cfg.get("ff_labels") else []).items():
        labels.setdefault(k, {})["ff_label"] = v.get("ff_label", v.get("label"))
    sens = sorted({f"sensitivity.{s}.{f}" for v in labels.values() for s, d in (v.get("sensitivity") or {}).items()
                   for f in ("chiral", "mixed", "improper_achiral") if f in (d or {})})
    keys = cfg.get("strata") or ["K"] + [k for k in ("z_mcf", "class", "chiral", "mixed", "improper_achiral",
                                                    "legacy_unrep", "formula_disjoint", "omc25_nonoverlap",
                                                    "ff_label") if any(k in v for v in labels.values())] + sens
    rep["table2"] = strata_table(h, arms, refcodes, items, labels, keys, {ours, ours_n2, spec["H2"]})
    idx = [i for i, r in enumerate(refcodes) if labels.get(r, {}).get("omc25_nonoverlap") is True]
    if idx and spec["H2"] in h2_arr and (ours in h2_arr or ours_n2 in h2_arr):   # §3.3: descriptive H2 there
        rep["h2_omc25_nonoverlap"] = {"form": rep["h2_form"], "n": len(idx), **{
            f"{o} - {spec['H2']}": contrast(h2_arr[o][idx], h2_arr[spec["H2"]][idx], n_boot, test=False)
            for o in (ours, ours_n2) if o in h2_arr and spec["H2"] in h2_arr}}
    if "picked1_mean_pct" in table1.get(ours, {}):      # (a BASIN-only entry exists when OURS' rows are missing)
        rep["abstract_inputs"] = {"ours_picked1_mean_pct": table1[ours]["picked1_mean_pct"],
                                  "ours_picked1_sd_pct": table1[ours]["picked1_sd_pct"],
                                  "h2_basin_sentence": rep["h2_form"] == "BASIN",
                                  "gate_ca_clause": bool(cfg.get("gate_ca_failed"))}
        for k in ("H1", "H2", "H3"):
            f = rep.get("family_primary", {}).get(k)
            if f:
                rep["abstract_inputs"][k] = {"verdict": f.get("verdict"), "delta_pp": f["delta_pp"],
                                             "bca_pp": f["bca_pp"]}
    out = rel(cfg["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1, default=float)
    print(f"wrote {out}")
    for name, e in table1.items():
        if "picked1_mean_pct" in e:
            print(f"  {name:>9s}: picked@1 {e['picked1_mean_pct']:6.2f}% (seeds {e['picked1_per_seed_pct']})  "
                  f"draw@1 {e['draw1_mean_pct']:.2f}%  any@16 {e['any16_mean_pct']:.2f}%")
    for fam in ("family_primary", "family_n2"):
        for k in ("H1", "H2", "H3"):
            f = rep.get(fam, {}).get(k)
            if f:
                print(f"  {fam} {k}: delta {f['delta_pp']:.2f} pp  p {f['p']:.4g}  "
                      f"p_holm {f.get('p_holm', float('nan')):.4g}  -> {f['verdict']}")
    if missing:
        print("  missing arms:", missing)
    return rep


def wording(rep, cfg, h, h2_arr, items, refcodes, spec, ours, n_boot):
    """§2.4 / §4 wording consequences that depend only on the numbers (the text itself is written by hand)."""
    w = {}
    P, N = rep["family_primary"], rep["family_n2"]
    for k in ("H1", "H2", "H3"):
        if k in P and k in N and "supported" in P[k] and "supported" in N[k]:
            if P[k]["supported"] and not N[k]["supported"]:
                w[k + "_n2"] = "holds for the model of record, not for the 30-epoch retrain"
            if N[k]["supported"] and not P[k]["supported"]:
                w[k + "_n2"] = "holds only for OURS-N2: not claimed"
    if cfg.get("gate_ca_failed") and "H1" in P:
        w["H1_harness_clause"] = "every H1 sentence carries the §3.2 Gate C-a harness clause"
    ff = {k: v.get("ff_label", v.get("label")) for k, v in _read_labels([cfg["ff_labels"]]).items()} \
        if cfg.get("ff_labels") else {}
    h2 = spec["H2"]
    if ff and ours in h2_arr and h2 in h2_arr:
        sub = {}
        for lab in ("ANCHORED", "DRIFTED", "UNRESOLVED"):
            idx = [i for i, r in enumerate(refcodes) if ff.get(r) == lab]
            if idx:
                sub[lab] = contrast(h2_arr[ours][idx], h2_arr[h2][idx], n_boot, test=False)
        w["H2_by_ff_label"] = sub
        if P.get("H2", {}).get("supported") and rep["h2_form"] == "EXACT" and "ANCHORED" in sub:
            lo, hi = sub["ANCHORED"]["bca_pp"]
            w["H2_claim"] = ("better than physics-based search" if lo > 0 else
                             "on FF*-held crystals the physics search reproduces more" if hi < 0 else
                             "OURS reproduces more experimental poses; on FF*-held crystals not resolved")
    if P.get("H1", {}).get("supported"):
        r_name, g_name = spec["H1"]
        ii_s = cfg.get("ii_s", "ii-S")
        if ii_s in h:
            a = contrast(h[ii_s], h[g_name], n_boot, test=False)     # §4: paired BCa CI, no p-value
            b = contrast(h[ours], h[ii_s], n_boot, test=False)
            w["ii_S_decomposition"] = {f"{ii_s} - {g_name}": a, f"{ours} - {ii_s}": b,
                                       "learner": ("adds" if b["bca_pp"][0] > 0 else "below" if b["bca_pp"][1] < 0
                                                   else "not resolved"),
                                       "expansion": "recovers" if a["bca_pp"][0] > 0 else "not attributed"}
        zl = _read_labels(cfg.get("labels"))
        idx = [i for i, r in enumerate(refcodes) if (zl.get(r, {}).get("z_mcf") or 0) >= 3]
        if idx:
            dr = 100 * (h[ours][idx].mean() - h[r_name][idx].mean())
            dg = 100 * (h[ours][idx].mean() - h[g_name][idx].mean())
            w["zmcf_ge3_min_delta_pp"] = round(min(dr, dg), 3)
            w["advantage_concentrates_in_zmcf_le2"] = min(dr, dg) <= 0
    return w


# ============================================================================== oracles (input-side)
def mcf_entry_to_cell(e):
    """One MCF data dict (export pickle entry, or a predictions sample with the same keys) -> a 'cells' draw:
    whole molecules R_k @ local_k + trans_k @ lattice (MCF's assemble_coords), species via MCF_IDX_TO_Z."""
    f = lambda v: np.asarray(v.detach().cpu().numpy() if torch.is_tensor(v) else v, dtype=np.float64)
    L = f(e["lattice_1"])
    R, t, loc = f(e["rotmats_1"]), f(e["trans_1"]), f(e["local_coords"])
    nb = np.asarray(e["bb_num_vec"]).astype(int).reshape(-1)
    off = np.concatenate([[0], np.cumsum(nb)])
    cart = np.concatenate([loc[off[k]:off[k + 1]] @ R[k].T + t[k] @ L for k in range(len(nb))])
    types = np.asarray(e["atom_types"]).astype(int).reshape(-1)
    species = np.array([MCF_IDX_TO_Z[int(x)] for x in types])
    mol = np.repeat(np.arange(len(nb)), nb)
    return {"lattice": torch.tensor(L), "cart": torch.tensor(cart), "species": torch.tensor(species),
            "mol": torch.tensor(mol)}


def _oracle_summary(bundle_paths, matchers, items):
    summ = {}
    for sm in matchers:
        b = C.load_bundle(bundle_paths[0])
        o = arm_outcomes([b], [load_match(bundle_paths[0], sm)], items, "lj")
        summ[sm] = {"picked1": int(o["picked"].sum()), "draw1": float(o["draw1"].sum()),
                    "any16": int(o["any"].sum()), "n": len(items),
                    "fail_refcodes": [items[i]["refcode"] for i in range(len(items)) if o["draw1"][i, 0] < 1]}
    ok = all(v["picked1"] == v["n"] and v["any16"] == v["n"] and abs(v["draw1"] - v["n"]) < 1e-9
             for v in summ.values())
    return summ, ok


def _oracle_bundle(path, set_name, kind, draws, source, valid=None, meta=None):
    """Truth-derived (input-side) bundle; draws[i] = list of S per-crystal draws (R or cells). A bundle built
    from a different source is replaced and its stale match files removed (for a reference bundle: the match
    files of every bundle matched against it)."""
    if os.path.exists(path) and C.load_bundle(path)["meta"].get("source") == source:
        return
    stem = os.path.splitext(os.path.basename(path))[0]
    for f in os.listdir(ORACLE_DIR) if os.path.isdir(ORACLE_DIR) else []:
        if f"~{stem}." in f:
            os.remove(os.path.join(ORACLE_DIR, f))
    for f in os.listdir(ORACLE_DIR) if os.path.isdir(ORACLE_DIR) else []:
        if f.startswith(os.path.splitext(os.path.basename(path))[0] + "__"):
            os.remove(os.path.join(ORACLE_DIR, f))
    items = C.load_set(set_name)
    n, S = len(items), len(draws[0])
    assert len(draws) == n and all(len(d) == S for d in draws)
    kw = {"R": torch.stack([torch.stack(d) for d in draws])} if kind == "rasym" else {"cells": draws}
    C.save_bundle(path, arm="oracle", seed=-1, set_name=set_name, refcodes=[a["refcode"] for a in items],
                  kind=kind, valid=torch.ones(n, S, dtype=torch.bool) if valid is None else valid,
                  sel={"lj": torch.zeros(n, S, dtype=torch.float64)},
                  meta={"oracle": True, "source": source, "cmd": sys.argv, **(meta or {})}, **kw)


def cmd_oracle(args):
    items = C.load_set(args.set)
    path = os.path.join(ORACLE_DIR, f"oracle_rasym_{args.set}.pt")
    _oracle_bundle(path, args.set, "rasym", [[a["R0"].double()] * args.draws for a in items], "expand(a, R0)")
    for sm in args.matchers:
        run_match([path], sm, workers=args.workers, slow=args.slow, allow_blind=True)
    summ, ok = _oracle_summary([path], args.matchers, items)
    print(json.dumps({k: {kk: v[kk] for kk in ("picked1", "draw1", "any16", "n", "fail_refcodes")}
                      for k, v in summ.items()}), "\nORACLE", "PASS" if ok else "FAIL")
    if getattr(args, "reference", False):
        _oracle_reference(args, path, items)


def _oracle_reference(args, cand_path, items):
    """Reference-bundle oracle (§3.3 BASIN matching): the truth as a one-draw reference bundle, in both match
    orders, must give 100% for the truth candidates (expand(a, R0) x S) under every matcher; and with the
    reference of every 10th crystal marked invalid (no minimum), rule (iii) must give exactly those crystals
    h = 0 ('ref_invalid' rows, no fit) and every other crystal h = 1."""
    n = len(items)
    R0 = [[a["R0"].double()] for a in items]
    res, ok_all = {}, True
    for order in ("ref_cand", "cand_ref"):
        rp = os.path.join(ORACLE_DIR, f"oracleref_{order}_{args.set}.pt")
        _oracle_bundle(rp, args.set, "rasym", R0, f"truth-as-bundle expand(a, R0); order {order}",
                       meta={"ref_order": order})
        for sm in args.matchers:
            run_match([cand_path], sm, workers=args.workers, slow=args.slow, allow_blind=True, reference=rp)
            d = basin_outcomes([cand_path], rp, "lj", sm)
            ok = (d["picked1_mean_pct"] == 100.0 and all(x == 1.0 for row in d["any"] for x in row)
                  and all(x == 1.0 for row in d["draw1"] for x in row))
            res[f"{order}/{sm}"] = {"picked1_pct": d["picked1_mean_pct"], "ok": ok}
            ok_all &= ok
    valid = torch.ones(n, 1, dtype=torch.bool)
    valid[::10] = False
    rp = os.path.join(ORACLE_DIR, f"oracleref_iii_{args.set}.pt")
    _oracle_bundle(rp, args.set, "rasym", R0, "truth-as-bundle expand(a, R0); every 10th reference invalid",
                   valid=valid, meta={"ref_order": "ref_cand"})
    run_match([cand_path], "primary", workers=args.workers, slow=args.slow, allow_blind=True, reference=rp)
    d = basin_outcomes([cand_path], rp, "lj", "primary")
    rows = load_match(cand_path, mkey("primary", ref_info(rp)))
    want_h = [0.0 if i % 10 == 0 else 1.0 for i in range(n)]
    ok3 = ([row[0] for row in d["h"]] == want_h and d["ref_invalid_refcodes"] == [a["refcode"] for a in items[::10]]
           and all(r["status"] == "ref_invalid" for (i, j), r in rows.items() if i % 10 == 0)
           and all(r["status"] in ("ok", "dup") and r["ok"] for (i, j), r in rows.items() if i % 10 != 0))
    res["rule_iii/primary"] = {"picked1_pct": d["picked1_mean_pct"], "expected_pct": round(100 * sum(want_h) / n, 3),
                               "ref_invalid": len(d["ref_invalid_refcodes"]), "ok": ok3}
    ok_all &= ok3
    out = os.path.join(C.PRIVATE, f"oracle_reference_{args.set}.json")
    json.dump({"oracle": "reference-bundle matching (truth as reference, truth as candidate)", "set": args.set,
               "n": n, "pass": bool(ok_all), "results": res}, open(out, "w"), indent=1)
    print(json.dumps(res), "\nREFERENCE ORACLE", "PASS" if ok_all else "FAIL", "->", out)


def _close(a, b, atol):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return a.shape == b.shape and bool(torch.allclose(a, b, rtol=0.0, atol=atol))


def check_oracle_predictions(pred, entries, atol=1e-5, atol_cart=1e-3):
    """Gate F6 input check (§1.3 blinding): every draw of an MCF predictions dict must be the pickle's own pose,
    in the fields the adapter scores (lattices, atom_types, num_atoms, num_bbs, cart_coords) and in MCF's pose
    tensors (pred_rotmats, pred_trans). Returns None, or the first discrepancy as text."""
    ref = {"num_atoms": torch.tensor([int(torch.as_tensor(e["local_coords"]).shape[0]) for e in entries]),
           "num_bbs": torch.tensor([int(torch.as_tensor(e["bb_num_vec"]).reshape(-1).shape[0]) for e in entries]),
           "atom_types": torch.cat([torch.as_tensor(e["atom_types"]).long().reshape(-1) for e in entries]),
           "lattices": torch.stack([torch.as_tensor(e["lattice_1"]).double() for e in entries]),
           "pred_rotmats": torch.cat([torch.as_tensor(e["rotmats_1"]).double() for e in entries]),
           "pred_trans": torch.cat([torch.as_tensor(e["trans_1"]).double() for e in entries]),
           "cart_coords": torch.cat([mcf_entry_to_cell(e)["cart"] for e in entries])}
    missing = [k for k in ref if k not in pred]
    if missing:
        return f"no {missing} in the predictions file"
    for s in range(int(pred["num_atoms"].shape[0])):
        for k in ("num_atoms", "num_bbs", "atom_types"):
            if not torch.equal(torch.as_tensor(pred[k][s]).long().reshape(-1), ref[k]):
                return f"draw {s}: {k} differs from the pickle"
        for k, tol in (("lattices", atol), ("pred_rotmats", atol), ("pred_trans", atol), ("cart_coords", atol_cart)):
            if not _close(pred[k][s], ref[k], tol):
                return f"draw {s}: {k} is not the pickle's own pose (atol {tol})"
    return None


def cmd_gate_f6(args):
    """Gate F6 (§3.2). Default route (the protocol's): the export's predictions file whose every draw is the
    pickle's own pose (n3_mcf_export.oracle_predictions), split into 'cells' draws with the export unit's
    predictions_cells. --pickle decodes the pickle entries with this module's own mcf_entry_to_cell instead
    (independent cross-check of the decode). Either way the pickle must be the set's export (sidecar SHA-256)
    and a predictions file must pass check_oracle_predictions before anything is matched."""
    import n3_mcf_export as X
    assert X.MCF_IDX_TO_Z == MCF_IDX_TO_Z, "export and scorer inverse type maps differ"
    items = C.load_set(args.set)
    refs = [a["refcode"] for a in items]
    tag = "pickle" if args.pickle else "predictions"
    path = os.path.join(ORACLE_DIR, f"f6_{tag}_{args.set}.pt")
    src = {"route": tag}
    P = X.paths(args.set)
    side = json.load(open(rel(args.sidecar) if args.sidecar else P["sidecar"]))
    order = side["refcodes"] if isinstance(side, dict) else side
    assert order == refs, "sidecar order differs from the frozen set order"
    pp = rel(args.predictions) if args.predictions else P["oracle"]
    pk = rel(args.pickle) if args.pickle else P["pickle"]
    pk_sha = C.sha256(pk)
    if not isinstance(side, dict) or pk_sha != side.get("sha256"):
        raise SystemExit(f"gate-f6: {pk} is not the {args.set} export named by its sidecar (SHA-256 differs)")
    with gzip.open(pk, "rb") as f:
        entries = pickle.load(f)
    assert len(entries) == len(items)
    if args.pickle or not os.path.exists(pp):
        src.update(pickle=os.path.relpath(pk, C.REPO).replace("\\", "/"), pickle_sha256=pk_sha)
    if args.pickle:
        draws = [[mcf_entry_to_cell(e)] * args.draws for e in entries]
    else:
        if os.path.exists(pp):
            pred = torch.load(pp, weights_only=False)
            src.update(predictions=os.path.relpath(pp, C.REPO).replace("\\", "/"), predictions_sha256=C.sha256(pp))
        else:
            pred = X.oracle_predictions(entries, args.draws)
            src["predictions"] = "n3_mcf_export.oracle_predictions (in memory)"
        bad = check_oracle_predictions(pred, entries)
        if bad:
            raise SystemExit(f"gate-f6 refuses {pp}: not an oracle predictions file of the {args.set} export "
                             f"({bad}); arm output is matched only by 'match' under the blinding rules")
        S = int(pred["num_atoms"].shape[0])
        per_draw = [X.predictions_cells(pred, s) for s in range(S)]
        assert all(len(d) == len(items) for d in per_draw)
        draws = [[per_draw[s][i] for s in range(S)] for i in range(len(items))]
    src["sidecar_sha256"] = side.get("sha256") if isinstance(side, dict) else None
    _oracle_bundle(path, args.set, "cells", draws, json.dumps(src, sort_keys=True))
    for sm in args.matchers:
        run_match([path], sm, workers=args.workers, slow=args.slow, allow_blind=True)
    summ, ok = _oracle_summary([path], args.matchers, items)
    out = os.path.join(C.PRIVATE, f"gateF6_{args.set}.json" if tag == "predictions" else f"gateF6_{tag}_{args.set}.json")
    json.dump({"gate": "F6", "set": args.set, "pass": ok, **src, "checked_against_pickle": {
        "pickle": os.path.relpath(pk, C.REPO).replace("\\", "/"), "sha256": pk_sha, "oracle_content_check": "pass"},
        "matchers": summ}, open(out, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("picked1", "draw1", "any16", "n")} for k, v in summ.items()}))
    print("GATE F6", "PASS" if ok else "FAIL", "->", out)


# ============================================================================== picks (truth-free)
def cmd_picks(args):
    """Each seed's and the ensemble's selector pick {refcode, seed, j}; reads selector inputs only (no truth,
    no match), so it is allowed on every set. j None = no valid draw (arm MISS)."""
    paths = [rel(p) for p in args.bundles]
    bundles = [C.load_bundle(p) for p in paths]
    sets = {b["set"] for b in bundles}
    assert len(sets) == 1 and all(b["refcodes"] == bundles[0]["refcodes"] for b in bundles), "one set, one order"
    set_name = sets.pop()
    items = C.load_set(set_name)
    per_seed, ens = [], []
    for i, ref in enumerate(bundles[0]["refcodes"]):
        cr = Crystal(bundles, items, i, args.mode_tiebreak)
        for k, b in enumerate(bundles):
            per_seed.append({"refcode": ref, "i": i, "seed": b["seed"], "bundle": k, "j": cr.pick(args.selector, k)})
        kj = cr.pick(args.selector) if len(bundles) > 1 else None
        if len(bundles) > 1:
            ens.append({"refcode": ref, "i": i, "seed": None if kj is None else bundles[kj[0]]["seed"],
                        "bundle": None if kj is None else kj[0], "j": None if kj is None else kj[1]})
    arm = bundles[0]["arm"]
    out = rel(args.out) if args.out else os.path.join(C.PRIVATE, "picks", f"{arm}_{set_name}_{args.selector}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"arm": arm, "set": set_name, "selector": args.selector, "mode_tiebreak": args.mode_tiebreak,
               "bundles": {os.path.relpath(p, C.REPO).replace("\\", "/"): C.sha256(p) for p in paths},
               "rule": "argmin, non-finite last, exact ties -> lowest draw index (MODE@r: §2.3 ties)",
               "per_seed": per_seed, "ensemble": ens}, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(per_seed)} seed picks ({sum(p['j'] is None for p in per_seed)} arm MISS)"
          + (f", {len(ens)} ensemble picks" if ens else ""))


# ============================================================================== BASIN outcomes (reference bundle)
def basin_outcomes(bundle_paths, reference, selector, matcher="primary", mode_tiebreak=None):
    """Per-crystal, per-seed picked@1 of one arm against a reference bundle (§3.3): H2 FORM = BASIN (reference
    'fftruth'; candidates basin-ours / basin-ours_n2 (one draw per seed), iii-s draw 0 (selector 'ff'), i-E
    (selector 'ff')) or the steric BASIN P1 (reference 'p1ref'; candidates p1-ours / iii-p draws; selector from
    P1_MENU). Reads only the rows `match --reference` wrote (made from these bundles and this reference).
    Rule (iii): a crystal whose reference has no minimum is a non-match for every arm (h = 0, so d_i = 0) and
    is not anchored. Returns the h2_basin JSON dict {refcodes, h, ...} that `report` reads."""
    ref = ref_info(reference)
    paths = [rel(p) for p in bundle_paths]
    bundles = [C.load_bundle(p) for p in paths]
    set_name = bundles[0]["set"]
    assert all(b["set"] == set_name and b["refcodes"] == bundles[0]["refcodes"] for b in bundles), "one set, one order"
    if ref["bundle"]["set"] != set_name:
        raise SystemExit(f"reference {ref['rel']} is a '{ref['bundle']['set']}' bundle, candidates '{set_name}'")
    refcodes = list(bundles[0]["refcodes"])
    if any(r not in ref["row"] for r in refcodes):
        raise SystemExit(f"reference {ref['rel']} lacks some of the candidates' crystals")
    key = mkey(matcher, ref)
    results = []
    for p in paths:
        if rows_current(p, key, ref) is None:
            raise SystemExit(f"no rows for {p} against {ref['rel']}: run `match --reference {ref['rel']} "
                             f"--bundles ... --matcher {matcher}` first")
        results.append(load_match(p, key))
    items = C.load_set(set_name)
    try:
        o = arm_outcomes(bundles, results, items, selector, mode_tiebreak, need_draws=True)
        draws_ok = True
    except Missing:
        if selector == "random":
            raise SystemExit("RANDOM is scored as its expectation: match every draw against the reference first")
        try:
            o = arm_outcomes(bundles, results, items, selector, mode_tiebreak, need_draws=False)
        except Missing as e:
            raise SystemExit(f"no {key} row for (crystal, draw) {e.args[0]}: match the picks against the reference")
        draws_ok = False
    rb = ref["bundle"]
    bad = [i for i, r in enumerate(refcodes) if not ref_ok(ref, ref["row"][r])]
    h, ens = o["picked"].copy(), o["ens"].copy()
    h[bad, :] = 0.0                                   # rule (iii): d_i = 0 for every arm
    ens[bad] = 0.0
    conv = rb["meta"].get("converged")
    unconv = [r for r in refcodes if conv is not None and ref_ok(ref, ref["row"][r])
              and not bool(torch.as_tensor(conv)[ref["row"][r], 0])]
    status = {}
    for res in results:
        for row in res.values():
            status[row.get("status", "ok")] = status.get(row.get("status", "ok"), 0) + 1
    out = {"arm": bundles[0]["arm"], "set": set_name, "selector": selector, "mode_tiebreak": mode_tiebreak,
           "matcher": matcher, "form": "BASIN",
           "reference": ref["rel"], "reference_sha256": ref["sha256"], "reference_arm": ref["arm"],
           "reference_order": ref["order"],
           "match_rule": ("StructureMatcher() fit(reference, candidate)" if ref["order"] == "ref_cand" else
                          "StructureMatcher() fit(candidate, reference)") if matcher == "primary" else key,
           "bundles": {os.path.relpath(p, C.REPO).replace("\\", "/"): C.sha256(p) for p in paths},
           "seeds": [b["seed"] for b in bundles], "n": len(refcodes), "refcodes": refcodes,
           "h": h.tolist(), "h_ensemble": ens.tolist(),
           "picked1_per_seed_pct": [_pp(x) for x in h.mean(0)], "picked1_mean_pct": _pp(h.mean()),
           "draw1": o["draw1"].tolist() if draws_ok else None, "any": o["any"].tolist() if draws_ok else None,
           "pick": o["pick"],
           "ref_invalid_refcodes": [refcodes[i] for i in bad],
           "ref_unconverged_refcodes": unconv,
           "rule_iii": "reference without a minimum: every arm is a non-match (d_i = 0); the crystal is not anchored",
           "arm_miss_refcodes_per_seed": o["miss_refcodes_per_seed"], "row_status_counts": status}
    return out


def cmd_basin(args):
    b0 = C.load_bundle(rel(args.bundles[0]))
    sel_rec = guard(b0["set"], args.step9, args.selection, args.selection_sha256)
    d = basin_outcomes(args.bundles, args.reference, args.selector, args.matcher, args.mode_tiebreak)
    d["selection"] = sel_rec
    ref_stem = os.path.splitext(os.path.basename(args.reference))[0]
    out = rel(args.out) if args.out else os.path.join(BASIN_DIR, f"{d['arm']}_{d['set']}__{ref_stem}_{args.selector}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(d, open(out, "w"), indent=1)
    print(f"wrote {out}: {d['arm']} vs {d['reference_arm']} ({d['match_rule']}), selector {args.selector}: "
          f"picked@1 per seed {d['picked1_per_seed_pct']}%; reference invalid (rule iii) "
          f"{len(d['ref_invalid_refcodes'])}, reference unconverged {len(d['ref_unconverged_refcodes'])}")
    return d


# ============================================================================== Gate C-a (released data)
GATECA_DIR = os.path.join(C.PRIVATE, "gateCa")
GATECA_STOL = {"mcf08": 0.8, "mcf09": 0.9, "mcf10": 1.0}
FIG3A_PCT = {"mcf09": 30.0, "mcf10": 66.0}           # §3.2 C-a (ii): 5-run median within +-5 pp; mcf08 reported


def _fp(counts, v_per_atom):
    """Crystal fingerprint that survives the primitive/conventional choice: reduced composition, V per atom."""
    from math import gcd
    from functools import reduce
    g = reduce(gcd, counts.values())
    return tuple(sorted((z, c // g) for z, c in counts.items())), v_per_atom


def _blinded_fingerprints():
    """{set: {composition: sorted V/atom}} for the blinded N3 sets that can be loaded here."""
    import collections
    out = {}
    for s in BLIND:
        try:
            items = C.load_set(s)
        except Exception:  # noqa: BLE001 -- inputs of this set are not on this machine
            continue
        d = collections.defaultdict(list)
        for a in items:
            cnt = collections.Counter(int(z) for z in a["Z"])
            n_at = sum(cnt.values()) * int(a["K"])
            comp, v = _fp(cnt, abs(float(np.linalg.det(a["orig"]["lattice"].double().numpy()))) / n_at)
            d[comp].append(v)
        out[s] = {k: np.sort(np.array(v)) for k, v in d.items()}
    return out


def gateca_guard(truths, pickle_path=None):
    """Refuse a Gate C-a input that is an N3 SEL / TEST-B / DEV-TEST (or VALSEL) export: its arm outputs are
    matched only by 'match' under §1.3. Checked by the export sidecars' SHA-256 and by >= 50% of the crystals
    matching a blinded set (reduced composition + volume per atom to 1e-3 relative); returns the overlaps."""
    import collections
    if pickle_path is not None:
        sha = C.sha256(pickle_path)
        import n3_mcf_export as X
        for s in ("sel", "testB", "devtest", "valsel"):
            sc = X.paths(s)["sidecar"]
            if os.path.exists(sc) and json.load(open(sc)).get("sha256") == sha:
                raise SystemExit(f"gate-ca: {pickle_path} is the N3 '{s}' export; refusing (blinding, protocol 1.3)")
    fps = _blinded_fingerprints()
    over = {}
    for s, d in fps.items():
        hit = 0
        for t in truths:
            cnt = collections.Counter(int(z) for z in t["species"])
            comp, v = _fp(cnt, abs(float(np.linalg.det(t["lattice"]))) / len(t["species"]))
            arr = d.get(comp)
            if arr is not None and len(arr):
                k = np.searchsorted(arr, v)
                near = [arr[q] for q in (k - 1, k) if 0 <= q < len(arr)]
                hit += any(abs(x - v) <= 1e-3 * v for x in near)
        over[s] = hit
        if hit >= 0.5 * len(truths):
            raise SystemExit(f"gate-ca: {hit}/{len(truths)} crystals are N3 '{s}' crystals; refusing (blinding, protocol 1.3)")
    return {"overlap_counts": over, "sets_checked": sorted(fps)}


def _gt_payloads(pred, mcf_reading=False):
    """MCF's scorer truth (visualize_predictions): gt_data_batch gt_coords / atom_types / lattice_1, split by
    num_atoms; species through the one inverse map."""
    g = pred["gt_data_batch"]
    na = [int(v) for v in g["num_atoms"]]
    cart = torch.as_tensor(g["gt_coords"]).double().split(na)
    types = torch.as_tensor(g["atom_types"]).long().split(na)
    L = torch.as_tensor(g["lattice_1"]).double()
    return [_cells_payload(L[b], cart[b], types[b], mcf_reading) for b in range(len(na))]


def _cells_payload(L, cart, types, mcf_reading=False):
    """mcf_reading (diagnostic only): emulate MCF's xyz round trip -- coordinates and lattice written at 6 dp,
    read back by ASE, the lattice REBUILT from its six parameters (Lattice.from_parameters, pymatgen's standard
    orientation) while the Cartesian coordinates stay in the original frame (xyz_matching.load_extxyz /
    process_one). For a cell not already in that orientation this places the atoms in a rotated lattice."""
    Z = np.array([MCF_IDX_TO_Z.get(int(t), -1) for t in types])
    if (Z < 0).any():
        raise SystemExit(f"atom type index outside the inverse map: {sorted(set(int(t) for t in types))}")
    L, X = L.double().numpy(), cart.double().numpy()
    if mcf_reading:
        from pymatgen.core import Lattice
        L = Lattice.from_parameters(*Lattice(np.round(L, 6)).parameters).matrix
        X = np.round(X, 6)
    return {"lattice": L, "cart": X, "species": Z}


def _pred_payloads(pred, s, mcf_reading=False):
    """Draw s: the same split as n3_mcf_export.predictions_cells (cart_coords / atom_types by num_atoms,
    lattices) without its equal-copy-size assertion (released data may have Z' > 1; 'mol' is not scored)."""
    na = [int(v) for v in pred["num_atoms"][s]]
    cart = torch.as_tensor(pred["cart_coords"][s]).double().split(na)
    types = torch.as_tensor(pred["atom_types"][s]).long().split(na)
    L = torch.as_tensor(pred["lattices"][s]).double()
    return [_cells_payload(L[b], cart[b], types[b], mcf_reading) for b in range(len(na))]


def _mcf_csv(d, S, stol):
    """MCF's run_structure_matching.py per-sample RMSD table -> [n][S] bools (a value = a match)."""
    import csv
    name = f"all_atom_rmsd_results_{S}_stol{float(stol)}.csv"
    for p in (os.path.join(d, "results", name), os.path.join(d, name)):
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))
            return [[r[f"sample_{k}"].strip().lower() not in ("", "nan", "none") for k in range(S)] for r in rows], p
    raise SystemExit(f"no {name} under {d}")


def cmd_gate_ca(args):
    import statistics
    import n3_harness as H
    warnings.filterwarnings("ignore")
    os.makedirs(GATECA_DIR, exist_ok=True)
    matchers = args.matchers
    entries = None
    if args.pickle:
        with gzip.open(rel(args.pickle), "rb") as f:
            entries = pickle.load(f)
    runs, truths = [], None
    for r, pp in enumerate(args.predictions):
        pred = torch.load(rel(pp), weights_only=False)
        S = int(pred["num_atoms"].shape[0])
        if S != args.num_samples:
            raise SystemExit(f"{pp}: {S} samples, expected --num-samples {args.num_samples}")
        gt = _gt_payloads(pred)
        if truths is None:
            truths = gt
            guard_info = gateca_guard(truths, rel(args.pickle) if args.pickle else None)
            if args.mcf_reading:
                truths_scored = _gt_payloads(pred, True)
            if entries is not None:                   # the file's truth must be this pickle's, in order
                if len(entries) != len(gt):
                    raise SystemExit(f"{pp}: {len(gt)} crystals, pickle {len(entries)}")
                for i, e in enumerate(entries):
                    if not (_close(e["lattice_1"], gt[i]["lattice"], 1e-4) and _close(e["gt_coords"], gt[i]["cart"], 1e-3)):
                        raise SystemExit(f"{pp}: crystal {i} ground truth differs from the pickle's")
        elif not all(_close(a["lattice"], b["lattice"], 1e-6) and _close(a["cart"], b["cart"], 1e-6)
                     for a, b in zip(truths, gt)) or len(gt) != len(truths):
            raise SystemExit(f"{pp}: ground truth differs from run 0 (all runs must use the same pickle)")
        runs.append((pp, pred, S))
    if not args.mcf_reading:
        truths_scored = truths
    n = len(truths) if args.limit is None else min(args.limit, len(truths))
    for r, (pp, pred, S) in enumerate(runs):
        draws = [_pred_payloads(pred, s, args.mcf_reading) for s in range(S)]
        for m in matchers:
            stem = os.path.join(GATECA_DIR, f"{args.label}_r{r}__{m}")
            meta = {"predictions": os.path.abspath(rel(pp)), "predictions_sha256": C.sha256(rel(pp)), "S": S,
                    "matcher": m, "n_file": len(truths)}
            if os.path.exists(stem + ".meta.json"):
                old = json.load(open(stem + ".meta.json"))
                if old["predictions_sha256"] != meta["predictions_sha256"]:
                    raise SystemExit(f"{stem}.jsonl belongs to another predictions file: delete it first")
            else:
                json.dump(meta, open(stem + ".meta.json", "w"), indent=1)
            H.repair_jsonl(stem + ".jsonl")
            done = {(x["i"], x["s"]) for x in H.read_jsonl(stem + ".jsonl")}
            tasks, dups, first = [], {}, {}
            for i in range(n):
                for s in range(S):
                    if (i, s) in done:
                        continue
                    p = draws[s][i]
                    h = payload_hash(p)
                    if (i, h) in first:
                        dups.setdefault((i, first[(i, h)]), []).append(s)
                        continue
                    first[(i, h)] = s
                    tasks.append(((i, s), (truths_scored[i], p, m)))
            print(f"gate-ca {args.label} run {r} {m}: {len(tasks)} fits "
                  f"({sum(len(v) for v in dups.values())} duplicates)", flush=True)
            with open(stem + ".jsonl", "a") as fh, open(stem + ".events.jsonl", "a") as ev:
                def log(e, ev=ev):
                    key = e.get("key")
                    ev.write(json.dumps({"t": round(time.time(), 1), **e,
                                         "key": list(key) if isinstance(key, tuple) else key}) + "\n")
                    ev.flush()
                for (i, s), st, res, info in H.run(_pair_task, tasks, workers=args.workers, slow=args.slow,
                                                   init=_winit, log=log):
                    row = {"i": i, "s": s, "ok": bool(res["ok"]) if st == "ok" else False,
                           "rms": res["rms"] if st == "ok" else None, "status": st,
                           "attempts": info.get("attempts", 1)}
                    for rr in [row] + [{**row, "s": ss, "status": "dup", "dup_of": s} for ss in dups.get((i, s), [])]:
                        fh.write(json.dumps(rr) + "\n")
                    fh.flush()
    # outcomes, rates, medians, agreement
    summ = {"gate": "C-a", "label": args.label, "n": len(truths), "n_scored": n, "complete": n == len(truths),
            "mcf_reading": bool(args.mcf_reading),
            "num_samples": args.num_samples, "pickle": args.pickle,
            "pickle_sha256": C.sha256(rel(args.pickle)) if args.pickle else None, "blinding_guard": guard_info,
            "settings": "StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol) get_rms_dist is not None",
            "runs": [], "outcomes_any": {}}
    any_ = {}
    for r, (pp, pred, S) in enumerate(runs):
        ent = {"predictions": pp, "sha256": C.sha256(rel(pp)), "any_pct": {}, "per_draw_pct": {}, "det_fail": {}}
        for m in matchers:
            rows = H.read_jsonl(os.path.join(GATECA_DIR, f"{args.label}_r{r}__{m}.jsonl"))
            ok = np.zeros((n, S), bool)
            for x in rows:
                if x["i"] < n:
                    ok[x["i"], x["s"]] = x["ok"]
            got = {(x["i"], x["s"]) for x in rows}
            assert all((i, s) in got for i in range(n) for s in range(S)), f"run {r} {m}: rows missing"
            any_[(r, m)] = ok.any(1)
            ent["any_pct"][m] = round(100.0 * float(ok.any(1).mean()), 3)
            ent["per_draw_pct"][m] = [round(100.0 * float(ok[:, s].mean()), 3) for s in range(S)]
            ent["det_fail"][m] = [[x["i"], x["s"]] for x in rows if x["status"] == "det_fail"]
            summ["outcomes_any"].setdefault(f"r{r}", {})[m] = ok.any(1).astype(int).tolist()
        summ["runs"].append(ent)
    summ["median_any_pct"] = {m: statistics.median([e["any_pct"][m] for e in summ["runs"]]) for m in matchers}
    summ["fig3a_ref_pct"] = FIG3A_PCT
    gated = [m for m in FIG3A_PCT if m in matchers]
    summ["ii_within_5pp"] = {m: abs(summ["median_any_pct"][m] - FIG3A_PCT[m]) <= 5.0 for m in gated}
    summ["ii_pass"] = bool(gated) and all(summ["ii_within_5pp"].values()) and len(runs) == 5
    summ["agreement"] = None
    if args.mcf_results:
        if len(args.mcf_results) != len(runs):
            raise SystemExit("--mcf-results needs one MCF output dir per --predictions file, same order")
        tot = agree = 0
        dis, per_draw = [], {"total": 0, "agree": 0}
        for r, d in enumerate(args.mcf_results):
            S = runs[r][2]
            for m in matchers:
                mcf, src = _mcf_csv(rel(d), S, GATECA_STOL[m])
                if len(mcf) != len(truths):
                    raise SystemExit(f"{src}: {len(mcf)} rows, predictions {len(truths)} crystals")
                rows = {(x["i"], x["s"]): x["ok"] for x in
                        H.read_jsonl(os.path.join(GATECA_DIR, f"{args.label}_r{r}__{m}.jsonl"))}
                for i in range(n):
                    a, b = bool(any_[(r, m)][i]), any(mcf[i])
                    tot += 1
                    agree += a == b
                    if a != b:
                        dis.append({"run": r, "i": i, "stol": GATECA_STOL[m], "n3_score": a, "mcf": b,
                                    "n3_draws": [k for k in range(S) if rows[(i, k)]],
                                    "mcf_draws": [k for k in range(S) if mcf[i][k]]})
                    for k in range(S):
                        per_draw["total"] += 1
                        per_draw["agree"] += rows[(i, k)] == mcf[i][k]
                        if rows[(i, k)] != mcf[i][k]:
                            per_draw.setdefault("disagreements", []).append(
                                {"run": r, "i": i, "s": k, "stol": GATECA_STOL[m], "n3_score": rows[(i, k)],
                                 "mcf": mcf[i][k]})
        summ["agreement"] = {"outcomes": tot, "agree": agree, "fraction": round(agree / tot, 6),
                             "per_draw": per_draw, "disagreements": dis}
        summ["i_pass"] = agree / tot >= 0.99
    out = os.path.join(C.PRIVATE, f"gateCa_{args.label}.json")
    json.dump(summ, open(out, "w"), indent=1)
    for m in matchers:
        print(f"  {m} (stol {GATECA_STOL[m]}): any-of-{args.num_samples} per run "
              f"{[e['any_pct'][m] for e in summ['runs']]}  median {summ['median_any_pct'][m]:.2f}%"
              + (f"  (Fig. 3a {FIG3A_PCT[m]:.0f}%)" if m in FIG3A_PCT else ""))
    if summ["agreement"]:
        a = summ["agreement"]
        print(f"  agreement with MCF's scorer: {a['agree']}/{a['outcomes']} (crystal, stol) outcomes "
              f"= {100 * a['fraction']:.2f}%  -> (i) {'PASS' if summ['i_pass'] else 'FAIL'}")
    print(f"  (ii) {'PASS' if summ['ii_pass'] else 'not met'}" + ("" if len(runs) == 5 else f" (needs 5 runs; {len(runs)} given)")
          + ("" if summ["complete"] else f"  [--limit {n}: not a gate result]"), "->", out)


# ============================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match")
    m.add_argument("--bundles", nargs="+", required=True)
    m.add_argument("--matcher", required=True, help="primary | mcf05 | mcf08 | mcf09 | mcf10")
    m.add_argument("--draws", choices=["all", "picks"], default="all")
    m.add_argument("--selector", default=None)
    m.add_argument("--mode-tiebreak", default=None)
    m.add_argument("--crystals", default=None, help="a:b subset (tests / timing)")
    m.add_argument("--workers", type=int, default=2)
    m.add_argument("--slow", type=float, default=1200.0, help="s before a pool fit is re-run alone, no limit")
    m.add_argument("--step9", action="store_true")
    m.add_argument("--selection", default=SELECTION)
    m.add_argument("--selection-sha256", default=None, help="step 9 without git: SHA-256 of the committed file")
    m.add_argument("--reference", default=None,
                   help="reference bundle (fftruth / p1ref): match against its draw 0 instead of the truth (§3.3)")
    bs = sub.add_parser("basin")
    bs.add_argument("--reference", required=True, help="fftruth (H2 BASIN) or p1ref (steric BASIN P1) bundle")
    bs.add_argument("--bundles", nargs="+", required=True, help="one arm's seed bundles (one set)")
    bs.add_argument("--selector", required=True, help="e.g. ff (basin-*, iii-s, i-E); lj / lj_pre / random (P1)")
    bs.add_argument("--matcher", default="primary")
    bs.add_argument("--mode-tiebreak", default=None)
    bs.add_argument("--out", default=None, help="default results/n3/private/basin/<arm>_<set>__<ref>_<selector>.json")
    bs.add_argument("--step9", action="store_true")
    bs.add_argument("--selection", default=SELECTION)
    bs.add_argument("--selection-sha256", default=None)
    s = sub.add_parser("select")
    s.add_argument("--config", required=True)
    r = sub.add_parser("report")
    r.add_argument("--config", required=True)
    r.add_argument("--step9", action="store_true")
    r.add_argument("--selection", default=SELECTION)
    r.add_argument("--selection-sha256", default=None)
    k = sub.add_parser("picks")
    k.add_argument("--bundles", nargs="+", required=True)
    k.add_argument("--selector", required=True)
    k.add_argument("--mode-tiebreak", default=None)
    k.add_argument("--out", default=None, help="default results/n3/private/picks/<arm>_<set>_<selector>.json")
    for name in ("oracle", "gate-f6"):
        o = sub.add_parser(name)
        o.add_argument("--set", required=True, choices=["val", "devtest", "sel", "testB", "valsel", "train"])
        o.add_argument("--matchers", nargs="+", default=["primary", "mcf05", "mcf08", "mcf09", "mcf10"])
        o.add_argument("--draws", type=int, default=16)
        o.add_argument("--workers", type=int, default=2)
        o.add_argument("--slow", type=float, default=1200.0)
        if name == "oracle":
            o.add_argument("--reference", action="store_true",
                           help="also test reference-bundle matching: truth-as-bundle references in both orders")
        if name == "gate-f6":
            o.add_argument("--predictions", default=None, help="default: n3_mcf_export.paths(set)['oracle']")
            o.add_argument("--pickle", default=None, help="decode this pickle with mcf_entry_to_cell instead")
            o.add_argument("--sidecar", default=None, help="default: n3_mcf_export.paths(set)['sidecar']")
    g = sub.add_parser("gate-ca")
    g.add_argument("--label", required=True, help="e.g. test / val (released Thurlemann pickles)")
    g.add_argument("--predictions", nargs="+", required=True, help="predictions_10.pt of each run (seeds 0-4)")
    g.add_argument("--pickle", default=None, help="the released pickle the runs sampled (truth cross-check)")
    g.add_argument("--mcf-results", nargs="+", default=None,
                   help="run_structure_matching.py output dir of each run (same order): scorer agreement (i)")
    g.add_argument("--matchers", nargs="+", default=["mcf08", "mcf09", "mcf10"])
    g.add_argument("--num-samples", type=int, default=10)
    g.add_argument("--limit", type=int, default=None, help="first N crystals only (smoke; not a gate result)")
    g.add_argument("--mcf-reading", action="store_true",
                   help="diagnostic: score MCF's xyz reading (lattice rebuilt from parameters); use its own --label")
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--slow", type=float, default=1200.0)
    args = ap.parse_args()
    if args.cmd == "match":
        cr = tuple(int(x) for x in args.crystals.split(":")) if args.crystals else None
        run_match([rel(p) for p in args.bundles], args.matcher, args.draws, args.selector, cr, args.workers,
                  args.slow, args.step9, args.selection, mode_tiebreak=args.mode_tiebreak,
                  selection_sha=args.selection_sha256, reference=args.reference)
    elif args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "picks":
        cmd_picks(args)
    elif args.cmd == "basin":
        cmd_basin(args)
    elif args.cmd == "oracle":
        cmd_oracle(args)
    elif args.cmd == "gate-ca":
        cmd_gate_ca(args)
    else:
        cmd_gate_f6(args)


if __name__ == "__main__":
    main()
