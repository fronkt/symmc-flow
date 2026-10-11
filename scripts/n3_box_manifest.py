"""N3 box upload manifests (protocol §7 steps 5 and 7), tar pack / unpack with SHA-256 verification on arrival,
the step-8 pull-back, and the pip constraint files of the box environments.

    python scripts/n3_box_manifest.py list   --step 5                   # laptop: every file, group, bytes, SHA-256
    python scripts/n3_box_manifest.py list   --step 7 [--new-box]
    python scripts/n3_box_manifest.py pack   --step 5 --out <dir>       # laptop: assert, then <dir>/n3_step5.tar
    python3 scripts/n3_box_manifest.py unpack --tar n3_step5.tar --root <box root>   # box: stdlib only
    python3 scripts/n3_box_manifest.py verify --step 5 --root <box root>             # box: stdlib only
    python3 scripts/n3_box_manifest.py pullback --root <box root> --out <dir> --label step7   # box, step 8
    python scripts/n3_box_manifest.py unpack --tar n3_pullback_step7.tar --root . --no-overwrite   # laptop
    python scripts/n3_box_manifest.py pins                              # laptop: scripts/n3_box_constraints_*.txt
    python scripts/n3_box_manifest.py wheels                            # laptop BEFORE rental: CUDA wheel availability

Step 5 (training + VALSEL) carries TRAIN / VAL / SEL inputs only. Step 7 (after n3_selection.json is committed
and pushed) adds TEST-B and DEV-TEST inputs, including the stored N1 / N2 DEV-TEST draws; --new-box adds the step-5
set and the step-5 products pulled back from the old box (§7 Failures: "re-run step 7 on a new box").

Step-5 assertions (pack refuses otherwise):
  * PATH: no file whose repo path names DEV-TEST / TEST-B or any full cache (devtest, testB, n1_test / n2_test,
    n3_mcf/test/, n3_mcf/devtest/, oracle dirs, match rows, g2_asym.pt, n3_asym_pool.pt, ds_o3.pt, data/csd_*);
  * HASH: no file whose SHA-256 equals that of a DEV-TEST / TEST-B input (per-set caches, MCF test / devtest
    pickles and sidecars, iv-P items, stored and converted DEV-TEST draws, DEV-TEST picks) or of a full cache;
  * CONTENT: no DEV-TEST / TEST-B refcode as a pickle string token in any .pt / .pkl.gz file or as a word in any
    text file, with a positive control (the same scanner finds them in set_caches/devtest.pt and testB.pt).
The forbidden (size, SHA-256) list travels in the step-5 manifest (hashes only), so `verify` on the box also
proves that no file under the root has one of those hashes.

Every file carries a `csd` flag (CSD-derived or licence-restricted: must be deleted at step 8). Manifests are
written into the tar as .n3_box/manifests/<label>.{manifest.json,sha256,tsv}; unpack also copies them to
$N3_MANIFEST_DIR (default ~/n3_manifests, outside the root), where scripts/n3_box_cleanup.sh reads them after the
root is gone. The .tsv lines are "<csd 0|1>\t<bytes>\t<sha256>\t<path>".
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen
from html.parser import HTMLParser

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN_SUBDIR = os.path.join(".n3_box", "manifests")
FORMAT = "n3-box-manifest/1"
SET_DIR = "data/n3_sets"
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


def safe_path(path, root):
    """A canonical repo-relative regular-file path, contained in root (including symlink parents)."""
    if not isinstance(path, str) or not path or "\\" in path or any(c in path for c in "\r\n\t\0"):
        raise ValueError(f"invalid manifest path {path!r}")
    if os.path.islink(root):
        raise ValueError("manifest root cannot be a symlink")
    parts = PurePosixPath(path).parts
    if PurePosixPath(path).is_absolute() or ":" in path or any(p in (".", "..") for p in parts) \
            or str(PurePosixPath(path)) != path:
        raise ValueError(f"non-canonical manifest path {path!r}")
    base, dest = os.path.realpath(root), os.path.realpath(os.path.join(root, *parts))
    if os.path.commonpath([base, dest]) != base or dest == base:
        raise ValueError(f"manifest path escapes root: {path!r}")
    current = os.path.abspath(root)
    for p in parts:
        current = os.path.join(current, p)
        if os.path.islink(current):
            raise ValueError(f"symlink in manifest path: {path!r}")
    return dest


def validate_manifest(man, root, deployable=True):
    """Fail closed before extraction, verification or using a cleanup inventory."""
    if not isinstance(man, dict) or man.get("format") != FORMAT:
        raise ValueError("unknown manifest format")
    label = man.get("label", "")
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("unsafe manifest label")
    if man.get("step") not in (4, 5, 7, 8):
        raise ValueError("invalid manifest step")
    if deployable and (man.get("deployable") is not True or man.get("problems") or man.get("missing")
                       or man.get("groups_filter")):
        raise ValueError("manifest is incomplete or not deployable")
    files = man.get("files")
    if not isinstance(files, list) or (man["step"] != 8 and not files):
        raise ValueError("manifest files must be a nonempty list")
    seen = set()
    for f in files + man.get("forbidden", []):
        if not isinstance(f, dict):
            raise ValueError("invalid file inventory row")
        safe_path(f.get("path"), root)
        if type(f.get("bytes")) is not int or f["bytes"] < 0 or not HEX256.fullmatch(f.get("sha256", "")):
            raise ValueError("invalid size or SHA-256 in inventory")
    for f in files:
        if type(f.get("csd")) is not bool or f["path"] in seen or f["path"].startswith(".n3_box/"):
            raise ValueError("duplicate/reserved path or invalid CSD flag")
        seen.add(f["path"])
        if man["step"] in (4, 5) and boundary_rule(man["step"]).search(f["path"]) and f.get("group") != "fork":
            raise ValueError("step-5 manifest contains a held-out/full-cache path")
    if man["step"] in (4, 5):
        forbidden = {f["sha256"] for f in man.get("forbidden", [])}
        if not forbidden or any(f["sha256"] in forbidden for f in files):
            raise ValueError("step-5 forbidden-hash inventory absent or violated")
    if man["step"] == 7 and not (man.get("checks", {}).get("n3_selection_complete") is True
                                and man["checks"].get("n3_selection_tracked") is True
                                and man["checks"].get("n3_selection_unchanged") is True
                                and man["checks"].get("head_on_remote") is True):
        raise ValueError("step-7 manifest lacks a complete committed/pushed selection gate")
    return man

# repo-relative entry scripts that run on the box; their local-import closure is shipped
BOX_ENTRY = ["n3_common", "n3_harness", "n3_tpool", "n3_stats", "n3_score", "n3_ours", "n3_classical", "n3_ffstar",
             "n3_chirality", "n3_old_symmc", "n3_mcf_lib", "n3_mcf_export", "n3_mcf_run", "n3_mcf_resid",
             "n3_mcf_gauge", "n3_mcf_sg", "n3_mcf_diag", "n3_mcf_gateC", "n3_mcf_smoke", "n3_mcf_fork",
             "n3_make_set_caches", "n3_timing_gate", "n3_box_manifest", "g2_asym_baselines", "g2_learned_flow",
             "g2_rank", "eval_orient_matchrate"]
BOX_SHELL = ["scripts/n3_box_setup.sh", "scripts/n3_box_cleanup.sh", "scripts/n3_box_constraints_symmc.txt",
             "scripts/n3_box_constraints_mcf.txt", "scripts/n3_box_constraints_ff.txt"]

# step-5 path rule (case-insensitive, on the repo-relative POSIX path)
FORBIDDEN_PATH_5 = re.compile(
    r"(devtest|testb|/test/|n1_test|n2_test|(^|/)oracle|/match/|g2_asym\.pt|n3_asym_pool|ds_o3|(^|/)data/csd_|"
    r"(^|/)results/n3/gatef\.json$|labels_|table_r|legacy_oracle)", re.I)

TEXT_EXT = (".json", ".jsonl", ".txt", ".csv", ".yaml", ".yml", ".log", ".md", ".tsv")
FORBIDDEN_PATH_4 = re.compile(FORBIDDEN_PATH_5.pattern + r"|(^|/)(sel|valsel)([_./]|$)|_sel\.|_valsel\.|valsel_as_test", re.I)


def boundary_rule(step):
    return FORBIDDEN_PATH_4 if step == 4 else FORBIDDEN_PATH_5


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def set_dir():
    path = rel(os.path.abspath(os.environ.get("N3_SET_CACHES", os.path.join(REPO, SET_DIR))))
    safe_path(path + "/train.pt", REPO)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


# ------------------------------------------------------------------------------------------ file groups
def code_closure():
    """scripts/*.py modules reachable by local import from BOX_ENTRY (AST, both `import x` and `from x import`)."""
    import ast
    sdir = os.path.join(REPO, "scripts")
    local = {f[:-3] for f in os.listdir(sdir) if f.endswith(".py")}
    todo, seen = [m for m in BOX_ENTRY if m in local], set()
    missing = [m for m in BOX_ENTRY if m not in local]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        tree = ast.parse(open(os.path.join(sdir, m + ".py"), encoding="utf-8").read())
        for n in ast.walk(tree):
            names = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else \
                [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            todo += [x for x in names if x in local and x not in seen]
    return [f"scripts/{m}.py" for m in sorted(seen)], missing


def fork_files():
    """external/mcf as it is (MCF 3c493f8 + patches/mcf_n3.patch): tracked + untracked-not-ignored files."""
    fork = os.path.join(REPO, "external", "mcf")
    top = subprocess.run(["git", "-C", fork, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    head = subprocess.run(["git", "-C", fork, "rev-parse", "HEAD"], capture_output=True, text=True)
    if top.returncode or os.path.normcase(os.path.realpath(top.stdout.strip())) != os.path.normcase(os.path.realpath(fork)) \
            or head.stdout.strip() != "3c493f8ddcfe8f3ea2d1e9bdba82c26b0f43516e":
        raise SystemExit("external/mcf must be the frozen 3c493f8 fork")
    r = subprocess.run(["git", "-C", fork, "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"external/mcf is not the git fork: {r.stderr.strip()}")
    files = [f for f in r.stdout.split("\n") if f and not f.startswith(("csp-pipeline/", "assets/"))]
    if any(f.lower().endswith((".pt", ".ckpt", ".cif", ".pkl", ".pkl.gz")) for f in files):
        raise SystemExit("MCF code fork contains a data/weight file outside excluded release assets")
    return [f"external/mcf/{f}" for f in sorted(set(files))]


def n2_chosen():
    p = os.path.join(REPO, "results", "n2_selection.json")
    if not os.path.isfile(p):
        raise SystemExit("required results/n2_selection.json is missing")
    chosen = json.load(open(p))["chosen"]
    if not isinstance(chosen, list) or len(chosen) != 3:
        raise SystemExit("N2 selection must contain exactly three checkpoint filenames")
    return chosen


def groups(step, new_box=False):
    """[(group, csd, required, [repo paths])]. `required` False = produced during the run (listed when present)."""
    P = "results/n3/private"
    S = set_dir()
    code, _ = code_closure()
    sym = sorted(rel(os.path.join(d, f)) for d, _, fs in os.walk(os.path.join(REPO, "symmc_flow"))
                 for f in fs if f.endswith(".py"))
    g5 = [
        ("code", False, True, code + sym + ["pyproject.toml", "patches/mcf_n3.patch", "tasks/n3_protocol.md"] + BOX_SHELL),
        ("frozen_lists", False, True, ["tasks/n3_sets/sel_refcodes.txt", "results/n2_selection.json"]),
        ("gate_records", False, True, [f"results/n3/gateF_{s}.json" for s in ("train", "val", "valsel")]),
        ("fork", False, True, fork_files()),
        ("uma", True, True, ["external/uma/uma-s-1p1.pt", "external/uma/iso_atom_elem_refs.yaml"]),
        ("ff_configuration", True, True, [f"{P}/classical/ffstar_config.json"]),
        ("ours_checkpoints", True, True, [f"results/vast_g2/eval_s{s}.pt" for s in range(3)]
         + [f"results/vast_n12/{c}" for c in n2_chosen()]),
        ("set_caches", True, True, [f"{S}/{s}.{e}" for s in ("train", "val", "sel", "valsel")
                                    for e in ("pt", "json")]),
        ("mcf_exports", True, True, ["n3_mcf/trainval/train_molcrystal_normalized.pkl.gz",
                                     "n3_mcf/trainval/train_sidecar.json",
                                     "n3_mcf/trainval/val_molcrystal_normalized.pkl.gz",
                                     "n3_mcf/trainval/val_sidecar.json",
                                     "n3_mcf/valsel_as_test/test_molcrystal_normalized.pkl.gz",
                                     "n3_mcf/valsel_as_test/test_sidecar.json"]),
        ("ivp_items", True, True, [f"{P}/old/items_{s}.pt" for s in ("train", "val", "sel")]
         + [f"{P}/old/coset_table.json"]),
        ("ours_val_draws", True, True, [f"{P}/draws/{a}_s{s}_val.pt" for a in ("ours", "ours_n2") for s in range(3)]),
    ]
    if step == 4:
        keep = {"code", "fork", "uma", "ours_checkpoints", "set_caches", "mcf_exports", "ivp_items", "gate_records"}
        result = []
        for group, csd, required, paths in g5:
            if group not in keep:
                continue
            if group == "ours_checkpoints":
                paths = [p for p in paths if p.startswith("results/vast_g2/")]
            else:
                paths = [p for p in paths if not FORBIDDEN_PATH_4.search(p)]
            result.append((group, csd, required, paths))
        return result
    if step == 5:
        return g5
    g7 = [
        ("public_protocol_7", False, True, ["tasks/n3_protocol.md"]),
        ("frozen_lists_7", False, True, ["tasks/n3_sets/testB_refcodes.txt", "results/n3/n3_selection.json"]),
        ("gate_records_7", False, True, [f"results/n3/gateF_{s}.json" for s in ("testB", "devtest")]),
        ("set_caches_7", True, True, [f"{S}/{s}.{e}" for s in ("testB", "devtest") for e in ("pt", "json")]),
        ("mcf_exports_7", True, True, ["n3_mcf/test/test_molcrystal_normalized.pkl.gz", "n3_mcf/test/test_sidecar.json",
                                       "n3_mcf/devtest/test_molcrystal_normalized.pkl.gz",
                                       "n3_mcf/devtest/test_sidecar.json"]),
        ("ivp_items_7", True, True, [f"{P}/old/items_{s}.pt" for s in ("testB", "devtest")]
         + [f"{P}/old/ivP_frozen.json"]),
        ("ours_devtest_draws", True, True, [f"{P}/draws/{a}_s{s}_devtest.pt" for a in ("ours", "ours_n2")
                                            for s in range(3)]
         + [f"{P}/picks/ours_devtest_lj.json", f"{P}/picks/ours_n2_devtest_lj.json"]),
    ]
    if not new_box:
        return g7
    # a new box for step 7 (§7 Failures): the step-5 upload plus the step-5 products pulled back at the old box's
    # end, all of which step 7 reads: the frozen MCF / iv-P checkpoints, the FF* configuration, its batched-LJ gate,
    # the VALSEL FF* truths and anchor (TEST-B / DEV-TEST truths require them), and the OURS SEL bundles.
    # n3_selection.json's "step7_inputs" (written at step 6) names the frozen MCF checkpoints and their
    # config.yaml; ivP_frozen.json names the frozen iv-P checkpoints.
    carry = []
    sel = os.path.join(REPO, "results", "n3", "n3_selection.json")
    if not os.path.isfile(sel):
        raise SystemExit("a new step-7 box requires the frozen selection input inventory")
    inputs = json.load(open(sel)).get("step7_inputs")
    if not isinstance(inputs, list) or not inputs:
        raise SystemExit("selection has no complete step7_inputs inventory")
    for row in inputs:
        if not isinstance(row, dict) or not row.get("role") or not HEX256.fullmatch(row.get("sha256", "")):
            raise SystemExit("invalid selection step7_inputs row")
        path = safe_path(row.get("path"), REPO)
        if not os.path.isfile(path) or sha256(path) != row["sha256"]:
            raise SystemExit(f"frozen step-7 input missing or changed: {row.get('path')}")
        carry.append(row["path"])
    fz = os.path.join(REPO, P, "old", "ivP_frozen.json")
    if os.path.exists(fz):
        for c in json.load(open(fz)).get("checkpoints", {}).values():
            carry.append(c if not os.path.isabs(c) else rel(c))
    carry += [f"{P}/classical/ffstar_config.json", f"{P}/classical/anchor_valsel.json",
              f"{P}/draws/fftruth_s-1_valsel.pt"]
    carry += [rel(os.path.join(d, f)) for d, _, fs in os.walk(os.path.join(REPO, P, "classical"))
              for f in fs if f.startswith("lbgate")]
    return g5 + g7 + [("step5_carryover", True, True, sorted(set(carry)))]


# ------------------------------------------------------------------------------------------ forbidden (step 5)
def forbidden_files(step=5):
    """Files whose content must never reach a step-5 box (DEV-TEST / TEST-B inputs and the full caches)."""
    P = "results/n3/private"
    S = set_dir()
    out = [f"{S}/{s}.{e}" for s in ("testB", "devtest") for e in ("pt", "json")]
    out += ["n3_mcf/test/test_molcrystal_normalized.pkl.gz", "n3_mcf/test/test_sidecar.json",
            "n3_mcf/devtest/test_molcrystal_normalized.pkl.gz", "n3_mcf/devtest/test_sidecar.json",
            f"{P}/old/items_testB.pt", f"{P}/old/items_devtest.pt",
            "results/vast_n12/n1_test.jsonl", "results/vast_n12/n2_test.jsonl",
            "data/csd_mol/g2_asym.pt", "data/csd_testB/n3_asym_pool.pt", "data/csd_mol/ds_o3.pt",
            "data/csd_testB/ds_o3.pt", "tasks/n3_sets/testB_refcodes.txt"]
    out += [f"{P}/draws/{a}_s{s}_devtest.pt" for a in ("ours", "ours_n2") for s in range(3)]
    out += [f"{P}/picks/ours_devtest_lj.json", f"{P}/picks/ours_n2_devtest_lj.json"]
    if step == 4:
        out += [f"{S}/{s}.{e}" for s in ("sel", "valsel") for e in ("pt", "json")]
        out += ["n3_mcf/valsel_as_test/test_molcrystal_normalized.pkl.gz", "n3_mcf/valsel_as_test/test_sidecar.json",
                f"{P}/old/items_sel.pt", "tasks/n3_sets/sel_refcodes.txt"]
    return [p for p in out if os.path.exists(os.path.join(REPO, p))]


def blinded_refcodes(step=5):
    """TEST-B + DEV-TEST refcodes (public identifiers; used only to scan files locally, never written out)."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import n3_common as C
    rc = set(C.frozen_refcodes("testB"))
    if step == 4:
        rc |= set(C.frozen_refcodes("sel"))
    js = os.path.join(REPO, set_dir(), "devtest.pt")
    if os.path.exists(js):
        import torch
        rc |= set(torch.load(js, weights_only=False)["refcodes"])
    else:
        rc |= {a["refcode"] for a in C.load_set("devtest")}
    return rc


def scan_refcodes(path, refcodes):
    """Number of `refcodes` spelled in the file: pickle string tokens in .pt / .pkl(.gz), words in text files."""
    from n3_make_set_caches import string_tokens
    p = os.path.join(REPO, path)
    if path.endswith((".pt", ".ckpt", ".pkl")):
        return len(string_tokens(open(p, "rb").read()) & refcodes)
    if path.endswith(".pkl.gz"):
        return len(string_tokens(gzip.open(p, "rb").read()) & refcodes)
    if path.endswith(TEXT_EXT) or path.endswith(".py"):
        words = set(re.findall(r"[A-Za-z0-9]+", open(p, encoding="utf-8", errors="replace").read()))
        return len(words & refcodes)
    return 0


# ------------------------------------------------------------------------------------------ build manifest
def build(step, new_box=False, only_groups=None, allow_missing=False, allow_unpatched=False, scan=True):
    t0 = time.time()
    label = f"step{step}" + ("_newbox" if new_box else "")
    files, missing, seen = [], [], set()
    problems = []
    if step not in (4, 5, 7) or new_box and step != 7:
        raise ValueError("--new-box is valid only for step 7")
    if only_groups or allow_missing or allow_unpatched or not scan:
        problems.append("diagnostic overrides are not deployable; production pack requires all groups/checks")
    for g, csd, req, paths in groups(step, new_box):
        if only_groups and g not in only_groups:
            continue
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            ap = safe_path(p, REPO)
            if not os.path.isfile(ap):
                missing.append({"group": g, "path": p, "required": req})
                continue
            files.append({"group": g, "csd": csd, "path": p, "bytes": os.path.getsize(ap), "sha256": sha256(ap)})
    if missing:
        problems.append(f"{len(missing)} listed files are missing: {[m['path'] for m in missing][:8]}")
    # the box needs n3_common's per-set cache loader (results/n3/private/n3_common_setcache.patch)
    src = open(os.path.join(REPO, "scripts", "n3_common.py"), encoding="utf-8").read()
    patched = "def set_cache_only" in src and "def set_cache_dir" in src
    if not patched:
        problems.append("scripts/n3_common.py has no per-set cache loader: apply "
                        "results/n3/private/n3_common_setcache.patch first (a box without it cannot load any set)")
    code_missing = code_closure()[1]
    if code_missing and "code" in (only_groups or ["code"]):
        problems.append(f"entry scripts not found: {code_missing}")
    forb = []
    checks = {}
    if step in (4, 5):
        forb = [{"path": p, "bytes": os.path.getsize(os.path.join(REPO, p)), "sha256": sha256(os.path.join(REPO, p))}
                for p in forbidden_files(step)]
        bad_path = [f["path"] for f in files if boundary_rule(step).search(f["path"]) and f["group"] != "fork"]
        fh = {f["sha256"] for f in forb}
        bad_hash = [f["path"] for f in files if f["sha256"] in fh]
        checks.update(path_rule_violations=bad_path, hash_violations=bad_hash, forbidden_hashes=len(forb))
        if bad_path:
            problems.append(f"step-5 path rule: {bad_path}")
        if bad_hash:
            problems.append(f"step-5 content hash equals a DEV-TEST / TEST-B / full-cache file: {bad_hash}")
        if scan:
            rc = blinded_refcodes(step)
            controls = ("devtest", "testB", "sel") if step == 4 else ("devtest", "testB")
            pos = {p: scan_refcodes(p, rc) for p in [f"{set_dir()}/{s}.pt" for s in controls]
                   if os.path.exists(os.path.join(REPO, p))}
            hits = {}
            for f in files:
                if f["group"] in ("uma",) or f["path"].startswith("external/mcf/"):
                    continue                             # third-party weights / code: no CSD refcodes to scan for
                n = scan_refcodes(f["path"], rc)
                if n:
                    hits[f["path"]] = n
            public_spec = "tasks/n3_protocol.md"
            data_hits = {p: n for p, n in hits.items() if not p.endswith(".py") and p != public_spec}
            checks.update(refcode_scan_blinded=len(rc), refcode_positive_control=pos, refcode_hits_data=data_hits,
                          refcode_hits_code=sorted(p for p in hits if p.endswith(".py") or p == public_spec))
            if len(pos) != len(controls) or min(pos.values()) == 0:
                problems.append(f"refcode scanner positive control failed: {pos}")
            if data_hits:
                problems.append(f"DEV-TEST / TEST-B refcodes found in step-5 data files: {data_hits}")
    rc_head = git("rev-parse", "HEAD")[1]
    dirty = git("status", "--porcelain", "--", "scripts", "symmc_flow", "patches", "pyproject.toml")[1]
    if step == 7:
        sel = "results/n3/n3_selection.json"
        selected = json.load(open(os.path.join(REPO, sel))) if os.path.isfile(os.path.join(REPO, sel)) else {}
        complete = selected.get("complete") is True and selected.get("schema_version") == 1
        tracked = git("ls-files", "--error-unmatch", sel)[0] == 0
        clean = git("diff", "--quiet", "HEAD", "--", sel)[0] == 0
        pushed = bool(git("branch", "-r", "--contains", "HEAD")[1])
        checks.update(n3_selection_complete=complete, n3_selection_tracked=tracked, n3_selection_unchanged=clean,
                      head_on_remote=pushed)
        if not (complete and tracked and clean and pushed):
            problems.append("results/n3/n3_selection.json must be committed, unchanged and pushed before step 7 (§7 step 6)")
    man = {"format": FORMAT, "label": label, "step": step, "new_box": new_box, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "git_head": rc_head, "code_dirty": dirty.splitlines(), "n3_common_setcache": patched,
           "groups_filter": only_groups, "files": files, "missing": missing, "forbidden": forb, "checks": checks,
           "problems": problems, "deployable": not problems and not only_groups and not allow_missing,
           "sec": round(time.time() - t0, 1)}
    return man


def summarize(man):
    by = {}
    for f in man["files"]:
        g = by.setdefault(f["group"], {"files": 0, "MB": 0.0, "csd": f["csd"]})
        g["files"] += 1
        g["MB"] += f["bytes"] / 1e6
    for g, v in by.items():
        print(f"  {g:20s} files {v['files']:5d}  {v['MB']:9.1f} MB  csd={int(v['csd'])}")
    tot = sum(f["bytes"] for f in man["files"]) / 1e6
    print(f"  {'TOTAL':20s} files {len(man['files']):5d}  {tot:9.1f} MB")
    for m in man["missing"]:
        print(f"  missing{' (REQUIRED)' if m['required'] else ''}: {m['path']}  [{m['group']}]")
    c = man["checks"]
    if man["step"] in (4, 5) and c:
        print(f"  step-5 path rule violations {len(c['path_rule_violations'])}, hash violations "
              f"{len(c['hash_violations'])} (vs {c['forbidden_hashes']} forbidden files)")
        if "refcode_scan_blinded" in c:
            print(f"  refcode scan over {c['refcode_scan_blinded']} TEST-B+DEV-TEST refcodes: data hits "
                  f"{len(c['refcode_hits_data'])}, code files mentioning one {c['refcode_hits_code']}, positive "
                  f"control {c['refcode_positive_control']}")
    for p in man["problems"]:
        print("  PROBLEM:", p)
    print(f"  deployable: {man['deployable']}  ({man['sec']} s)")


def write_manifest_files(man, d):
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, man["label"])
    json.dump(man, open(base + ".manifest.json", "w"), indent=1)
    with open(base + ".sha256", "w", newline="\n") as fh:
        fh.writelines(f"{f['sha256']}  {f['path']}\n" for f in man["files"])
    with open(base + ".tsv", "w", newline="\n") as fh:
        rows = [(int(f["csd"]), f["bytes"], f["sha256"], f["path"]) for f in man["files"]]
        rows += [(1, f["bytes"], f["sha256"], "FORBIDDEN:" + f["path"]) for f in man.get("forbidden", [])]
        fh.writelines(f"{a}\t{b}\t{c}\t{p}\n" for a, b, c, p in rows)
    return [base + e for e in (".manifest.json", ".sha256", ".tsv")]


def inventory_text(man):
    hashes = "".join(f"{f['sha256']}  {f['path']}\n" for f in man["files"])
    rows = [(int(f["csd"]), f["bytes"], f["sha256"], f["path"]) for f in man["files"]]
    rows += [(1, f["bytes"], f["sha256"], "FORBIDDEN:" + f["path"]) for f in man.get("forbidden", [])]
    return hashes, "".join(f"{a}\t{b}\t{c}\t{p}\n" for a, b, c, p in rows)


def external_manifest_dir(root):
    mdir = os.path.realpath(os.path.expanduser(os.environ.get("N3_MANIFEST_DIR", "~/n3_manifests")))
    rroot = os.path.realpath(root)
    if os.path.commonpath([rroot, mdir]) == rroot:
        raise SystemExit("N3_MANIFEST_DIR must be outside the box root so cleanup retains its receipts")
    return mdir


# ------------------------------------------------------------------------------------------ commands
def cmd_list(args):
    man = build(args.step, args.new_box, args.groups, scan=not args.no_scan)
    if args.files:
        for f in man["files"]:
            print(f"{f['sha256']}  {f['bytes']:>12d}  {f['group']:18s} {f['path']}")
    summarize(man)


def cmd_pack(args):
    man = build(args.step, args.new_box, args.groups, args.allow_missing, args.allow_unpatched)
    summarize(man)
    if man["problems"] or not man["deployable"]:
        raise SystemExit("pack refused: " + "; ".join(man["problems"]))
    validate_manifest(man, REPO)
    os.makedirs(args.out, exist_ok=True)
    mdir = os.path.join(args.out, "manifests")
    mfiles = write_manifest_files(man, mdir)
    tar = os.path.join(args.out, f"n3_{man['label']}.tar")
    with tarfile.open(tar + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        for f in man["files"]:
            ti = tf.gettarinfo(os.path.join(REPO, f["path"]), arcname=f["path"])
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            with open(os.path.join(REPO, f["path"]), "rb") as fh:
                tf.addfile(ti, fh)
        for m in mfiles:
            tf.add(m, arcname=f"{MAN_SUBDIR}/{os.path.basename(m)}".replace("\\", "/"))
    os.replace(tar + ".tmp", tar)
    th = sha256(tar)
    with open(tar + ".sha256", "w", newline="\n") as fh:
        fh.write(f"{th}  {os.path.basename(tar)}\n")
    # re-read the tar: every member hash equals the manifest (catches a file changed while packing)
    got = {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if m.isfile() and not m.name.startswith(".n3_box/"):
                got[m.name] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    bad = [f["path"] for f in man["files"] if got.get(f["path"]) != f["sha256"]]
    if bad or len(got) != len(man["files"]):
        os.remove(tar)
        raise SystemExit(f"tar re-read differs from the manifest: {bad[:5]}")
    print(f"wrote {tar} ({os.path.getsize(tar) / 1e6:.1f} MB, sha256 {th[:16]}...) + {tar}.sha256 + {mdir}")
    print("upload (manual-approval mode for CSD uploads, §7 step 4):")
    print(f"  scp -P <port> {tar} {tar}.sha256 scripts/n3_box_manifest.py root@<host>:~/n3_upload/")
    print(f"  box: cd ~/n3_upload && sha256sum -c {os.path.basename(tar)}.sha256 && "
          f"python3 n3_box_manifest.py unpack --tar {os.path.basename(tar)} --root $N3_BOX_ROOT")


def _safe_members(tf, root):
    seen = set()
    for m in tf.getmembers():
        safe_path(m.name, root)
        if not m.isfile() or m.name in seen:
            raise SystemExit(f"tar member {m.name!r} is non-regular or duplicated")
        seen.add(m.name)
        yield m


def cmd_unpack(args):
    tar, root = os.path.abspath(args.tar), os.path.abspath(args.root)
    side = tar + ".sha256"
    if not os.path.isfile(side):
        raise SystemExit(f"required transfer checksum is missing: {side}")
    want = open(side).read().split()[0]
    if not HEX256.fullmatch(want) or sha256(tar) != want:
        raise SystemExit(f"FAIL: {tar} SHA-256 differs from {side} (transfer corrupted)")
    print(f"tar SHA-256 ok ({want[:16]}...)")
    mdir = external_manifest_dir(root)
    with tarfile.open(tar) as tf:
        members = list(_safe_members(tf, root))
        mans = [m for m in members if m.name.startswith(".n3_box/manifests/") and m.name.endswith(".manifest.json")]
        if len(mans) != 1:
            raise SystemExit(f"expected one manifest in the tar, found {[m.name for m in mans]}")
        man = validate_manifest(json.loads(tf.extractfile(mans[0]).read()), root)
        mb = f".n3_box/manifests/{man['label']}"
        expected = {f["path"] for f in man["files"]} | {mb + e for e in (".manifest.json", ".sha256", ".tsv")}
        found = {m.name for m in members}
        if found != expected:
            raise SystemExit("archive members differ from the complete manifest inventory")
        for f in man["files"]:
            member = tf.getmember(f["path"])
            if member.size != f["bytes"] or hashlib.sha256(tf.extractfile(member).read()).hexdigest() != f["sha256"]:
                raise SystemExit(f"archive member failed checksum before extraction: {f['path']}")
        for ext, content in zip((".sha256", ".tsv"), inventory_text(man)):
            if tf.extractfile(tf.getmember(mb + ext)).read() != content.encode("utf-8"):
                raise SystemExit(f"manifest inventory sidecar disagrees with JSON: {ext}")
        if args.no_overwrite:
            clash = [m.name for m in members if m.isfile() and os.path.exists(os.path.join(root, m.name))
                     and sha256(os.path.join(root, m.name)) != hashlib.sha256(tf.extractfile(m).read()).hexdigest()]
            if clash:
                raise SystemExit(f"--no-overwrite: {len(clash)} existing files differ, e.g. {clash[:5]}")
        os.makedirs(root, exist_ok=True)
        for m in members:
            try:
                tf.extract(m, root, filter="data")      # python >= 3.12 (and security backports)
            except TypeError:
                tf.extract(m, root)                      # members were already checked by _safe_members
    os.makedirs(mdir, exist_ok=True)
    for e in (".manifest.json", ".sha256", ".tsv"):
        src = os.path.join(root, MAN_SUBDIR, man["label"] + e)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(mdir, man["label"] + e))
    print(f"extracted {len(members)} members into {root}; manifests copied to {mdir}")
    ok = verify(man, root)
    receipt = {"label": man["label"], "tar_sha256": want, "files": len(man["files"]), "verified": ok,
               "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"), "root": root}
    json.dump(receipt, open(os.path.join(mdir, man["label"] + ".unpack_receipt.json"), "w"), indent=1)
    raise SystemExit(0 if ok else 1)


def verify(man, root, check_boundary=True):
    validate_manifest(man, root)
    t0 = time.time()
    bad, miss = [], []
    for f in man["files"]:
        p = safe_path(f["path"], root)
        if not os.path.isfile(p):
            miss.append(f["path"])
        elif os.path.getsize(p) != f["bytes"] or sha256(p) != f["sha256"]:
            bad.append(f["path"])
    extra = []
    if man["step"] in (4, 5) and check_boundary:
        # step 5: nothing under the root may have a DEV-TEST / TEST-B / full-cache hash or a forbidden path
        sizes = {f["bytes"] for f in man["forbidden"]}
        fh = {f["sha256"] for f in man["forbidden"]}
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x not in (".git", "__pycache__")]
            if any(os.path.islink(os.path.join(d, x)) for x in dirs):
                extra.append(("symlink-directory", os.path.relpath(d, root)))
            for fn in fs:
                p = os.path.join(d, fn)
                rp = os.path.relpath(p, root).replace(os.sep, "/")
                if rp.startswith(".n3_box/manifests/"):
                    continue
                if os.path.islink(p):
                    extra.append(("symlink-file", rp))
                elif boundary_rule(man["step"]).search(rp) and not rp.startswith("external/mcf/"):
                    extra.append(("path", rp))
                elif os.path.getsize(p) in sizes and sha256(p) in fh:
                    extra.append(("hash", rp))
    ok = not bad and not miss and not extra
    print(f"verify {man['label']}: {len(man['files'])} files, missing {len(miss)}, hash mismatch {len(bad)}"
          + (f", step-5 forbidden-by-path/hash under the root {len(extra)}" if man.get("forbidden") else "")
          + f" ({time.time() - t0:.0f} s)")
    for x in (miss[:10] + bad[:10] + [f"{k}: {p}" for k, p in extra[:10]]):
        print("   ", x)
    print("PASS" if ok else "FAIL")
    return ok


def cmd_verify(args):
    root = os.path.abspath(args.root)
    label = args.label or f"step{args.step}"
    p = os.path.join(root, MAN_SUBDIR, label + ".manifest.json")
    if not os.path.exists(p):
        raise SystemExit(f"no manifest {p}")
    man = json.load(open(p))
    check_boundary = True
    if man.get("step") in (4, 5):
        # Step-5 hashes stay required after step 7, but the held-out input boundary has been opened.
        later = [os.path.join(root, MAN_SUBDIR, f) for f in os.listdir(os.path.join(root, MAN_SUBDIR))
                 if f.endswith(".manifest.json") and f != os.path.basename(p)
                 and json.load(open(os.path.join(root, MAN_SUBDIR, f))).get("step", 0) in (5, 7)
                 and json.load(open(os.path.join(root, MAN_SUBDIR, f))).get("step", 0) > man["step"]]
        later.sort(key=lambda path: json.load(open(path))["step"], reverse=True)
        later = later[:1]
        for path in later:
            if not verify(json.load(open(path)), root):
                raise SystemExit("step-7 inventory failed; cannot open the step-5 data boundary")
        check_boundary = not later
    raise SystemExit(0 if verify(man, root, check_boundary) else 1)


# the box-side output trees pulled back at step 8 (everything CSD-derived the runs wrote)
PULL_ROOTS = ["results/n3", "n3_mcf/runs", "n3_mcf/infer", "n3_mcf/_gateC", "n3_mcf/_timing", "n3_mcf/_smoke",
              "external/mcf/outputs"]
PULL_SKIP = ("__pycache__",)


def pull_link(p, root):
    """A symlink met by pullback -> {path, target}; refuses one whose target resolves outside the box root."""
    rp = os.path.relpath(p, root).replace(os.sep, "/")
    base, dest = os.path.realpath(root), os.path.realpath(p)
    if os.path.commonpath([base, dest]) != base:
        raise SystemExit(f"symlink leaves the box root: {rp} -> {os.readlink(p)}")
    return {"path": rp, "target": os.readlink(p)}


def cmd_pullback(args):
    """Box, §7 step 8: tar every file the run wrote under PULL_ROOTS (uploaded files with unchanged hashes are left
    out), with a manifest; all of it is CSD-derived (csd=1) and is deleted by n3_box_cleanup.sh afterwards."""
    root = os.path.abspath(args.root)
    home = external_manifest_dir(root)
    uploaded = {}
    mdir = os.path.join(root, MAN_SUBDIR)
    if not os.path.isdir(mdir):
        raise SystemExit("pullback requires the verified upload manifests")
    upload_manifests = [fn for fn in sorted(os.listdir(mdir)) if fn.endswith(".manifest.json")]
    if not upload_manifests:
        raise SystemExit("pullback requires the verified upload manifests")
    for fn in upload_manifests:
        if fn.endswith(".manifest.json"):
            man = validate_manifest(json.load(open(os.path.join(mdir, fn))), root)
            for f in man["files"]:
                uploaded[f["path"]] = f["sha256"]
    files, links = [], []
    for pr in PULL_ROOTS:
        base = os.path.join(root, pr)
        for d, dirs, fs in os.walk(base):
            dirs[:] = [x for x in dirs if x not in PULL_SKIP]
            # wandb offline runs leave pointer symlinks (latest-run, debug*.log): they carry no data of their own.
            # A link inside the box root is listed, not packed (its target is pulled as a file or deleted with the
            # root at cleanup); a link leaving the root stops the pullback.
            for x in [x for x in dirs + fs if os.path.islink(os.path.join(d, x))]:
                links.append(pull_link(os.path.join(d, x), root))
            dirs[:] = [x for x in dirs if not os.path.islink(os.path.join(d, x))]
            for fn in fs:
                p = os.path.join(d, fn)
                if os.path.islink(p):
                    continue
                rp = os.path.relpath(p, root).replace(os.sep, "/")
                safe_path(rp, root)
                h = sha256(p)
                if uploaded.get(rp) == h:
                    continue
                files.append({"group": "pullback", "csd": True, "path": rp, "bytes": os.path.getsize(p), "sha256": h})
    label = f"pullback_{args.label}"
    man = {"format": FORMAT, "label": label, "step": 8, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "files": files, "missing": [], "forbidden": [], "checks": {}, "problems": [], "roots": PULL_ROOTS,
           "deployable": True}
    validate_manifest(man, root)
    os.makedirs(args.out, exist_ok=True)
    mfiles = write_manifest_files(man, mdir)
    os.makedirs(home, exist_ok=True)
    for m in mfiles:
        shutil.copy2(m, os.path.join(home, os.path.basename(m)))
    if links:
        json.dump(links, open(os.path.join(home, label + ".skipped_symlinks.json"), "w"), indent=1)
        print(f"{len(links)} in-root symlinks listed, not packed -> {os.path.join(home, label + '.skipped_symlinks.json')}")
    tar = os.path.join(args.out, f"n3_{label}.tar")
    with tarfile.open(tar + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        for f in files:
            tf.add(os.path.join(root, f["path"]), arcname=f["path"], recursive=False)
        for m in mfiles:
            tf.add(m, arcname=f"{MAN_SUBDIR}/{os.path.basename(m)}", recursive=False)
    with tarfile.open(tar + ".tmp") as tf:
        for f in files:
            m = tf.getmember(f["path"])
            if m.size != f["bytes"] or hashlib.sha256(tf.extractfile(m).read()).hexdigest() != f["sha256"]:
                raise SystemExit(f"pullback input changed while packing: {f['path']}; tar not published")
    os.replace(tar + ".tmp", tar)
    th = sha256(tar)
    open(tar + ".sha256", "w").write(f"{th}  {os.path.basename(tar)}\n")
    open(os.path.join(home, label + ".archive_sha256"), "w").write(th + "\n")
    print(f"{len(files)} files ({sum(f['bytes'] for f in files) / 1e6:.1f} MB) -> {tar} (sha256 {th[:16]}...); "
          f"manifest also in {home}")
    print(f"laptop: scp the tar + .sha256, then python scripts/n3_box_manifest.py unpack --tar {os.path.basename(tar)} "
          "--root . --no-overwrite")


def cleanup_plan(root, upload, mdir, extras=(), require_pullback=True, receipt=None):
    """Validate explicit private deletion targets and retained inventories; this function deletes nothing."""
    root, upload, mdir = (os.path.realpath(os.path.abspath(p)) for p in (root, upload, mdir))
    protected = {os.path.abspath(os.sep), os.path.expanduser("~"), "/root", "/home", "/tmp", "/dev", "/dev/shm",
                 "/usr", "/opt", "/var", "/etc", "/proc", "/sys", "/run"}
    targets = []
    for p in (root, upload, *extras):
        if not isinstance(p, str) or not p or any(c in p for c in "\r\n\t\0") or not os.path.isabs(p):
            raise ValueError("cleanup targets must be explicit absolute paths, one per line")
        dest = os.path.realpath(p)
        if dest in protected or os.path.dirname(dest) == os.path.abspath(os.sep) \
                or any(os.path.commonpath([dest, keep]) == dest for keep in (mdir, os.path.expanduser("~"))):
            raise ValueError(f"refusing protected cleanup target: {p}")
        if os.path.islink(p):
            raise ValueError(f"refusing symlink cleanup target: {p}")
        if p not in (root, upload) and not any(os.path.commonpath([base, dest]) == base for base in (root, upload)) \
                and not os.path.basename(dest).startswith("n3_"):
            raise ValueError("extra targets outside private root/upload must have an n3_ name")
        targets.append(dest)
    if root == upload or os.path.commonpath([root, upload]) in (root, upload):
        raise ValueError("box root and upload directory must be distinct, non-nested private directories")
    inventory = []
    labels = []
    if not os.path.isdir(mdir):
        raise ValueError("retained manifest directory is missing")
    for name in sorted(os.listdir(mdir)):
        if not name.endswith(".manifest.json"):
            continue
        man = validate_manifest(json.load(open(os.path.join(mdir, name))), root)
        base = os.path.join(mdir, man["label"])
        for ext, wanted in zip((".sha256", ".tsv"), inventory_text(man)):
            if not os.path.isfile(base + ext) or open(base + ext, encoding="utf-8").read() != wanted:
                raise ValueError(f"cleanup manifest sidecar is missing or inconsistent: {base + ext}")
        labels.append(man["label"])
        inventory += [(f["bytes"], f["sha256"]) for f in man["files"] if f["csd"]]
        inventory += [(f["bytes"], f["sha256"]) for f in man.get("forbidden", [])]
    if not inventory or not any(l.startswith(("step4", "step5")) for l in labels):
        raise ValueError("cleanup requires a timing/training upload manifest and private-file hash inventory")
    if require_pullback:
        if not receipt or not os.path.isfile(receipt):
            raise ValueError("cleanup requires the successful laptop pullback unpack receipt")
        rec = json.load(open(receipt))
        label = rec.get("label", "")
        hp = os.path.join(mdir, label + ".archive_sha256")
        if label not in labels or not label.startswith("pullback_") or rec.get("verified") is not True \
                or not os.path.isfile(hp) or rec.get("tar_sha256") != open(hp).read().strip():
            raise ValueError("pullback receipt is not a verified copy of this box's output archive")
        if any(os.path.commonpath([p, os.path.realpath(receipt)]) == p for p in targets):
            raise ValueError("the retained pullback receipt lies inside a deletion target")
    return {"root": root, "upload": upload, "manifest_dir": mdir, "targets": sorted(set(targets)),
            "labels": labels, "csd_hashes": sorted(set(inventory)), "pullback_receipt": receipt,
            "discard_outputs": not require_pullback}


def cmd_cleanup_plan(args):
    extras = []
    if args.extras_file:
        extras = [line.rstrip("\r\n") for line in open(args.extras_file, encoding="utf-8") if line.strip()]
    plan = cleanup_plan(args.root, args.upload, args.manifest_dir, extras, not args.discard_outputs,
                        args.pullback_receipt)
    json.dump(plan, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in plan.items() if k != "csd_hashes"}))


# ------------------------------------------------------------------------------------------ pins
SYMMC_DIRECT = ["numpy", "scipy", "pymatgen", "spglib", "monty", "ase", "rdkit", "gemmi", "networkx", "pandas",
                "psutil", "pyyaml", "pytest"]
MCF_DIRECT = ["torch-geometric", "pytorch-lightning", "hydra-core", "omegaconf", "wandb", "rdkit", "ase", "pymatgen",
              "spglib", "monty", "numpy", "scipy", "pandas", "networkx", "h5py", "tqdm", "rich", "matplotlib",
              "plotly", "seaborn", "psutil", "pyyaml", "p-tqdm", "gputil", "setuptools"]
TORCH_FAMILY = re.compile(r"^(torch|torchvision|torchaudio|triton|nvidia-.*|pytorch-triton.*)$", re.I)


def _norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def closure_pins(direct):
    """name==version for `direct` and their requirements, as installed in THIS interpreter (torch family left
    out: the box pins it from the CUDA index)."""
    import importlib.metadata as md
    from packaging.requirements import Requirement
    dists = {_norm(d.metadata["Name"]): d for d in md.distributions()}
    out, todo, absent = {}, [_norm(x) for x in direct], []
    while todo:
        n = todo.pop()
        if n in out or TORCH_FAMILY.match(n):
            continue
        d = dists.get(n)
        if d is None:
            absent.append(n)
            continue
        out[n] = f"{d.metadata['Name']}=={d.version.split('+')[0]}"
        for r in d.requires or []:
            try:
                req = Requirement(r)
            except Exception:  # noqa: BLE001
                continue
            if req.marker and not req.marker.evaluate({"extra": "", "sys_platform": "linux",
                                                       "platform_system": "Linux", "os_name": "posix"}):
                continue
            todo.append(_norm(req.name))
    return sorted(out.values(), key=str.lower), sorted(set(absent))


def cmd_pins(args):
    head = ("# generated by scripts/n3_box_manifest.py pins on {host} {t} from {src}\n"
            "# pip constraints (-c) for the '{env}' box env: the versions the N3 units tested locally\n")
    t = time.strftime("%Y-%m-%d %H:%M")
    import platform
    for env, direct in (("symmc", SYMMC_DIRECT), ("mcf", MCF_DIRECT)):
        pins, absent = closure_pins(direct)
        p = os.path.join(REPO, "scripts", f"n3_box_constraints_{env}.txt")
        with open(p, "w", newline="\n") as fh:
            fh.write(head.format(host=platform.node(), t=t, src=sys.executable, env=env))
            if absent:
                fh.write(f"# not installed locally (left unpinned): {' '.join(absent)}\n")
            fh.writelines(x + "\n" for x in pins)
        print(f"{rel(p)}: {len(pins)} pins; unpinned {absent}")
    ffpy = os.path.join(REPO, "external", "venv_fairchem", "Scripts" if os.name == "nt" else "bin",
                        "python.exe" if os.name == "nt" else "python")
    r = subprocess.run([ffpy, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"pip freeze in {ffpy} failed: {r.stderr[:300]}")
    lines = []
    for ln in r.stdout.splitlines():
        if not ln or ln.startswith("#") or " @ " in ln or "==" not in ln:
            continue
        name, ver = ln.split("==", 1)
        if _norm(name) in ("pywin32", "pywinpty", "pywin32-ctypes"):
            continue
        lines.append(f"{name}=={ver.split('+')[0]}")
    p = os.path.join(REPO, "scripts", "n3_box_constraints_ff.txt")
    with open(p, "w", newline="\n") as fh:
        fh.write(head.format(host=platform.node(), t=t, src=ffpy, env="ff"))
        fh.writelines(x + "\n" for x in sorted(lines, key=str.lower))
    print(f"{rel(p)}: {len(lines)} pins (full freeze of external/venv_fairchem)")


class _WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)


def read_url(url, timeout):
    """Read metadata only; never download a wheel or model artifact."""
    with urlopen(Request(url, headers={"User-Agent": "SymMC-N3-wheel-preflight/1"}), timeout=timeout) as response:
        return response.read().decode("utf-8")


def check_wheels(timeout=30):
    """Fail closed before rental when a pinned Linux CPython 3.12 x86_64 CUDA wheel is absent."""
    cu128, cu130 = "https://download.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/cu130"
    pyg = "https://data.pyg.org/whl/torch-2.7.0+cu128.html"
    targets = [
        ("mcf torch", cu128 + "/torch/", "torch", "2.7.1+cu128"),
        ("mcf torchvision", cu128 + "/torchvision/", "torchvision", "0.22.1+cu128"),
        ("ff torch", cu130 + "/torch/", "torch", "2.13.0+cu130"),
        ("symmc torch", cu128 + "/torch/", "torch", "2.11.0+cu128"),
        ("mcf torch-scatter", pyg, "torch_scatter", "2.1.2+pt27cu128"),
        ("mcf torch-cluster", pyg, "torch_cluster", "1.6.3+pt27cu128"),
    ]
    pages, found = {}, {}
    for label, index, package, version in targets:
        if index not in pages:
            parser = _WheelLinks()
            parser.feed(read_url(index, timeout))
            pages[index] = parser.links
        matches = []
        prefix = package + "-" + version + "-"
        for link in pages[index]:
            filename = unquote(urlparse(link).path.rsplit("/", 1)[-1])
            if filename.startswith(prefix) and "-cp312-cp312-" in filename and \
                    re.search(r"-(?:manylinux[^-]*|linux)_x86_64\.whl$", filename):
                matches.append(urljoin(index, link))
        if not matches:
            raise ValueError(f"{label}: no pinned Linux cp312 x86_64 CUDA wheel at {index}")
        found[label] = sorted(set(matches))
    metadata_url = "https://pypi.org/pypi/fairchem-core/2.23.0/json"
    metadata = json.loads(read_url(metadata_url, timeout))
    torch_reqs = [r for r in metadata["info"].get("requires_dist", [])
                  if re.match(r"torch\s*[~<>=!([]", r, re.I)]
    if len(torch_reqs) != 1 or re.sub(r"\s", "", torch_reqs[0]) != "torch~=2.13.0":
        raise ValueError(f"fairchem-core 2.23.0 torch dependency differs from the reviewed pin: {torch_reqs}")
    ff_wheels = [u["url"] for u in metadata.get("urls", []) if u.get("packagetype") == "bdist_wheel"
                 and u.get("filename", "").endswith("-py3-none-any.whl") and not u.get("yanked")]
    if not ff_wheels:
        raise ValueError("fairchem-core 2.23.0 has no available universal Python wheel")
    found["ff fairchem-core"] = ff_wheels
    return {"available": True, "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target": "Linux CPython 3.12 x86_64", "min_driver_cuda": "13.0", "wheels": found,
            "fairchem_torch_requirement": torch_reqs[0],
            "scope": "Pinned CUDA and PyG wheels plus fairchem metadata; no install or CUDA execution"}


def cmd_wheels(args):
    try:
        result = check_wheels(args.timeout)
    except Exception as e:  # HTTP errors and missing pins both stop deployment.
        raise SystemExit(f"wheel preflight FAILED; do not rent: {e}") from e
    print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--step", type=int, required=True, choices=[4, 5, 7])
        p.add_argument("--new-box", action="store_true", help="step 7 on a new box: + the step-5 set and carry-over")
        p.add_argument("--groups", nargs="+", default=None, help="test only: restrict to these groups (not deployable)")
        if name == "list":
            p.add_argument("--files", action="store_true")
            p.add_argument("--no-scan", action="store_true")
        else:
            p.add_argument("--out", required=True)
            p.add_argument("--allow-missing", action="store_true", help="test only (not deployable)")
            p.add_argument("--allow-unpatched", action="store_true", help="test only: n3_common without set caches")
    p = sub.add_parser("unpack")
    p.add_argument("--tar", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--no-overwrite", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("--step", type=int, choices=[4, 5, 7], default=5)
    p.add_argument("--label", default=None, help="manifest label (default step<N>)")
    p.add_argument("--root", required=True)
    p = sub.add_parser("pullback")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    p = sub.add_parser("cleanup-plan")
    p.add_argument("--root", required=True)
    p.add_argument("--upload", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extras-file")
    p.add_argument("--pullback-receipt")
    p.add_argument("--discard-outputs", action="store_true", help="protocol stop: discard all arm outputs unscored")
    sub.add_parser("pins")
    p = sub.add_parser("wheels", help="read-only CUDA wheel availability check before paid provisioning")
    p.add_argument("--timeout", type=float, default=30)
    args = ap.parse_args()
    {"list": cmd_list, "pack": cmd_pack, "unpack": cmd_unpack, "verify": cmd_verify, "pullback": cmd_pullback,
     "pins": cmd_pins, "cleanup-plan": cmd_cleanup_plan, "wheels": cmd_wheels}[args.cmd](args)


if __name__ == "__main__":
    main()
