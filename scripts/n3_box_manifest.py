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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN_SUBDIR = os.path.join(".n3_box", "manifests")
FORMAT = "n3-box-manifest/1"

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


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


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
    r = subprocess.run(["git", "-C", fork, "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"external/mcf is not the git fork: {r.stderr.strip()}")
    files = [f for f in r.stdout.split("\n") if f and not f.startswith(("csp-pipeline/", "assets/"))]
    return [f"external/mcf/{f}" for f in files if os.path.isfile(os.path.join(fork, f))]


def n2_chosen():
    p = os.path.join(REPO, "results", "n2_selection.json")
    return json.load(open(p))["chosen"] if os.path.exists(p) else []


def groups(step, new_box=False):
    """[(group, csd, required, [repo paths])]. `required` False = produced during the run (listed when present)."""
    P = "results/n3/private"
    code, _ = code_closure()
    sym = sorted(rel(os.path.join(d, f)) for d, _, fs in os.walk(os.path.join(REPO, "symmc_flow"))
                 for f in fs if f.endswith(".py"))
    g5 = [
        ("code", False, True, code + sym + ["pyproject.toml", "patches/mcf_n3.patch"] + BOX_SHELL),
        ("frozen_lists", False, True, ["tasks/n3_sets/sel_refcodes.txt", "results/n2_selection.json"]),
        ("gate_records", False, True, [f"results/n3/gateF_{s}.json" for s in ("train", "val", "valsel")]),
        ("fork", False, True, fork_files()),
        ("uma", True, True, ["external/uma/uma-s-1p1.pt", "external/uma/iso_atom_elem_refs.yaml"]),
        ("ours_checkpoints", True, True, [f"results/vast_g2/eval_s{s}.pt" for s in range(3)]
         + [f"results/vast_n12/{c}" for c in n2_chosen()]),
        ("set_caches", True, True, [f"{P}/set_caches/{s}.{e}" for s in ("train", "val", "sel", "valsel")
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
    if step == 5:
        return g5
    g7 = [
        ("frozen_lists_7", False, True, ["tasks/n3_sets/testB_refcodes.txt", "results/n3/n3_selection.json"]),
        ("gate_records_7", False, True, [f"results/n3/gateF_{s}.json" for s in ("testB", "devtest")]),
        ("set_caches_7", True, True, [f"{P}/set_caches/{s}.{e}" for s in ("testB", "devtest") for e in ("pt", "json")]),
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
    if os.path.exists(sel):
        carry += list(json.load(open(sel)).get("step7_inputs", []))
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
def forbidden_files():
    """Files whose content must never reach a step-5 box (DEV-TEST / TEST-B inputs and the full caches)."""
    P = "results/n3/private"
    out = [f"{P}/set_caches/{s}.{e}" for s in ("testB", "devtest") for e in ("pt", "json")]
    out += ["n3_mcf/test/test_molcrystal_normalized.pkl.gz", "n3_mcf/test/test_sidecar.json",
            "n3_mcf/devtest/test_molcrystal_normalized.pkl.gz", "n3_mcf/devtest/test_sidecar.json",
            f"{P}/old/items_testB.pt", f"{P}/old/items_devtest.pt",
            "results/vast_n12/n1_test.jsonl", "results/vast_n12/n2_test.jsonl",
            "data/csd_mol/g2_asym.pt", "data/csd_testB/n3_asym_pool.pt", "data/csd_mol/ds_o3.pt",
            "data/csd_testB/ds_o3.pt", "tasks/n3_sets/testB_refcodes.txt"]
    out += [f"{P}/draws/{a}_s{s}_devtest.pt" for a in ("ours", "ours_n2") for s in range(3)]
    out += [f"{P}/picks/ours_devtest_lj.json", f"{P}/picks/ours_n2_devtest_lj.json"]
    return [p for p in out if os.path.exists(os.path.join(REPO, p))]


def blinded_refcodes():
    """TEST-B + DEV-TEST refcodes (public identifiers; used only to scan files locally, never written out)."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import n3_common as C
    rc = set(C.frozen_refcodes("testB"))
    js = os.path.join(C.PRIVATE, "set_caches", "devtest.pt")
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
    for g, csd, req, paths in groups(step, new_box):
        if only_groups and g not in only_groups:
            continue
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            ap = os.path.join(REPO, p)
            if not os.path.isfile(ap):
                missing.append({"group": g, "path": p, "required": req})
                continue
            files.append({"group": g, "csd": csd, "path": p, "bytes": os.path.getsize(ap), "sha256": sha256(ap)})
    problems = []
    if missing and not allow_missing:
        problems.append(f"{len(missing)} listed files are missing: {[m['path'] for m in missing][:8]}")
    # the box needs n3_common's per-set cache loader (results/n3/private/n3_common_setcache.patch)
    src = open(os.path.join(REPO, "scripts", "n3_common.py"), encoding="utf-8").read()
    patched = "def set_cache_only" in src
    if not patched and not allow_unpatched:
        problems.append("scripts/n3_common.py has no per-set cache loader: apply "
                        "results/n3/private/n3_common_setcache.patch first (a box without it cannot load any set)")
    code_missing = code_closure()[1]
    if code_missing and "code" in (only_groups or ["code"]):
        problems.append(f"entry scripts not found: {code_missing}")
    forb = []
    checks = {}
    if step == 5:
        forb = [{"path": p, "bytes": os.path.getsize(os.path.join(REPO, p)), "sha256": sha256(os.path.join(REPO, p))}
                for p in forbidden_files()]
        bad_path = [f["path"] for f in files if FORBIDDEN_PATH_5.search(f["path"]) and f["group"] != "fork"]
        fh = {f["sha256"] for f in forb}
        bad_hash = [f["path"] for f in files if f["sha256"] in fh]
        checks.update(path_rule_violations=bad_path, hash_violations=bad_hash, forbidden_hashes=len(forb))
        if bad_path:
            problems.append(f"step-5 path rule: {bad_path}")
        if bad_hash:
            problems.append(f"step-5 content hash equals a DEV-TEST / TEST-B / full-cache file: {bad_hash}")
        if scan:
            rc = blinded_refcodes()
            pos = {p: scan_refcodes(p, rc) for p in ("results/n3/private/set_caches/devtest.pt",
                                                      "results/n3/private/set_caches/testB.pt")
                   if os.path.exists(os.path.join(REPO, p))}
            hits = {}
            for f in files:
                if f["group"] in ("uma",) or f["path"].startswith("external/mcf/"):
                    continue                             # third-party weights / code: no CSD refcodes to scan for
                n = scan_refcodes(f["path"], rc)
                if n:
                    hits[f["path"]] = n
            data_hits = {p: n for p, n in hits.items() if not p.endswith(".py")}
            checks.update(refcode_scan_blinded=len(rc), refcode_positive_control=pos, refcode_hits_data=data_hits,
                          refcode_hits_code=sorted(p for p in hits if p.endswith(".py")))
            if not pos or min(pos.values()) == 0:
                problems.append(f"refcode scanner positive control failed: {pos}")
            if data_hits:
                problems.append(f"DEV-TEST / TEST-B refcodes found in step-5 data files: {data_hits}")
    rc_head = git("rev-parse", "HEAD")[1]
    dirty = git("status", "--porcelain", "--", "scripts", "symmc_flow", "patches", "pyproject.toml")[1]
    if step == 7:
        sel = "results/n3/n3_selection.json"
        tracked = git("ls-files", "--error-unmatch", sel)[0] == 0
        clean = git("diff", "--quiet", "HEAD", "--", sel)[0] == 0
        pushed = bool(git("branch", "-r", "--contains", "HEAD")[1])
        checks.update(n3_selection_tracked=tracked, n3_selection_unchanged=clean, head_on_remote=pushed)
        if not (tracked and clean and pushed) and not allow_missing:
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
    if man["step"] == 5 and c:
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


# ------------------------------------------------------------------------------------------ commands
def cmd_list(args):
    man = build(args.step, args.new_box, args.groups, allow_missing=True, allow_unpatched=True, scan=not args.no_scan)
    if args.files:
        for f in man["files"]:
            print(f"{f['sha256']}  {f['bytes']:>12d}  {f['group']:18s} {f['path']}")
    summarize(man)


def cmd_pack(args):
    man = build(args.step, args.new_box, args.groups, args.allow_missing, args.allow_unpatched)
    summarize(man)
    if man["problems"]:
        raise SystemExit("pack refused: " + "; ".join(man["problems"]))
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
    rroot = os.path.realpath(root)
    for m in tf.getmembers():
        dest = os.path.realpath(os.path.join(root, m.name))
        if os.path.isabs(m.name) or ".." in m.name.split("/") or not (dest == rroot or dest.startswith(rroot + os.sep)):
            raise SystemExit(f"unsafe tar member {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise SystemExit(f"tar member {m.name!r} is not a regular file or directory")
        yield m


def cmd_unpack(args):
    tar, root = os.path.abspath(args.tar), os.path.abspath(args.root)
    side = tar + ".sha256"
    if os.path.exists(side):
        want = open(side).read().split()[0]
        if sha256(tar) != want:
            raise SystemExit(f"FAIL: {tar} SHA-256 differs from {side} (transfer corrupted)")
        print(f"tar SHA-256 ok ({want[:16]}...)")
    else:
        print(f"warning: no {side}; the tar hash is not checked (member hashes still are)")
    os.makedirs(root, exist_ok=True)
    with tarfile.open(tar) as tf:
        members = list(_safe_members(tf, root))
        mans = [m for m in members if m.name.startswith(".n3_box/manifests/") and m.name.endswith(".manifest.json")]
        if len(mans) != 1:
            raise SystemExit(f"expected one manifest in the tar, found {[m.name for m in mans]}")
        man = json.loads(tf.extractfile(mans[0]).read())
        if args.no_overwrite:
            clash = [m.name for m in members if m.isfile() and os.path.exists(os.path.join(root, m.name))
                     and sha256(os.path.join(root, m.name)) != hashlib.sha256(tf.extractfile(m).read()).hexdigest()]
            if clash:
                raise SystemExit(f"--no-overwrite: {len(clash)} existing files differ, e.g. {clash[:5]}")
        for m in members:
            try:
                tf.extract(m, root, filter="data")      # python >= 3.12 (and security backports)
            except TypeError:
                tf.extract(m, root)                      # members were already checked by _safe_members
    mdir = os.path.expanduser(os.environ.get("N3_MANIFEST_DIR", "~/n3_manifests"))
    os.makedirs(mdir, exist_ok=True)
    for e in (".manifest.json", ".sha256", ".tsv"):
        src = os.path.join(root, MAN_SUBDIR, man["label"] + e)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(mdir, man["label"] + e))
    print(f"extracted {len(members)} members into {root}; manifests copied to {mdir}")
    ok = verify(man, root)
    raise SystemExit(0 if ok else 1)


def verify(man, root):
    t0 = time.time()
    bad, miss = [], []
    for f in man["files"]:
        p = os.path.join(root, f["path"])
        if not os.path.isfile(p):
            miss.append(f["path"])
        elif os.path.getsize(p) != f["bytes"] or sha256(p) != f["sha256"]:
            bad.append(f["path"])
    extra = []
    if man.get("forbidden"):
        # step 5: nothing under the root may have a DEV-TEST / TEST-B / full-cache hash or a forbidden path
        sizes = {f["bytes"] for f in man["forbidden"]}
        fh = {f["sha256"] for f in man["forbidden"]}
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x not in (".git", "__pycache__")]
            for fn in fs:
                p = os.path.join(d, fn)
                rp = os.path.relpath(p, root).replace(os.sep, "/")
                if rp.startswith(".n3_box/"):
                    continue
                if FORBIDDEN_PATH_5.search(rp) and not rp.startswith("external/mcf/"):
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
    raise SystemExit(0 if verify(json.load(open(p)), root) else 1)


# the box-side output trees pulled back at step 8 (everything CSD-derived the runs wrote)
PULL_ROOTS = ["results/n3", "n3_mcf/runs", "n3_mcf/infer", "n3_mcf/_gateC", "n3_mcf/_timing", "n3_mcf/_smoke",
              "external/mcf/outputs"]
PULL_SKIP = ("__pycache__",)


def cmd_pullback(args):
    """Box, §7 step 8: tar every file the run wrote under PULL_ROOTS (uploaded files with unchanged hashes are left
    out), with a manifest; all of it is CSD-derived (csd=1) and is deleted by n3_box_cleanup.sh afterwards."""
    root = os.path.abspath(args.root)
    uploaded = {}
    mdir = os.path.join(root, MAN_SUBDIR)
    for fn in sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []:
        if fn.endswith(".manifest.json"):
            for f in json.load(open(os.path.join(mdir, fn)))["files"]:
                uploaded[f["path"]] = f["sha256"]
    files = []
    for pr in PULL_ROOTS:
        base = os.path.join(root, pr)
        for d, dirs, fs in os.walk(base):
            dirs[:] = [x for x in dirs if x not in PULL_SKIP]
            for fn in fs:
                p = os.path.join(d, fn)
                rp = os.path.relpath(p, root).replace(os.sep, "/")
                h = sha256(p)
                if uploaded.get(rp) == h:
                    continue
                files.append({"group": "pullback", "csd": True, "path": rp, "bytes": os.path.getsize(p), "sha256": h})
    label = f"pullback_{args.label}"
    man = {"format": FORMAT, "label": label, "step": 8, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "files": files, "missing": [], "forbidden": [], "checks": {}, "problems": [], "roots": PULL_ROOTS}
    os.makedirs(args.out, exist_ok=True)
    mfiles = write_manifest_files(man, mdir)
    home = os.path.expanduser(os.environ.get("N3_MANIFEST_DIR", "~/n3_manifests"))
    os.makedirs(home, exist_ok=True)
    for m in mfiles:
        shutil.copy2(m, os.path.join(home, os.path.basename(m)))
    tar = os.path.join(args.out, f"n3_{label}.tar")
    with tarfile.open(tar + ".tmp", "w", format=tarfile.PAX_FORMAT) as tf:
        for f in files:
            tf.add(os.path.join(root, f["path"]), arcname=f["path"], recursive=False)
        for m in mfiles:
            tf.add(m, arcname=f"{MAN_SUBDIR}/{os.path.basename(m)}", recursive=False)
    os.replace(tar + ".tmp", tar)
    th = sha256(tar)
    open(tar + ".sha256", "w").write(f"{th}  {os.path.basename(tar)}\n")
    print(f"{len(files)} files ({sum(f['bytes'] for f in files) / 1e6:.1f} MB) -> {tar} (sha256 {th[:16]}...); "
          f"manifest also in {home}")
    print(f"laptop: scp the tar + .sha256, then python scripts/n3_box_manifest.py unpack --tar {os.path.basename(tar)} "
          "--root . --no-overwrite")


# ------------------------------------------------------------------------------------------ pins
SYMMC_DIRECT = ["numpy", "scipy", "pymatgen", "spglib", "monty", "ase", "rdkit", "gemmi", "networkx", "pandas",
                "psutil", "pyyaml", "pytest"]
MCF_DIRECT = ["torch-geometric", "pytorch-lightning", "hydra-core", "omegaconf", "wandb", "rdkit", "ase", "pymatgen",
              "spglib", "monty", "numpy", "scipy", "pandas", "networkx", "h5py", "tqdm", "rich", "matplotlib",
              "plotly", "seaborn", "psutil", "pyyaml", "p-tqdm", "gputil"]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--step", type=int, required=True, choices=[5, 7])
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
    p.add_argument("--step", type=int, choices=[5, 7], default=5)
    p.add_argument("--label", default=None, help="manifest label (default step<N>)")
    p.add_argument("--root", required=True)
    p = sub.add_parser("pullback")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    sub.add_parser("pins")
    args = ap.parse_args()
    {"list": cmd_list, "pack": cmd_pack, "unpack": cmd_unpack, "verify": cmd_verify, "pullback": cmd_pullback,
     "pins": cmd_pins}[args.cmd](args)


if __name__ == "__main__":
    main()
