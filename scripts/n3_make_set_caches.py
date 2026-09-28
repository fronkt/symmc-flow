"""N3 per-set input caches, so a box never receives a set before its step (protocol §7 step 5: "Upload TRAIN,
VAL and SEL inputs only; TEST-B and DEV-TEST inputs stay off the box").

n3_common.load_set reads the FULL dev cache data/csd_mol/g2_asym.pt (TRAIN + VAL + DEV-TEST) for train / val /
devtest and the pool cache data/csd_testB/n3_asym_pool.pt (SEL + TEST-B) for sel / testB. Neither file may go to
the box before step 7. This script writes one file per set holding exactly that set's asym items in frozen order:

    <dir>/<set>.pt     torch.save({"format", "set", "n", "refcodes", "items", "content_sha256", "source"})
    <dir>/<set>.json   sidecar: file SHA-256 and size, n, content_sha256, refcodes_sha256, source file identity
                       (path, SHA-256, size, mtime_ns) -- what the patched load_set checks before trusting a cache
<dir> = results/n3/private/set_caches (gitignored; override with N3_SET_CACHES). Sets: train val sel valsel testB
devtest. content_sha256 is a canonical digest of the item tensors (dtype, shape, bytes) and python values, so it
is independent of torch.save's container bytes.

Build-time guarantees (each asserted, the build stops otherwise):
  * the items equal, object for object, what the CURRENT load_set returns (source = dev_split / the pool cache in
    frozen-list order, never an existing cache);
  * no tensor storage of a set's items is shared with an item outside that set (torch.save writes the WHOLE
    storage behind a view, so a shared storage would carry another set's bytes);
  * the written file re-loads to the same content digest, contains every own refcode as a pickle string token
    (positive control) and no refcode of any other set (negative scan over the raw file bytes).

    python scripts/n3_make_set_caches.py build                       # all six sets
    python scripts/n3_make_set_caches.py check                       # any machine: sidecar SHA, digest, order
    python scripts/n3_make_set_caches.py simulate --patched <n3_common.py with n3_common_setcache.patch applied>

`simulate` (laptop only; needs the full caches) imports the patched n3_common under another module name, points it
at temp cache dirs, and checks (1) load_set with all six caches == the current load_set item by item (bitwise
tensor equality, dtype, shape, python values) for every set, and that neither full cache is opened; (2) the same
with no caches (the fallback path is unchanged); (3) with ONLY train/val/sel caches and N3_SET_CACHE_ONLY=1,
load_set('devtest') and load_set('testB') raise, dev_split() raises, load_set('valsel') == val + sel, and neither
full cache is opened. CSD-derived content is never printed: only counts, hashes and pass flags.
"""
import argparse
import builtins
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

import numpy as np
import torch

FORMAT = "n3-set-cache/1"
CACHE_SETS = ("train", "val", "sel", "valsel", "testB", "devtest")
BASE_SETS = ("train", "val", "devtest", "sel", "testB")
DEFAULT_DIR = os.path.join(C.PRIVATE, "set_caches")


def cache_dir():
    return os.environ.get("N3_SET_CACHES", DEFAULT_DIR)


# ------------------------------------------------------------------------------------------ digests
def _feed(h, x):
    """Canonical, type-tagged digest of nested python / torch / numpy values."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous()
        h.update(b"T" + str(t.dtype).encode() + repr(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    elif isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h.update(b"N" + str(a.dtype).encode() + repr(a.shape).encode() + a.tobytes())
    elif isinstance(x, dict):
        h.update(b"D" + str(len(x)).encode())
        for k in sorted(x, key=repr):
            _feed(h, k)
            _feed(h, x[k])
    elif isinstance(x, (list, tuple)):
        h.update((b"L" if isinstance(x, list) else b"U") + str(len(x)).encode())
        for v in x:
            _feed(h, v)
    elif isinstance(x, (str, int, float, bool, type(None), np.generic)):
        h.update(b"S" + type(x).__name__.encode() + repr(x).encode())
    else:
        raise TypeError(f"no canonical digest for {type(x).__name__}")
    h.update(b";")


def content_sha256(items):
    h = hashlib.sha256()
    _feed(h, items)
    return h.hexdigest()


def list_sha256(xs):
    return hashlib.sha256("\n".join(xs).encode()).hexdigest()


def deep_equal(a, b, path="item"):
    """None if a == b exactly (tensors bitwise, same dtype and shape; same types and keys), else the first
    difference as a path string (no values)."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return f"{path}: dtype/shape"
        ab = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        bb = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return None if torch.equal(ab, bb) else f"{path}: tensor bytes"
    if isinstance(a, np.ndarray):
        ok = a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == \
            np.ascontiguousarray(b).tobytes()
        return None if ok else f"{path}: ndarray"
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return f"{path}: keys"
        for k in a:
            d = deep_equal(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            d = deep_equal(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: value"


def _tensors(x, out):
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            _tensors(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _tensors(v, out)
    return out


_STR_TOKEN = re.compile(rb"X([\x01-\x40])\x00\x00\x00|\x8c([\x01-\x40])")


def string_tokens(raw):
    """Every short ASCII string a pickle stream in `raw` spells as a length-prefixed token (protocol 2
    BINUNICODE 'X'+<u32 len>, protocols 4-5 SHORT_BINUNICODE 0x8c+<u8 len>); one regex pass over the file.
    False starts inside tensor bytes only add strings that are no refcode."""
    found = set()
    for m in _STR_TOKEN.finditer(raw):
        n = (m.group(1) or m.group(2))[0]
        s = raw[m.end():m.end() + n]
        if len(s) == n and s.isascii():
            found.add(s.decode("ascii"))
    return found


# ------------------------------------------------------------------------------------------ sources
def source_items():
    """Every set exactly as the CURRENT n3_common.load_set builds it, from the full caches (never a set cache)."""
    if os.environ.get("N3_SET_CACHE_ONLY") == "1":
        raise SystemExit("N3_SET_CACHE_ONLY=1: this is a box; caches are built on the laptop from the full caches")
    for p in (C.DEV_CACHE, C.POOL_CACHE):
        if not os.path.exists(p):
            raise SystemExit(f"missing {os.path.relpath(p, C.REPO)}: caches are built on the laptop only")
    dev = C.dev_split()
    if "pool" not in C._CACHE:                    # the memo load_set's pool path uses (same objects)
        C._CACHE["pool"] = {a["refcode"]: a for a in torch.load(C.POOL_CACHE, weights_only=False)}
    pool = C._CACHE["pool"]
    out = {"train": dev["train"], "val": dev["val"], "devtest": dev["devtest"],
           "sel": [pool[r] for r in C.frozen_refcodes("sel")], "testB": [pool[r] for r in C.frozen_refcodes("testB")]}
    out["valsel"] = out["val"] + out["sel"]
    return out


def file_ident(path):
    st = os.stat(path)
    return {"path": os.path.relpath(path, C.REPO).replace("\\", "/"), "sha256": C.sha256(path),
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def set_sources(name):
    if name in ("train", "val", "devtest"):
        return [C.DEV_CACHE]
    if name in ("sel", "testB"):
        return [C.POOL_CACHE]
    return [C.DEV_CACHE, C.POOL_CACHE]


# ------------------------------------------------------------------------------------------ build
def cmd_build(args):
    t0 = time.time()
    src = source_items()
    # the items must be the ones load_set returns without caches: the same objects (both memoise the full caches);
    # a patched n3_common is pointed at an empty cache dir for this comparison so it takes its fallback path
    env_prev = os.environ.get("N3_SET_CACHES")
    empty = tempfile.mkdtemp(prefix="n3_setcache_empty_")
    os.environ["N3_SET_CACHES"] = empty
    try:
        for s in CACHE_SETS:
            cur = C.load_set(s)
            assert len(cur) == len(src[s]) and all(x is y for x, y in zip(cur, src[s])), f"{s}: source != load_set"
    finally:
        if env_prev is None:
            os.environ.pop("N3_SET_CACHES", None)
        else:
            os.environ["N3_SET_CACHES"] = env_prev
        os.rmdir(empty)
    ref_sets = {s: [a["refcode"] for a in src[s]] for s in CACHE_SETS}
    for s in ("sel", "testB"):
        assert ref_sets[s] == C.frozen_refcodes(s), f"{s}: not the frozen list"
    for s in BASE_SETS:
        assert len(set(ref_sets[s])) == len(ref_sets[s]), f"{s}: duplicate refcode"
    base_union = [r for s in BASE_SETS for r in ref_sets[s]]
    assert len(set(base_union)) == len(base_union), "a refcode occurs in two base sets"
    # storage ownership: storage pointer -> refcodes of the items that reference it
    owner = {}
    for s in BASE_SETS:
        for a in src[s]:
            for t in _tensors(a, []):
                owner.setdefault(t.untyped_storage().data_ptr(), set()).add(a["refcode"])
    d = cache_dir()
    os.makedirs(d, exist_ok=True)
    sets = args.sets or list(CACHE_SETS)
    summary = {}
    for s in sets:
        items, refs = src[s], ref_sets[s]
        mine = set(refs)
        foreign_storage = 0
        for a in items:
            for t in _tensors(a, []):
                if not owner[t.untyped_storage().data_ptr()] <= mine:
                    foreign_storage += 1
        assert foreign_storage == 0, f"{s}: {foreign_storage} tensors share storage with items of another set"
        digest = content_sha256(items)
        srcs = [file_ident(p) for p in set_sources(s)]
        blob = {"format": FORMAT, "set": s, "n": len(items), "refcodes": refs, "items": items,
                "content_sha256": digest, "source": srcs,
                "frozen_list_sha256": {k: C.sha256(os.path.join(C.SETS_DIR, f"{k}_refcodes.txt"))
                                       for k in ("sel", "testB") if k == s or (s == "valsel" and k == "sel")}}
        pt = os.path.join(d, f"{s}.pt")
        torch.save(blob, pt + ".tmp")
        os.replace(pt + ".tmp", pt)
        # re-load: same content; raw-byte refcode scan (positive control on own, negative on every other set)
        back = torch.load(pt, weights_only=False)
        assert back["refcodes"] == refs and content_sha256(back["items"]) == digest, f"{s}: re-load differs"
        toks = string_tokens(open(pt, "rb").read())
        own_missing = len(mine - toks)
        others = set(base_union) - mine
        foreign_hits = others & toks
        assert own_missing == 0, f"{s}: positive control failed ({own_missing} own refcodes not found as tokens)"
        assert not foreign_hits, f"{s}: {len(foreign_hits)} refcodes of other sets occur in the file"
        side = {"format": FORMAT, "set": s, "file": f"{s}.pt", "sha256": C.sha256(pt), "bytes": os.path.getsize(pt),
                "n": len(items), "content_sha256": digest, "refcodes_sha256": list_sha256(refs), "source": srcs,
                "frozen_list_sha256": blob["frozen_list_sha256"], "torch": torch.__version__,
                "script_sha256": C.sha256(os.path.abspath(__file__)), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checks": {"same_objects_as_load_set": True, "no_foreign_storage": True, "reload_digest": True,
                           "own_refcode_tokens_found": len(refs), "foreign_refcode_tokens_found": 0,
                           "foreign_refcodes_scanned": len(others)}}
        tmp = os.path.join(d, f"{s}.json.tmp")
        json.dump(side, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(d, f"{s}.json"))
        summary[s] = {"n": len(items), "MB": round(side["bytes"] / 1e6, 2), "sha256": side["sha256"][:16],
                      "content_sha256": digest[:16]}
        print(s, json.dumps(summary[s]), flush=True)
    print(f"wrote {len(sets)} caches to {os.path.relpath(d, C.REPO)} in {time.time() - t0:.0f} s")


# ------------------------------------------------------------------------------------------ check
def check_one(d, s, deep=True):
    """Verify one cache against its sidecar (works on the box: needs no full cache). Returns a dict."""
    pt, js = os.path.join(d, f"{s}.pt"), os.path.join(d, f"{s}.json")
    r = {"set": s, "present": os.path.exists(pt)}
    if not r["present"]:
        return r
    side = json.load(open(js))
    r["file_sha_ok"] = C.sha256(pt) == side["sha256"]
    if deep:
        blob = torch.load(pt, weights_only=False)
        refs = [a["refcode"] for a in blob["items"]]
        r["n"] = len(refs)
        r["header_ok"] = blob["format"] == FORMAT and blob["set"] == s and blob["n"] == len(refs) == side["n"]
        r["refcodes_ok"] = refs == blob["refcodes"] and list_sha256(refs) == side["refcodes_sha256"]
        r["digest_ok"] = content_sha256(blob["items"]) == side["content_sha256"] == blob["content_sha256"]
        lists = {k: os.path.join(C.SETS_DIR, f"{k}_refcodes.txt") for k in ("sel", "testB")}
        if s in ("sel", "testB") and os.path.exists(lists[s]):
            r["frozen_order_ok"] = refs == C.frozen_refcodes(s)
        if s == "valsel" and os.path.exists(lists["sel"]):
            r["frozen_order_ok"] = len(refs) == 400 and refs[100:] == C.frozen_refcodes("sel")
    r["pass"] = all(v for k, v in r.items() if k.endswith("_ok"))
    return r


def cmd_check(args):
    d = cache_dir()
    res = [check_one(d, s) for s in (args.sets or CACHE_SETS)]
    for r in res:
        print(json.dumps(r))
    present = [r for r in res if r["present"]]
    ok = bool(present) and all(r["pass"] for r in present)
    print("PASS" if ok else "FAIL", f"({len(present)} caches present in {d})")
    if not ok:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------ simulate
class _OpenLog:
    """Record every path opened through builtins.open / torch.load while active."""

    def __init__(self):
        self.paths = []

    def __enter__(self):
        self._open, self._tload = builtins.open, torch.load
        log = self.paths

        def op(file, *a, **k):
            if isinstance(file, (str, bytes, os.PathLike)):
                log.append(os.path.abspath(os.fsdecode(file)))
            return self._open(file, *a, **k)

        def tl(f, *a, **k):
            if isinstance(f, (str, bytes, os.PathLike)):
                log.append(os.path.abspath(os.fsdecode(f)))
            return self._tload(f, *a, **k)
        builtins.open, torch.load = op, tl
        return self

    def __exit__(self, *exc):
        builtins.open, torch.load = self._open, self._tload
        return False

    def touched(self, path):
        p = os.path.normcase(os.path.abspath(path))
        return any(os.path.normcase(x) == p for x in self.paths)


def _load_patched(path, tag):
    spec = importlib.util.spec_from_file_location(f"n3_common_patched_{tag}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # the copy lives outside the repo: point its repo-relative globals at this checkout
    m.REPO, m.DEV_CACHE, m.POOL_CACHE = C.REPO, C.DEV_CACHE, C.POOL_CACHE
    m.SETS_DIR, m.PRIVATE = C.SETS_DIR, C.PRIVATE
    return m


def _compare_all(ref, m, sets):
    out = {}
    for s in sets:
        a, b = ref[s], m.load_set(s)
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if deep_equal(x, y)]
        out[s] = {"n": len(b), "n_ref": len(a), "identical": len(a) == len(b) and not diffs,
                  "first_diff": deep_equal(a[diffs[0]], b[diffs[0]]) if diffs else None}
    return out


def cmd_simulate(args):
    t0 = time.time()
    src_dir = cache_dir()
    for s in CACHE_SETS:
        if not os.path.exists(os.path.join(src_dir, f"{s}.pt")):
            raise SystemExit(f"build the caches first (missing {s}.pt in {src_dir})")
    ref = {s: C.load_set(s) for s in CACHE_SETS}            # the CURRENT (unpatched) load_set
    env0 = {k: os.environ.get(k) for k in ("N3_SET_CACHES", "N3_SET_CACHE_ONLY")}
    tmp = tempfile.mkdtemp(prefix="n3_setcache_sim_", dir=args.tmp or None)
    report = {"patched": os.path.abspath(args.patched), "patched_sha256": C.sha256(args.patched)}
    try:
        full, part, none = (os.path.join(tmp, x) for x in ("all6", "train_val_sel", "none"))
        for dd in (full, part, none):
            os.makedirs(dd)
        for s in CACHE_SETS:
            for ext in (".pt", ".json"):
                shutil.copy2(os.path.join(src_dir, s + ext), os.path.join(full, s + ext))
                if s in ("train", "val", "sel"):
                    shutil.copy2(os.path.join(src_dir, s + ext), os.path.join(part, s + ext))

        # (1) all six caches, strict mode: identical, full caches never opened
        os.environ.update(N3_SET_CACHES=full, N3_SET_CACHE_ONLY="1")
        m = _load_patched(args.patched, "all6")
        with _OpenLog() as lg:
            cmp1 = _compare_all(ref, m, CACHE_SETS)
        report["1_all_caches_strict"] = {"sets": cmp1, "dev_cache_opened": lg.touched(C.DEV_CACHE),
                                         "pool_cache_opened": lg.touched(C.POOL_CACHE)}
        # (1b) all six caches, default (non-strict) mode: identical, full caches not loaded
        os.environ.pop("N3_SET_CACHE_ONLY")
        m = _load_patched(args.patched, "all6_default")
        with _OpenLog() as lg:
            cmp1b = _compare_all(ref, m, CACHE_SETS)
        report["1b_all_caches_default"] = {"sets": cmp1b, "dev_cache_opened": lg.touched(C.DEV_CACHE),
                                           "pool_cache_opened": lg.touched(C.POOL_CACHE)}
        # (2) no caches, default mode: the fallback path is the old one
        os.environ["N3_SET_CACHES"] = none
        m = _load_patched(args.patched, "none")
        report["2_no_caches_fallback"] = {"sets": _compare_all(ref, m, CACHE_SETS)}
        # (3) the step-5 box: only train / val / sel caches, strict
        os.environ.update(N3_SET_CACHES=part, N3_SET_CACHE_ONLY="1")
        m = _load_patched(args.patched, "part")
        r3 = {}
        with _OpenLog() as lg:
            for s in ("devtest", "testB"):
                try:
                    m.load_set(s)
                    r3[f"load_set_{s}"] = "LOADED (FAIL)"
                except Exception as e:  # noqa: BLE001 -- the point is that it raises
                    r3[f"load_set_{s}"] = f"raised {type(e).__name__}"
            try:
                m.dev_split()
                r3["dev_split"] = "LOADED (FAIL)"
            except Exception as e:  # noqa: BLE001
                r3["dev_split"] = f"raised {type(e).__name__}"
            r3["valid_sets"] = _compare_all(ref, m, ("train", "val", "sel", "valsel"))
            try:                                              # save_bundle's frozen-order check works on the box
                import numpy as _np  # noqa: F401
                p = os.path.join(tmp, "b.pt")
                refs = [a["refcode"] for a in ref["valsel"]]
                m.save_bundle(p, arm="sim", seed=0, set_name="valsel", refcodes=refs, kind="rasym",
                              valid=torch.ones(len(refs), 1, dtype=torch.bool),
                              sel={"lj": torch.zeros(len(refs), 1)}, meta={}, R=torch.eye(3).expand(len(refs), 1, 3, 3))
                r3["save_bundle_valsel"] = "ok"
            except Exception as e:  # noqa: BLE001
                r3["save_bundle_valsel"] = f"raised {type(e).__name__}: {str(e)[:120]}"
        r3["dev_cache_opened"], r3["pool_cache_opened"] = lg.touched(C.DEV_CACHE), lg.touched(C.POOL_CACHE)
        report["3_step5_box_strict"] = r3
        # (4) a tampered cache is refused
        bad = os.path.join(tmp, "tamper")
        os.makedirs(bad)
        for ext in (".pt", ".json"):
            shutil.copy2(os.path.join(src_dir, "val" + ext), os.path.join(bad, "val" + ext))
        with open(os.path.join(bad, "val.pt"), "ab") as fh:
            fh.write(b"\0")
        os.environ.update(N3_SET_CACHES=bad, N3_SET_CACHE_ONLY="1")
        m = _load_patched(args.patched, "tamper")
        try:
            m.load_set("val")
            report["4_tampered_cache"] = "LOADED (FAIL)"
        except Exception as e:  # noqa: BLE001
            report["4_tampered_cache"] = f"raised {type(e).__name__}"
    finally:
        for k, v in env0.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)
    r1, r1b, r2, r3 = (report[k] for k in ("1_all_caches_strict", "1b_all_caches_default", "2_no_caches_fallback",
                                           "3_step5_box_strict"))
    checks = {
        "all_caches_identical_strict": all(v["identical"] for v in r1["sets"].values()),
        "all_caches_full_caches_untouched": not (r1["dev_cache_opened"] or r1["pool_cache_opened"]),
        "all_caches_identical_default": all(v["identical"] for v in r1b["sets"].values()),
        "all_caches_default_full_caches_not_loaded": not (r1b["dev_cache_opened"] or r1b["pool_cache_opened"]),
        "no_caches_identical": all(v["identical"] for v in r2["sets"].values()),
        "step5_devtest_raises": r3["load_set_devtest"].startswith("raised"),
        "step5_testB_raises": r3["load_set_testB"].startswith("raised"),
        "step5_dev_split_raises": r3["dev_split"].startswith("raised"),
        "step5_train_val_sel_valsel_identical": all(v["identical"] for v in r3["valid_sets"].values()),
        "step5_save_bundle_valsel_ok": r3["save_bundle_valsel"] == "ok",
        "step5_full_caches_untouched": not (r3["dev_cache_opened"] or r3["pool_cache_opened"]),
        "tampered_cache_refused": report["4_tampered_cache"].startswith("raised"),
    }
    report["checks"], report["pass"] = checks, all(checks.values())
    report["sec"] = round(time.time() - t0, 1)
    out = args.out or os.path.join(src_dir, "simulate_report.json")
    json.dump(report, open(out, "w"), indent=1)
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    print("counts:", {s: v["n"] for s, v in r1["sets"].items()}, "| step-5 box:",
          {k: r3[k] for k in ("load_set_devtest", "load_set_testB", "dev_split")})
    print("PASS" if report["pass"] else "FAIL", f"-> {out} ({report['sec']} s)")
    if not report["pass"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--sets", nargs="+", choices=CACHE_SETS)
    p = sub.add_parser("check")
    p.add_argument("--sets", nargs="+", choices=CACHE_SETS)
    p = sub.add_parser("simulate")
    p.add_argument("--patched", required=True, help="a copy of scripts/n3_common.py with the set-cache patch applied")
    p.add_argument("--tmp", default=None, help="parent dir for the temporary cache dirs (deleted afterwards)")
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "simulate": cmd_simulate}[args.cmd](args)


if __name__ == "__main__":
    main()
