"""Canonical identity and validation for private N3 per-set input caches."""
import hashlib
import json
import os

import numpy as np
import torch

FORMAT = "n3-set-cache/1"


def _feed(h, x):
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


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_cache(path, name, expected_n, expected_refs=None, frozen_hashes=None):
    """Validate the file, sidecar, canonical payload and frozen ordering before using items.

    The sidecar travels inside the checksummed upload manifest; it is not a substitute for that manifest.
    This function needs no full dev/pool cache, so it works on a blinded box.
    """
    sidepath = os.path.splitext(path)[0] + ".json"
    with open(sidepath, encoding="utf-8") as f:
        side = json.load(f)
    if side.get("format") != FORMAT or side.get("set") != name or side.get("file") != os.path.basename(path):
        raise RuntimeError(f"{path}: wrong sidecar format/set/file")
    if side.get("bytes") != os.path.getsize(path) or side.get("sha256") != file_sha256(path):
        raise RuntimeError(f"{path}: file differs from its sidecar")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != FORMAT or blob.get("set") != name:
        raise RuntimeError(f"{path}: wrong cache format/set")
    items = blob.get("items")
    if not isinstance(items, list) or len(items) != expected_n:
        raise RuntimeError(f"{path}: expected {expected_n} items")
    refs = [a.get("refcode") for a in items]
    if (any(not isinstance(r, str) or not r for r in refs) or len(set(refs)) != len(refs)
            or refs != blob.get("refcodes") or blob.get("n") != len(refs) or side.get("n") != len(refs)):
        raise RuntimeError(f"{path}: inconsistent count/refcodes")
    if expected_refs is not None and refs != expected_refs:
        raise RuntimeError(f"{path}: not the frozen order of {name}")
    if side.get("refcodes_sha256") != list_sha256(refs):
        raise RuntimeError(f"{path}: refcode digest mismatch")
    digest = content_sha256(items)
    if blob.get("content_sha256") != digest or side.get("content_sha256") != digest:
        raise RuntimeError(f"{path}: payload digest mismatch")
    if blob.get("source") != side.get("source") or not isinstance(side.get("source"), list) or not side["source"]:
        raise RuntimeError(f"{path}: source provenance mismatch")
    if (blob.get("frozen_list_sha256", {}) != side.get("frozen_list_sha256", {})
            or (frozen_hashes is not None and side.get("frozen_list_sha256", {}) != frozen_hashes)):
        raise RuntimeError(f"{path}: frozen-list provenance mismatch")
    return items, side
