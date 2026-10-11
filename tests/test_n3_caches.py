"""Portable cache integrity and step-5 held-out isolation; entirely synthetic inputs."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
import n3_common as C
import n3_make_set_caches as M
from n3_set_cache import FORMAT, content_sha256, file_sha256, list_sha256


@pytest.fixture
def caches(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_CACHE", {})
    monkeypatch.setattr(C, "SET_SIZES", {"train": 1, "val": 2, "sel": 1, "valsel": 3, "testB": 1, "devtest": 1})
    monkeypatch.setattr(C, "SETS_DIR", str(tmp_path / "lists"))
    os.makedirs(C.SETS_DIR)
    for name, ref in (("sel", "SEL001"), ("testB", "TEST01")):
        with open(os.path.join(C.SETS_DIR, name + "_refcodes.txt"), "w") as f:
            f.write(ref + "\n")
    monkeypatch.setenv("N3_SET_CACHES", str(tmp_path / "sets"))
    monkeypatch.setenv("N3_SET_CACHE_ONLY", "1")
    os.makedirs(C.set_cache_dir())

    def write(name, refs, values=None):
        path = os.path.join(C.set_cache_dir(), name + ".pt")
        items = [{"refcode": r, "value": torch.tensor([float(i if values is None else values[i])])}
                 for i, r in enumerate(refs)]
        source = [{"path": "absent-source.pt", "sha256": "0" * 64, "size": 0, "mtime_ns": 0}]
        frozen = {s: file_sha256(os.path.join(C.SETS_DIR, s + "_refcodes.txt")) for s in ("sel", "testB")
                  if name == s or (name == "valsel" and s == "sel")}
        digest = content_sha256(items)
        blob = {"format": FORMAT, "set": name, "n": len(items), "items": items, "refcodes": refs,
                "source": source, "content_sha256": digest, "frozen_list_sha256": frozen}
        torch.save(blob, path)
        side = {k: blob[k] for k in ("format", "set", "n", "source", "content_sha256", "frozen_list_sha256")}
        side.update(file=name + ".pt", sha256=file_sha256(path), bytes=os.path.getsize(path),
                    refcodes_sha256=list_sha256(refs))
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(side, f)
        return path, items
    return write


def test_step5_loads_allowed_sets_without_touching_full_caches(caches, monkeypatch):
    caches("train", ["TRAIN1"])
    caches("val", ["VAL001", "VAL002"])
    caches("sel", ["SEL001"])
    def full_cache_forbidden():
        raise AssertionError("full cache opened")
    monkeypatch.setattr(C, "dev_split", full_cache_forbidden)
    assert [x["refcode"] for x in C.load_set("valsel")] == ["VAL001", "VAL002", "SEL001"]
    for name in ("devtest", "testB"):
        with pytest.raises(FileNotFoundError, match="forbidden"):
            C.load_set(name)


def test_strict_mode_blocks_previously_memoized_full_dev(caches):
    C._CACHE["dev"] = {"val": [{"refcode": "SECRET"}]}
    with pytest.raises(RuntimeError, match="DEV-TEST"):
        C.dev_split()


def test_replaced_cache_invalidates_memo_and_tampering_is_refused(caches):
    p, _ = caches("val", ["VAL001", "VAL002"], [1, 2])
    assert C.load_set("val")[0]["value"].item() == 1
    caches("val", ["VAL001", "VAL002"], [8, 9])
    assert C.load_set("val")[0]["value"].item() == 8
    with open(p, "ab") as f:
        f.write(b"tamper")
    with pytest.raises(RuntimeError, match="differs"):
        C.load_set("val")


def test_payload_header_and_digest_fail_closed(caches):
    p, _ = caches("val", ["VAL001", "VAL002"])
    blob = torch.load(p, weights_only=False)
    blob["items"][0]["value"] += 1
    torch.save(blob, p)
    sidepath = os.path.splitext(p)[0] + ".json"
    side = json.load(open(sidepath))
    side.update(sha256=file_sha256(p), bytes=os.path.getsize(p))
    json.dump(side, open(sidepath, "w"))
    with pytest.raises(RuntimeError, match="payload digest"):
        C.load_set("val")


def test_valsel_requires_exact_val_prefix_and_payload(caches):
    caches("val", ["VAL001", "VAL002"])
    caches("sel", ["SEL001"])
    caches("valsel", ["OTHER1", "VAL002", "SEL001"])
    with pytest.raises(RuntimeError, match="VAL followed by SEL"):
        C.load_set("valsel")


def test_local_valsel_cache_can_verify_full_source_fallback(caches, monkeypatch):
    _, val = caches("val", ["VAL001", "VAL002"], [0, 1])
    _, sel = caches("sel", ["SEL001"], [2])
    caches("valsel", ["VAL001", "VAL002", "SEL001"], [0, 1, 2])
    for name in ("val", "sel"):
        for suffix in (".pt", ".json"):
            os.remove(os.path.join(C.set_cache_dir(), name + suffix))
    monkeypatch.delenv("N3_SET_CACHE_ONLY")
    monkeypatch.setattr(C, "dev_split", lambda: {"val": val})
    C._CACHE["pool"] = {a["refcode"]: a for a in sel}
    assert len(C.load_set("valsel")) == 3
    val[0]["value"] += 1
    with pytest.raises(RuntimeError, match="VAL followed by SEL"):
        C.load_set("valsel")


def test_incomplete_valsel_cache_is_refused(caches):
    caches("val", ["VAL001", "VAL002"])
    caches("sel", ["SEL001"])
    path, _ = caches("valsel", ["VAL001", "VAL002", "SEL001"])
    os.remove(path)
    with pytest.raises(RuntimeError, match="must both exist"):
        C.load_set("valsel")


def test_check_requires_all_requested_caches(caches):
    caches("train", ["TRAIN1"])
    with pytest.raises(SystemExit):
        M.cmd_check(SimpleNamespace(sets=["train", "val", "sel"]))


def test_frozen_refcodes_and_count_are_verified(caches):
    caches("sel", ["OTHER1"])
    with pytest.raises(RuntimeError, match="frozen order"):
        C.load_set("sel")
    caches("val", ["VAL001"])
    with pytest.raises(RuntimeError, match="expected 2"):
        C.load_set("val")
