"""N3 / G3 benchmark: shared plumbing for every arm (protocol: tasks/n3_protocol.md).

Frozen sets (protocol §1.1, §2.2)
    train / val / devtest   seed-0 split of data/csd_mol/g2_asym.pt (perm[300:], perm[200:300], perm[:200])
    sel / testB             fresh pool, in the order of tasks/n3_sets/{sel,testB}_refcodes.txt; items come
                            from data/csd_testB/n3_asym_pool.pt (written by scripts/n3_build_testB.py)
    valsel                  val followed by sel (i = 0..399)
Split codes for seeds: valsel/val/sel 0, testB 1, devtest 2.

Draw bundle (one file per arm x seed x set; torch.save of a dict; CSD-derived -> results/n3/private/)
    arm, seed, set          str, int (training seed / Haar seed base; -1 = deterministic arm), str
    refcodes                list[str], exactly the frozen order of `set`
    kind                    "rasym": R is a float64 tensor [n, S, 3, 3] of asym orientations, expanded with
                                     the crystal's own spglib ops (expand());
                            "cells": cells[i][j] = dict(lattice [3,3], cart [N,3] whole molecules (never
                                     wrapped per atom), species [N] atomic numbers, mol [N] copy index)
    valid                   bool [n, S]; False = arm failure for that draw (a crystal with no valid draw is
                            an arm MISS, protocol §1.3)
    sel                     dict name -> float tensor [n, S] of selector inputs ("lj", "resid", "ff", ...);
                            non-finite values rank last
    meta                    dict: command line, code versions, config values, timings
Every scorer adapter turns a bundle draw into a pymatgen Structure; truth is always to_structure(orig).
"""
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import torch

DEV_CACHE = os.path.join(REPO, "data", "csd_mol", "g2_asym.pt")
POOL_CACHE = os.path.join(REPO, "data", "csd_testB", "n3_asym_pool.pt")
SETS_DIR = os.path.join(REPO, "tasks", "n3_sets")
PRIVATE = os.path.join(REPO, "results", "n3", "private")
SPLIT_CODE = {"valsel": 0, "val": 0, "sel": 0, "testB": 1, "devtest": 2}
SETS = ("train", "val", "devtest", "sel", "testB", "valsel")

_CACHE = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dev_split():
    """The seed-0 split of g2_asym.pt, exactly as G2/N1/N2 used it."""
    if "dev" not in _CACHE:
        elig = torch.load(DEV_CACHE, weights_only=False)
        perm = torch.randperm(len(elig), generator=torch.Generator().manual_seed(0)).tolist()
        _CACHE["dev"] = {"devtest": [elig[i] for i in perm[:200]],
                         "val": [elig[i] for i in perm[200:300]],
                         "train": [elig[i] for i in perm[300:]]}
    return _CACHE["dev"]


def frozen_refcodes(name):
    path = os.path.join(SETS_DIR, f"{name}_refcodes.txt")
    return [l.strip() for l in open(path) if l.strip()]


def load_set(name):
    """List of asym items (g2_asym_baselines.asym_item dicts) in the frozen order of `name`."""
    if name in ("train", "val", "devtest"):
        return dev_split()[name]
    if name == "valsel":
        return load_set("val") + load_set("sel")
    if name in ("sel", "testB"):
        if "pool" not in _CACHE:
            _CACHE["pool"] = {a["refcode"]: a for a in torch.load(POOL_CACHE, weights_only=False)}
        pool = _CACHE["pool"]
        return [pool[r] for r in frozen_refcodes(name)]
    raise ValueError(f"unknown set {name!r}; one of {SETS}")


def truth_structure(a):
    from g2_asym_baselines import to_structure
    o = a["orig"]
    return to_structure(o["lattice"], o["centroid"], o["orient"], o["local"], o["Z"],
                        o["atom_mask"], o["mol_mask"])


def rasym_structure(a, R):
    """Structure built from one asym orientation by exact space-group expansion."""
    from g2_asym_baselines import expand, to_structure
    return to_structure(*expand(a, R.double()))


def cell_structure(c):
    """Structure from a 'cells' draw (whole molecules; pymatgen wraps sites, which is harmless for matching)."""
    from pymatgen.core import Lattice, Structure
    lat = Lattice(c["lattice"].double().numpy())
    return Structure(lat, [int(z) for z in c["species"]], c["cart"].double().numpy(),
                     coords_are_cartesian=True)


def save_bundle(path, *, arm, seed, set_name, refcodes, kind, valid, sel, meta, R=None, cells=None):
    expect = [a["refcode"] for a in load_set(set_name)]
    assert list(refcodes) == expect, f"{arm}/{set_name}: refcodes not in frozen order"
    assert kind in ("rasym", "cells")
    n = len(refcodes)
    if kind == "rasym":
        assert R is not None and R.shape[:1] == (n,) and R.shape[-2:] == (3, 3)
    else:
        assert cells is not None and len(cells) == n
    assert valid.shape[0] == n and all(v.shape == valid.shape for v in sel.values())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"arm": arm, "seed": seed, "set": set_name, "refcodes": list(refcodes), "kind": kind,
                "R": R, "cells": cells, "valid": valid, "sel": sel, "meta": meta}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_bundle(path):
    return torch.load(path, weights_only=False)


def bundle_path(arm, seed, set_name):
    return os.path.join(PRIVATE, "draws", f"{arm}_s{seed}_{set_name}.pt")
