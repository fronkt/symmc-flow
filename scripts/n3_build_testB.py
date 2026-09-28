"""N3: build the fresh confirmatory TEST-B and the selection supplement SEL (protocol §1.1, steps 2-6).

Input: the G1 O(3) parse of the 4,500 pool CIFs (rows 3501-8000 of data/csd_mol_scale_big/manifest.csv),
already run as build step 1:
    python scripts/g1_o3_reparse.py --cif-dir data/csd_testB/cif --manifest data/csd_testB/manifest.csv \
        --cache data/csd_testB/ds_o3.pt --out results/n3/g1_o3_reparse_testB.json --workers 5
Steps (all input-side; nothing is sampled or scored):
  2  asym_item (Z'=1, n_mol = K = #spglib ops), drop G1-oracle failures (part-file field 4), then the
     EXPAND ORACLE: (i) orbit check -- every W_k c0 + t_k within 1e-3 (max-abs fractional, min image) of a
     stored copy centroid and no two orbit points within 1e-3; (ii) StructureMatcher().fit(truth,
     expand(a, R0)), no time limit
  3  de-leak: 6-letter family vs the 3,500 dev-corpus refcodes; molecule WL hash (n3_chirality.graph)
     vs the 1,987 g2_asym.pt items (TRAIN, VAL, DEV-TEST)
  4  sort by refcode, perm = randperm(seed 20261001), keep the first item of each 6-letter family -> P
  5  SEL = P[:300], TEST-B = P[300:1300]
  6  write tasks/n3_sets/{sel,testB}_refcodes.txt (committed), data/csd_testB/n3_asym_pool.pt (CSD,
     gitignored) and results/n3/n3_build_testB.json (counts + SHA-256, committed)

    python scripts/n3_build_testB.py --workers 6
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

import torch

POOL_DIR = os.path.join(C.REPO, "data", "csd_testB")


def _check(it):
    """Worker: steps 2 for one parsed item -> (refcode, reason_or_None, asym_item_or_None, wl_hash)."""
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from g2_asym_baselines import asym_item
    from n3_chirality import wl_hash
    ref = it["refcode"]
    a = asym_item(it)
    if a is None:
        return ref, "not Z'=1 general-position (asym_item)", None, None
    real = [m for m in range(it["mol_mask"].shape[0]) if bool(it["mol_mask"][m])]
    cents = it["centroid"][real].double()
    orb = torch.einsum("kij,j->ki", a["W"], a["c0"]) + a["t"]
    wrap = lambda x: x - torch.round(x)
    d_oc = wrap(orb[:, None, :] - cents[None, :, :]).abs().amax(-1)          # (K, n_mol)
    d_oo = wrap(orb[:, None, :] - orb[None, :, :]).abs().amax(-1)
    d_oo.fill_diagonal_(1.0)
    if float(d_oc.min(1).values.max()) > 1e-3 or float(d_oo.min()) < 1e-3:
        return ref, "expand oracle (i): orbit", None, None
    if not bool(StructureMatcher().fit(C.truth_structure(a), C.rasym_structure(a, a["R0"]))):
        return ref, "expand oracle (ii): StructureMatcher", None, None
    return ref, None, a, wl_hash(a["local"].numpy(), a["Z"].numpy())


def _hash(a):
    warnings.filterwarnings("ignore")
    from n3_chirality import wl_hash
    return wl_hash(a["local"].numpy(), a["Z"].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-sel", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.multiprocessing.set_sharing_strategy("file_system")
    from multiprocessing import Pool

    cache = os.path.join(POOL_DIR, "ds_o3.pt")
    parts = sorted(glob.glob(cache + ".parts/part_*.pt"))
    part_rows = [row for p in parts for row in torch.load(p, weights_only=False)]
    g1_ok = {r[0]: bool(r[4]) for r in part_rows if r[2] is not None}
    items = torch.load(cache, weights_only=False)["items"]
    counts = collections.OrderedDict(pool_cifs=len(part_rows), o3_kept=len(items))

    items = [it for it in items if g1_ok.get(it["refcode"], False)]
    counts["after_g1_oracle"] = len(items)
    with Pool(args.workers) as pool:
        res = pool.map(_check, items, chunksize=4)
    reasons = collections.Counter(r[1] for r in res if r[1])
    surv = [(r[2], r[3]) for r in res if r[1] is None]
    counts["step2_drops"] = dict(reasons)
    counts["after_step2"] = len(surv)

    dev_refs = [r["refcode"] for r in csv.DictReader(open(os.path.join(C.REPO, "data", "csd_mol", "manifest.csv")))]
    dev_fam = {r[:6] for r in dev_refs}
    dev_items = torch.load(C.DEV_CACHE, weights_only=False)
    with Pool(args.workers) as pool:
        dev_hash = set(pool.map(_hash, dev_items, chunksize=8))
    fam_drop = [a for a, h in surv if a["refcode"][:6] in dev_fam]
    surv = [(a, h) for a, h in surv if a["refcode"][:6] not in dev_fam]
    wl_drop = [a for a, h in surv if h in dev_hash]
    surv = [(a, h) for a, h in surv if h not in dev_hash]
    counts["step3_family_drops"] = len(fam_drop)
    counts["step3_wl_drops"] = len(wl_drop)
    counts["after_step3"] = len(surv)
    hcount = collections.Counter(h for _, h in surv)
    counts["pool_items_sharing_wl_species_within_pool"] = sum(c for c in hcount.values() if c > 1)

    surv.sort(key=lambda x: x[0]["refcode"])
    perm = torch.randperm(len(surv), generator=torch.Generator().manual_seed(20261001)).tolist()
    seen, P = set(), []
    for i in perm:
        a = surv[i][0]
        if a["refcode"][:6] in seen:
            continue
        seen.add(a["refcode"][:6])
        P.append(a)
    counts["P_after_family_dedup"] = len(P)
    if len(P) < 900:
        raise SystemExit(f"|P| = {len(P)} < 900: stop and re-plan by amendment (protocol §1.1 step 5)")
    sel, test = P[:args.n_sel], P[args.n_sel:args.n_sel + args.n_test]
    counts["SEL"], counts["TEST_B"] = len(sel), len(test)

    os.makedirs(C.SETS_DIR, exist_ok=True)
    for name, lst in (("sel", sel), ("testB", test)):
        with open(os.path.join(C.SETS_DIR, f"{name}_refcodes.txt"), "w", newline="\n") as fh:
            fh.write("\n".join(a["refcode"] for a in lst) + "\n")
    torch.save(sel + test, C.POOL_CACHE + ".tmp")
    os.replace(C.POOL_CACHE + ".tmp", C.POOL_CACHE)

    import gemmi
    def cls(n):
        g = gemmi.find_spacegroup_by_number(n)
        return "centro" if g.is_centrosymmetric() else "sohncke" if g.is_sohncke() else "other"
    desc = {name: {"K": dict(sorted(collections.Counter(int(a["K"]) for a in lst).items())),
                   "class": dict(collections.Counter(cls(int(a["sg"])) for a in lst))}
            for name, lst in (("sel", sel), ("testB", test))}
    out = {"counts": counts, "describe": desc,
           "sha256": {"sel_refcodes.txt": C.sha256(os.path.join(C.SETS_DIR, "sel_refcodes.txt")),
                      "testB_refcodes.txt": C.sha256(os.path.join(C.SETS_DIR, "testB_refcodes.txt")),
                      "g1_cache ds_o3.pt": C.sha256(cache),
                      "g1_parts": {os.path.basename(p): C.sha256(p) for p in parts},
                      "n3_asym_pool.pt": C.sha256(C.POOL_CACHE)},
           "seeds": {"perm": 20261001}}
    os.makedirs(os.path.join(C.REPO, "results", "n3"), exist_ok=True)
    json.dump(out, open(os.path.join(C.REPO, "results", "n3", "n3_build_testB.json"), "w"), indent=1)
    print(json.dumps(counts, indent=1))
    print(json.dumps(desc, indent=1))


if __name__ == "__main__":
    main()
