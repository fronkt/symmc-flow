"""G1 (Omega rebuild): re-parse CIFs with mirror-image copies allowed and measure what it recovers.

For each CIF, parse twice -- legacy det+1 alignment and `allow_mirror=True` -- and record per
space group how many crystals survive the rigidity gate. For every crystal kept under
`allow_mirror`, rebuild it from the rigid-body factorization and check it against the source
structure with StructureMatcher (the oracle round-trip: must be ~100% or the parity encoding is
wrong). Writes the O(3) dataset cache and a JSON summary.

    python scripts/g1_o3_reparse.py --cif-dir data/csd_mol/cif --manifest data/csd_mol/manifest.csv \
        --cache data/csd_mol/ds_o3.pt --out results/g1_o3_reparse.json
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_REG = {}


def _work(task):
    """Parse one CIF both ways; rebuild the O(3) item and oracle-match it to the source.
    Each worker keeps its own conformer registries (the reference only fixes a gauge)."""
    warnings.filterwarnings("ignore")
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from symmc_flow.molcrystal import (_ConformerRegistry, _parse_structure, read_cif,
                                       rigid_to_structure)
    path, common = task
    ref = os.path.splitext(os.path.basename(path))[0]
    if not _REG:
        _REG.update(old=_ConformerRegistry(), new=_ConformerRegistry())
    try:
        st = read_cif(path)
        old, _ = _parse_structure(st, _REG["old"], *common, allow_mirror=False)
        new, why = _parse_structure(st, _REG["new"], *common, allow_mirror=True)
    except Exception as e:
        return ref, False, None, f"error: {type(e).__name__}: {e}", False
    if new is None:
        return ref, old is not None, None, why, False
    rebuilt = rigid_to_structure(new["lattice"], new["Z"], new["local"], new["centroid"],
                                 new["orient"], new["atom_mask"], new["mol_mask"])
    ok = bool(StructureMatcher().fit(rebuilt, st))
    new["refcode"] = ref
    return ref, old is not None, new, None, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mols", type=int, default=16)
    ap.add_argument("--max-atoms", type=int, default=64)
    ap.add_argument("--conf-tol", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--part-size", type=int, default=250)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import gemmi
    import torch

    sg_of = {r["refcode"]: gemmi.find_spacegroup_by_name(r["spacegroup"]).number
             for r in csv.DictReader(open(args.manifest))}
    paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    if args.limit:
        paths = paths[:args.limit]

    from multiprocessing import Pool
    common = (args.max_mols, args.max_atoms, 0.1, args.conf_tol)
    # checkpoint every `--part-size` CIFs so a killed run resumes instead of restarting;
    # recycle workers per part to cap memory (each keeps a growing conformer registry)
    part_dir = args.cache + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    results = []
    for s in range(0, len(paths), args.part_size):
        part = os.path.join(part_dir, f"part_{s:05d}.pt")
        if os.path.exists(part):
            results += torch.load(part, weights_only=False)
            continue
        with Pool(args.workers) as pool:
            chunk = pool.map(_work, [(p, common) for p in paths[s:s + args.part_size]],
                             chunksize=4)
        torch.save(chunk, part + ".tmp")
        os.replace(part + ".tmp", part)
        results += chunk
        print(f"{len(results)}/{len(paths)} done", flush=True)

    tot, kept_old, kept_new, match = (collections.Counter() for _ in range(4))
    n_mirror_copies = n_copies = 0
    items, skipped, fails = [], [], []
    for i, (ref, old_ok, new, why, ok) in enumerate(results):
            sg = sg_of.get(ref)
            tot[sg] += 1
            kept_old[sg] += int(old_ok)
            if new is None:
                skipped.append((ref, why))
            else:
                kept_new[sg] += 1
                mm = new["mol_mask"]
                n_copies += int(mm.sum())
                n_mirror_copies += int(((new["parity"] == -1) & mm).sum())
                match[sg] += int(ok)
                if not ok:
                    fails.append(ref)
                items.append(new)

    torch.save({"items": items, "skipped": skipped}, args.cache)

    def cls(n):
        g = gemmi.find_spacegroup_by_number(n)
        return "centro" if g.is_centrosymmetric() else "sohncke" if g.is_sohncke() else "other"

    rows = []
    for n, t in tot.most_common():
        if n is None:
            continue
        rows.append({"sg": n, "hm": gemmi.find_spacegroup_by_number(n).hm, "class": cls(n),
                     "total": t, "kept_det1": kept_old[n], "kept_o3": kept_new[n],
                     "oracle_match": match[n]})
    agg = {}
    for c in ("centro", "sohncke", "other"):
        r = [x for x in rows if x["class"] == c]
        agg[c] = {k: sum(x[k] for x in r) for k in ("total", "kept_det1", "kept_o3", "oracle_match")}
    summary = {"n_cifs": len(paths), "by_class": agg, "by_sg": rows,
               "copies": n_copies, "mirror_copies": n_mirror_copies,
               "oracle_fail_refcodes": fails,
               "skip_reasons": collections.Counter(
                   (w or "").split(" rmsd")[0].split("=")[0].split(":")[0] for _, w in skipped
               ).most_common()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1, default=int)
    for c, a in agg.items():
        print(f"{c:8s} total {a['total']:5d}  det+1 {a['kept_det1']:5d} ({a['kept_det1']/max(a['total'],1):.1%})"
              f"  O(3) {a['kept_o3']:5d} ({a['kept_o3']/max(a['total'],1):.1%})"
              f"  oracle {a['oracle_match']}/{a['kept_o3']}")
    print(f"mirror copies {n_mirror_copies}/{n_copies}; skip reasons {summary['skip_reasons'][:6]}")


if __name__ == "__main__":
    main()
