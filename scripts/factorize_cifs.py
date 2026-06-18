"""Factorize a directory of CIFs into rigid molecular blocks via `MolCrystalDataset`.

Source-agnostic consumer: point it at any CIF dir (CSD export from `csd_export.py`, COD
download from `fetch_cod_molcrystals.py`, or hand-collected CIFs) and it reports how many
become rigid-body molecular crystals, with skip reasons and Z'/atoms-per-molecule
histograms. Runs under the symmc-flow env (torch/pymatgen), which the `ccdc` interpreter
lacks -- this is the second stage of the CSD pipeline.

    python scripts/factorize_cifs.py --cif-dir data/csd_mol/cif --cache data/csd_mol/ds.pt
"""
import argparse
import collections
import glob
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--cache", default=None, help="optional dataset cache .pt path")
    ap.add_argument("--max-mols", type=int, default=16)
    ap.add_argument("--max-atoms", type=int, default=64)
    ap.add_argument("--conf-tol", type=float, default=0.3)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    from symmc_flow.molcrystal import MolCrystalDataset, rigid_to_frac, read_cif

    paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    print(f"== {len(paths)} CIFs in {args.cif_dir} ==")

    structs, parse_fail = [], 0
    for p in paths:
        try:
            structs.append(read_cif(p))
        except Exception:
            parse_fail += 1
    print(f"  {len(structs)} parsed, {parse_fail} unreadable")

    ds = MolCrystalDataset(structures=structs, max_mols=args.max_mols,
                           max_atoms=args.max_atoms, conf_tol=args.conf_tol,
                           cache_path=args.cache)
    print(f"\nKEPT {len(ds)} / {len(structs)}   SKIPPED {len(ds.skipped)}")

    reasons = collections.Counter()
    for _, why in ds.skipped:
        key = why.split(" rmsd")[0].split("=")[0].split(" has ")[0].strip()
        reasons[key] += 1
    print("\nskip reasons:")
    for why, k in reasons.most_common():
        print(f"  {k:4d}  {why}")

    zprime, natoms = collections.Counter(), collections.Counter()
    for i in range(len(ds)):
        it = ds[i]
        zprime[int(it["mol_mask"].sum())] += 1
        for m in range(it["mol_mask"].shape[0]):
            if bool(it["mol_mask"][m]):
                natoms[int(it["atom_mask"][m].sum())] += 1
        recon = rigid_to_frac(it["lattice"], it["local"], it["centroid"], it["orient"])
        assert torch.isfinite(recon).all(), f"non-finite reconstruction at {i}"

    print("\nZ' (molecules/cell) over kept:")
    for z, k in sorted(zprime.items()):
        print(f"  Z'={z:2d}: {k}")
    print("\natoms/molecule over kept (min={}, max={}):".format(
        min(natoms) if natoms else 0, max(natoms) if natoms else 0))
    multi = sum(k for a, k in natoms.items() if a > 1)
    mono = natoms.get(1, 0)
    print(f"  multi-atom rigid blocks: {multi}   monatomic (orient=I): {mono}")

    if args.cache:
        print(f"\ncached dataset -> {args.cache}")
    print(f"\nOK: {len(ds)} crystals factorized into shared-conformer rigid blocks.")


if __name__ == "__main__":
    main()
