"""Fetch real molecular crystals from the open Crystallography Open Database (COD) and run
them through `MolCrystalDataset`, while CSD/CCDC access is pending.

COD is fully open (no license), so its CIFs are redistributable and feed the *exact same*
pymatgen -> MolCrystalDataset pipeline the CSD corpus will. This script is the license-free
companion path: it (1) gathers candidate COD ids (an element-restricted query, with a small
built-in fallback list of known organic molecular crystals), (2) downloads the CIFs with a
local cache, (3) parses + factorizes them into rigid molecular blocks, and (4) prints a
detection summary (kept vs skipped reasons, Z' and atoms-per-molecule histograms).

    python scripts/fetch_cod_molcrystals.py --n 60 --out data/cod_mol

Nothing here is CSD-specific; when the CSD Python API arrives it swaps in at the
`detect_fn` / structure-source layer and the rest is unchanged.
"""
import argparse
import collections
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COD_CIF = "https://www.crystallography.net/cod/{id}.cif"
COD_QUERY = "https://www.crystallography.net/cod/result"

# Known organic molecular crystals (refcode-ish names for sanity), as a robust fallback if
# the REST query is unavailable. These are classic rigid(-ish) organics.
FALLBACK_IDS = [
    1101307, 1508981, 2300017, 2208992, 1200029, 1100094, 1508653, 7050000,
    1509118, 2010205, 2012107, 2100147, 2104857, 2016332, 1515581, 4509231,
    7011403, 7150001, 1100094, 2228564,
]


def _http_get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "symmc-flow/molcrystal (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cod_query_ids(n, elements=("C", "H", "N", "O")):
    """Ask COD for ids of structures built *exactly* from `elements`: must contain each
    (el1..elN) and have no more than N distinct elements (strictmax), i.e. pure organics
    with no metals/counter-ions. Returns the full id list; caller dedups/truncates."""
    params = "&".join(f"el{i+1}={e}" for i, e in enumerate(elements))
    params += f"&strictmax={len(elements)}&format=lst"
    try:
        raw = _http_get(f"{COD_QUERY}?{params}").decode("utf-8", "replace")
        ids = [int(line) for line in raw.split() if line.strip().isdigit()]
        return ids
    except Exception as e:
        print(f"  [warn] COD query failed ({e}); using fallback id list")
        return []


def fetch_cifs(ids, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for cid in ids:
        p = os.path.join(cache_dir, f"{cid}.cif")
        if not os.path.exists(p) or os.path.getsize(p) < 200:
            try:
                data = _http_get(COD_CIF.format(id=cid))
                with open(p, "wb") as f:
                    f.write(data)
                time.sleep(0.15)            # be polite to the public server
            except Exception as e:
                print(f"  [warn] download {cid} failed: {e}")
                continue
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="how many COD ids to try")
    ap.add_argument("--out", default="data/cod_mol", help="cache dir for CIFs + dataset cache")
    ap.add_argument("--max-mols", type=int, default=12)
    ap.add_argument("--max-atoms", type=int, default=64)
    ap.add_argument("--conf-tol", type=float, default=0.3)
    ap.add_argument("--elements", default="C,H,N,O")
    args = ap.parse_args()

    from pymatgen.core import Structure
    from symmc_flow.molcrystal import MolCrystalDataset, rigid_to_frac
    import torch

    cif_dir = os.path.join(args.out, "cif")
    elements = tuple(args.elements.split(","))

    print(f"== querying COD for {elements} organics ==")
    ids = cod_query_ids(args.n, elements)
    if not ids:
        ids = FALLBACK_IDS
    ids = list(dict.fromkeys(ids))                  # dedup, keep order
    if len(ids) > args.n:                           # representative draw, not just oldest ids
        import random
        random.Random(0).shuffle(ids)
        ids = ids[: args.n]
    print(f"  {len(ids)} candidate ids (seeded sample)")

    print("== downloading CIFs (cached) ==")
    paths = fetch_cifs(ids, cif_dir)
    print(f"  {len(paths)} CIFs on disk")

    # Parse to Structures first so a single bad CIF can't abort the batch.
    structs, parse_fail = [], 0
    for p in paths:
        try:
            structs.append(Structure.from_file(p))
        except Exception:
            parse_fail += 1
    print(f"  {len(structs)} parsed, {parse_fail} unreadable")

    print("== factorizing into rigid molecular blocks ==")
    ds = MolCrystalDataset(structures=structs, max_mols=args.max_mols,
                           max_atoms=args.max_atoms, conf_tol=args.conf_tol)
    print(f"\nKEPT {len(ds)} / {len(structs)} structures   SKIPPED {len(ds.skipped)}")

    # skip-reason histogram (collapse the numeric tails so reasons group)
    reasons = collections.Counter()
    for _, why in ds.skipped:
        key = why.split(" rmsd")[0].split("=")[0].split(" has ")[0].strip()
        reasons[key] += 1
    print("\nskip reasons:")
    for why, k in reasons.most_common():
        print(f"  {k:4d}  {why}")

    # Z' (molecules per cell) and atoms-per-molecule histograms over KEPT structures
    zprime = collections.Counter()
    natoms = collections.Counter()
    max_rt = 0.0
    for i in range(len(ds)):
        it = ds[i]
        zprime[int(it["mol_mask"].sum())] += 1
        for m in range(it["mol_mask"].shape[0]):
            if bool(it["mol_mask"][m]):
                natoms[int(it["atom_mask"][m].sum())] += 1
        # round-trip check on real data: reconstruct -> compare to nothing planted, but the
        # factorization must at least be self-consistent (finite, wrapped)
        recon = rigid_to_frac(it["lattice"], it["local"], it["centroid"], it["orient"])
        assert torch.isfinite(recon).all(), f"non-finite reconstruction at {i}"

    print("\nZ' (molecules/cell) over kept:")
    for z, k in sorted(zprime.items()):
        print(f"  Z'={z:2d}: {k}")
    print("\natoms/molecule over kept:")
    for a, k in sorted(natoms.items()):
        print(f"  {a:3d} atoms: {k}")

    if len(ds):
        print(f"\nOK: {len(ds)} real COD molecular crystals factorized into rigid blocks "
              f"(shared-conformer SO(3) representation), reconstruction finite.")
    else:
        print("\nNo structures survived detection — widen --elements or --max-atoms.")


if __name__ == "__main__":
    main()
