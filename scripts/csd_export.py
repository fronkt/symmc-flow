"""Export a filtered rigid-body molecular-crystal corpus from the licensed CSD.

MUST be run with the CSD Python API interpreter (it bundles the `ccdc` package), NOT the
symmc-flow env::

    "C:/Users/frank/CCDC/ccdc-software/csd-python-api/miniconda/python.exe" \
        scripts/csd_export.py --n 200 --sample 8000 --out data/csd_mol

It draws a *seeded random sample* of CSD entries (reproducible given the CSD version + seed),
keeps the ones suitable for a rigid-body molecular-crystal benchmark, and writes:
  - <out>/cif/<REFCODE>.cif         one CIF per kept entry
  - <out>/manifest.csv             refcode, formula, z_prime, r_factor, n_atoms, spacegroup

IMPORTANT (licensing): CSD-derived coordinates are NOT redistributable. The CIFs stay local
(`data/csd_mol/` is gitignored); the *manifest* (refcodes + the filter protocol) IS
shareable and lets anyone with CSD access reproduce the exact corpus. The CIFs are then
factorized by `scripts/factorize_cifs.py` under the symmc-flow env (torch/pymatgen), which
the `ccdc` interpreter does not have.

Filter (organic molecular crystals, quality-gated):
  organic & not organometallic; no disorder; 3D coords; all atoms sited; not polymeric;
  R-factor <= --rmax; 0 < Z' <= --zmax; largest component <= --max-atoms heavy+H atoms.
"""
import argparse
import csv
import os
import random


def passes(entry, max_atoms, rmax, zmax):
    """Return (ok, reason). Cheap checks first so we bail before touching geometry."""
    if not entry.is_organic or entry.is_organometallic:
        return False, "not pure-organic"
    if entry.has_disorder:
        return False, "disordered"
    if not entry.has_3d_structure:
        return False, "no 3d coords"
    rf = entry.r_factor
    if rf is None or rf > rmax:
        return False, "r_factor"
    cr = entry.crystal
    zp = cr.z_prime
    if zp is None or zp <= 0 or zp > zmax:
        return False, "z_prime"
    mol = entry.molecule
    if mol.is_polymeric or not mol.all_atoms_have_sites:
        return False, "polymeric / missing sites"
    comps = mol.components
    if not comps:
        return False, "no components"
    if max(len(c.atoms) for c in comps) > max_atoms:
        return False, "too many atoms"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="target number of kept entries")
    ap.add_argument("--sample", type=int, default=8000, help="random candidates to scan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/csd_mol")
    ap.add_argument("--max-atoms", type=int, default=64)
    ap.add_argument("--rmax", type=float, default=7.5, help="max crystallographic R-factor (%)")
    ap.add_argument("--zmax", type=float, default=2.0, help="max Z'")
    args = ap.parse_args()

    import ccdc
    from ccdc.io import EntryReader

    reader = EntryReader("CSD")
    try:
        ver = ccdc.io.csd_version()
    except Exception:
        ver = getattr(ccdc, "__version__", "?")
    total = len(reader)
    print(f"CSD: {total} entries (api/db version {ver}); seed={args.seed}")

    idx = list(range(total))
    random.Random(args.seed).shuffle(idx)

    cif_dir = os.path.join(args.out, "cif")
    os.makedirs(cif_dir, exist_ok=True)

    kept, scanned, reasons = [], 0, {}
    for i in idx:
        if len(kept) >= args.n or scanned >= args.sample:
            break
        scanned += 1
        try:
            entry = reader[i]
            ok, why = passes(entry, args.max_atoms, args.rmax, args.zmax)
        except Exception as e:
            ok, why = False, f"error:{type(e).__name__}"
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        ref = entry.identifier
        try:
            cif = entry.crystal.to_string("cif")
        except Exception as e:
            reasons[f"cif:{type(e).__name__}"] = reasons.get(f"cif:{type(e).__name__}", 0) + 1
            continue
        with open(os.path.join(cif_dir, f"{ref}.cif"), "w", encoding="utf-8") as f:
            f.write(cif)
        mol = entry.molecule
        kept.append({
            "refcode": ref,
            "formula": entry.formula or mol.formula,
            "z_prime": entry.crystal.z_prime,
            "r_factor": entry.r_factor,
            "n_atoms": max(len(c.atoms) for c in mol.components),
            "spacegroup": entry.crystal.spacegroup_symbol,
        })

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["refcode", "formula", "z_prime", "r_factor",
                                          "n_atoms", "spacegroup"])
        w.writeheader()
        w.writerows(kept)

    print(f"\nscanned {scanned} candidates -> KEPT {len(kept)} CIFs in {cif_dir}")
    print("reject reasons:")
    for why, k in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:5d}  {why}")
    print(f"\nmanifest: {os.path.join(args.out, 'manifest.csv')} (shareable; CIFs are not)")


if __name__ == "__main__":
    main()
