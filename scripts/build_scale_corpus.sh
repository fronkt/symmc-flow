#!/bin/bash
# Phase E / E2 -- build a LARGER molecular-crystal corpus (the "modest scale" fix for JCIM).
# LOCAL ONLY: csd_export.py needs the CSD-licensed `ccdc` interpreter; factorize_cifs.py uses
# pymatgen and runs on the normal python. After this, scp the ds.pt to the Vast box and run
# scripts/run_phaseE.sh there.
#
#   N=10000 SAMPLE=90000 bash scripts/build_scale_corpus.sh
#
# The workshop corpus was N=1127 (scanned 14,874). Keep-rate is ~1/8, so SAMPLE ~= 8*N.
# Bigger N -> stronger absolute end-to-end + a defensible scale claim, but longer export.
set -eu
N=${N:-10000}
SAMPLE=${SAMPLE:-90000}
OUT=${OUT:-data/csd_mol_scale}
CSD_PY="${CSD_PY:-C:/Users/frank/CCDC/ccdc-software/csd-python-api/miniconda/python.exe}"
PY="${PY:-python}"

echo "[1/2] CSD export  N=$N  sample=$SAMPLE  -> $OUT   (CSD interp: $CSD_PY)"
"$CSD_PY" scripts/csd_export.py --n "$N" --sample "$SAMPLE" --out "$OUT"

echo "[2/2] factorize CIFs -> $OUT/ds.pt"
"$PY" scripts/factorize_cifs.py --cif-dir "$OUT/cif" --cache "$OUT/ds.pt"

echo "DONE. Corpus at $OUT/ds.pt"
echo "Next: scp $OUT/ds.pt to the box's data/csd_mol_scale/ds.pt, then"
echo "      bash scripts/run_phaseE.sh data/csd_mol_scale/ds.pt"
