#!/bin/bash
# Phase F2 (gated smoke retrain): crystal-family-masked lattice (logmetric6) + coset orientation.
#
# Trains the DEPLOYABLE coset model with the O(3)-invariant log-metric lattice repr + a
# crystal-family mask + an informed volume prior, then scores match@10 / cell-volume RMAD /
# crystal-family angle-spike in MolCrystalFlow's units (eval_e4). This is the first end-to-end
# test of whether extending symmetry conditioning from orientation (the coset) to the unit cell
# (the family mask) unblocks packing.
#
# CPU is ~19 s/step (~4 h for 800 steps) -> RUN ON A GPU BOX. See feedback_vast_workflow: cap
# workers on many-core boxes, install torch via the cu128 index on RTX 5090, check Inet first.
#
#   PY=/venv/main/bin/python bash scripts/run_phaseF.sh [match-k=10] [workers=8]
set -u
cd /workspace/symmc-flow 2>/dev/null || cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
OUT=gpu_results/phaseF
mkdir -p "$OUT" checkpoints
MK=${1:-10}; W=${2:-8}; LVSTD=${LVSTD:-0.11}; STEPS=${STEPS:-800}
ST=$OUT/_status.log
echo "START $(date)  match-k=$MK workers=$W logvol-std=$LVSTD steps=$STEPS" | tee "$ST"

# --- 1. train the Phase F model: coset ON, log-metric lattice + family mask + informed prior ---
echo "[F train: coset-ON logmetric6 + family-mask] $(date)" | tee -a "$ST"
$PY scripts/diag_orient_coset.py --deployable --lattice-logmetric --family-mask \
    --logvol-std "$LVSTD" --steps "$STEPS" --batch-size 16 --seed 0 \
    --ckpt checkpoints/coset_fammask_s0.pt > "$OUT/train_fammask_s0.log" 2>&1
grep -E "NON-REF drop|cosets:|lattice " "$OUT/train_fammask_s0.log" | tee -a "$ST"

# --- 2. (ablation) repr swap WITHOUT the mask, to isolate the mask from the informed prior ------
if [ "${ABLATE:-0}" = "1" ]; then
  echo "[F ablation: logmetric6, NO family-mask] $(date)" | tee -a "$ST"
  $PY scripts/diag_orient_coset.py --deployable --lattice-logmetric \
      --logvol-std "$LVSTD" --steps "$STEPS" --batch-size 16 --seed 0 \
      --ckpt checkpoints/coset_logmetric_nomask_s0.pt > "$OUT/train_nomask_s0.log" 2>&1
  $PY scripts/eval_e4_molcrystalflow.py --ckpt checkpoints/coset_logmetric_nomask_s0.pt \
      --match-k "$MK" --workers "$W" --out "$OUT/nomask" > "$OUT/eval_nomask.log" 2>&1
  grep -E "TAG e4|angle spike|reference" "$OUT/eval_nomask.log" | tee -a "$ST"
fi

# --- 3. match@10 / RMAD / crystal-family angle-spike in MolCrystalFlow units --------------------
echo "[F eval: match@10 + RMAD + angle-spike] $(date)" | tee -a "$ST"
$PY scripts/eval_e4_molcrystalflow.py --ckpt checkpoints/coset_fammask_s0.pt \
    --match-k "$MK" --workers "$W" --out "$OUT" > "$OUT/eval_fammask.log" 2>&1
grep -E "TAG e4|angle spike|reference|coset ON|coset OFF" "$OUT/eval_fammask.log" | tee -a "$ST"

echo "" | tee -a "$ST"
echo "GATE (tasks/phaseF_spec.md): angle-spike should rise from ~3% toward the reference ~72%," | tee -a "$ST"
echo "     AND match@10 (coset ON, stol<=1.0) should move off 0 (E4 shape10 baseline ~0-0.8%)," | tee -a "$ST"
echo "     OR cell-volume RMAD < ~10% with the angle-spike restored (partial win worth scaling)." | tee -a "$ST"
echo "     Otherwise -> F4 fallback (E5 write-up + honest necessary-not-sufficient finding)." | tee -a "$ST"
echo "ALL DONE $(date)" | tee -a "$ST"
