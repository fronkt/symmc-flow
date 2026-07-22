#!/bin/bash
# Phase C (stretch): de-novo coset predictor (C4) + SO(3)-averaged objective (C5).
# Run AFTER run_phaseB.sh has passed the gate (needs checkpoints/coset_deploy_s0.pt from it).
#
#   bash scripts/run_phaseC.sh
set -u
cd /workspace/symmc-flow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python
OUT=gpu_results/phaseC
mkdir -p $OUT checkpoints
ST=$OUT/_status.log
echo "START $(date)" | tee $ST

# --- C4: train the packing-only coset predictor; measure the closable inference gap ----------
echo "[C4 train coset predictor] $(date)" | tee -a $ST
$PY scripts/train_coset_predictor.py --steps 3000 --batch-size 16 \
    --ckpt checkpoints/coset_predictor.pt > $OUT/predictor.log 2>&1
grep -E "top-1 accuracy|majority|recoverable" $OUT/predictor.log | tee -a $ST

# --- C4: orientation-isolated reconstruction with the PREDICTED coset (de-novo, no template) --
# Compare against the ground-truth deployable coset (--coset) and no coset from phase B.
if [ -f checkpoints/coset_deploy_s0.pt ]; then
  echo "[C4 predicted-coset orient-isolated] $(date)" | tee -a $ST
  $PY scripts/eval_orient_matchrate.py --ckpt checkpoints/coset_deploy_s0.pt \
      --predictor-ckpt checkpoints/coset_predictor.pt --match-k 8 \
      > $OUT/predicted_coset_orient.log 2>&1
  grep -E "trained|oracle|PREDICTOR" $OUT/predicted_coset_orient.log | tee -a $ST
else
  echo "[skip predicted-coset eval: run run_phaseB.sh first (need coset_deploy_s0.pt)]" | tee -a $ST
fi

# --- C5: SO(3)-averaged objective (K=4) vs standard (K=1), deployable coset, 3 seeds ---------
for s in 0 1 2; do
  echo "[C5 so3-avg K=4 seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_coset.py --deployable --so3-avg-k 4 --steps 800 --batch-size 16 \
      --seed $s --ckpt checkpoints/coset_avg_s$s.pt > $OUT/coset_avg_s$s.log 2>&1
  grep -E "NON-REF drop" $OUT/coset_avg_s$s.log | tee -a $ST
done

echo "ALL DONE $(date)" | tee -a $ST
echo "READ: C4 predictor accuracy (closable inference gap) + predicted-vs-given-vs-none orient rate;" | tee -a $ST
echo "      C5 so3-avg K=4 non-ref drop vs the K=1 deployable-coset baseline from phase B." | tee -a $ST
