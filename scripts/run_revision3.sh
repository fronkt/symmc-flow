#!/bin/bash
# Digital Discovery revision — remaining GPU items, FULL-CARD (batch 16) config.
# Launched by watch_and_run.sh only once the correlative sweep has fully released the GPU,
# so batch 16 (== the published +27.5% baseline) is safe and the grouped split is directly
# comparable. Runs sequentially; logs to gpu_results/revision/. headline sweep first,
# long all-atom baseline last.
set -u
cd /workspace/symmc-flow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python
OUT=gpu_results/revision
mkdir -p $OUT
NOISED=checkpoints/diag_orient_relative_noised.pt
ST=$OUT/_status3.log

echo "START $(date)" | tee $ST

# --- T2.6 / T10: orientation-isolated match@1 + tolerance sweep, fit-based criterion ---
echo "[orient sweep fit] $(date)" | tee -a $ST
$PY scripts/eval_orient_matchrate.py --ckpt $NOISED --match-k 8 --sweep \
    > $OUT/orient_sweep.log 2>&1

# --- T1.5: species-grouped split x3 (batch 16 = published config) ---
for s in 0 1 2; do
  echo "[grouped split-seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_relative_grouped.py --steps 800 --split-seed $s --batch-size 16 \
      --ckpt checkpoints/rel_grouped_s$s.pt > $OUT/grouped_s$s.log 2>&1
done

# --- T1.5: one grouped match-rate read (seed 0) under leakage control ---
echo "[grouped match-rate s0] $(date)" | tee -a $ST
$PY scripts/eval_orient_matchrate.py --ckpt checkpoints/rel_grouped_s0.pt --match-k 8 \
    > $OUT/grouped_matchrate_s0.log 2>&1

# --- T1.3: all-atom + random-prior de-novo baseline (the long pole) ---
echo "[all-atom baseline] $(date)" | tee -a $ST
$PY scripts/baseline_allatom_denovo.py --steps 8000 --batch-size 64 --match-k 20 --workers 8 \
    --ckpt checkpoints/baseline_allatom.pt > $OUT/baseline_allatom.log 2>&1

echo "ALL DONE $(date)" | tee -a $ST
