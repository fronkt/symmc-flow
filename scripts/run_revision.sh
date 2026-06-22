#!/bin/bash
# Digital Discovery revision roadmap — full GPU run battery.
# Runs sequentially, logs each to gpu_results/revision/. Launch in tmux.
set -u
cd /workspace/symmc-flow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python
OUT=gpu_results/revision
mkdir -p $OUT
NOISED=checkpoints/diag_orient_relative_noised.pt
BIG=checkpoints/diag_orient_relative_big.pt

echo "START $(date)" | tee $OUT/_status.log

# --- T1.2 probe (3 seeds) ---
for s in 0 1 2; do
  echo "[probe seed $s] $(date)" | tee -a $OUT/_status.log
  $PY scripts/probe_rasym_conditional.py --epochs 300 --seed $s > $OUT/probe_s$s.log 2>&1
done

# --- T1.4 + T2.6 orientation-isolated: match@1, tolerance sweep, matched RMSD ---
echo "[orient-isolated sweep] $(date)" | tee -a $OUT/_status.log
$PY scripts/eval_orient_matchrate.py --ckpt $NOISED --match-k 8 --sweep \
    > $OUT/orient_sweep.log 2>&1

# --- T1.4 de-novo match@1 (base + big) ---
echo "[denovo match@1 base] $(date)" | tee -a $OUT/_status.log
$PY scripts/eval_denovo_matchrate.py --ckpt $NOISED --match-k 20 --workers 16 \
    --tag denovo_base > $OUT/denovo_base.log 2>&1
echo "[denovo match@1 big] $(date)" | tee -a $OUT/_status.log
$PY scripts/eval_denovo_matchrate.py --ckpt $BIG --match-k 20 --workers 16 \
    --tag denovo_big > $OUT/denovo_big.log 2>&1

# --- T1.5 species-grouped split x3 ---
for s in 0 1 2; do
  echo "[grouped split-seed $s] $(date)" | tee -a $OUT/_status.log
  $PY scripts/diag_orient_relative_grouped.py --steps 800 --split-seed $s \
      --ckpt checkpoints/rel_grouped_s$s.pt > $OUT/grouped_s$s.log 2>&1
done
# one grouped match-rate read (seed 0) for a reconstruction number under leakage control
echo "[grouped match-rate s0] $(date)" | tee -a $OUT/_status.log
$PY scripts/eval_orient_matchrate.py --ckpt checkpoints/rel_grouped_s0.pt --match-k 8 \
    > $OUT/grouped_matchrate_s0.log 2>&1

# --- T1.3 all-atom baseline (the long pole) ---
echo "[all-atom baseline] $(date)" | tee -a $OUT/_status.log
$PY scripts/baseline_allatom_denovo.py --steps 8000 --match-k 20 --workers 16 \
    --batch-size 64 --ckpt checkpoints/baseline_allatom.pt > $OUT/baseline_allatom.log 2>&1

echo "ALL DONE $(date)" | tee -a $OUT/_status.log
