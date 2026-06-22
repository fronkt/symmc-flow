#!/bin/bash
# Digital Discovery revision — remaining GPU items, batch-8 budget.
# The box is SHARED with a correlative-microscopy RoMa finetune holding ~25 GiB, so only
# ~6.8 GiB is free: every job runs at batch 8 (batch 16 OOMs). Headline sweep first, the
# long all-atom baseline last. Runs sequentially; logs to gpu_results/revision/. tmux.
set -u
cd /workspace/symmc-flow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python
OUT=gpu_results/revision
mkdir -p $OUT
NOISED=checkpoints/diag_orient_relative_noised.pt
ST=$OUT/_status2.log

echo "START $(date)" | tee $ST

# --- T2.6 / T10: orientation-isolated match@1 + tolerance sweep, fit-based criterion ---
echo "[orient sweep fit] $(date)" | tee -a $ST
$PY scripts/eval_orient_matchrate.py --ckpt $NOISED --match-k 8 --sweep --batch-size 8 \
    > $OUT/orient_sweep.log 2>&1

# --- T1.5: species-grouped split x3 (batch 8) ---
for s in 0 1 2; do
  echo "[grouped split-seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_relative_grouped.py --steps 800 --split-seed $s --batch-size 8 \
      --ckpt checkpoints/rel_grouped_s$s.pt > $OUT/grouped_s$s.log 2>&1
done

# --- T1.5 control: batch-MATCHED ungrouped x3 (so grouped vs ungrouped isn't confounded
#     by the batch-16 -> batch-8 reduction; published +27.5% was at batch 16) ---
for s in 0 1 2; do
  echo "[ungrouped b8 seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_relative.py --steps 800 --seed $s --batch-size 8 \
      --ckpt checkpoints/rel_ungrouped_b8_s$s.pt > $OUT/ungrouped_b8_s$s.log 2>&1
done

# --- T1.5: one grouped match-rate read (seed 0) under leakage control ---
echo "[grouped match-rate s0] $(date)" | tee -a $ST
$PY scripts/eval_orient_matchrate.py --ckpt checkpoints/rel_grouped_s0.pt --match-k 8 \
    --batch-size 8 > $OUT/grouped_matchrate_s0.log 2>&1

# --- T1.3: all-atom + random-prior de-novo baseline (the long pole) ---
echo "[all-atom baseline] $(date)" | tee -a $ST
$PY scripts/baseline_allatom_denovo.py --steps 8000 --batch-size 8 --match-k 20 --workers 8 \
    --ckpt checkpoints/baseline_allatom.pt > $OUT/baseline_allatom.log 2>&1

echo "ALL DONE $(date)" | tee -a $ST
