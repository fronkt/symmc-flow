#!/bin/bash
# GPU sweep for the molecular-crystal orientation paper (run on the vast.ai 5090).
# Covers: B (3-seed error bars for the relative + coset diagnostics) and
# C / reviewer M1 (de-novo joint-generation match rate), plus a capacity run and
# an orientation-isolated consistency check. NOT `set -e`: one failure must not
# abort the whole multi-hour sweep.
set -uo pipefail
source /venv/main/bin/activate
cd "$(dirname "$0")/.."
mkdir -p checkpoints results
log() { echo "[$(date +%H:%M:%S)] $*"; }
export OMP_NUM_THREADS=8   # cap BLAS threads (256-core box) per training process

log "=== B: relative-orientation error bars (seeds 0,1,2) ==="
for s in 0 1 2; do
  log "RELATIVE seed $s"
  python -u scripts/diag_orient_relative.py --steps 800 --seed "$s" \
      --ckpt checkpoints/rel_s$s.pt > results/rel_s$s.log 2>&1 || log "rel s$s FAILED"
done

log "=== B: coset-conditioning error bars (seeds 0,1,2) ==="
for s in 0 1 2; do
  log "COSET seed $s"
  python -u scripts/diag_orient_coset.py --steps 800 --seed "$s" \
      --ckpt checkpoints/coset_s$s.pt > results/coset_s$s.log 2>&1 || log "coset s$s FAILED"
done

log "=== C / M1: de-novo joint-generation match@20 (seeds 0,1,2) ==="
for s in 0 1 2; do
  log "DENOVO seed $s"
  python -u scripts/eval_denovo_matchrate.py --ckpt checkpoints/rel_s$s.pt \
      --match-k 20 --sampler-steps 100 --workers 32 --seed "$s" --tag denovo_s$s \
      > results/denovo_s$s.log 2>&1 || log "denovo s$s FAILED"
done

log "=== C2: capacity run (d256/6L/5000 steps) + de-novo ==="
python -u scripts/diag_orient_relative.py --steps 5000 --d-model 256 --n-attn-layers 6 \
    --egnn-layers 5 --seed 0 --ckpt checkpoints/rel_big_s0.pt \
    > results/rel_big_s0.log 2>&1 || log "rel_big FAILED"
python -u scripts/eval_denovo_matchrate.py --ckpt checkpoints/rel_big_s0.pt \
    --match-k 20 --sampler-steps 100 --workers 32 --seed 0 --tag denovo_big \
    > results/denovo_big.log 2>&1 || log "denovo_big FAILED"

log "=== orientation-isolated consistency (seeds 0,1,2; reproduces CPU 16.8%) ==="
for s in 0 1 2; do
  log "ORIENT-ISO seed $s"
  python -u scripts/eval_orient_matchrate.py --ckpt checkpoints/rel_s$s.pt \
      --match-k 8 --steps 50 --seed "$s" > results/orientiso_s$s.log 2>&1 || log "iso s$s FAILED"
done

log "=== summary ==="
grep -h "NON-REF drop\|nonref\|match rate\|TAG\|VERDICT" results/*.log 2>/dev/null | tail -60
log "SWEEP DONE"
