#!/bin/bash
# Phase B (strengthening): DEPLOYABLE symmetry-coset conditioning -- the gate.
#
# Question: does the leak-free, template-available coset (assign_symmetry_cosets: the generating
# space-group operation, recovered from centroids only) realize the orientation gain that the
# OLD clustered codebook reached as an upper bound (+50.6% single / +47.9+-3.2% 3-seed), while a
# paired no-coset control stays at the +27% relative baseline? And does supplying that template
# at GENERATION time beat unconditioned (MolCrystalFlow-style) rigid-body generation?
#
# Batch 16 == the published +27.5% baseline config, so numbers are directly comparable.
# Runs sequentially; logs to gpu_results/phaseB/. Launch on a dedicated GPU (or via
# watch_and_run.sh if the box is shared with the correlative job -- never kill a co-tenant).
#
#   bash scripts/run_phaseB.sh            # match-k=20, workers=8 (tune per box)
set -u
cd /workspace/symmc-flow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/venv/main/bin/python
OUT=gpu_results/phaseB
mkdir -p $OUT checkpoints
MK=${1:-20}        # match-k for the end-to-end template eval
W=${2:-8}          # matcher worker processes (cap on many-core boxes)
# unconditioned baseline for the templated delta: prefer the paired no-coset control trained in
# step 2 on the IDENTICAL corpus/split (n_cosets=0 -> eval runs it unconditioned); else a
# standalone relative checkpoint if one exists.
UNCOND=checkpoints/coset_deploy_off_s0.pt
[ -f "$UNCOND" ] || UNCOND=checkpoints/diag_orient_relative_noised.pt
ST=$OUT/_status.log
echo "START $(date)  match-k=$MK workers=$W" | tee $ST

# --- 1. DEPLOYABLE coset ON, 3 seeds (the core result) -----------------------------------
for s in 0 1 2; do
  echo "[coset-deploy ON  seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_coset.py --deployable --steps 800 --batch-size 16 --seed $s \
      --ckpt checkpoints/coset_deploy_s$s.pt > $OUT/coset_deploy_s$s.log 2>&1
  grep -E "NON-REF drop|cosets:" $OUT/coset_deploy_s$s.log | tee -a $ST
done

# --- 2. Paired control: same cosetted corpus, embedding OFF, 3 seeds (isolates the coset) --
for s in 0 1 2; do
  echo "[coset-deploy OFF seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_coset.py --deployable --no-coset --steps 800 --batch-size 16 --seed $s \
      --ckpt checkpoints/coset_deploy_off_s$s.pt > $OUT/coset_deploy_off_s$s.log 2>&1
  grep -E "NON-REF drop" $OUT/coset_deploy_off_s$s.log | tee -a $ST
done

# --- 3. Orientation-isolated PHYSICAL reconstruction with the deployable coset (+ sweep) ---
echo "[orient-isolated + deployable coset, seed 0] $(date)" | tee -a $ST
$PY scripts/eval_orient_matchrate.py --ckpt checkpoints/coset_deploy_s0.pt --coset --match-k 8 \
    --sweep > $OUT/orient_isolated_coset.log 2>&1
grep -E "trained|oracle" $OUT/orient_isolated_coset.log | tee -a $ST

# --- 4. End-to-end TEMPLATE-BASED generation: conditioned vs unconditioned (the delta) -----
echo "[templated end-to-end: conditioned] $(date)" | tee -a $ST
$PY scripts/eval_templated_matchrate.py --ckpt checkpoints/coset_deploy_s0.pt --match-k $MK \
    --workers $W --tag templated > $OUT/templated_conditioned.log 2>&1
grep -E "TAG|match rate|component" $OUT/templated_conditioned.log | tee -a $ST

if [ -f "$UNCOND" ]; then
  echo "[templated end-to-end: unconditioned baseline] $(date)" | tee -a $ST
  $PY scripts/eval_templated_matchrate.py --ckpt $UNCOND --match-k $MK --workers $W \
      --tag unconditioned > $OUT/templated_unconditioned.log 2>&1
  grep -E "TAG|match rate|component" $OUT/templated_unconditioned.log | tee -a $ST
else
  echo "[skip unconditioned baseline: $UNCOND not present -- train diag_orient_relative first]" | tee -a $ST
fi

echo "ALL DONE $(date)" | tee -a $ST
echo "GATE: compare NON-REF drop  coset-ON (step 1)  vs  OFF control (step 2)  vs  48% clustered upper bound;" | tee -a $ST
echo "      and templated (step 4 conditioned) match rate vs unconditioned baseline." | tee -a $ST
