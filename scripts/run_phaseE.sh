#!/bin/bash
# Phase E / E2 (scale) -- retrain coset + control + predictor on a LARGER corpus and re-run the
# gate + the templated (conditioned vs coset-off) end-to-end + the predictor top-k read.
# Answers the JCIM objections: does the +41% coset gain HOLD at scale (O3), does more data lift
# END-TO-END (O1), and does the predictor sharpen enough for template-free use (O2/E3)?
#
# Needs a bigger dataset built first (scripts/build_scale_corpus.sh, LOCAL) and scp'd to $DS.
#   bash scripts/run_phaseE.sh [ds_path] [steps] [match-k] [workers]
set -u
cd /workspace/symmc-flow
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/venv/main/bin/python}   # override on a shared box, e.g. PY=/workspace/symmc-flow/.venv/bin/python
DS=${1:-data/csd_mol_scale/ds.pt}
STEPS=${2:-3000}          # more steps for the bigger corpus (workshop used 800 at N=1127)
MK=${3:-20}
W=${4:-16}                # matcher workers (cap on many-core boxes)
OUT=gpu_results/phaseE_scale
mkdir -p $OUT checkpoints
ST=$OUT/_status.log
[ -f "$DS" ] || { echo "MISSING dataset $DS -- build_scale_corpus.sh + scp it first" | tee $ST; exit 1; }
echo "START $(date)  ds=$DS steps=$STEPS mk=$MK w=$W" | tee $ST

# --- 1. GATE at scale: deployable coset ON vs paired no-coset control, 3 seeds ----------------
for s in 0 1 2; do
  echo "[scale coset ON  seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_coset.py --deployable --cache $DS --steps $STEPS --batch-size 32 \
      --seed $s --ckpt checkpoints/scale_coset_s$s.pt > $OUT/coset_s$s.log 2>&1
  grep -E "NON-REF drop|cosets:" $OUT/coset_s$s.log | tee -a $ST
done
for s in 0 1 2; do
  echo "[scale coset OFF seed $s] $(date)" | tee -a $ST
  $PY scripts/diag_orient_coset.py --deployable --no-coset --cache $DS --steps $STEPS \
      --batch-size 32 --seed $s --ckpt checkpoints/scale_off_s$s.pt > $OUT/off_s$s.log 2>&1
  grep -E "NON-REF drop" $OUT/off_s$s.log | tee -a $ST
done

# --- 2. END-TO-END at scale (E1 at scale): templated conditioned vs same-model coset-off -------
echo "[templated conditioned] $(date)" | tee -a $ST
$PY scripts/eval_templated_matchrate.py --ckpt checkpoints/scale_coset_s0.pt --cache $DS \
    --match-k $MK --workers $W --tag scale_cond > $OUT/templated_cond.log 2>&1
grep -E "match rate|component|TAG" $OUT/templated_cond.log | tee -a $ST
echo "[templated coset-off (ablate)] $(date)" | tee -a $ST
$PY scripts/eval_templated_matchrate.py --ckpt checkpoints/scale_coset_s0.pt --cache $DS \
    --ablate-no-coset --match-k $MK --workers $W --tag scale_uncond > $OUT/templated_uncond.log 2>&1
grep -E "match rate|component|TAG" $OUT/templated_uncond.log | tee -a $ST

# --- 3. Predictor at scale (E3): top-1 accuracy + top-k marginalization read -------------------
echo "[predictor scale train] $(date)" | tee -a $ST
$PY scripts/train_coset_predictor.py --cache $DS --steps 5000 --batch-size 32 \
    --ckpt checkpoints/scale_predictor.pt > $OUT/predictor.log 2>&1
grep -E "top-1|majority|recoverable" $OUT/predictor.log | tee -a $ST
echo "[predicted-coset orient-isolated: template / top-1 / top-3 / top-5] $(date)" | tee -a $ST
$PY scripts/eval_orient_matchrate.py --ckpt checkpoints/scale_coset_s0.pt --cache $DS --coset \
    --match-k 8 > $OUT/orient_template.log 2>&1
echo " template(gt): $(grep -E 'trained' $OUT/orient_template.log | tail -1)" | tee -a $ST
for tk in 1 3 5; do
  $PY scripts/eval_orient_matchrate.py --ckpt checkpoints/scale_coset_s0.pt --cache $DS \
      --predictor-ckpt checkpoints/scale_predictor.pt --predictor-topk $tk --match-k 8 \
      > $OUT/pred_top$tk.log 2>&1
  echo " top-$tk: $(grep -E 'trained' $OUT/pred_top$tk.log | tail -1)" | tee -a $ST
done

echo "ALL DONE $(date)" | tee -a $ST
echo "GATE: (1) coset-ON vs OFF non-ref drop at scale (does +41% hold/grow?);" | tee -a $ST
echo "      (2) templated conditioned vs coset-off (does end-to-end lift with data?);" | tee -a $ST
echo "      (3) predictor top-1 acc + does top-k marginalization approach the template read?" | tee -a $ST
