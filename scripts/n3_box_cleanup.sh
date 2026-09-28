#!/usr/bin/env bash
# N3 §7 step 8, last part: delete every CSD-derived file on the box and prove by search that none remains.
# Run ONLY after the pull-back tar was copied to the laptop and `n3_box_manifest.py unpack --no-overwrite` there
# printed PASS (tasks/n3_runbook.md, step 8). Then destroy the box.
#
#   bash scripts/n3_box_cleanup.sh                       # asks for the word DELETE (N3_YES=1 skips the prompt)
#   N3_CLEANUP_VERIFY_ONLY=1 bash scripts/n3_box_cleanup.sh   # search only, delete nothing
#
# Deletes: the box root (code, inputs, exports, per-set caches, checkpoints, draws, match rows, wandb offline
# dirs, hydra outputs), the upload dir (tars), ~/.cache/huggingface, wandb caches/configs, /tmp and /dev/shm
# leftovers of our jobs (torch file_system sharing, tempfile dirs), and $N3_EXTRA_DELETE.
# Verifies, over $N3_FIND_ROOTS (default /; /proc /sys /dev /run pruned; /dev/shm searched):
#   (a) NAME: no file or directory matching the CSD-derived name patterns below (site-packages / conda pkgs are
#       pruned for this part only, where generic names like train.pt or wandb/ are package files);
#   (b) HASH: no regular file anywhere whose (size, SHA-256) equals a csd=1 row of any manifest in
#       $N3_MANIFEST_DIR (the step-5 / step-7 upload manifests and the pull-back manifest; only files of a
#       listed size are hashed).
# Prints PASS or FAIL (exit 0 / 1). Paths printed on FAIL are box paths, never file content.
set -uo pipefail

if [ -z "${N3_CLEANUP_REEXEC:-}" ]; then        # run from a copy: the original is deleted with the root
  cp "$0" /tmp/n3_box_cleanup.sh 2>/dev/null && N3_CLEANUP_REEXEC=1 exec bash /tmp/n3_box_cleanup.sh "$@"
fi

ROOT=${N3_BOX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}
MDIR=${N3_MANIFEST_DIR:-$HOME/n3_manifests}
UPLOAD=${N3_UPLOAD_DIR:-$HOME/n3_upload}
FIND_ROOTS=${N3_FIND_ROOTS:-/}
TMP_DIRS=${N3_TMP_DIRS:-/tmp}
SHM_DIR=${N3_SHM_DIR:-/dev/shm}
VERIFY_ONLY=${N3_CLEANUP_VERIFY_ONLY:-0}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/cleanupwork.XXXXXX")   # name outside every delete pattern below
trap 'rm -rf "$WORK"' EXIT

say() { echo "[n3-cleanup] $*"; }

# ---------------------------------------------------------------------------------------------- preconditions
shopt -s nullglob
tsvs=("$MDIR"/*.tsv)
[ ${#tsvs[@]} -gt 0 ] || { say "FAIL: no manifests (*.tsv) in $MDIR; the hash search needs them"; exit 1; }
pulls=("$MDIR"/pullback_*.tsv)
if [ ${#pulls[@]} -eq 0 ] && [ "${N3_ALLOW_NO_PULLBACK:-0}" != 1 ] && [ "$VERIFY_ONLY" != 1 ]; then
  say "refusing: no pull-back manifest in $MDIR (run n3_box_manifest.py pullback and verify it on the laptop first;"
  say "N3_ALLOW_NO_PULLBACK=1 deletes anyway, e.g. after a stop under §7 'Overrun', where step-7 output is discarded)"
  exit 1
fi
say "root $ROOT | manifests: $(printf '%s ' "${tsvs[@]##*/}")"

if [ "$VERIFY_ONLY" != 1 ]; then
  # our jobs must be finished: list them (no pgrep -f / pkill -f; nothing is killed here)
  if ps -eo pid=,args= >/dev/null 2>&1; then
    busy=$(ps -eo pid=,args= | grep -E "python[0-9.]* .*(n3_|molcrystalflow|inference\.py|train\.py)" \
           | grep -v -E "grep|n3_box_cleanup" || true)
    if [ -n "$busy" ] && [ "${N3_FORCE:-0}" != 1 ]; then
      say "refusing: N3 jobs still running (kill them by PID first, or N3_FORCE=1):"; echo "$busy"; exit 1
    fi
  else
    say "note: ps -eo unavailable; running-job check skipped"
  fi
  if [ "${N3_YES:-0}" != 1 ]; then
    read -r -p "[n3-cleanup] type DELETE to remove every CSD-derived file on this box: " ans
    [ "$ans" = DELETE ] || { say "aborted"; exit 1; }
  fi
  # ------------------------------------------------------------------------------------------- delete
  del() {
    for p in "$@"; do
      [ -e "$p" ] || [ -L "$p" ] || continue
      case "$p" in "$WORK"|"$0") continue ;; esac
      rm -rf -- "$p" && say "deleted $p"
    done
  }
  case "$ROOT" in /|/root|/home|/home/*/|"$HOME"|"") say "FAIL: refusing to delete root '$ROOT'"; exit 1 ;; esac
  del "$ROOT" "$UPLOAD"
  del "$HOME/.cache/huggingface" "$HOME/.cache/wandb" "$HOME/.config/wandb" "$HOME/.local/share/wandb" \
      "$HOME/.netrc"
  for t in $TMP_DIRS; do
    del "$t"/wandb* "$t"/n3_* "$t"/n3* "$t"/torch_* "$t"/pymp-* "$t"/tmp* "$t"/*.tar "$t"/*.pkl.gz "$t"/*.pt
  done
  del "$SHM_DIR"/torch_* "$SHM_DIR"/n3* "$SHM_DIR"/__KMP_REGISTERED_LIB_*
  # shellcheck disable=SC2086
  [ -n "${N3_EXTRA_DELETE:-}" ] && del $N3_EXTRA_DELETE
fi

# ------------------------------------------------------------------------------------------------ verify
PRUNE=( -path /proc -o -path /sys -o -path /dev -o -path /run -o -path "$MDIR" -o -path "$WORK" )
NAME_PRUNE=( -o -path '*/site-packages' -o -path '*/pkgs' -o -path '*/conda-meta' -o -path '*/dist-packages' )
fpat=( -name '*_molcrystal_normalized*' -o -name 'g2_asym.pt' -o -name 'n3_asym_pool.pt' -o -name 'ds_o3.pt*'
       -o -name '*.cif' -o -name 'items_*.pt' -o -name '*_sidecar.json' -o -name 'predictions_*.pt'
       -o -name 'resid_*.pt' -o -name '*.ckpt' -o -name 'eval_s[0-9].pt' -o -name 'n2c_s*.pt'
       -o -name 'uma-s-1p1.pt' -o -name 'iso_atom_elem_refs.yaml' -o -name 'n3_*.tar' -o -name '*.pkl.gz'
       -o -name '*_valsel.pt' -o -name '*_testB.pt' -o -name '*_devtest.pt' -o -name '*_val.pt' -o -name '*_sel.pt'
       -o -name '*_train.pt' -o -name '*_valsel.jsonl' -o -name '*_testB.jsonl' -o -name '*_devtest.jsonl'
       -o -name '*_val.jsonl' -o -name '*_sel.jsonl' -o -name 'train.pt' -o -name 'val.pt' -o -name 'test.pt'
       -o -name 'sel.pt' -o -name 'valsel.pt' -o -name 'testB.pt' -o -name 'devtest.pt' -o -name 'fftruth_*'
       -o -name 'ffstar_config.json' -o -name 'n3_selection.json' -o -name 'ivP_frozen.json' )
dpat=( -name 'n3_mcf' -o -name 'symmc-flow' -o -name 'set_caches' -o -name 'offline-run-*' -o -name 'run-*-n3_*'
       -o -name 'wandb' -o -name 'mcf_release' -o -name 'thurlemann23' -o -name '.n3_box' -o -name 'n3_upload'
       -o -name 'n3_timing' -o -name '_timing' )
: > "$WORK/name_hits"
for r in $FIND_ROOTS "$SHM_DIR"; do
  [ -d "$r" ] || continue
  find "$r" \( "${PRUNE[@]}" "${NAME_PRUNE[@]}" \) -prune -o \( -type f \( "${fpat[@]}" \) -print \) \
       -o \( -type d \( "${dpat[@]}" \) -print -prune \) 2>/dev/null >> "$WORK/name_hits"
done
sort -u -o "$WORK/name_hits" "$WORK/name_hits"

# (size, sha256) of every csd=1 manifest row
awk -F'\t' '$1==1 {print $2"\t"$3}' "${tsvs[@]}" | sort -u > "$WORK/csd_rows"
cut -f1 "$WORK/csd_rows" | sort -u > "$WORK/sizes"
cut -f2 "$WORK/csd_rows" | sort -u > "$WORK/shas"
: > "$WORK/all_files"
for r in $FIND_ROOTS "$SHM_DIR"; do
  [ -d "$r" ] || continue
  find "$r" \( "${PRUNE[@]}" \) -prune -o -type f -printf '%s\t%p\n' 2>/dev/null >> "$WORK/all_files"
done
awk -F'\t' 'NR==FNR {s[$1]=1; next} ($1 in s) {print substr($0, index($0, "\t")+1)}' \
    "$WORK/sizes" "$WORK/all_files" | sort -u > "$WORK/candidates"
: > "$WORK/hash_hits"
while IFS= read -r f; do
  h=$(sha256sum -- "$f" 2>/dev/null | cut -d' ' -f1) || continue
  grep -qx -- "$h" "$WORK/shas" && echo "$f" >> "$WORK/hash_hits"
done < "$WORK/candidates"

n_files=$(wc -l < "$WORK/all_files"); n_cand=$(wc -l < "$WORK/candidates")
n_rows=$(wc -l < "$WORK/csd_rows"); n_name=$(wc -l < "$WORK/name_hits"); n_hash=$(wc -l < "$WORK/hash_hits")
say "searched $FIND_ROOTS + $SHM_DIR: $n_files files; $n_rows csd manifest hashes; $n_cand size-matched files hashed"
say "name-pattern hits: $n_name   hash hits: $n_hash"
if [ "$n_name" -gt 0 ]; then say "NAME hits:"; head -50 "$WORK/name_hits"; fi
if [ "$n_hash" -gt 0 ]; then say "HASH hits:"; head -50 "$WORK/hash_hits"; fi
if [ "$n_name" -eq 0 ] && [ "$n_hash" -eq 0 ]; then
  say "PASS: no CSD-derived file remains (by name and by manifest hash)"
  if [ "$VERIFY_ONLY" != 1 ] && [ "${N3_KEEP_MANIFESTS:-0}" != 1 ]; then rm -rf -- "$MDIR"; say "deleted $MDIR"; fi
  say "now destroy the box (vastai destroy instance <id>) and record it"
  exit 0
fi
say "FAIL: CSD-derived files remain (delete them, then re-run with N3_CLEANUP_VERIFY_ONLY=1)"
exit 1
