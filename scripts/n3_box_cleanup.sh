#!/usr/bin/env bash
# N3 §7 step 8: dedicated Linux box only; verified pullback, cleanup, then destroy.
# Deletes ONLY explicit, canonical private root/upload and N3_EXTRA_DELETE_FILE targets.
# Shared HOME caches, credentials and broad /tmp or /dev/shm globs are never deletion targets.
# Setup confines caches/tmp/wandb to the box root. Searches cover N3_FIND_ROOTS (newline-separated,
# default /) plus /dev/shm. Any failed search or hash makes verification FAIL.
# Manifests, laptop receipt and timestamped cleanup receipt remain in N3_MANIFEST_DIR.
#
# N3_PULLBACK_RECEIPT=/root/n3_manifests/pullback_step7.unpack_receipt.json bash scripts/n3_box_cleanup.sh
# N3_CLEANUP_PLAN_ONLY=1 bash scripts/n3_box_cleanup.sh   # no deletion or search
# N3_CLEANUP_VERIFY_ONLY=1 bash scripts/n3_box_cleanup.sh # no deletion
# N3_DISCARD_OUTPUTS=1: only a protocol stop/box failure; discard outputs unscored.
set -euo pipefail
say() { echo "[n3-cleanup] $*"; }
die() { say "FAIL: $*" >&2; exit 1; }
[ "$(uname -s)" = Linux ] || die "cleanup runs only on the dedicated Linux box"

# Resolve the root before re-executing a copy outside the deletion tree.
ROOT=${N3_BOX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MDIR=${N3_MANIFEST_DIR:-$HOME/n3_manifests}
UPLOAD=${N3_UPLOAD_DIR:-$HOME/n3_upload}
VERIFY_ONLY=${N3_CLEANUP_VERIFY_ONLY:-0}
PLAN_ONLY=${N3_CLEANUP_PLAN_ONLY:-0}
if [ -z "${N3_CLEANUP_SCRIPT_DIR:-}" ]; then
  self=$(mktemp -d /tmp/n3-cleanup-script.XXXXXX)
  cp "$0" "$self/cleanup.sh"
  cp "$(dirname "${BASH_SOURCE[0]}")/n3_box_manifest.py" "$self/n3_box_manifest.py"
  export N3_CLEANUP_SCRIPT_DIR=$self N3_BOX_ROOT=$ROOT
  exec bash "$self/cleanup.sh" "$@"
fi
SELF_DIR=$N3_CLEANUP_SCRIPT_DIR
[ "$(dirname "$SELF_DIR")" = /tmp ] && [[ "$(basename "$SELF_DIR")" =~ ^n3-cleanup-script\.[A-Za-z0-9]+$ ]] \
  && [ ! -L "$SELF_DIR" ] && [ "$(realpath -- "$SELF_DIR")" = "$SELF_DIR" ] \
  || die "unsafe cleanup-script temporary directory"
WORK=$(mktemp -d /tmp/n3-cleanup-work.XXXXXX)
trap 'rm -rf -- "$WORK" "$SELF_DIR"' EXIT
opts=(--root "$ROOT" --upload "$UPLOAD" --manifest-dir "$MDIR" --out "$WORK/plan.json")
[ -z "${N3_EXTRA_DELETE_FILE:-}" ] || opts+=(--extras-file "$N3_EXTRA_DELETE_FILE")
[ -z "${N3_PULLBACK_RECEIPT:-}" ] || opts+=(--pullback-receipt "$N3_PULLBACK_RECEIPT")
if [ "${N3_DISCARD_OUTPUTS:-0}" = 1 ] || [ "$VERIFY_ONLY" = 1 ] || [ "$PLAN_ONLY" = 1 ]; then
  opts+=(--discard-outputs)
fi
python3 "$SELF_DIR/n3_box_manifest.py" cleanup-plan "${opts[@]}" || die "target/inventory/pullback preflight failed"
MDIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_dir"])' "$WORK/plan.json")
if [ "$PLAN_ONLY" = 1 ]; then
  say "plan validated; no files deleted"
  exit 0
fi
mapfile -t targets < <(python3 - "$WORK/plan.json" <<'PY'
import json, sys
for p in json.load(open(sys.argv[1]))["targets"]:
    print(p)
PY
)
if [ "$VERIFY_ONLY" != 1 ]; then
  ps -eo pid=,args= > "$WORK/processes" || die "cannot check whether jobs are still running"
  busy=$(awk '$0 ~ /python[0-9.]* .*n3_|python[0-9.]* .*molcrystalflow|python[0-9.]* .*inference\.py|python[0-9.]* .*train\.py/ && $0 !~ /n3_box_cleanup|n3_box_manifest|cleanup.sh/ {print}' "$WORK/processes")
  [ -z "$busy" ] || { say "refusing: jobs still running; finish or stop by explicit PID"; echo "$busy"; exit 1; }
  if [ "${N3_YES:-0}" != 1 ]; then
    read -r -p "[n3-cleanup] type DELETE to remove the listed private directories: " ans
    [ "$ans" = DELETE ] || die "aborted"
  fi
  for target in "${targets[@]}"; do
    [ ! -L "$target" ] || die "target became a symlink: $target"
    [ "$(realpath -m -- "$target")" = "$target" ] || die "target resolution changed: $target"
    rm -rf -- "$target" || die "failed to remove explicit private target $target"
    say "deleted $target"
  done
fi

# Roots are read without shell word splitting or glob expansion.
roots=()
while IFS= read -r r; do
  [ -n "$r" ] && roots+=("$r")
done <<< "${N3_FIND_ROOTS:-/}"
roots+=(/dev/shm)
PRUNE=( -path /proc -o -path /sys -o -path /dev -o -path /run -o -path "$MDIR" -o -path "$WORK" -o -path "$SELF_DIR" )
NAME_PRUNE=( -o -path '*/site-packages' -o -path '*/pkgs' -o -path '*/conda-meta' -o -path '*/dist-packages' )
fpat=( -name '*_molcrystal_normalized*' -o -name '*.cif' -o -name '*.pkl.gz' -o -name '*_sidecar.json'
       -o -name 'g2_asym.pt' -o -name 'n3_asym_pool.pt' -o -name 'ds_o3.pt*' -o -name 'items_*.pt'
       -o -name 'predictions_*.pt' -o -name 'resid_*.pt' -o -name '*.ckpt' -o -name 'eval_s[0-9].pt'
       -o -name 'n2c_s*.pt' -o -name 'uma-s-1p1.pt' -o -name 'iso_atom_elem_refs.yaml' -o -name 'n3_*.tar'
       -o -name '*_valsel.pt' -o -name '*_testB.pt' -o -name '*_devtest.pt' -o -name '*_val.pt'
       -o -name '*_sel.pt' -o -name '*_train.pt' -o -name 'train.pt' -o -name 'val.pt' -o -name 'sel.pt'
       -o -name 'valsel.pt' -o -name 'testB.pt' -o -name 'devtest.pt' -o -name 'fftruth_*' )
dpat=( -name 'n3_mcf' -o -name 'set_caches' -o -name 'n3_sets' -o -name 'offline-run-*'
       -o -name 'run-*-n3_*' -o -name 'wandb' -o -name 'mcf_release' -o -name 'thurlemann23'
       -o -name '.n3_box' -o -name 'n3_upload' )
: > "$WORK/name_hits"
: > "$WORK/all_files"
: > "$WORK/search_errors"
search_failed=0
for r in "${roots[@]}"; do
  [ -d "$r" ] || { echo "missing search root: $r" >> "$WORK/search_errors"; search_failed=1; continue; }
  find "$r" \( "${PRUNE[@]}" "${NAME_PRUNE[@]}" \) -prune \
      -o \( -type f \( "${fpat[@]}" \) -print \) \
      -o \( -type d \( "${dpat[@]}" \) -print -prune \) \
      >> "$WORK/name_hits" 2>> "$WORK/search_errors" || search_failed=1
  find "$r" \( "${PRUNE[@]}" \) -prune -o -type f -printf '%s\t%p\n' \
      >> "$WORK/all_files" 2>> "$WORK/search_errors" || search_failed=1
done
sort -u -o "$WORK/name_hits" "$WORK/name_hits"
sort -u -o "$WORK/all_files" "$WORK/all_files"

# Compare exact (size,hash) pairs; retain a receipt on PASS or FAIL.
python3 - "$WORK" "$MDIR" "$VERIFY_ONLY" "$search_failed" <<'PY'
import datetime, hashlib, json, os, sys
work, mdir, verify_only, search_failed = sys.argv[1:]
plan = json.load(open(os.path.join(work, "plan.json")))
# A zero-byte inventory entry carries no data, and its hash matches every empty file on the box (/.dockerenv,
# lock files, empty __init__.py: 3,274 false hits on the 2026-10-11 timing box), so it is not searched.
empty = sum(1 for size, _ in plan["csd_hashes"] if int(size) == 0)
private = {(int(size), sha) for size, sha in plan["csd_hashes"] if int(size) > 0}
sizes = {s for s, _ in private}
hits, errors, candidates = [], [], 0
lines = open(os.path.join(work, "all_files"), encoding="utf-8").read().splitlines()
for line in lines:
    try:
        size, path = line.split("\t", 1)
        size = int(size)
        if size not in sizes:
            continue
        candidates += 1
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if (size, h.hexdigest()) in private:
            hits.append(path)
    except Exception as e:
        errors.append(f"file search/hash failed: {type(e).__name__}")
names = open(os.path.join(work, "name_hits"), encoding="utf-8").read().splitlines()
search_errors = open(os.path.join(work, "search_errors"), encoding="utf-8").read().splitlines()
ok = not (names or hits or errors or search_errors or int(search_failed))
rec = {"verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "pass": ok,
       "verify_only": verify_only == "1", "targets": plan["targets"], "labels": plan["labels"],
       "files_searched": len(lines), "manifest_hashes": len(private), "empty_inventory_entries": empty,
       "files_hashed": candidates,
       "name_hits": names, "hash_hits": hits, "errors": errors + search_errors,
       "discard_outputs": plan["discard_outputs"], "pullback_receipt": plan["pullback_receipt"]}
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out = os.path.join(mdir, "cleanup_receipt_" + stamp + ".json")
json.dump(rec, open(out, "w"), indent=1)
print(f"[n3-cleanup] searched {len(lines)} files; hashed {candidates}; name hits {len(names)}; hash hits {len(hits)}; errors {len(rec['errors'])}")
print(f"[n3-cleanup] {'PASS' if ok else 'FAIL'}; retained receipt: {out}")
for p in (names + hits + rec["errors"])[:50]:
    print("[n3-cleanup]", p)
sys.exit(0 if ok else 1)
PY
say "now destroy the box by its instance ID and retain the destruction receipt"
