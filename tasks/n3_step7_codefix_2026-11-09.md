# N3 step-7 code fix, 2026-11-09 (Frank's OK, 2026-11-09; recorded before any TEST-B or DEV-TEST input reaches the box)

Like `tasks/n3_step7_clarification_2026-11-08.md`, this is kept out of `tasks/n3_protocol.md`, because the step-6
freeze (`results/n3/n3_selection.json`, 772a36e) records the protocol's SHA-256 and `n3_score.py` refuses a
changed protocol.

It concerns execution only. It changes no run, seed, command, menu, endpoint, hypothesis, never-reduced item or
outcome-to-claim rule. No listed run is added or removed. MCF-R and MCF-A behave exactly as before.

## What failed

The step-7 pre-upload smoke (Anvil job 22088788, gpu-debug, 2026-11-09 12:41-12:47 EST, VAL / VALSEL inputs
only; `results/n3/private/anvil_jobs/step7_smoke.sh`) ran each never-exercised step-7 path in its frozen form.

- Passed:
  - the prelude verify;
  - `n3_classical.py ie` (2 crystals, 0 MISS);
  - `basin --from-ie`;
  - `basin` direct;
  - `n3_mcf_run.py prior_floor --seed 0 --set valsel` (400 crystals x 16 draws) and its 6400 LJ evaluations.
- Failed: `n3_mcf_run.py to_bundle` on the floor run, with `KeyError: 'info'` at
  `chk["info"].get("resid_nonfinite_on_finite_draws", 0)`.

## Cause

Amendment A3's commit (a25a444, 2026-10-19: a non-finite RESID on a finite MCF draw is recorded, not failed) added
that read to `cmd_to_bundle` for every arm.

- `run_check`, used by MCF-R and MCF-A, returns an `info` dict.
- `run_check_floor`, used by the MCF-A prior floor, never had one.

The laptop floor smoke passed on 2026-09-28, before A3. No test ran the floor through `to_bundle` after A3.
Without a fix, every `MCF-A-floor/*` step-7 run would fail at `to_bundle`.

## Fix

`run_check_floor` now returns `"info": {}` (the floor has no RESID), so `to_bundle` records
`resid_nonfinite_on_finite_draws: 0` for the floor, which is the value its `.get(..., 0)` default intended.

| `scripts/n3_mcf_run.py` | SHA-256 (CRLF working tree, as shipped) |
|---|---|
| On the box since the step-5 upload (step-5 manifest, b50fb74; same content at the freeze head 0ce4beb) | `a4c99a1db5af870597e2ec98b7beef24211367973480086de2502f87eccbc6be` |
| Fixed | `86dfe34deb96bf4b8e5524d5db153bfb904284ec32de20a220a8fb11718c0ead` |

## Evidence

- Laptop, with the frozen script unedited (`to_bundle` with only `run_check_floor` patched in memory) on the
  09-28 laptop floor smoke run: the unpatched code raises the same `KeyError`. Patched, the bundle writes with
  4/4 LJ ok and the expected `arm_failures`.
- `tests/test_n3_mcf_floor_bundle.py` covers three things:
  - `run_check_floor` returns and writes `info`, tested hermetically;
  - `to_bundle` runs end to end on a tmp copy of the laptop floor smoke run;
  - the step-7 pack re-ships the script.
- N3 suite: 203 passed, 10 skipped.

## How the fix reaches the box

- The step-7 pack (`scripts/n3_box_manifest.py`, group `code_fix_7`) re-ships `scripts/n3_mcf_run.py` over the
  step-5 copy.
- `step7_verify.py` judges every re-shipped path by the step-7 manifest and every other step-5 file by the
  step-5 manifest, as it already does for `tasks/n3_protocol.md`.
- Before the upload, a second smoke (gpu-debug) re-runs only the failed `to_bundle` on the VALSEL floor run. It
  uses the fixed file from `$SCRATCH/n3_ops/fix7/scripts/`, outside the box root, so the step-5 verify still
  passes. Its result is recorded in `tasks/n3_step4_record_2026-10-11.md`.
