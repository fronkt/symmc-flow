# N3 step-9 clarification, 2026-11-15 (Frank's decisions, 2026-11-15; recorded before any step-9 command runs)

Like the step-7 clarification, this is kept out of `tasks/n3_protocol.md` (P). The step-6 freeze
(`results/n3/n3_selection.json`, SEL, 772a36e, sha256 `553f1dc9…e1f491`) records P's SHA-256 (`9a483a41…`), and
`n3_score.py` refuses a freeze whose protocol file has changed.

It concerns execution only. It changes no run, seed, matcher, selector, estimand, stratum, bootstrap count,
hypothesis, H2 FORM or outcome-to-claim rule. No listed step-9 command is added or removed. Section 3 adds one
aggregation that P §3.2 asks for and no frozen command performs.

## 1. Where step 9 runs (Frank, 2026-11-15: "Anvil GPU job, 32 cores")

- P §7 step 9: "CPU matching may run locally or on a CPU box under the same hygiene."
- Step 9 runs on Anvil as two Slurm jobs in the GPU partition, because Anvil CPU SU are STS-only:
  - **9T**: TEST-B.
  - **9D**: DEV-TEST.
- Each job gets 1 A100 and 32 cores, W = 28.
- The GPU is not given to the container (no `--nv`, `CUDA_VISIBLE_DEVICES` empty). No step-9 command uses it, and no
  FF* evaluation runs after step 8.
- **Same hygiene as steps 5-8:**
  - The inputs go up in one manual-approval-mode session (`n3_box_manifest.py --step 9`) into `$SCRATCH/n3_box`.
  - The outputs come back the same way (`pullback --label step9`).
  - The box root and the upload directory are then deleted and searched for, as at step 8.
- **Scoring environment:** `$PROJECT/n3_conda/envs/symmc` inside the step-4 image, the environment VALSEL was scored
  in. All 109 VALSEL match metas record torch 2.11.0+cu128, pymatgen 2026.5.4, scipy 1.17.1 and numpy 2.3.5. It is
  recorded at each set's first start, and a resume under any other environment stops (runbook Step 9: one CPU
  environment for the whole of step 9).
- **Code:**
  - The scoring code is byte-identical to 772a36e (the pack refuses otherwise).
  - The one post-freeze change in the shipped closure is `n3_mcf_run.py`'s code_fix_7
    (`tasks/n3_step7_codefix_2026-11-09.md`). Diagnostic (c) uses only that file's `bundle_arm` / `smoke_subset`.

## 2. How the frozen commands run

- **Driver:** `scripts/n3_step9.py`.
  - It runs `runs.step9` for the set (MCF diagnostics (b)-(d) and the Haar floor), then `runs.step9_scoring[set]`:
    92 commands per set, in SEL's order.
  - It refuses any other script or subcommand, smoke / oracle / force flags, an n3_score command without `--step9`,
    and a command naming the other set.
- **Rewrites permitted:**
  - `{W}` → 28;
  - `python` → `$PY_SYMMC`;
  - `--selection-sha256 553f1dc9…` appended to n3_score match / haar-floor / basin / report and n3_mcf_diag b / d.
    Runbook Step 9: "A CPU box without git also requires --selection-sha256 with the laptop-verified published
    freeze hash." n3_mcf_diag c and n3_classical anchor take no such flag and check no selection.
- **Report config:** `results/n3/report_<set>.json` is written verbatim from `report_configs[set]` (runbook Step 9).
- **Blinding preflight:** before a set's first command, no match, basin, anchor, report, oracle-aligned bundle, (c) LJ
  row or diagnostic (b)-(d) output of that set may exist under the root.
- **Logs:**
  - `report`, `basin`, `anchor` and the diagnostics print outcomes. Each command's stdout / stderr therefore goes to
    its own log inside the box root (`results/n3/private/step9/logs/`), which is pulled back and deleted with the
    root.
  - The Slurm log in `$SCRATCH/n3_ops` gets start / done lines and, on a failure, the traceback only.
  - Progress is followed from sacct and the events file, never from the command logs.

## 3. The diagnostic (c) summary (an aggregation P §3.2 asks for)

- **The gap:** the frozen list makes the oracle-aligned MCF-R bundles (`n3_mcf_diag.py c`) and the scorer's rows for
  them (primary, mcf05, mcf08; every draw), but no frozen command aggregates them.
- **What P §3.2 (c) asks:** "Report draw@1, any@16 and picked@1 per z_MCF and matcher. It is not computed for
  z_MCF = 1, where it is exact by construction (counted)."
- **What runs:** `scripts/n3_step9_summary.py c --set <set> --step9` runs as each set's last unit, after the frozen
  list, in the same job and environment.
- **What it does:**
  - reads only the scorer's current rows (refused unless they match the bundle, the scoring identity and the
    freeze, as `report` refuses them);
  - uses MCF-R's frozen selector from `report_configs` (LJ, recomputed after alignment by (c)) and the three MCF-R
    seeds;
  - writes `results/n3/private/mcf/diag_c_<set>.json`, labelled ORACLE ITEM, not a result.
- It makes no match and no choice.

## 4. Stops and resumption (Frank, 2026-11-15: "Resume")

1. **A job stop is not a second scoring.** This covers timeout, node failure, preemption, OOM and a kill. Every
   step-9 command is deterministic: n3_score is worker-count invariant, every bootstrap uses a fixed seed
   (`n3_stats`, diagnostic d), and (c) is closed form. So a resumed command gives the numbers an uninterrupted one would.
2. **A command with a done record is skipped.** Otherwise it runs again:
   - n3_score match and n3_mcf_diag b continue from their per-fit / per-(crystal, matcher) rows;
   - n3_mcf_diag c deletes its own unrecorded output bundle and re-makes it (its LJ rows resume);
   - haar-floor, basin, anchor, report, diagnostic d and the (c) summary re-run whole.
   - Nothing recorded is deleted.
3. **Every stop and restart goes in the run record** (`tasks/n3_step4_record_2026-10-11.md`): job, cause, the
   command affected and how it continued.
4. **No output is read before both sets' reports and (c) summaries exist.** Then `scripts/n3_results.py` writes
   `results/n3/n3_results.json`, and it is committed and pushed together with `report_<set>.json` and
   `report_<set>.out.json` before any interpretation.

## 5. A failure that needs a code change

If a command fails for a code or environment reason (not a stop):
- The fix may only make the frozen computation run as specified. No matcher, selector, estimand, seed, bootstrap
  count, stratum, H2 FORM or report configuration changes.
- The fix is recorded in a dated file and pushed before anything runs again.
- Rows or outputs the fix affects are deleted and re-made.
- The fix and what was re-made are disclosed in the run record and the SI.
- Only the traceback is shown. Outcome lines printed before the failure stay unread in the command log.

## 6. Known in advance

- **The floor arm's mcf10 block in `report` will read "missing".** The frozen list matches iii-u under mcf10 only at
  its LJ picks, while the floor arm (selector random) needs every draw. No run is added. This is the continuous-RMS
  endpoint of the random floor, an SI item. Primary, mcf05 and mcf08 are complete for every arm.
- DEV-TEST is labelled "used during development, not confirmatory" (P §1.4) in `n3_results.json`.
