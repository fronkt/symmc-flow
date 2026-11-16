<!-- Archived 2026-11-15: the laptop version of the N3 step 9 plan (commit b7c50ea). Superseded the same day
     by the Anvil version at the top of tasks/todo.md (Frank chose an Anvil GPU job). -->

# N3 step 9 plan — 2026-11-15 (Frank 11-15: "Go ahead" = write this plan; nothing built or run yet; awaits his OK)

Step 9 unblinds TEST-B (1000 crystals) and DEV-TEST (200). P §7 step 9: "Score TEST-B and DEV-TEST once ... Write
`results/n3/n3_results.json`. Commit and push." Abbreviations as in the step-7 plan below.

**State on 11-15:** every input is on the laptop (step-7 pullback verified, 98 TEST-B/DEV-TEST bundles in
`results/n3/private/draws/`, MCF-R run dirs in `n3_mcf/infer/mcfR/`). There are 0 TEST-B/DEV-TEST match files, so
nothing has been scored. The scoring code (n3_score, n3_common, n3_set_cache, n3_harness, g2_asym_baselines,
molcrystal, n3_mcf_diag, n3_mcf_lib, n3_classical) has not changed since the freeze commit 772a36e.

## 1. What runs: the frozen list, verbatim, in SEL order

- `runs.step9`: 16 runs, 42 commands. Per set: MCF diagnostic (c) on the 3 MCF-R seeds, the Haar floor, (b),
  then (d) for primary / mcf05 / mcf08. TEST-B first, then DEV-TEST.
- `runs.step9_scoring`: 69 commands per set (61 match, 6 basin, 1 anchor, 1 report), TEST-B then DEV-TEST.
- **154 commands in all.** Some match commands repeat (iii-u primary/mcf05/mcf08 appear in the Haar-floor run and
  again in scoring). Match rows resume per fit, so a repeat finds its rows and returns. Run verbatim, no dedupe.
- Only `{W}` is replaced (W = 6). The report configs are written from `report_configs[set]` to
  `results/n3/report_<set>.json`, as R says; `frozen_report` refuses any scientific difference.
- Diagnostics (b)-(d) do not load the MCF fork (only (a) does), so no shim is involved.
- **Cost:** 164 CPU-h, about 27 h on 6 cores (REC, 10-15 projection). It is unmeasured on these bundles, and a
  multi-worker run on this laptop has come in 1.3-2x over a projection before, so plan for 27-55 h.

## 2. Driver `scripts/n3_step9.py` (modelled on `n3_step7.py`)

- [ ] `plan`: read-only. Prints all 154 commands with W filled in and runs the preflight. Executes nothing.
- [ ] `run --workers 6`: runs the commands in order and stops at the first non-zero exit.
- [ ] Allowlist: n3_score `match` / `haar-floor` / `basin` / `report` with `--step9`, n3_mcf_diag `b` / `c` / `d`,
      n3_classical `anchor`. Anything else (sampling, FF*, smoke / oracle / force flags) stops it before anything
      runs.
- [ ] Preflight, all before the first command:
  - SEL is sha `553f1dc9…`, committed, unchanged and pushed (n3_score's guard checks this again per command);
  - the scoring code above is byte-identical to 772a36e (working tree included), and P is sha `9a483a41…`;
  - every input named by a command exists, and its sha256 equals its entry in the laptop unpack receipts
    (`C:/Users/frank/n3_manifests/*.unpack_receipt.json`) where it has one;
  - on the first start, 0 TEST-B/DEV-TEST match, basin, anchor, report or diagnostic (b)-(d) outputs exist. On a
    resume, only the ones the driver's own events recorded;
  - the environment is recorded: Python 3.12.10 (`Python312/python.exe`), pymatgen 2026.5.4, scipy 1.17.1, numpy
    2.3.5, torch 2.11.0+cpu. On a resume, a different environment stops the driver (R: one CPU environment for
    the whole of step 9);
  - AC power is online and at least 50 GB is free (279 GB today).
- [ ] Events go to `results/n3/private/step9/events.jsonl`: start, done (exit code, wall time, sha256 of each
      output), stop and resume for every command. The log holds exit codes, row counts and harness / deterministic
      failure counts. It never holds hit rates or picked values.
- [ ] Windows: it clears power throttling (SetProcessInformation) for itself and every descendant every 30 s,
      retries its status writes (WinError 5), and is launched detached, so a Claude Code shell reap can't kill it.
- [ ] **Resume after a stop** (reboot, kill, crash): restart at the first command without a done record.
      - Match commands continue from their per-fit rows; n3_score trims a cut-off final line itself.
      - basin, anchor, haar-floor, report and diagnostics (b) / (d) re-run whole. They are deterministic (bootstrap
        seed 0), so the result is the same.
      - Diagnostic (c) deletes its partial bundle and re-runs.
      - Nothing scored is discarded, no setting changes, and every stop is listed in REC.
      - Recorded before the first command in `tasks/n3_step9_clarification_2026-11-15.md`. P is not edited.
- [ ] Tests `tests/test_n3_step9.py`, on stub commands and a synthetic tree (no TEST data is read). They cover:
      - the order, a stop on failure, and a resume that skips done commands;
      - the allowlist refusals and an environment change;
      - preflight FAIL on planted controls: a fake `match/zz_testB.jsonl`, a changed SEL, a changed scorer, and a
        changed bundle sha.
      Then run the whole N3 suite.

## 3. Unblinding and results

- Nothing is read until the last command (the DEV-TEST report) finishes. Progress is followed by row counts only.
- [ ] `scripts/n3_results.py` assembles `results/n3/n3_results.json` from:
      - both report outputs (the `report_<set>.out.json` files are committed as well);
      - the diagnostic (b) / (c) / (d) summaries per z_MCF stratum and the Haar-floor summary;
      - the Gate C, diagnostic (a), degrade and H2 FORM (EXACT) records from SEL;
      - step-9 provenance: driver, code and SEL sha256, the environment, an events summary and any stops.
- [ ] Hygiene check on every committed JSON: numbers, labels and refcodes only. No coordinates, cells or CIF text.
- [ ] Commit and push `n3_results.json`, `report_<set>.json` and `report_<set>.out.json` right away, before any
      interpretation. Then report to Frank: H1-H3 outcomes with Δ and CI, worded by P §6's abstract rules.
- Later and separate: the manuscript rebuild (R "Step 9") and the Tier-2 (§8) decision.

## 4. Laptop conditions (memory: Windows background compute)

- For 1-2 days: AC in, lid open, don't touch the power button. Modern Standby freezes the run.
- Windows Update: a 7-day pause plus `NoAutoRebootWithLoggedOnUsers` needs one UAC prompt from Frank. Without it,
  a ~04:30 restart costs only the fits in flight, and the run resumes after his sign-in.
- No other heavy local job at the same time (the Claude Code memory reaper).

## 5. Timeline

Build and tests take about half a day after the OK. The run takes 27-55 h. Results land about 11-17 to 11-19.

## 6. Decisions for Frank (recommended first)

1. Where: **the laptop, W = 6** / an Anvil GPU job on its 32 cores. Anvil needs a new CSD upload in manual mode
   and a second cleanup.
2. Stops: **resume as in section 2, recorded in a dated clarification before the first command** / strict: any
   stop discards every step-9 row and scoring restarts from zero.
3. Updates: **approve one UAC prompt for a 7-day pause and no auto-reboot** / leave updates on.
