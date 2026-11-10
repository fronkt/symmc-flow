# N3 step 7 plan — 2026-11-08 (Frank 11-08: decisions 1-7 as recommended; build approved; ask before sbatch)

**Decision 1 is recorded in `tasks/n3_step7_clarification_2026-11-08.md`, not as a §9 clarification in P.**
SEL records P's SHA-256, and `n3_score.py` `selection_data` refuses a freeze whose protocol file has changed
(`scripts/n3_score.py:165-167`). An edit to P after the freeze would invalidate it.

The freeze is published: 772a36e, `results/n3/n3_selection.json`, sha256 `553f1dc9…e1f491`. Step 7 runs exactly
the 76 runs in `runs.step7` on TEST-B (1000 crystals) and DEV-TEST (200 crystals), with their committed seeds.
Only `{W}`, `{G}` and `{GPU}` may be replaced, and they affect scheduling only. Abbreviations: P =
`tasks/n3_protocol.md`, R = `tasks/n3_runbook.md`, SEL = `results/n3/n3_selection.json`, REC =
`tasks/n3_step4_record_2026-10-11.md`, M = `scripts/n3_box_manifest.py`.

**Maintenance does not block step 7.** `Nov_16_Flush` (11-16 to 11-23) reserves only the CPU, big-memory and H100
nodes (a, b, h). The A100 nodes g[000-015] stay up (RCAC news 7947: CPU and AI nodes are down 11-16 04:00 to
11-18 06:00; login, A100 GPU nodes and storage stay online). This replaces the 5b plan's "Step 7 runs after
11-23" line.

## 1. Pack and upload (Frank in manual-approval mode)

- **DONE 11-10 (Frank: "Go ahead", manual mode): packed, uploaded, unpacked, verified, dry-run PASS. Details are in
  REC, "Step-7 upload and dry run (2026-11-10)".**
- [x] `python scripts/n3_box_manifest.py pack --step 7 --out results/n3/private/uploads/step7` (not `--new-box`:
      step 7 reuses the step-5 box root `$SCRATCH/n3_box`).
      - `list --step 7` (11-08): deployable, 24 files, 70.3 MB.
      - Contents: protocol, `testB_refcodes.txt` + SEL, `gateF_{testB,devtest}`, TEST-B/DEV-TEST set caches, the
        MCF test/devtest exports, `items_{testB,devtest}` + `ivP_frozen.json`, and the stored N1/N2 DEV-TEST
        draws + LJ picks.
      - The pack checks that SEL is complete, tracked, unchanged and pushed. It ships no code: the box keeps the
        step-5 code.
      - All 56 step-5 code files equal HEAD except `tasks/n3_protocol.md`, which the 11-06 clarifications changed
        and the step-7 pack re-ships.
- [x] scp the tar + `.sha256` to `$SCRATCH/n3_upload/` (CSD: manual mode only).
- [x] Unpack inside the image on the login node (its python3 is 3.6), with
      `N3_MANIFEST_DIR=$SCRATCH/n3_manifests`; then verify. Copy the receipt back to `C:/Users/frank/n3_manifests`.
- [x] Check that every `step7_inputs` row (29: frozen MCF/iv-P/OURS checkpoints, MCF configs, OURS/N2 DEV-TEST
      draws) is on the box with its SEL sha256.

## 2. Prelude and verify for step 7 (new private files, no shipped script edited)

`step5_verify.py` refuses once a step-7 manifest exists, by design. Step 7 gets `anvil_step7_prelude.sh` and
`step7_verify.py` in `results/n3/private/anvil_jobs/`. The verify:
- [x] runs M `verify` on the step-7 inventory;
- [x] verifies the step-5 inventory, taking any path the step-7 manifest re-ships from the step-7 manifest.
      **The shipped `verify --step 5` would FAIL after the step-7 unpack**, because the step-7 copy of
      `tasks/n3_protocol.md` differs from the step-5 one. Handle that here, not by editing M.
- [x] refuses if the box SEL sha256 is not `553f1dc9…`, and logs the inner script, driver and freeze sha256 values;
- [x] **refuses any match, anchor, report, label or truth-based diagnostic output on TEST-B / DEV-TEST** under the
      root (R:235): e.g. `private/match/*testB*|*devtest*`, `report_*`, `labels_*`, `anchor_{testB,devtest}*`,
      `diag_[bcd]_*`. Nothing on the box may call match, anchor, report or diagnostics (b)-(d) on test outputs;
- [x] refuses symlinks, as step 5 does: list them and leave a removal for Frank, if MCF inference leaves wandb
      or hydra links.
- (11-08, laptop) Built: `anvil_step7_prelude.sh`, `anvil_step7.sbatch` (new; `anvil_step5.sbatch` sources the
  step-5 prelude) and `step7_verify.py`. A synthetic-root smoke passed: clean PASS; planted controls all FAIL
  (TEST-B match rows, anchor_testB, report_devtest, diag_b_testB, oracle_aligned in a TEST-B run dir,
  ff_labels_devtest, a changed step7_inputs file, a changed step-5 file, a wrong freeze sha, a step-7 pullback
  manifest); VALSEL match rows and the fftruth TEST-B bundle pass. The same synthetic root confirms that the
  shipped `verify --step 5` FAILs on the re-shipped protocol. The box check waits for the dry run.

## 3. Driver and jobs

- [x] `scripts/n3_step7.py` (new, stdlib only, not shipped; built on `n3_step5b.py`). (11-08; `jobs`, `plan`, `run`;
      the two uploaded DEV-TEST conversions sit in 7C as check-only runs, so 7C lists 27.)
      - It reads the run list from the box SEL, runs each run's `commands` verbatim, and substitutes only
        `{W}`/`{G}`/`{GPU}`. It refuses a command or run id that is not in SEL.
      - Python per script, as 5b: `n3_mcf_run`/`n3_mcf_gauge`/`n3_mcf_sg` → `PY_MCF`; `n3_ours`/`n3_classical`/
        `n3_old_symmc` → `PY_SYMMC`; FF* through `N3_FF_PYTHON`.
      - After each run it checks that every listed output exists and writes its sha256 to a per-job checksum
        list (R:231). It also checks that each MCF `check.json` passes and that each production bundle carries
        the full frozen refcode order (R:234; refcode order only, no truth).
      - A finished run (all outputs present and checksummed) is never redone.
- [x] Tests with a fake executor, as in 5b: verbatim commands, the placeholder rule, refusal of unlisted runs,
      dependency order, the checksum list, and the restart rule of section 5. (`tests/test_n3_step7.py`, 32
      cases on the real freeze after the review fixes; N3 suite 200 passed, 10 skipped.)
- [x] Review it (an independent reviewer on the code). 11-08: no crash or protocol violation on the normal path.
      The 8 build fixes it asked for are in the driver and private files:
      - trim a cut-off shard line before a resume;
      - `--dependency=singleton` plus a fixed job name, and `--open-mode=append`;
      - a failing `check.json` is kept in the event;
      - one in-job retry for single-call runs;
      - side outputs are checksummed;
      - the `(mcf/)?sg_oracle` and `mcfRoal_` verify patterns;
      - the prelude logs its own sha256;
      - a real recorded-file test.
- [x] Login-node dry run: prelude + `plan --job X` (needs the step-7 upload, manual mode). 11-10: STEP7_VERIFY_PASS;
      `jobs` lists 76 runs in 9 jobs; all 9 `plan` calls exit 0. 7E0-7E2 wait on 7O's TEST-B draws, as designed.
- [x] **Smoke 22088788 RAN 11-09 12:41-12:47 (gpu-debug): verify, ie, basin --from-ie, basin direct, prior_floor and its
      6400 LJ evaluations PASSED; floor `to_bundle` FAILED (`KeyError: 'info'`).** Cause: A3 (a25a444) made to_bundle
      read `chk["info"]`, and `run_check_floor` had none. Frank's OK 11-09: "Fix + re-smoke". The fix, record and
      re-ship route are in `tasks/n3_step7_codefix_2026-11-09.md`; tests are in `tests/test_n3_mcf_floor_bundle.py`;
      N3 suite: 203 passed, 10 skipped.
- [x] **Smoke 2 22112588 PASSED 11-09 19:54-19:55 EST (gpu-debug, 46 s)** (`step7_smoke2.sh` `c6d493e6…`): the floor
      `to_bundle` with the fixed script from `$OPS/fix7/scripts/` (SHA-256 `86dfe34d…`, checked in the job), outside
      the box root. The prelude verify passed first (209 files, unexcused 0). It wrote `draws/mcfAfloor_s0_valsel.pt`
      (n 400, S 16, 6400 valid; LJ ok 6395, spacing_inf 5), and `check.json` passed with `"info": {}`.
      `sbatch -p gpu-debug -t 00:30:00 -J n3-s7smoke2 anvil_step5.sbatch step7_smoke2.sh`
- [x] (11-10: both confirmed.) At the step-7 pack: confirm that `list --step 7` shows group `code_fix_7` and that the box-side
      `step7_verify` prints `scripts/n3_mcf_run.py` among the re-shipped paths.
- (Smoke 1 plan, as approved 11-08:) **one gpu-debug smoke of the FF* / floor paths no test has run**, on VAL / VALSEL
      only, BEFORE the step-7 upload.
      - The step-5 prelude refuses once a step-7 manifest exists, so the smoke runs under `anvil_step5.sbatch`:
        `sbatch -p gpu-debug -t 00:30:00 -J n3-s7smoke anvil_step5.sbatch step7_smoke.sh`.
      - `step7_smoke.sh` would be written only after the OK. Its commands, with where each output lands; none
        can collide with a step-7 output (`_val` / `_valsel` names; no 5b product of these names exists):
        ```
        "$PY_SYMMC" scripts/n3_classical.py ie --draws results/n3/private/draws/ours_s0_val.pt --limit 2 --before-form --workers 1
          # -> draws/i-E-ours_s0_val.partial.pt; shard classical/i-E-ours_s0_val.smoke.jsonl; classical/ff_logs/i-E-ours_s0_val/
        "$PY_SYMMC" scripts/n3_classical.py basin --draws results/n3/private/draws/i-E-ours_s0_val.partial.pt --from-ie --limit 2
          # -> draws/basin-ours_s0_val.partial.pt (no FF*, no shard)
        "$PY_SYMMC" scripts/n3_classical.py basin --draws results/n3/private/draws/ours_n2_s0_val.pt --limit 2 --before-form --workers 1
          # -> draws/basin-ours_n2_s0_val.partial.pt; shard classical/basin-ours_n2_s0_val.smoke.jsonl
        "$PY_MCF" scripts/n3_mcf_run.py prior_floor --seed 0 --set valsel
        "$PY_MCF" scripts/n3_mcf_run.py to_bundle --run_dir n3_mcf/infer/mcfA_floor/valsel/s0 --workers 28
          # -> n3_mcf/infer/mcfA_floor/valsel/s0/ and draws/mcfAfloor_s0_valsel.pt
        ```
      - Recommendation: drop `--before-form`. The H2 form exists (`classical/anchor_valsel.json`), so without the
        flag `ie` and the direct `basin` run step 7's own form guard and FF* configuration check.
      - Nothing is matched. These are CSD-derived VAL / VALSEL products, deleted with the root at step 8.
      - The prior floor runs on all 400 VALSEL crystals; its time is unmeasured, so if 30 min is short it goes on
        `gpu` with `-t 01:00:00`.
- **Submit commands, after the OK** (from `$SCRATCH/n3_ops`). The `-J` name must be the fixed one: each inner
  script refuses any other, and `--dependency=singleton` keys on the name.
  ```
  sbatch -J n3-s7O  -t 08:00:00 anvil_step7.sbatch step7_7O.sh
  sbatch -J n3-s7C  -t 08:00:00 anvil_step7.sbatch step7_7C.sh
  sbatch -J n3-s7M  -t 12:00:00 anvil_step7.sbatch step7_7M.sh
  sbatch -J n3-s7S  -t 2-00:00:00 anvil_step7.sbatch step7_7S.sh
  sbatch -J n3-s7Sd -t 2-00:00:00 anvil_step7.sbatch step7_7Sd.sh
  sbatch -J n3-s7Ed -t 2-00:00:00 anvil_step7.sbatch step7_7Ed.sh
  sbatch -J n3-s7E0 -t 2-00:00:00 --dependency=afterok:<7O id>,singleton anvil_step7.sbatch step7_7E0.sh   # same for 7E1, 7E2
  ```
  - A command-line `--dependency` replaces the script's `singleton` line, so the 7E jobs repeat `,singleton`.
  - **If 7O is resubmitted**, the queued 7E0-7E2 still wait on the old 7O id (afterok on a failed job never
    clears). Cancel them and resubmit them with `--dependency=afterok:<new 7O id>,singleton`.

**Job grouping.** One A100 and 32 cores per job (W = 28, G = 1, as timed). The 2 DEV-TEST OURS/N2 conversions are
done; their bundles arrive in the step-7 pack.

| job | runs | deps | est GPU-h (scaled from measured) | worst case (A3 projection) | limit |
|---|---|---|---|---|---|
| 7O | ours/testB, ours_n2/testB, p1-ours/testB/s0-s2 (5) | none | ~1.5 | ~3 | 8 h |
| 7C | iii-u x6 (haar+press), p1ref x2, p1-ours/devtest x3, iv-P x6, iv-S x6, fftruth x2 (25) | none | ~2 | ~6 | 8 h |
| 7M | MCF-R x6, MCF-R+G x6 (2 infer + 6 gauge scans), ii-S x6, MCF-A x6, MCF-A floor x6 (30) | none | ~5 | ~8 | 12 h |
| 7S | iii-s/testB (screen + search) (1) | none | ~25 | ~190 | 48 h |
| 7Sd | iii-s/devtest (1) | none | ~5 | ~38 | 48 h |
| 7E0/7E1/7E2 | i-E+basin-ours/testB/sk and basin-ours_n2/testB/sk (2 each) | afterok 7O | ~6.5 each | ~50 each | 48 h |
| 7Ed | i-E+basin-ours/devtest x3, basin-ours_n2/devtest x3 (6) | none | ~4 | ~30 | 48 h |

- The jobs cover 74 runs, plus the 2 conversions already done, for 76. The order inside a job follows SEL's
  order: iii-u then press; iii-s screen then search; MCF-R infer then gauge then ii-S; iv-P then iv-S.
- **Estimates are guesses scaled from measured 5b rates:**
  - iii-s VAL search: 100 crystals in 8723 s at G = 1, i.e. 87 s/crystal and 1.36 s per minimisation;
  - MCF call: about 10 min per 400 crystals;
  - OURS sampling: 300 crystals in 11.3 min;
  - VALSEL truths: 400 in a 9-min job.
- i-E is unmeasured. Its starts are generated draws, not screened starts, so they may need more evaluations.
- The A3 projection costs a minimisation at about 10.6 s, 7.8x the measured 1.36 s. It is the worst case,
  roughly every minimisation at its 100-evaluation cap.
- **Total:** about 60-120 GPU-h likely (point estimate ~62), and at most about 475 (A3: step 7 is 473-482 box-h).
- **Cap:** 700 GPU-h over steps 5-7. Steps 5a+5b used 27.7 GPU-h (sacct, elapsed x GPUs, n3-* jobs since 10-19),
  so about 672 are left. Both cases fit. Balance: 1,429 GPU-h (`mybalance`, 11-08).
- **48 h wall:** at the worst-case rate, 7S needs about 4 job lengths and each 7E about 1. That works only
  under the restart rule of section 5 (per-crystal resume) or with a larger G.
- **The 12-GPU per-user cap is shared with pxrd array 22019698** (%8, 2 h tasks). At most 8 N3 jobs run at once
  (7E waits for 7O). Queue starts have run at a few per day, so the cap rarely binds.

## 4. Runs labelled CPU or "local after step 8"

- ii-S (6), iv-S (6), the +G gauge scans (6 inside MCF-R+G), the MCF-A prior floor (6), iii-u/press (6), p1ref
  (2) and p1-ours (6) are CPU work. Anvil CPU SU are STS-only, so they run **inside the GPU jobs** above, on 28
  of the 32 cores. That keeps every step-7 output in one step-8 pullback.
- "Local after step 8" is also allowed for the gauge scans and ii-S. It is not needed: they cost under 1 box-h.
- No FF* call runs after step 8 (P step 7). Every FF* run (iii-s, fftruth, i-E, both BASIN rows) is on the box.

## 5. Failure handling: record before any test output exists

P (§7 Failures and the Budget overrun paragraph) and R:237-239 say: if the box dies or the cap forces a stop during
step 7, discard **all** partial test outputs unscored and re-run the complete list with the same seeds.

**Open question:** on a persistent Anvil box, does one Slurm job ending early (timeout, node failure, preemption,
OOM) count as a "box death"? Options, to record as a dated §9 clarification before the upload:
- **A (recommended).**
  - A job stop is not a box death. Its finished runs stay; its unfinished runs continue with the committed seeds.
  - Per-crystal resumable runs continue from their complete records, which are cfg-checked and refuse another
    configuration. These are `n3_classical` haar/press/screen/search/truths/ie/basin/p1basin and the
    `n3_ours sample` part file.
  - Single-call runs delete their partial outputs and re-run from the start: MCF infer (which refuses a run dir
    holding partial outputs), iv-P sample, prior_floor, gauge, ii-S and iv-S.
  - Every stop and restart is listed in REC. Nothing is matched or inspected before step 9.
  - A box death (box root lost or corrupted, a verify failure, a scratch purge, account loss) or a cap stop still
    means the full discard and full re-run as written.
- B. Strict reading: any job stop means discarding every step-7 output and re-running all 76 runs. The cost is up
  to 2x step 7, and with 9 jobs one stop is not unlikely.
- C. Per-run restart only, with no per-crystal resume. This is infeasible for 7S at the worst-case rate (one run
  over 48 h) unless G >= 4.

## 6. Steps 8 and 9 (brief)

- [ ] Step 8, after every listed run is checksummed:
      - On the box, `pullback --label step7` with `N3_MANIFEST_DIR=$SCRATCH/n3_manifests` set explicitly (the
        env.sh default does not exist). The size is a guess, about 10-20 GB.
      - scp in manual mode, then `unpack --no-overwrite` on the laptop, then send the receipt back.
      - `N3_CLEANUP_PLAN_ONLY=1 bash scripts/n3_box_cleanup.sh`: review the targets, then run the cleanup.
      - Targets: `$SCRATCH/n3_box` (inputs, exports, checkpoints, outputs, wandb/hydra dirs), `$SCRATCH/n3_upload`
        (including the step-6 pullback tar). Add `$SCRATCH/n3_ops` logs via `N3_EXTRA_DELETE_FILE` if the
        search finds names in them.
      - Keep the receipts in `$SCRATCH/n3_manifests`. "Destroy the box" on Anvil means deleting the root;
        `$PROJECT` envs and images hold no CSD (check with the cleanup search).
- [ ] Step 9, on the laptop: `runs.step9` (16: MCF diagnostics (b)-(d) and the Haar floor), then
      `runs.step9_scoring` (2 reports, 10,000 bootstrap draws). The projection gives about 164 CPU-h, ~27 h on 6
      cores. Write `results/n3/n3_results.json`, commit and push.
- Tier-2 (§8) runs only after step 9 and is not in the step-7 list. D-c0 uses TEST-B but OURS + iii-u on CPU
  locally.

## Timeline

- Build, tests, review and dry run: about 1 day after the OK. Pack + upload: 1 manual-mode session, at the dry run.
- Queue waits have run 9-25 h. The longest job is ~25 h (guess), so step 7 ends about 3-5 days after submission.
  Steps 8 + 9 take about 2-3 days.
- **Results about 11-16 to 11-20** if submitted by ~11-11. At worst-case rates with resubmits, about 2 weeks
  later. Either way it is well inside the 2027-01-22 window.

## Decisions for Frank (recommended first)

1. Failure rule: **A** (job stop ≠ box death; per-crystal resume or a per-run restart with committed seeds),
   recorded as a dated §9 clarification before the upload / B (strict) / C (per-run only).
2. Build: **build the driver, prelude, verify and tests; review; dry run; ask before sbatch** / plan changes first.
3. Upload timing: **at the dry run, in one manual-mode session** / now.
4. FF* workers: **G = 1, as timed in A3** / G = 2 for 7S and 7E (invariant; halves wall if the GPU has room,
   unmeasured).
5. pxrd sharing the cap: **leave array 22019698 at %8** / lower it to %4 while step 7 runs.
6. CPU-labelled runs: **inside the GPU jobs on the box** / gauge + ii-S locally after step 8.
7. Limits: **48 h for the FF* jobs (7S, 7Sd, 7E\*)** / 24 h + resume (needs 1A; shorter jobs backfill sooner).

## Verification

- [ ] `pack --step 7` deployable; upload sha OK; unpack + verify PASS (receipt on the laptop).
- [ ] The step-7 prelude passes on the box and fails on the planted controls (a fake `match/zz_testB.jsonl`,
      a changed SEL).
- [ ] Every job log carries the inner, driver and SEL sha256 values; every listed output exists with a checksum.
- [ ] No match, anchor, report or diagnostic (b)-(d) output on TEST-B / DEV-TEST exists on the box at step 8.

---

# N3 step 5b plan — 2026-11-05 (Frank: "yes go ahead"; 2026-11-06: decisions 1-3 as recommended, build approved, ask before sbatch)

Step 5a is done (FD, Gate C all_pass, H2 form EXACT, iv-P x5, MCF-R/A x3). Step 5b means sampling every
VALSEL candidate, scoring and selecting each arm, and running diagnostic (a) (runbook lines 176-196). Step 6,
the laptop freeze, comes after it. Abbreviations: P = `tasks/n3_protocol.md`, S = `scripts/n3_select.py`,
REC = `tasks/n3_step4_record_2026-10-11.md`.

**Already on the box, so no re-upload.** The deployed step-5 manifest (b50fb74, 209 files) matches HEAD on
all 56 code files. It carries every 5b input: set caches, the valsel MCF export, the iv-P items, the OURS
checkpoints and converted VAL bundles, UMA and the FF* config. verify ignores files that are not listed
(`n3_box_manifest.py:599-636`), so new scripts and config JSONs can go up without a re-pack. Editing a
**shipped** script would break verify and force the CSD re-pack with Frank present, so 5b edits none.

## Decisions (Frank, 2026-11-06: (b) set reading; NaN ranks last; iii-s runs in 5b)

1. **`last.ckpt` twins.** In all 6 MCF runs, last.ckpt is SHA-identical to the latest top-3 checkpoint
   (REC:462-470). The protocol says "top-3 + last" (P:518, 553, 579, 1234) and has no dedup rule. S requires a
   `last` label among 1-4 candidates (S:461-464).
   - **(a) Literal.** List 4 labels and sample the twin twice.
     - Cost: +6 MCF calls (~0.5 GPU-h).
     - Both copies use the same weights and the same seed, so the choice between them is decided by GPU noise (P:1483-1485).
   - **(b) Set reading (recommended).** List the 3 distinct checkpoints and label the twin `last`.
     - Each set of weights is sampled once, and S accepts it unchanged.
     - Disclose: "last coincided with a top-3 checkpoint in every run".
2. **iv-P NaN minimum.** `S:610-612` needs three *finite* seed-0 minima. The box record has `lr1e-3: NaN`,
   because that run diverged in epoch 1, so step 6 would refuse `ivP_window`.
   - **Fix (recommended):** a non-finite minimum ranks last, matching the existing non-finite-last rule (P:258,
     1394-1397). The choice is still lr 1e-4.
   - It is a laptop-only fix with a new test, because S is not shipped to the box.
   - Add a dated clarification line and push it **before any VALSEL score exists**. P:1054-1055 freezes
     amendments once a score exists.
3. **iii-s VAL sanity row** (P:667-668). It is booked in step 5 at 17.8 GPU-h on the A100, the largest 5b
   item. It is not in the runbook's 5b text, and S's step-7 list leaves out VAL, so if 5b skips it, it never
   runs. **Recommend: run it in 5b as its own parallel job.**

Readings I'll apply unless Frank objects:
- MCF-A Stage-1 final tie-break: MCF's logged valid/loss, as P:606 says. MCF-R Stage B uses the fixed-draw
  VAL loss (P:554).
- OURS-N2: sample SEL and concat as R:177 says, although nothing in S consumes it.
- +G diagnostic (a): run it only when +G picks a different checkpoint (S:765-768).

## Build (laptop)

- [x] B0. **Designation record.** From `train_report`, before any score:
  - MCF-R Stage A uses each seed's lowest *logged* valid/loss: s0 ep1471, s1 ep715, s2 ep1067.
  - MCF-A lowest per seed: s0 ep1155, s1 ep1070, s2 ep1534. For s0 and s2, that is the `last` twin.
  - Stage B / Stage 1 candidates: per decision 1.
  - Fixed-draw val loss for each candidate.
  - Commit a numeric copy under `results/n3/` and push it.
- [x] B1. `results/n3/private/select/degrade.json` = `{"items": [], "amendment": "A3 item 7"}`.
- [x] B2. S NaN fix plus test (decision 2), with the dated clarification line in the protocol. Push it.
- [x] B3. Run `n3_select.py --write-configs` on the laptop. It writes the iii-u, iii-p, p1-ours and p1-iii-p
  configs, which then get copied to the box.
- [x] B4. **New `scripts/n3_step5b.py`**, the largest item. It writes the 8 MCF configs that nothing writes
  today: `mcfR_A/B`, `mcfRG_A/B`, `iiS`, `mcfA_1/2/3`. They follow exactly the shape S:409-574 validates. It
  also reads each select record to make the next round's commands; for example, Stage B runs at Stage A's
  chosen s_uR.
  - Resumable: skip any stage whose outputs exist and match their recorded SHA, and never re-sample an
    existing bundle (P:1018 resumes step 5 with the same seeds).
  - Every bundle path is repo-relative.
  - Pass `--workers $W` everywhere; the defaults are 1-4, and press at 1 worker would take about a day.
  - Unit tests feed synthetic records through S's own validators.
- [x] B5. Inner job scripts `step5_V{R,A,C,S}.sh` (private `anvil_jobs/`, in the step5_*.sh pattern). Each
  logs its own sha256 and n3_step5b.py's sha256, uses `$PY_MCF` for the `n3_mcf_*` scripts and `$PY_SYMMC` for
  everything else, and touches a STEP5_V*_DONE marker.
- [x] B6. The N3 suite and the new tests pass locally.
- [x] B7. Commit and push (explicit paths). Upload n3_step5b.py, the configs and the inner scripts to the box.
  None of them holds CSD content.
- [x] B8. Dry run on the login node, inside the image: the prelude passes, and each job prints its full plan
  without touching the GPU.

## Jobs: 4 in parallel on `gpu` (1 GPU + 32 cores each), one queue wait

CPU work runs inside these GPU jobs, as the approved projection assumed. Anvil CPU SU stay reserved for STS,
and `anvil_step5_cpu.sbatch` is debug-only (2 h).

| Job | Contents (in order) | MCF/iv-P calls | Est. | Limit |
|---|---|---|---|---|
| V-R | MCF-R Stage A, 5 s_uR x 3 seeds → to_bundle → +G gauge → match → select `mcfR_A`, `mcfRG_A` → Stage B at the chosen s_uR (+6 more if +G picks a different s_uR) → gauge → match → select `mcfR_B`, `mcfRG_B` → ii-S → match → select `iiS` → diag (a) for the selected R (+G) checkpoints at their frozen s_uR, plus the Thurlemann provenance repeat | 15 + 6 (+6) | 6-8 h | 16 h |
| V-A | MCF-A Stage 1 at (9,3) → match primary/mcf10/mcf08 → select `mcfA_1` → Stage 2: the remaining 8 grid cells x 3 seeds → match → select `mcfA_2` → select `mcfA_3` (rerank only) → diag (a) x3 | 9 + 24 | 6-8 h | 16 h |
| V-C | iv-P sample x6 → match → select `ivP` → `select ivs` → ivs x3 → match → select `ivS`; OURS and OURS-N2 SEL sampling → concat; iii-u haar x3 → iii-p press x3; p1basin; p1-ours press x3; match (P1 against the pressed truth) → select iii-u, iii-p, p1-ours, p1-iii-p | 6 | 3-5 h | 12 h |
| V-S | iii-s VAL sanity row: `screen --set val` → `search --set val` | n/a | ~18 h | 30 h |

- **Counts.** MCF calls total 54 under decision (b), or 60 under (a).
- **Cost.** About 45 GPU-h billed, from wall time x 1 GPU.
- **Budget.** Balance is 1,451 GPU-h. Steps 5-7 run about 75 + ~500 against the 700 GPU-h cap.

## Timeline

- Build and test: 11-06 to 11-07. Submit by 11-08.
- Latest safe submit is about 11-10. A 30-h job must *start* by 11-14 22:00, or Slurm holds it until after
  `Nov16Flush` (11-16 04:00 to 11-23 04:00). Queue waits have run 9-25 h.
- Step 6 (pullback, `n3_select.py --dry-run`, freeze, push) can run on the laptop during the flush.
- Step 7 (~500 GPU-h FF*) runs after 11-23. The resubmission window is open to 2027-01-22.

## Verification

- [x] After pullback, `n3_select.py --dry-run` lists no 5b role as missing. (11-08: 27 records present,
      0 problems, diag (a) PASS, H2 FORM EXACT, 76 step-7 runs.)
- [x] Every VALSEL cell is recorded, losing choices included (R:189-190). (Every cell is in the select
      records and in the preview's `valsel_cell_scores`; the validator recomputes each winner, 0 problems.)
- [x] The diag (a) records carry the frozen s_uR/s_uF (S:808-811). (Checked by the dry run: MCF-R/+G s_uR 5,
      MCF-A s_uF 9 / s_uR 3.)
- [x] Job logs carry both sha256 values, and the prelude verify excused only VALSEL and Gate C hits. (All
      four: inner + driver sha256 lines; STEP5_VERIFY_PASS, unexcused 0.)
- [x] Frank reviews the freeze preview, then `n3_select.py`, then commit + push `results/n3/n3_selection.json`. (11-08, Frank: "yes go ahead"; 772a36e, remote confirmed.)

---

# N3 Gate C fix — 2026-11-02 (Frank: "yes, start on those")

Gate C (21571227, 10-22) failed C-a (i), C-a (ii) and C-b cuda. iv-P (21580267) stopped in the prelude, and M
(21571229) could never start.

- [x] Cancel M 21571229.
- [x] Diagnose C-a. `ca_match` gated (i) and (ii) on n3_score's own reading, against A2 item 7.
- [x] Re-score locally from the box predictions (SHA-checked):
  - --mcf-reading vs MCF's scorer is 100.00% on TEST and VAL, so (i) passes;
  - MCF-scorer medians are TEST 29.73 / 66.53 and VAL 30.22 / 65.85, so (ii) passes.
- [x] `n3_score.py gate-ca` writes the `mcf_scorer` block. `n3_mcf_gateC.py ca_match` runs both readings.
  Re-summarised and the record keys checked.
- [x] Prelude: Gate C run dirs are excused only with the released pickle SHAs, and the overlap is ≤ 1%. A CPU
  sbatch and `step5_GCa.sh` are written (private anvil_jobs).
- [x] Draft A4 in `tasks/n3_amendment_A4_DRAFT.md`. Only the commit hash is left.
- [x] Frank OK'd A4 and chose C-b option (b) (2026-11-02).
- [x] Option (b) as first worded (max of 5 noise draws) would still fail a perfect patch most of the time.
  Replaced by `cb_envelope`: 1 patched + 19 unpatched runs, failing only if the patched run is the strict max
  (≤ 5% false failure). `tests/test_n3_gateC_envelope.py` has 7 tests passing, and the full N3 suite passes.
  `step5_GCb.sh` is written.
- [x] Frank OK'd the corrected (b) wording and re-queuing M behind C-b (2026-11-02). A4 is appended to protocol §9.
- [x] Commit and push the code with A4 → re-pack the step-5 tar → upload (Frank in manual mode) → copy the
  ops files to `$SCRATCH/n3_ops`.
- [x] Box re-score: `sbatch -J n3-gateCa -t 01:30:00 anvil_step5_cpu.sbatch step5_GCa.sh`. It must reproduce
  the laptop numbers.
- [x] Run C-b per Frank's option, then resubmit iv-P. Anvil is down for MAINT on all nodes 11-16 → 11-23.

---

# ACS Omega continuation — 2026-10-11

Decision on record: ao-2026-09173w rejected 2026-09-24, resubmission permitted through 2027-01-22.
Resume the frozen N3 v3 protocol plus A1/A2 from `tasks/n3_next_steps.md`; G0–G2 and the input-side TEST-B work are complete.

Scope for this continuation: finish the unreviewed round-2 implementation, validate its protocol and blinding boundaries locally, and publish the completed phase. No benchmark claims change without results. Preserve existing manuscript/workshop edits. Any paid timing run follows a concrete cost and upload proposal; the full benchmark requires the measured projection specified in the handoff.

- [x] Review/fix VALSEL selection and BASIN scoring against the frozen protocol.
- [x] Finish per-set cache loading and prove held-out caches cannot load on the step-5 box.
- [x] Review/fix upload manifests, environment setup, timing/cost gate, and verified cleanup.
- [x] Write an executable runbook for steps 4–9, with exact prerequisites and review gates.
- [x] Run meaningful local checks and independent protocol review; resolve blockers.
- [x] Record evidence and remaining paid-run prerequisites; commit explicit paths and push this phase.

### Plan check-in

Continuing at round 2, rather than repeating completed dataset parsing or G2 studies. Selection uses VALSEL only; TEST-B/DEV-TEST arm outputs remain blinded until step 9. Verify each tool with synthetic/TRAIN/VAL inputs and cache identity checks; preserve the frozen protocol except for explicitly dated implementation amendments if needed.

### Review

The implementation and verification are complete: 137 tests passed, 3 heavy checks skipped; actual
input-cache checks and checksummed TRAIN/VAL-only archive extraction passed. Independent review
blockers are resolved. Code commit `783a1ab` was pushed to `origin/paper-draft` and its remote SHA
verified. Evidence is retained in `tasks/n3_round2_review_2026-10-11.md`. The next step is the
timing-only spend/TRAIN-VAL upload approval; no paid run or CSD upload has occurred.

---

# Digital Discovery revision roadmap — execution log (2026-06-21)

Target: **Digital Discovery (RSC)**. Items from the full 5-reviewer panel
(`paper/peer_review.md`); decision Major Revision. Tracks the Tier 1–3 roadmap.

## Tier 1 (gating)
- [x] **T1.1** Scope unlearnability (marginal≠conditional; under-featurized vs in-principle).
- [x] **T1.2** R_asym **conditional probe** (`scripts/probe_rasym_conditional.py`). 3-seed:
  probe 125.2° vs constant 122.4° vs Haar 123.5° → −2.4%, no conditional signal. In SI §S3.
- [~] **T1.3** All-atom + random-prior **de-novo baseline** (`scripts/baseline_allatom_denovo.py`).
  Random-prior floor 0% confirmed; trained all-atom run **OOM'd on a leak** → re-run pending.
- [x] **T1.4** **match@1** alongside best-of-k. De-novo match@1 = 0% (base & big). Orient-isolated
  match@1 pending the criterion re-run.
- [~] **T1.5** **Species-grouped split ×3** (`scripts/diag_orient_relative_grouped.py`;
  union-find on shared species — refcode grouping needs CSD API). Smoke-passed; **OOM'd** → re-run.

## Tier 2
- [~] **T2.6** Tolerance sweep + matched RMSD (`eval_orient_matchrate.py --sweep`). First run used
  lenient `get_rms_dist`; **fixed to `fit()`** → re-run pending.
- [x] **T2.7** Scope statement (rigid/CHNO; general-position; special-position exception).
- [x] **T2.8** Classical-CSP lineage paragraph.
- [x] **T2.9** Capacity → "capacity + training" (verified).

## Tier 3
- [ ] **T3.10** Two-head PoC — optional; decide after Tier 1–2.

## Results (`paper/gpu_results/revision/`)
- probe_s0/1/2: probe never beats constant (−1.1/−3.5/−2.5%).
- denovo_base: match@1 0%, match@20 0%, comps 0.46/0.345/57.3°.
- denovo_big:  match@1 0%, match@20 0%, comps 0.41/0.351/50.3°.
- orient_sweep: lenient criterion (discarded); grouped/baseline: OOM (leak) → pending.

## Blocker — GPU memory leak
Orphaned process (PID 26140) held ~24.7 GiB after the de-novo Pool eval, OOM-ing the grouped
splits + all-atom baseline. `kill -9` was blocked by the safety classifier (shared box; task
was the github push). **Needs user authorization** to kill the orphan, then re-run grouped ×3
+ baseline + fit-based sweep. Recurrence fix: matcher `Pool` → **spawn** context (no inherited
CUDA context in forked workers).

---

# Task: SymMC-Flow reference implementation

## Plan
- [x] Scaffold project (dirs, meta files, archive old todo)
- [x] Write PLAN.md design doc
- [x] `config.py` — dataclass configs
- [x] `periodic_table.py` — Z→(r,c) table + 2D embedding
- [x] `manifolds.py` — T³ / SO(3) / lattice exp·log·geodesic·prior·dist
- [x] `egnn.py` — equivariant GNN invariant encoder
- [x] `pair_bias_attention.py` — Clari-style attention + pair bias
- [x] `space_group.py` — symmetry ops + SGFM group averaging
- [x] `flow.py` — conditional paths + CFM loss
- [x] `model.py` — combined SymMCFlow module
- [x] `sampler.py` — RK4 ODE integrator (50 steps)
- [x] `data.py` — synthetic molecular-crystal dataset + real-data hooks
- [x] `train.py` — CUDA-aware training loop
- [x] tests for every module — `pytest -q` green (28 passed)
- [x] `scripts/train_demo.py`, `scripts/sample_demo.py` run on CPU
- [x] README.md + LICENSE
- [x] Init git, create GitHub repo, push
- [x] GPU phase setup: vast.ai RTX 5090, torch cu128, repo cloned, 28 tests pass on GPU
- [x] GPU phase: carbon-24 loader (pymatgen) + real training run + OT-coupling fix
- [x] Documented carbon-24 results in RESULTS.md (centroid fixed; lattice is next)
- [ ] GPU phase next: lattice reparametrization (log-vol + N^(1/3)) + SUN/match eval
- [ ] GPU phase later: MP-20 loader, full SUN benchmark vs diffusion baselines

# Task: Lattice reparametrization (log-volume + shape) + StructureMatcher eval

## Plan
- [x] `manifolds.py`: `lattice_to_param(L, n)` / `param_to_lattice(k, n)` — k = (per-atom
      log-volume, det-normalized 3x3 shape) in R^10; `prior_lattice_param` with
      configurable per-atom volume (V ∝ N built in)
- [x] `flow.py`: lattice path/velocity in param space (u_L: (B,10)); `sample_prior`
      draws the lattice prior in param space using n = mask.sum()
- [x] `model.py`: condition tokens on current lattice (k, 10-d) — before, forward()
      ignored the lattice entirely, so the field couldn't depend on lattice state;
      `head_lattice` outputs 10
- [x] `sampler.py`: integrate k, decode to 3x3 for each field evaluation
- [x] configs/train: replace `lattice_prior_scale` with `prior_vol_per_atom`
- [x] `train_carbon24.py`: StructureMatcher match-rate eval (CDVAE tolerances)
      alongside the C–C distance proxy; vol/atom reported gen vs ref
- [x] tests: param roundtrip, volume∝N scaling, prior validity; update shape asserts
- [x] full `pytest -q` green on CPU (32 passed, 1 data-dependent skip)
- [x] CPU demos verified: train_demo 76% loss drop, sample_demo valid det>0 cells
- [x] retrain on new vast.ai box (142.171.48.138:44563); RESULTS.md updated —
      vol/atom now 6.57 vs 6.30 ref (volume problem solved); C–C still 1.05 Å,
      match rate 0% → next bottleneck is the coordinate field (lattice-aware
      pair features / Fourier features)

## Review (2026-06-11)
- Decode always renormalizes det(shape)=1 and yields det(L)>0 by construction —
  the old "det>0 frac" metric is now trivially 100%.
- Found + fixed a latent bug: the network never consumed the lattice state at all
  (forward() ignored its `lattice` arg), so the lattice ODE field was not a
  function of lattice state. Added `lattice_in` (10-d param) token conditioning.
- GPU retrain DONE: volume problem solved (gen 6.5 vs ref 6.3 Å³/atom).

# Task: Fix position-flow under-dispersion (carbon-24 match rate still 0%)

## Findings (no code beyond diagnostics this round)
- [x] Lattice-aware pair features (Cartesian Δfrac@L + Fourier) — committed 99c4c98,
      verified active (pair_dim=20). NULL RESULT: centroid loss stays 0.09, C–C 1.05 Å,
      match 0%. The loss is at its OT-coupled irreducible floor — features can't help.
- [x] Eval sanity-checked: ref-vs-ref match 100%, ref-vs-shuffled 4.7% → 0% is real.
- [x] Root cause: mean-field under-dispersion. Generated NN dist median 1.11 Å (ref
      1.45), whole distribution contracted, 9% have <0.9 Å overlaps.

## Plan (all three behind config flags, run as an ablation ladder for attribution)
- [x] Lever 1 — stochastic sampler (`sampler_churn`). Implemented + tested.
- [x] Lever 2 — wrapped-normal centroid prior (`centroid_prior_std`). Implemented.
- [x] Lever 3 — fixed per-structure prior coupling (`fixed_prior` + PriorCache, idx
      plumbed into batches). Implemented + tested.
- [x] Tests: std-prior validity, churn dispersion+validity, fixed-prior determinism;
      34 pytest green on CPU + smoke through every training path.
- [x] GPU ablation ladder run (A churn / B fixed_prior / C1 prior / C2 all-three).
- [x] WINNER: lever 2 (wrapped-normal prior std=0.25). valid bonds 17%→84%, overlaps
      11%→2%, vol exactly on ref, match 0%→4.7%. Churn null; fixed_prior overfits;
      all-three worse than prior alone. Baked `--centroid-prior-std 0.25` as default.
      Best ckpt: checkpoints/carbon24_wn.pt. RESULTS.md updated.

## Review (2026-06-11, position-flow ablation)
- "Do all three" → ablation showed they are NOT additive: the concentrated prior alone
  is the fix; stochastic sampling does nothing; sharper coupling lowers the loss floor
  but overfits and even degrades the combination. Attribution was the whole value.
- Remaining: absolute match rate ~5% on a hard CSP benchmark. Next levers are
  capacity/steps/std-tuning, NOT collapse — see RESULTS "Next steps".

# Task: MP-20 loader + flow benchmark (multi-element CSP) — IN PROGRESS

## Plan
- [x] `symmc_flow/mp20.py`: MP20Dataset reads per-atom Z (carbon hardcoded C); same
      batch schema so model/training unchanged. Single-atom blocks, λ_orient=0.
- [x] `scripts/train_mp20.py`: flow objective (reuses train()) + match@k eval that
      builds StructureMatcher inputs from the REAL per-atom species (not "C").
- [x] `tests/test_mp20.py`: per-atom Z preserved, Z-aware NaCl identity match=100%,
      collate roundtrip. 3 pass on box (full suite green).
- [x] MP-20 CSVs downloaded to box (train ~27k, val/test ~9k).
- [x] train flow on MP-20 (30k steps, d_model 256/8-layer), eval match@1 + match@20.
      RESULT (256 val): match@1 26.2%, match@20 59.8%, 0% overlaps, vol on ref.
      ~7x carbon's match@1 (composition conditions strongly). match@20 beats DiffCSP's
      ~51% match@1; our match@1 ~half DiffCSP (headroom = more train/capacity, not collapse).
- [x] RESULTS.md MP-20 numbers vs CDVAE/DiffCSP. ckpt checkpoints/mp20.pt on box.

## Added RMSE metric (2026-06-13) + matcher discrepancy to resolve
- [x] match functions now also return mean matched RMSD (RMSE) — the 2nd standard CSP
      metric (DiffCSP reports match-rate AND RMSE). Wired into carbon/MP-20/diffusion,
      tests updated. MP-20 RMSE@20 = 0.1506 (DiffCSP ~0.06).
- [ ] RESOLVE NEXT SESSION: switching match test sm.fit -> get_rms_dist changed MP-20
      match@20 59.8% -> 80.5% (~20pp). Lock one matcher (get_rms_dist, break_on_match=False
      is more thorough) and re-run match@1/@20 for carbon24_big + mp20 to get canonical
      numbers. Did NOT overwrite the 59.8% headline until reconciled.
- [ ] carbon-24 RMSE eval not yet run (carbon24_big.pt) — quick eval-only next session.

# Task: Reconcile matcher + canonical match@1/@20/RMSE (2026-06-13, new box)

## Plan
- [x] New box 192.3.91.246:26436 (RTX 5090). Old box recycled; carbon24_big.pt restored
      from local copy; mp20.pt gone -> retrain.
- [x] Matcher already locked: both scripts use sm.get_rms_dist (pymatgen internally uses
      break_on_match=False -> thorough). Confirmed identical in carbon + mp20.
- [x] Added --seed to both train scripts (prior draws are the only RNG in eval).
- [x] Box setup (clone + torch cu128 + deps + carbon24/mp20 CSVs). Full test suite ran clean.
- [x] Retrain mp20.pt (30k steps, d_model 256/8-layer). Backed up to LOCAL repo (recycle-safe).
- [x] Canonical eval (seeds 0/1/2): carbon24_big + mp20, match@1 + match@20 + RMSE.
- [x] Reconcile -> overwrote RESULTS.md + README with get_rms_dist canonical numbers. Memory updated.

# Task: Close MP-20 match@1 gap to DiffCSP — capacity scaling (2026-06-13)

## Diagnosis
mp20 (d_model 256/8L/30k) val loss plateaus after ~20k (0.033->0.029) while TRAIN loss keeps
dropping -> capacity-bound, not steps-bound. (Carbon was conditioning-bound; MP-20 is not.)

## Plan
- [x] Train mp20_big: d_model 384, 8 layers, egnn_hidden 128, 40k steps, batch 256, seed 0.
      (~4.4h; box was degraded/bursty — same noisy-neighbor cause as the SSH flakiness.)
- [x] Canonical eval (get_rms_dist): match@1 43.4% (41.8–44.5, seeds 0/1/2), match@20 80.9%,
      RMSE@1 ~0.19, RMSE@20 0.141. Baseline was 41.4% / 80.5% / 0.20 / 0.145.
- [x] scp mp20_big.pt local; RESULTS/README/memory updated.

## Review (2026-06-13/14) — capacity scaling = diminishing returns
- ~3× compute (1.5× width × 1.33× steps) bought ~+2 pp match@1, flat match@20. Wider model's
  val loss tracked baseline's at every step (~0.0286 vs 0.0288 floor) -> near capacity/objective
  floor, like carbon-24. Scaling will NOT close the ~8 pp gap to DiffCSP (51%). Conclusion: the
  remaining gap is the MODELLING APPROACH (OT-coupling depth, post-hoc relaxation), not size.
  Next lever is method, not scale.

# Task: SUN metric (Stable / Unique / Novel) (2026-06-13)

## Plan
- [x] symmc_flow/sun.py: validity + within-formula unique + novel (vs train). 5 tests green.
- [x] symmc_flow/stability.py: CHGNet relax + CHGNet-consistent MP-chemsys convex hull ->
      E_above_hull (one energy model for candidate + hull anchors; sidesteps GGA/GGA+U
      DFT-correction mismatch). MP key via env, not committed.
- [x] scripts/eval_sun.py: generate per val composition, U/N/validity (+ --stability for S+SUN).
- [x] U/N/validity on mp20.pt (256, seed 0): valid 92.2%, unique 100%, novel 98.0%,
      valid∩unique∩novel 90.2%.
- [x] Full SUN with stability (CHGNet+MP, 252/256 scored, ~81min): stable 58.6%, **SUN 56.6%**,
      median E_hull 0.074 eV/atom. RESULTS.md SUN section + README + memory updated, pushed.

## Review (2026-06-13) — SUN
- mp20 SUN (256, seed 0): valid 92.2%, unique 100%, novel 98.0%, stable 58.6%, **SUN 56.6%**.
- Stability = CHGNet-relative E_above_hull on a CHGNet-energy MP-chemsys hull (one energy model
  for candidate + anchors -> no GGA/GGA+U correction mismatch). Caveat: this is COMPOSITION-
  CONDITIONED SUN (eval on known val compositions), NOT unconditional DNG-SUN — not directly
  comparable to CDVAE/FlowMM. Next: unconditional SUN (sample compositions) for that comparison.
- chgnet + mp-api installed on box; MP key passed via env (never committed).
- UNCONDITIONAL SUN DONE (--from-train, sample comps from train marginal; commit pending):
  valid 93.8%, unique 100%, novel 72.7%, stable 61.3%, **SUN 35.2%** (255/256, median E_hull
  0.071). Only novelty moves vs conditional (98%->73%) = honest de-novo signal (regenerates
  known structure ~27%). Caveats (NOT comparable to CDVAE/FlowMM DNG-SUN, optimistic): comps
  from train marginal (no novel-composition test), unique~1 is small-sample, CHGNet-relative
  metastability. RESULTS.md now has a 2-column conditional-vs-unconditional table.

## Review (2026-06-13) — matcher reconciliation task
- **Root cause:** "matcher discrepancy" was not a bug — `StructureMatcher.fit` (break_on_match=
  True, early-exit) vs `get_rms_dist` (break_on_match=False, exhaustive). VERIFIED via DiffCSP's
  official compute_metrics.py that the CSP-standard matcher is `get_rms_dist` (match = non-None,
  ltol .3/stol .5/angle_tol 10) — exactly what our code already used. Old headlines just predated it.
- **Canonical (get_rms_dist, 256 val, seeds 0/1/2):**
  - carbon24_big: match@1 **26.7%** (26.2–27.3), match@20 **79.7%**, RMSE@20 **0.350**.
  - mp20: match@1 **41.4%** (40.6–42.2), match@20 **80.5%**, RMSE@1 ~0.20, RMSE@20 **0.145**.
- **Impact:** the "~5% carbon plateau" was a DOUBLE artifact (match@1-vs-k + non-standard fit).
  Carbon match@1 26.7% > DiffCSP ~17%. MP-20 41.4% vs DiffCSP ~51% (now apples-to-apples = ~81%,
  not the old "half"). No capability ceiling; remaining gap = train/capacity + post-hoc relaxation.
- **Shipped:** --seed on both train scripts; scripts/diag_matcher.py (fit-vs-rms reconciliation);
  RESULTS.md + README corrected; mp20.pt + carbon24_big.pt backed up locally.
- **Lesson captured:** checkpoints are not committed (too large) — scp every new ckpt to the local
  repo immediately; the old box recycle lost mp20.pt + carbon24_diff.pt (carbon24_big survived only
  via a stale local copy).
- **Deferred:** diffusion row (carbon24_diff) get_rms_dist re-eval — ckpt lost with old box;
  relative flow≫diffusion conclusion unaffected. DiffCSP's own match@20; SUN metric.

## Note — diffusion follow-up deferred (not a quick fix)
A FAIR diffusion baseline can't be a knob-turn: VE diffusion needs a uniform prior at
t=T to be sampleable (can't init near f0 without knowing f0), so the flow's "concentrate
the prior" trick has no drop-in analog. Making it fair needs a bridge/stochastic-
interpolant reformulation WITH OT coupling (≈ flow-matching-with-noise) — real research,
uncertain payoff. Chose MP-20 (concrete, standard benchmark) over it for the warm box.

# Task: DiffCSP-style diffusion baseline (objective head-to-head vs flow)

## Plan
- [x] `diffusion.py`: DiffCSP two-process diffusion REUSING the SymMCFlow net unchanged
      (VP-DDPM lattice + wrapped-normal VE fractional score). Sampler: lattice DDIM
      (x0-clamped) + fractional predictor-corrector (ancestral SMLD + Langevin).
- [x] `tests/test_diffusion.py` (6 tests): wrapped-normal score correctness, q_sample
      validity, learnable toy fit, valid samples. Full suite 42 pass.
- [x] `scripts/train_carbon24_diffusion.py`: same data + same match@k eval as the flow.
- [x] trained on RTX 5090 (same config as flow: d_model 256, 8 layers, 15k steps).
- [x] RESULTS.md flow-vs-diffusion table.

## Review (2026-06-13) — NEGATIVE RESULT, well-diagnosed
- Flow wins decisively: match@1 3.9% vs ~1.6%, match@20 35.5% vs 14.1%, valid bonds
  82% vs 26–30%, overlaps 0% vs 10–13%.
- The diffusion COORDINATE field never learned: predict-zero DSM loss 2.538 == trained
  frac loss 2.54 (diag_diffusion_floor.py). Lattice head trained fine (vol on ref).
- Root cause = same under-dispersion the flow had: carbon frac marginal ~uniform, plain
  DSM has ~no learnable coordinate signal. Flow fixed it with OT coupling + concentrated
  prior; vanilla diffusion has neither. Langevin corrector lifted match@1 0%→1.6% but
  can't create unlearned signal. Not a code bug (unit-tested; lattice side works).
- STOPPED per "objective at its floor → change approach, don't scale": a fair diffusion
  baseline needs the structural levers ported into the noising (DiffCSP's full design).

# Task: Push carbon-24 match rate (std sweep + sampler steps + scale) — DONE

## Plan
- [x] Phase 1: centroid_prior_std sweep {0.15–0.35} @ 6k steps. 64-struct peak at
      0.30 (7.8%) was small-sample noise.
- [x] Phase 2: re-eval on 256 structures — std 0.25 ≈ 0.30 (both 3.1%); sampler
      steps 50/100/200 → no gain (ODE converged by 50). Added --eval-n.
- [x] Phase 3: scaled run (d_model 256, 8 layers, 15k steps) → 4.7%, ~82% valid.
      Scaling did NOT improve match rate. Exposed model size as CLI args.
- [x] RESULTS.md updated with full sweep + scaling. Default std 0.30. Best ckpt
      carbon24_big.pt.

## Conclusion
- Collapse SOLVED (valid bonds 17%→~82%, overlaps→2%, vol on ref). Match rate
  plateaus ~5%, NOT limited by prior std / sampler steps / model capacity (all
  ruled out). Gap is the modelling approach: conditions only on (n, space group),
  so generates *a* polymorph not *the* reference.

## Open next (architectural — not more of the same)
- [x] Best-of-k match metric (standard CSP eval) — one-to-one understates us.
      `--match-k K` in train_carbon24.py: draws K gens/ref, hit if any matches.
      Implemented + unit-tested on CPU (tests/test_match_topk.py). RESULT on fresh
      RTX 5090 (carbon24_big, 15k steps, 256 val): match@1 3.9% -> match@20 35.5%
      (~9x). The ~5% "plateau" was a match@1 metric artifact, not a capability cap.
- [ ] DiffCSP-style diffusion baseline; head-to-head vs the flow objective
- [ ] MP-20 loader, full SUN + match-rate benchmark vs CDVAE/DiffCSP

## Review
- **Completed:** 2026-06-10
- **What worked:**
  - All 28 unit tests green. Train demo: total CFM loss drops ~80% (8.0 → 1.5)
    with all three manifold heads (lattice/centroid/orient) learning.
  - Sample demo: 50-step manifold RK4 yields valid lattices (~5 Å near-cubic),
    wrapped centroids in [0,1), orientations with det≈1, cell volumes ~100–145 Å³.
  - SO(3) log made numerically robust via Shepperd's quaternion method (stable at θ=π).
- **What changed from plan:**
  - Synthetic orientation/centroid targets were initially pure noise → orientation
    loss was irreducible. Re-tied target pose to molecular composition so the
    conditional flow is genuinely learnable (lesson captured).
- **Known limitations:**
  - CPU reference impl on synthetic data; no real MP-20/carbon/WBM training yet.
  - Space-group op table is a curated subset (P1, P-1, P2, Pm, P222, Pmm2);
    GPU phase pulls full ops from pymatgen.
  - SGFM averaging is implemented for the centroid field; lattice/orient use the
    plain field (full joint averaging is a GPU-phase extension).

# Task: Journal hardening — relaxed match + strict DiffCSP head-to-head + mol-crystal (2026-06-16)

Target: npj Computational Materials (fallback Digital Discovery). Addresses the two reviewer-facing
weaknesses (apples-to-oranges DiffCSP baseline; loose matched cells / no relaxation) and the central
thesis gap (orientation off on both real benchmarks).

## P0 — done in repo (CPU), GPU runs pending
- [x] `--relax` / `--relax-steps` on train_carbon24.py + train_mp20.py: CHGNet-relaxes GENERATED
      structures (refs never relaxed) and prints a relaxed match rate + RMSE as a SEPARATE line,
      never replacing the canonical unrelaxed numbers. Reuses symmc_flow.stability.relax_structures
      (lazy chgnet import). Added `relax=`/`relax_steps=` kwargs to match_rate/match_rate_topk.
- [x] tests/test_match_topk.py: identity-relax monkeypatch test (relaxed score == unrelaxed when
      relaxation is identity; relax hook invoked). Skips if no pymatgen; no chgnet import. py_compile OK.
- [x] scripts/diffcsp_headtohead.md: runbook to run DiffCSP's released models at num_evals=20 and
      report match@1/@20 + RMSE with the same get_rms_dist matcher (no repo code; GPU-box task).
- [ ] GPU: scp carbon24_big.pt + mp20.pt to box; `pip install chgnet`; run --relax (seeds 0/1/2)
      on both; add relaxed match@1/@20 + RMSE as NEW columns in RESULTS.md (with carbon-topology caveat).
- [ ] GPU: run scripts/diffcsp_headtohead.md; replace the "~51% (match@1)"/"~17%" placeholders in
      RESULTS.md/README with DiffCSP's own match@20 from our run of the released checkpoint.

## P0-strategic — molecular-crystal benchmark (orientation ON) — SCOPED, not built
The thesis (rigid-body conformers + SO(3) orientation flow + SGFM) is validated only on synthetic
data; both real benchmarks use single-atom blocks with lambda_orient=0. Gated on data access.
- [ ] Confirm Purdue CCDC/CSD license access (Libraries / CCDC academic program).
- [ ] Choose a CSD-derived molecular-crystal benchmark set (align with MolCrystalFlow's protocol if
      available) so numbers are comparable.
- [ ] Then design symmc_flow/molcrystal.py — analog of mp20.py but multi-atom rigid blocks (PCA
      framing) with orientation ON (lambda_orient>0), exercising the SO(3) flow + SGFM on real data.

## Deferred (real research, uncertain payoff)
- [ ] Fair stochastic-interpolant diffusion baseline (OT-coupled / concentrated coordinate noising).
- [ ] Unconditional de-novo SUN with composition sampling (model can't generate compositions yet).

## Review (2026-06-17) — GPU runs DONE
- Checkpoints carbon24_big.pt + mp20.pt were LOST (not local, boxes recycled) -> RETRAINED on
  vast.ai RTX 5090 (108.240.82.27). Reproduced canonical: carbon match@20 78.9% (rec 79.7),
  mp20 match@20 82.4% (rec 80.5). Both scp'd to LOCAL repo checkpoints/ (recycle-safe this time).
- **Relaxed match@1 (mean seeds 0/1/2, --relax):** MP-20 41.8->44.7% (HELPS, RMSE 0.21->0.16);
  carbon 29.4->21.2% (HURTS - CHGNet drifts allotropes to graphite/diamond, RMSE 0.41->0.33).
  Consistent across all 3 seeds. RESULTS.md "Post-hoc relaxation" section added.
- **DiffCSP head-to-head DONE** (2x RTX 3090 box, torch-1.9 env, released ckpts, num_evals=20,
  256 test, get_rms_dist; recipe in scripts/diffcsp_headtohead.md):
  - carbon: DiffCSP m@1 18.8% (<ours 26.7), m@20 89.1% (>ours 79.7).
  - mp20:   DiffCSP m@1 48.0% (>ours 41.4), m@20 76.2% (<ours 80.5).
  - **MIXED - each wins 2 of 4 cells.** DiffCSP RMSE much tighter (0.06-0.21 vs 0.15-0.35).
    Old "match@20 beats DiffCSP match@1" framing WITHDRAWN (metric mismatch). Honest takeaway:
    competitive at ~50x fewer sampling steps, not a uniform win. RESULTS + README corrected.
- Still TODO: molecular-crystal benchmark (orientation ON) gated on CSD access (user reached out).

# Task: molcrystal.py — real rigid-body molecular-crystal loader (lambda_orient>0)

## Plan
- [x] `symmc_flow/molcrystal.py`: `MolCrystalDataset` — molecule detection via
      `StructureGraph.from_local_env_strategy(JmolNN)` + PBC-unwrap BFS over `to_jimage`;
      per-species conformer registry keyed by WL graph hash; Kabsch alignment
      (`manifolds.project_so3`) with automorphism-min mapping (VF2, element node_match);
      rigid-strictness skip gate (`conf_tol`, default 0.3 Å) recorded in `self.skipped`;
      monatomic units -> orient=I/local=0. Same batch dict as carbon24/mp20.
- [x] `rigid_to_frac` / `rigid_to_structure` — the exact inverse factorization, for
      round-trip validation and eval-time molecular StructureMatcher reconstruction
      (the single-atom `_to_structures` in train scripts can't expand molecules).
- [x] Re-export from `data.py` + `__init__.py`.
- [x] `tests/test_molcrystal.py` — 5 tests: round-trip recovers planted atoms (gauge-free,
      frac space), all copies share one conformer + recovered relative rotations match
      planted, monatomic=identity, non-rigid copy skipped, symmetric-molecule (water)
      automorphism guard. All green.
- [x] `scripts/train_molcrystal.py` — smoke train, lambda_orient=1.0, orientation tied to
      centroid (learnable field): orient loss 5.39 -> 3.38 (37% drop). First time the SO(3)
      head trains on real multi-atom rigid blocks.
- [x] full `pytest -q` green (no regression); molcrystal deprecation warning removed.

## Review
- Factorization convention locked: `cart_atom = c@L + R@local`; species share ONE
  reference conformer `local`, copies differ only by `R` — that shared conformer is what
  makes the SO(3) flow learnable (vs single-atom mp20/carbon24 where orient≡I).
- **No model change needed**: `model.encode_molecules` already runs the EGNN over A>1
  `local`; only real multi-atom local/orient had to reach it.
- CSD-independent by design: the loader takes pymatgen `Structure`s, so the full detection
  pipeline is already exercised on genuine Structure objects; CSD only swaps the *corpus*
  (a `detect_fn` hook lets the CSD Python API plug in later). Decision: reject non-rigid
  (flexible) molecules above `conf_tol` to keep the rigid-body benchmark honest.
- Lesson captured (tasks/lessons.md 2026-06-17): synthetic bond-graph tests must separate
  copies (large cell + spread centroids) or JmolNN cross-bonds them; flex (don't snap) a
  copy to exercise the conformer gate; a passing round-trip doesn't prove correctness —
  assert the shared-conformer invariant too.
- Still TODO (on CSD access): curated molecular-crystal benchmark corpus + the actual
  orientation-ON training run and match@k numbers for the paper.

## 2026-06-18 — Real CSD molecular-crystal benchmark (orientation-ON)
Full writeup: see `MOLCRYSTAL.md`.
- [x] CSD access obtained; export (`scripts/csd_export.py`, CCDC interpreter) + COD fallback.
- [x] Factorize real CIFs → `MolCrystalDataset` (`scripts/factorize_cifs.py`); tolerant `read_cif`.
- [x] Orientation-ON training on real corpus (`scripts/train_csd_molcrystal.py`).
- [x] Molecule-intrinsic gauge (`_canonical_frame`) + lr 3e-4 stability fix.
- [x] Scale corpus 250 → 1127 structures.

### Result (robust negative)
- lattice + centroid heads LEARN on real data; **SO(3) orientation head stays at its
  predict-zero floor (5.24)** across both corpus sizes and after the gauge fix.
- Not data sparsity (scaling didn't help). Leading cause: noised flow-time conditioning —
  orientation must be predicted from half-noised lattice+centroids.

### Next (not started — paused for thesis-framing discussion)
1. Two-stage / cleaner conditioning (generate lattice+centroid first, orientation after). [rec]
2. Diagnostic: condition orientation on TRUE lattice+centroids to confirm the cause.
3. Fallback: reframe paper as rigid-body lattice+centroid flow + orientation as open problem.
4. Deprioritized: min-over-symmetry orientation target (helps only symmetric minority).

# Task: Clean-packing diagnostic — does the SO(3) head floor on noised conditioning? (2026-06-18)

Ran the cheap diagnostic (#2 above) BEFORE building two-stage (#1), because a positive result
is literally two-stage's second stage and a negative result rules two-stage out — strictly
cheaper either way.

## Plan
- [x] `TrainConfig.cond_clean_packing` flag (config.py).
- [x] `train._step_loss`/`evaluate`/`train`: when set, feed the field the TRUE (z1)
      lattice+centroid instead of noised z_t; orientation stays noised, target unchanged
      (no leakage). Only the orient loss is meaningful under the flag.
- [x] `scripts/diag_orient_conditioning.py`: clean-packing run on the 1127 CSD corpus, orient
      pre→post on held-out val, CONFIRMED/NULL verdict, saves diag_orient_cleanpack.pt.
- [x] `tests/test_cond_clean_packing.py`: 2 plumbing tests (flag feeds true vs noised packing).
      Full suite green.

## Review (2026-06-18) — NULL, two-stage RULED OUT
- 800 steps, clean packing: lattice 1.20→0.044, centroid 0.355→0.254 (both learn), **orient
  5.35→5.18 (+3.3%) — still at the ~5.2 predict-zero floor; R train oscillates 4.69–5.90,
  no trend.** Conditioning on the TRUE packing is the best case for two-stage's 2nd stage, so
  two-stage cannot help → **RULED OUT.**
- Cause is deeper: the absolute per-molecule SO(3) target (even gauge-fixed) carries no
  learnable signal on this asymmetric, ~1-crystal-per-molecule corpus.
- **Decision:** lead with the honest reframe (rigid-body lattice+centroid flow + orientation
  as a well-diagnosed open problem). One technical lever left before fully committing: change
  the orientation TARGET (restrict to species recurring across crystals → relative target),
  not the conditioning. Full writeup in MOLCRYSTAL.md.

# Task: Relative-orientation target — decompose the floor (2026-06-18)

Ran the remaining lever: change the orientation TARGET (relative, not absolute), not the
conditioning. The absolute target factors R_m = rot(g_m)·R_asym (space-group part + free
asymmetric-unit orientation); re-gauge to cancel R_asym and see if the symmetry part learns.

## Plan
- [x] `molcrystal.relative_gauge_item` / `_species_groups` / `species_multiplicity`: re-gauge
      first-copy-as-reference (orient=I), others R'_m=R_m·R0^{-1}; round-trip preserved; adds
      `is_ref` mask. Test `test_relative_gauge_preserves_roundtrip_and_targets` (suite green).
- [x] `scripts/diag_orient_relative.py`: regauge + keep multi-copy crystals (1095/1127), train,
      report orient loss split ref vs NON-ref (guards trivial predict-identity), `--clean-packing`.
- [x] Ran the full 2×2 (absolute/relative × noised/clean), 800 steps each.

## Review (2026-06-18) — PARTIAL POSITIVE; floor = free asymmetric-unit orientation
- Non-reference (symmetry-determined) held-out orient loss, untrained→trained:
  - absolute: 5.37→5.18 (+3.3%, floor) both noised & clean.
  - **relative: 5.37→3.91 (+27.1%) noised; 5.37→3.94 (+26.7%) clean** — descends + generalizes.
  - Overall orient +34–36%; reference copies +56–61%.
- **Identical noised vs clean** → the relative signal rides on the space group (directly
  conditioned) + coarse centroids; robust to conditioning noise (re-confirms two-stage moot).
- **Conclusion:** the SO(3) flow DOES learn space-group-induced relative orientation; the floor
  is the FREE asymmetric-unit orientation R_asym — gauge-arbitrary, fundamentally unlearnable,
  not a broken flow/conditioning artifact. Orientation is *partially* learnable, precisely
  decomposed. → Reframe paper with this decomposition (see MOLCRYSTAL.md Next steps).

## Review (2026-06-20) — strengthening 2a/2b/2c DONE; reframe shipped
Ran the three follow-ups the partial positive needed, then reframed the docs.
- [x] **2a match metric** (`scripts/eval_orient_matchrate.py`): orientation-isolated
      (true lattice+centroid+conformer, sample only SO(3), best-of-8), rebuild via
      `rigid_to_structure`, StructureMatcher.fit. n=131 val: trained **16.8%** vs 0% floor /
      1.5% naive R=I; oracle 100%. Single deterministic draw collapses to the conditional mean
      (relatives are ≈180°), so best-of-k is the correct generative read.
- [x] **2b capacity/steps** (`diag_orient_relative.py` + `--d-model/--n-attn-layers/--egnn-layers`):
      d_model 192 / 2000 steps → non-ref 5.35→3.56 (**+33.4%**, ref +68%). Real but diminishing;
      plateaus above 0. First "big" config (256/6/5) OOM-swapped the CPU box (19 GB) → killed,
      reran modest. Lesson: this box swaps with parallel/oversized training; run sequentially.
- [x] **2c coset conditioning** (`scripts/diag_orient_coset.py`, `assign_cosets`, model
      `n_cosets`/`coset_embed`): 707-coset per-SG codebook; conditioned non-ref 5.36→**2.65
      (+50.6%)** vs `--no-coset` control 5.37→3.91 (+27.1%, reproduces baseline). → ceiling is
      **inference-limited, not representational**; residual = free R_asym + symmetric-top multimodality.
- [x] Tests: +`test_assign_cosets_*` (loader), +`test_coset_conditioning_optional_and_active`
      (model); fixed `test_cond_clean_packing` for new `coset=` kwarg. Suite **60 passed, 2 skipped**.
- [x] **Reframe (task 1)**: MOLCRYSTAL.md (TL;DR 3-way characterization, 2a/2b/2c subsections,
      components table, next steps, artifacts) + PLAN.md §1 reframe note.

---

# Phase D — Reframe as a METHOD paper (PLAN for sign-off, 2026-07-18)

**Status:** Phase B/C GPU runs DONE, gate PASSED (commit `e035b8d`, gpu_results/). This is the
rewrite PLAN only; **awaiting Frank's sign-off before editing `paper/main.tex`.**

## Venue recommendation (Claude's pick): MoML/MIT short paper (Sep 1) FIRST, then JCIM (archival)
- **Why MoML first:** the result is a focused, positive method+analysis -> ideal workshop short
  paper. MoML/MIT (Sep 1) is ALREADY committed for symmc-flow, is the exact domain audience
  (molecular ML), is non-archival (does not burn a later journal), and Sep 1 is a clean deadline.
- **Why not straight-to-journal (yet):** DD (AE Jung) and TMLR both desk-rejected the *negative*
  version on general-interest triage. A domain-workshop review judges it on molecular-ML merit,
  where SG-conditioning for rigid-body molecular-crystal flows is squarely on-topic. Land the
  workshop result, THEN extend.
- **Free second shot:** same paper co-submits to AI4Mat (NeurIPS ws, Aug 30) -- dual non-archival
  workshops OK per the committed plan.
- **Archival top-venue follow-on (Phase E):** extend to **JCIM (J. Chem. Inf. Model.)** -- strong
  molecular-informatics journal, domain review, less general-interest triage than DD. (DD round-2
  possible but already desk-rejected once; JCIM is the cleaner target.) Journal version wants the
  optional extra runs below.

## Target format
- [ ] **Confirm MoML 2026 template + page limit** (likely ~4 pp + refs, workshop LaTeX style).
- [ ] Rewrite from `paper/main.tex` (currently RSC single-column) into the MoML short format
      (keep a copy; do NOT overwrite the TMLR/RSC package under paper/submissions/).

## The reframe (diagnostic -> method)
- OLD title: "What an SO(3) orientation flow can and cannot learn in molecular-crystal SP"
- NEW title (draft): **"Deployable Symmetry-Coset Conditioning Recovers the Orientation Benefit
  of Space-Group Symmetry in Molecular-Crystal Flow Matching"** (trim as needed)
- OLD thesis (negative): orientation is partly unlearnable (free R_asym floor).
- NEW thesis (positive): the LEARNABLE part is the SG-determined relative rotation; we expose it
  as a **deployable, leak-free coset label** (the generating SG operation, available from a
  template at sampling time) and condition the orientation flow on it -> recover ~2/3 of the
  oracle-codebook benefit; **SO(3)-averaged training closes the rest**; de-novo (no-template)
  prediction is the remaining honest gap.

## Section-by-section rewrite
- [ ] **Title + abstract** -> method+result framing (numbers below).
- [ ] **Intro**: lead with the GAP -- MolCrystalFlow/MOFFlow skip SG conditioning; SG-conditioning
      proven inorganic-only (NextCrystal / WyckoffDiff / DiffCSP++ / SymmCD). State contributions:
      (1) deployable coset label; (2) conditioning recovers the orientation benefit; (3) SO(3)-avg
      reaches the ceiling; (4) honest de-novo gap.
- [ ] **Method** (expand from the diagnostic's decomposition):
      - R_m = rot(g_m) * R_asym decomposition (keep from current paper).
      - Deployable coset = generating SG op h = g_m * g_0^{-1}, recovered from centroids
        (min-image), LEAK-FREE + template-available at sampling (contrast DiffCSP++ atom templates).
      - Coset conditioning in the flow (coset_embed; threaded through sampler, C3a).
      - SO(3)-averaged FM objective (C5).
      - Packing-only coset predictor (C4).
- [ ] **Experiments / Results** (all from `e035b8d`):
      - **Table 1 (main):** held-out NON-REF orientation-loss drop, 3 seeds:
          control (no-coset)      +27.5% (29.4 / 24.4 / 28.8)
          deployable coset K=1    +41.1% (43.2 / 37.2 / 42.9)
          + SO(3)-avg K=4         +47.7% (49.2 / 43.4 / 50.5)
          leaky-codebook oracle   ~+48% (upper bound)
      - **Fig 1:** bar chart of Table 1 with the ~+48% ceiling line.
      - **Orient-isolated physical match** (best-of-8, coset template): up to 14.5%/37.4% at loose
        tol, 9.9%/27.5% at (0.3,0.5), median 41.1 deg; tol-sweep table.
      - **Predictor (C4):** 39.5% top-1 vs 10.3% majority (4x); predicted-coset reconstruction
        collapses to 64.8 deg -> de-novo gap.
      - **Templated end-to-end:** 0.0% match; error budget lattice 0.442 / centroid 0.346 /
        orient 13.4 deg -> scopes the contribution to the ORIENTATION mechanism (lattice+centroid
        are the end-to-end bottleneck, not orientation).
- [ ] **Limitations / Analysis**: de-novo predictor gap; end-to-end bottleneck = lattice+centroid;
      symmetric-top multimodality residual.
- [ ] **Conclusion + future work**: better/top-k coset predictor for template-free use; joint
      lattice-centroid improvement.
- [ ] **Related work**: add MolCrystalFlow (2602.16020), NextCrystal (2602.17176), MOFFlow,
      SO(3)-averaged FM (2507.09785), Genarris-3; keep DiffCSP++/WyckoffDiff/SymmCD.

## Figures / tables to produce
- [ ] Table 1 (main result) -- from logs.
- [ ] Fig 1 bar chart -- matplotlib from logs.
- [ ] Tol-sweep table -- from phaseB/orient_isolated_coset.log.
- [ ] (Optional) schematic of R_asym decomposition + coset conditioning (excalidraw skill; needs
      jsdelivr import fix + dangerouslyDisableSandbox per [[reference_excalidraw_skill]]).

## Claims discipline (do NOT overclaim)
- Coset helps ON THE ORIENTATION RESIDUAL; NOT yet end-to-end match.
- De-novo (no template) does NOT work yet (39.5% too noisy -> 64.8 deg).
- SO(3)-avg is the ceiling-reaching component.

## Optional extra runs (JCIM journal version, NOT the workshop)
- [ ] Unconditioned templated baseline (was skipped: run_phaseB UNCOND fallback missed
      coset_deploy_off_s0.pt at eval time). Rerun eval_templated_matchrate on an OFF ckpt
      (CPU-heavy matcher; short GPU rental or overnight CPU). Cheap; completes the templated delta.
- [ ] Stronger / calibrated coset predictor (bigger, or top-k conditioning) to shrink de-novo gap.
- [ ] End-to-end with improved lattice/centroid generation.

## Repro / data
- Numbers: `e035b8d`, gpu_results/. Checkpoints local (gitignored, 10x ~5.5MB). Zenodo v1.0.3
  DOI 10.5281/zenodo.21384130 (bump on submission).
- Author block per [[reference_author_identity]] (Frank Cai, Purdue, ORCID 0009-0003-0041-1459).

## Open decisions for Frank (sign-off)
1. Venue: MoML-first (my rec) vs straight-to-journal?
2. Title (draft above OK, or shorter)?
3. Include the excalidraw schematic figure?
4. Do the optional unconditioned-templated rerun now, or defer to the journal version?

---

# Task: Finish JCIM manuscript for publication + achemso formatting (2026-07-22)

Finalize `paper/submissions/JCIM/` to ACS J. Chem. Inf. Model. (jcisd8) publication format;
also fix MoML stale title + verify page count. LaTeX installed locally (MiKTeX 25.12).

## JCIM
- [ ] `article` -> ACS `achemso` (journal=jcisd8, manuscript=article); drop natbib/bibstyle (achemso owns it)
- [ ] Remove `\todo` macro + 6 markers; expand each to publication prose:
  - [ ] Intro opener; Methods rigid-body flow; R_asym derivation (Eq + gauge floor + Haar + probe + 2x2);
        orientation-isolated tolerance table; Limitations (from diagnostic Discussion)
- [ ] achemso author block (\affiliation/\email/\keywords); ORCID OUT of body (JCIM rule)
- [ ] Abstract <=250 words; TOC graphic (\begin{tocentry})
- [ ] "Data and Code" -> **Data and Software Availability** (JCIM-required); competing-interest Notes
- [ ] Appendices -> Supporting Information (si.tex, manuscript=suppinfo) + \begin{suppinfo} pointer
- [ ] Self-contained figures/ in package; compile clean (0 undefined refs/cites)

## MoML
- [ ] Fix stale title in comment header; compile -> real page count; trim to <=4 pp if over

## Guidelines confirmed (researcher-resources.acs.org, jcisd8, updated 2026-05-21)
- Abstract 150-250 words; TOC graphic required (Article); Data Availability Statement mandatory;
  ORCID must NOT appear in manuscript text (auto-added on accept); refs any style but complete w/ titles;
  COI: "The authors declare no competing financial interest."

## Review (2026-07-22) — DONE, compiled + verified locally (MiKTeX 25.12)
- **LaTeX installed**: winget MiKTeX.MiKTeX 25.12 (user scope), AutoInstall on; pdflatex/bibtex/latexmk
  at `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`. pdftoppm/pdftotext there too (Read-tool PDF render
  needs poppler, which the Read tool lacks — render via MiKTeX pdftoppm to PNG instead).
- **JCIM** `submissions/JCIM/main.tex`: article -> achemso (journal=jcisd8, manuscript=article). All 6
  `\todo` expanded (intro; rigid-body flow; full R_asym derivation + 2x2 Table; orientation-isolated
  tolerance table; limitations). achemso author block (\affiliation/\email/\keywords), ORCID out of body.
  Abstract 224 words (in range). TOC graphic (fig3_ladder). "Data and Software Availability" +
  competing-interest + \begin{suppinfo}. Extended methods moved to **new `si.tex`** (achemso
  manuscript=suppinfo, 5 SI sections). fig3_ladder.pdf copied into package. **Compiles CLEAN**:
  main.pdf 19 pp (ACS double-spaced), si.pdf 6 pp; 0 undefined refs/cites, 0 errors, 0 overfull>15pt,
  19/19 bibitems resolve.
- **MoML** `submissions/MoML/main.tex`: stale title comment fixed. Trimmed main text 4.4pp -> **exactly
  4 pp** (refs start top of p5; appendix p6) via prose tightening + smaller bar chart + caption/table
  spacing. Compiles clean, 6 pp total.
- **PUSHED** `fff13b3` (JCIM finalize + MoML trim) to origin/paper-draft.

## Follow-up (2026-07-22): TOC graphic + AI4Mat NeurIPS conversion (items 4 & 5)
- [x] **Item 4 — purpose-built JCIM TOC graphic**: `paper/figures/fig_toc.py` → `fig_toc.pdf`
      (decomposition R_m=rot(g_m)R_asym + coset/family-mask conditioning + 0%→6.9%/10.7% ladder vs
      MCF; Okabe-Ito, 3.25×1.75in). Wired into JCIM tocentry (replaces fig3_ladder); recompiled
      clean, 19 pp, TOC renders on last page.
- [x] **Item 5 — AI4Mat → NeurIPS style**: downloaded `neurips_2024.sty` (stable-layout basis);
      rebuilt `AI4Mat/main.tex` = MoML body under `\usepackage[preprint,nonatbib]{neurips_2024}` +
      numbered natbib + hyperref. Compiles clean: 6 pp, 20/20 bibitems, 0 undefined, 0 bad overfull,
      numbered cites + "Preprint. Under review." footer. SUBMISSION_NOTES updated (swap 2026 .sty +
      check double-blind at submit).
- **On flag #2 (tab:bench)**: NOT necessary to change. The bench (we're below MCF end-to-end at
  diagnostic scale) now explicitly sets up §ladder, which closes that same gap to parity — it reads
  as the honest "before" of the before→after arc, not an exposed weakness. Recommend keep as-is.
- **STILL HUMAN-GATED (Frank owns)**: (a) review the prose in JCIM derivation/limitations; (b) Zenodo
  DOI bump; (c) at AI4Mat submit: swap official 2026 .sty + set anonymity per the CFP.

## 2026-09-25: ACS Omega reject + resubmit (window to 2027-01-22)
- Plan + gates in `tasks/omega_rebuild_plan.md`. Measured: centro survival 18.1% vs Sohncke 74.1%
  (det+1 Kabsch drops enantiomeric copies). G0 comparator check in progress.
- 2026-09-28: N3/G3 benchmark protocol pre-registered in `tasks/n3_protocol.md` (v3). Next steps are in
  its §7:
  - A1 build of the fresh TEST-B;
  - local implementation and gates;
  - a timing gate, whose cost projection needs Frank's approval before the full Vast run.
