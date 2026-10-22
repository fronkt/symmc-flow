# N3 §7 step 4 (timing gate) record - 2026-10-11

Approval: Frank approved in chat on 2026-10-11 a **$5 timing-only cap** and the **TRAIN/VAL step-4 upload**
(`n3_step4.tar`, 1,278,146,560 B, SHA-256 `280c00bc…c544`). Nothing beyond step 4 is approved. No SEL, VALSEL,
TEST-B or DEV-TEST input left the laptop. No N3 arm output was sampled or scored on SEL/TEST-B/DEV-TEST.
**Spend: $3.11** (Vast credit 13.22 → 10.11). All rented instances are destroyed.

## Deviations (disclose in A3)

1. **Box A upload was made in auto mode.** Section 7 requires CSD uploads in manual-approval mode. Frank's
   "go" did not switch the session out of auto mode. The classifier allowed the rental and the upload
   (05:57-06:03Z) on his explicit chat approval, then blocked the next steps. Frank then left auto mode, and the
   box-B upload (17:22-17:26Z) was made in manual mode.
2. **Box A abandoned.**
   - Box A was 55345965 (RTX 4090, $0.501/h, CUDA 13.0, cgroup quota 30.7 CPUs).
   - The gate stopped at its first failure (MCF-A inference, item 4 below). The idle watchdog then stopped the box.
     Nine hours later its GPU had been re-rented, so the box could not restart ("resources unavailable").
   - The failure evidence was copied by `vastai copy` to a relay. It was pulled to the laptop and SHA-verified
     (146 files): `results/n3/private/timing_boxA_2026-10-11/`.
   - The relay was cleaned (search clean) and destroyed. Box A was destroyed without running
     `n3_box_cleanup.sh`, which needs a running box. Destroying the instance deleted its disk.
   - All box-A measurements are **discarded**: rates are host-fingerprinted and box B re-measured everything.
3. **Missing C compiler.** The PyTorch runtime image has no gcc. Triton in the FF env builds its CUDA launcher
   with a C compiler, even on the uncompiled `batch` path. On box A every FF call therefore failed
   (`Failed to find C compiler`), including both float64 FD checks.
   - Fix `3099448`: setup installs gcc, and `data` runs a CUDA FF selftest on a toy cell.
   - Box B installed gcc at boot.
4. **MCF-A timing inference: check failure on random-init weights.** On box A, the smoke_init MCF-A model
   diverged on 631/800 draws, and 13/50 crystals had no finite draw. Two checks failed:
   `resid_finite_on_finite_draws` and `resid_recomputes_from_saved_rotmats`. Sampling, RESID and the check
   themselves completed.
   - Fix `3099448`: the gate accepts exactly those two failures for timing and records them in the
     measurement. Every other check failure still fails the measurement.
   - **Open protocol question:** with trained weights, a single finite draw whose RESID forward is non-finite
     would fail `check`, which says "discard and regenerate". A deterministic regeneration would fail again.
     Decide before step 5 how such a draw is treated.
5. **Pullback symlinks.** wandb offline runs leave 11 pointer symlinks (`latest-run`, `debug*.log`,
   `debug-core.log`), and pullback refused them.
   - On box B the 11 links were deleted before pullback. Their list is retained in
     `returns/step4_box_receipts/n3_ops/removed_wandb_symlinks.txt`.
   - Fix `cae84bf`: in-root links are listed instead of packed, and out-of-root links still abort. Tested on
     the box (Linux).
6. **Cleanup false positive.**
   - The first verification FAILED on 3,274 hash hits. Every hit was a zero-byte file such as `/.dockerenv`,
     because one empty file in the inventory matches every empty file on the box.
   - The deletion itself had completed. Fix `cae84bf` skips zero-byte inventory entries.
   - Verify-only re-run: **PASS**. 237,831 files searched, 0 name hits, 0 hash hits, 0 errors. Receipt:
     `cleanup_receipt_20261011T193618368753Z.json`.
7. **Return unpacked to a separate root.** Thirteen pulled files would have overwritten local smoke artifacts,
   for example `n3_mcf/_smoke/ckpt_A/last.ckpt`, which earlier local smoke manifests hash. So the return was
   unpacked to `results/n3/private/returns/step4_box/`. The verified receipt covers 302 files, archive SHA-256
   `848d1e58…aa5b6`.

## Box B (55421479, RTX 4090, $0.501/h, CUDA 13.0, 30.7-CPU quota)

### Setup and checks

- **Setup:** passed: three envs, data check, smoke.
- **P6 equivalence** (MCF, TRAIN, CUDA, no shims):
  - prior bit-identical in 9/9 cases;
  - scatter pooling vs loop: max **relative** error 5.5e-7, but max **absolute** error 6.1e-5 on outputs of
    magnitude ~127 (float32 rounding).
  - It passes a relative 1e-5 tolerance and fails an absolute one. **A3 must state which.**
- **iv-P padded/trimmed/packed equivalence:** max |Δv_R| 8.9e-8 / 0.
- **FF finite-difference check** (float64, h = 1e-4 rad, frozen rule: every relative error ≤ 1e-4):
  - **FAIL** on both settings (`batch` and `default`), with an identical max error of 1.2893e-4.
  - Nine of ten rows are ≤ 5.4e-6. The failing row is VAL #0 (ZEVQOG, K = 4, 136 atoms) at its Haar pose: a
    clash pose, E = −504 eV vs −811 at the truth, |g| ≈ 2.8e3 eV/rad.
  - Laptop CPU float64 step series on that row: h = 1e-3 → 1.25e-2, 1e-4 → 1.29e-4, 1e-5 → 1.29e-6. That is
    pure O(h²) truncation error of the central difference, and the analytic gradient is correct.
  - All 5 crystals × {truth, Haar} at h = 1e-5: **pass**, max 1.29e-6 (`results/n3/private/fd_diag_2026-10-11/`).
  - **A3 must decide:** amend h to 1e-5, or accept on this evidence.

### Timing gate

The gate ran with W = 28, concurrency 1, iv-P cpb {1, 8}. All **23/23 measurements are OK**, and the rates
were published on the box (19:24Z).

| item | measured |
|---|---|
| MCF-R train, s/epoch | 4.84 (P6 off) / **2.00 (P6 on)** |
| MCF-A train, s/epoch | 5.26 / **2.47** |
| MCF infer (16 draws × 50 steps + RESID + check) | c0 15.0 s + 0.445 s/crystal (R); 0.449 (A) |
| iv-P train, s/epoch | trim 8.7 / **pack 5.2** |
| iv-P sample, GPU s/crystal | cpb 1: 0.44; **cpb 8: 0.076** |
| FF `batch` float32, s/structure | single 0.043-0.144; batched-64 0.056-0.376 (**slower per structure than single**) |
| FF `default` float32 (merge_mole + compile) | ~22 s recompile **per crystal**; batched ~0.6 s/structure |
| FF float64 | 5-10× float32 |
| FF* truth minimisation | 6.4 evaluations mean, 20/20 converged (both dtypes) |
| OURS sampling (CPU) | 71.5 CPU-s/crystal |

### Projection

Configuration: P6 on, iv-P pack cpb 8, FF `batch_float32`, tier-2 included, at the GPU box's $0.501/h.
Full output: `results/n3/private/timing/projection_A_gpubox_only.json`.

| category | GPU-h | box CPU-h | $ (alone on box) |
|---|---|---|---|
| training | 7.1 | 0 | 3.5 |
| Gate C | 0.7 | 0 | 0.4 |
| VALSEL sampling | 3.3 | 14.0 | 1.9 |
| TEST-B/DEV-TEST sampling | 1.0 | 41.9 | 1.3 |
| **FF*** | **528.9** | 0 | **265.0** |
| press / LJ / matching | 0 | 166 | 3.2 |
| tier-2 | 1.6 | 1.5 | 0.8 |

- **Step 5:** $19-22. 21.75 of its 32.9 GPU-h is the iii-s VAL sanity row.
- **Step 7:** $294-295.
- **Total:** **$313-318** (542 GPU-h; 624-634 box-h), plus about 99 local CPU-h at step 9.
- **All seven degrade items:** $190-194. iii-s (285 GPU-h) and the FF* ceiling are never reduced.
- The prior was $15-40. It assumed 2-3.5 ms per UMA evaluation; the measured end-to-end batched rate is about
  130 ms at the median cell size.

## Decisions needed from Frank (nothing past step 4 runs without them)

1. **Budget.** The measured plan is about 10× the prior and about 30× the current credit. Options:
   - approve the cost;
   - re-time a faster FF* configuration before A3, then re-project. `batch` turns activation checkpointing on,
     and batched-64 calls lose to single calls, so the measured FF path is likely mis-configured for inference.
     This would need a dated amendment for a new FF settings candidate and one more timing box (~$1-2);
   - stop before step 5 and disclose: protocol §7, "the cap still binds after item 7".
2. **FD check.** Choose h = 1e-5, or accept the O(h²) evidence.
3. **P6 tolerance.** Relative or absolute 1e-5.
4. **The non-finite-RESID-on-finite-draw rule** for trained MCF (deviation 4).

A3 (FF settings/dtype via `n3_classical.py ffconfig`, P6, iv-P encoder, degrade items) is **not** written or
frozen; it depends on decision 1.

## Anvil re-run (in progress, 2026-10-11 20:40Z)

**Frank's decision (2026-10-11):** run N3 on the ACCESS allocation **CHE260157** (Anvil, 150 GPU-h; Frank is PI).
No more Vast spend. Frank left auto mode before the CSD transfer.

- **New FF candidate `merged`** (`6749cbe`): MOLE merge, no torch.compile, no activation checkpointing, graph
  generation v2. The server restarts per crystal, and all iii-s starts of one crystal share one composition.
  It is timed beside `batch` and `default` and FD-checked in float64. The CPU selftest passes.
- **Archive re-packed** with the current code. It holds the same TRAIN/VAL inputs: 189 files, SHA-256
  `2d13322d…`. Its boundary scan is clean: 0 forbidden paths or hashes, 0 TEST-B/DEV-TEST refcode hits.
- **Uploaded** by scp under manual approval at 20:22-20:26Z to `$SCRATCH/n3_upload`, which like
  `$SCRATCH/n3_box` and `$SCRATCH/n3_manifests` is `drwx------`. The checksum verified on Anvil.
- **Unpacked and built on the login node** (`N3_NO_GPU_CHECK=1`). The environments live in `$PROJECT/n3_conda`
  with pinned torch 2.7.1+cu128 / 2.13.0+cu130 / 2.11.0+cu128. System `python3` is 3.6, so every box command
  uses the Miniforge python.
- **GPU job 21289469** (`gpu` partition, 1× A100, 32 cores, 120 GB, 5 h limit) runs check → smoke → data → P6,
  iv-P and FD checks for batch, default and merged → the timing gate → rates, all in one job and therefore on
  one host. Driver probe: 21287865 (gpu-debug). Queue at submission: 389 jobs pending on `gpu`; the probe's ETA
  was Oct 13.
- **Job 21289469 FAILED at `check`, after 4 s** (2026-10-12 10:05Z, node g007; 0.0 GPU-h charged).
  - The node has driver 615.71.09. Its `nvidia-smi` prints no "CUDA Version" banner, so the check read an
    empty version and refused it. The probe had already shown that, since its banner grep printed nothing.
    A 615 driver is far past the CUDA 13.0 minimum (R580).
  - The repo fix falls back to libcuda's `cuDriverGetVersion` and fails with an explicit message when both
    sources are empty. Four Linux tests cover it; they passed on the Anvil login node.
  - The archived box root keeps the old script. `data`-mode manifest verify hashes every archived file, so a
    patched copy would fail it.
  - For this run, `anvil_step4.sbatch` (sha256 `b1918de8…9197`):
    1. prints the banner;
    2. reads the driver CUDA version and libcuda path itself (≥ 13.0 required);
    3. sets `TRITON_LIBCUDA_PATH` if `ldconfig` lacks libcuda;
    4. runs the archived `check` with `N3_NO_GPU_CHECK=1` (disk, python, compiler).
    `smoke` still asserts a finite CUDA matmul in every env.
  - **Resubmitted as job 21318021** (2026-10-12 ~18:50Z). Queue: 315 pending / 36 running on `gpu`. The last
    two jobs waited 9.5 h and 13.5 h.
- **Job 21318021 FAILED at `smoke`, after 1 min 12 s** (started 2026-10-13 00:56Z on g005 by backfill; 0.0
  GPU-h charged).
  - The libcuda check worked: driver CUDA **13.4** (the banner reads "CUDA UMD Version: 13.4").
  - `smoke` then failed in the mcf env. The A2a-pinned PyG binaries, torch-scatter 2.1.2+pt27cu128 and
    torch-cluster 1.6.3+pt27cu128, need **GLIBC_2.32**, and Anvil runs Rocky 8.10 with glibc 2.28.
  - A scan of every `.so` in the three envs found only those 26 PyG files needing glibc above 2.28.
  - **Fix: the job runs in an Ubuntu 22.04.5 Apptainer image** (`docker://buildpack-deps:jammy`, glibc 2.35,
    gcc 11.4). Path: `$PROJECT/n3_images/jammy.sif`, SIF sha256 `386af7e8…7c7b`.
  - The package pins do not change, and neither do the envs or the archived box root. Only the OS userland
    under the conda envs changes, and Vast box B was also an Ubuntu 22.04 image.
  - Login-node tests in the image:
    - all smoke import lists load, with exact A2a versions and a CPU `radius_graph`/scatter;
    - the same test outside the image fails with the GLIBC_2.32 error;
    - the archived `check` passes in the image;
    - `--cleanenv` drops the host Spack `LD_LIBRARY_PATH`.
  - Scripts:
    - `anvil_step4.sbatch` (sha256 `bcbd3924…c1e4`) runs `apptainer exec --nv --cleanenv -B /anvil`.
    - `anvil_step4_inner.sh` (`ad0eaaf0…5758`) is the unchanged body, with a container-only PATH.
  - **Every later N3 Anvil job must run in the same image**, because host identity and the loader must match.
  - Repo: `check` now refuses glibc < 2.32 even on a login node, so an env build on such a host fails
    before any GPU job is queued.
  - **Resubmitted as job 21340353** (2026-10-14 ~00:52Z).
  - The exact job scripts are kept locally, gitignored (they name the cluster account), in
    `results/n3/private/anvil_jobs/`: both sbatch versions, the inner body, the env build and the import test.
- **Job 21340353 TIMEOUT** on g000, 2026-10-14 05:05:55Z to 10:06:21Z (backfill; 5.0 GPU-h charged).
  - In the image, every stage passed: check, smoke, data, and the P6 / iv-P / FD check runs (all rc 0).
  - The FD summaries still read `pass: false` at the frozen h = 1e-4: max relative error 1.289e-4, the same
    O(h²) row as box B.
  - By 08:14Z, 16 of the 24 measurements were done. (This line first said "15 of 23", a miscount: Anvil
    times 24, because `merged` adds two FF measurements.)
  - **`ff_default_float32` took 4,647 s** (box B: 1,277 s). The per-crystal recompile is CPU-bound, with
    about 100-150 s warm-up per crystal; GPU utilisation was 2 %.
  - `ff_default_float64` was projected to end about 09:23Z, which leaves too little of the 5 h limit for
    `merged`, `ff_min`, `cpu`, `lj_screen` and `ours_sample`.
  - `scontrol` refused to extend the time limit (user permission).
  - **Outcome: 17 of 24 measurements have successful records** (first recorded on 10-14 as "16 of 23", a
    miscount). `ff_default_float64` ended 09:20Z after
    5,408 s (box B: 1,883 s; warm-up 103-153 s per crystal, GPU utilisation 2 %).
  - `ff_merged_float32` was then cut off at 19 of 20 crystals. It had about 70-80 s warm-up per crystal and
    no record was written.
  - Not started: `ff_merged_float64`, `ff_min_truth_float32/64`, `cpu`, `lj_screen`, `ours_sample`.
  - **Reset dry run repeated on the final state (20:40Z):** it would move only the `merged` float32 shard
    (17,360 B) and its log. It keeps the four FF records, and the three script hashes are unchanged.
  - At 20:30Z, follow-up 21344282 was PENDING (Priority). All 4 GPUs on g000 are held by other users' jobs,
    and Slurm's estimated start is 2026-10-15 06:06Z.
  - Estimated length of the remainder: about 50 min for `merged` float32, about 1-2 h for float64, and under
    40 min for the short ones (8 min on box B). That fits in the 4 h limit.
- **Follow-up job 21344282** (`anvil_step4b.sbatch`, `-w g000`, `--dependency=afterany:21340353`, 4 h, same image):
  - **Same node, because rates refuse other hosts** and the fingerprint includes the node name. The finished
    measurements are reused, since `run` keeps any record whose fingerprint still matches.
  - **Reset first.** The harnesses resume their shards, so a killed measurement would otherwise be re-timed
    on cached work. `anvil_step4b_reset.py` renames every resumable output of a measurement that has no
    successful record to `*.partial_<job>`, and writes a receipt to `results/n3/private/timing_resume_<job>.json`.
    It stops if any failed record exists.
  - **Order.** It then times the unfinished measurements with `merged` first and `default` last, re-runs
    `run` to confirm every record is kept, and publishes `rates`.
  - The dry run on the live box root listed only the in-progress default-float64 shard and its log.
  - Script hashes:
    - sbatch `54e728c7…081a`
    - inner `b0f7f940…fd18`
    - reset `0b69b066…f1`
- **Platform deviation for A3.** Anvil is a persistent shared filesystem, not a destroyable box.
  - Hygiene becomes an explicit `n3_box_cleanup.sh` over `$SCRATCH/n3_box` and `$SCRATCH/n3_upload`, with
    `N3_FIND_ROOTS` set to `$HOME`, `$SCRATCH` and `$PROJECT`.
  - The cleanup script always adds `/dev/shm`, which is shared on login nodes. Run it in a compute job, or
    expect permission errors there.
- **Job 21344282 COMPLETED** on g000, 2026-10-14 23:21:11Z to 2026-10-15 01:12:04Z (backfill; it started
  about 7 h before Slurm's estimate).
  - The reset moved exactly the two files the dry run listed. `run` timed the 7 unfinished measurements
    (rc 0); the repeat `run` reported all 24 as "done (same configuration, kept)". `rates` returned rc 0.
  - `rates.json` sha256 is `4facd352…890c`.
  - GPU charged for all of step 4 on Anvil: **6.9 of 150 GPU-h**; 143.1 remain.
- **Pullback (2026-10-15 07:49Z, on Anvil):** 289 files, 1,920.6 MB, in `n3_pullback_step4anvil.tar` with
  sha256 `6db1fa12…efcac`.
  - The 11 in-root wandb links were listed, not packed (fix `cae84bf`), to
    `pullback_step4anvil.skipped_symlinks.json`.
  - **Transfer to the laptop has not been done.** It needs manual-approval mode (CSD-derived outputs).
- **Projection (2026-10-15, laptop, before the pullback).**
  - Only `rates.json` was copied, after checking it has no refcode-like tokens and no host paths. The sha256
    matches. `project` accepted its `code_identity` against laptop HEAD.
  - **Rate-model defect.** The published MCF-A inference slope is 0.0. On this host the MCF call is
    overhead-bound: R gives c0 150.1 s and c1 0.4 s/crystal. The A call at n = 50 took 126.3 s, below R's
    intercept, because random-init draws are non-finite and skip most of RESID (37 s vs 74 s). `rates`
    clamps the slope to 0, and `project` then refuses it.
  - For analysis only, a labelled copy sets A's slope to R's 0.4 s (`rates_Aslope_eq_R.json`). Setting it
    to about 0 instead moves the total by 3 GPU-h.
  - A real fix changes `n3_timing_gate.py`, which is inside `code_identity`. That invalidates these rates
    unless A3 records the substitution.
  - Settings: `--usd-per-hour 1 --cores 32` (one A100's share of a 128-core node) `--p6 on --ivp-encoder
    pack --ivp-cpb 8 --tier2`. "$" therefore reads as box-hours, i.e. GPU-h charged with the CPU stages
    run inside the GPU job.
  - Outputs are in `results/n3/private/anvil_projection_2026-10-15/` (gitignored).

  | FF* candidate | FF* GPU-h | total GPU-h | box-h (= GPU-h charged) | all 7 degrade items |
  |---|---|---|---|---|
  | **batch_float32** | **425.6** | **447.5** | **515-535** | **321-335** |
  | batch_float64 | 1,090.6 | 1,112.5 | 1,279-1,299 | 776-789 |
  | merged_float32 | 3,572.7 | 3,594.6 | 4,134-4,154 | 2,540-2,553 |
  | merged_float64 | 5,134.1 | 5,156.0 | 5,929-5,949 | 3,612-3,626 |
  | default_float32 | 6,876.0 | 6,897.8 | 7,933-7,953 | 4,851-4,865 |
  | default_float64 | 9,467.0 | 9,488.9 | 10,912-10,932 | 6,621-6,635 |

  - For batch_float32: step 5 is 42-53 box-h (36.2 GPU-h), and step 7 is 473-482. Step-9 matching adds
    164 CPU-h on the laptop (27 h on 6 cores). With P6 off the total is 536-556.
  - `merged` and `default` pay 70-120 s of per-crystal warm-up on this CPU, which FF* repeats for every
    crystal. Only `batch` is viable here.
  - **No candidate fits the 143.1 GPU-h left**, even with every degrade item. The cheapest is 2.2-2.3× over.
    The box-B figure for the same configuration was 624-634 box-h ($313-318), with FF* at 528.9 GPU-h.
- **Budget route: ACCESS credit exchange (Frank, 2026-10-15).** "check the unexchanged ACCESS credits and
  Anvil GPU rate... exchange them", then "do it".
  - Portal state before the exchange: 89,903 ACCESS credits unexchanged. Anvil GPU costs 67.308 credits per
    GPU-h; Anvil CPU costs 1 credit per SU.
  - **Submitted about 08:35Z, in manual mode:** 89,857 credits for **+1,335 Anvil GPU-h**. The GPU balance
    goes 143 → 1,478, and 46 credits are left. It is **pending** resource-provider approval; the August
    exchange was approved the next day.
  - The first attempt, in auto mode, was blocked by the classifier and nothing was entered. Frank then
    switched to manual mode.
  - Justification submitted (Frank approved the draft unchanged): "These GPU hours fund a pre-registered
    benchmark of generative models for molecular crystal structure prediction, for a manuscript I am
    revising for resubmission. A timing run on Anvil A100s (6.9 GPU-hours) measured the full protocol at
    515-535 GPU-hours. Most of that is machine-learned force-field relaxation; the rest is model training
    and sampling on held-out test sets. The 143 GPU-hours remaining cover about a quarter of it. These
    credits let me complete the protocol, including any reruns it requires."
  - CHE260157 is titled for the STS electrocatalyst work. The text states the symmc-flow use, and Frank was
    told the reviewer may see that mismatch.
  - With 1,478 GPU-h, `batch_float32` (515-535) fits with no degrade items. So does `batch_float64`
    (1,279-1,299), with little margin.
- **Pullback returned (2026-10-15, manual mode).** The scp'd tar's sha256 matches the box (`6db1fa12…efcac`).
  - `unpack --no-overwrite` into `results/n3/private/returns/step4_anvil/`: 289 files, 0 missing, 0 hash
    mismatches, **PASS**.
  - The unpacked `rates.json` has the same sha256 (`4facd352…890c`) as the copy the projection used.
  - The unpack receipt went back to `$SCRATCH/n3_manifests/` on Anvil (hashes match), so a later
    `n3_box_cleanup.sh` can verify it.
  - Cleanup itself waits until Frank decides whether step 5 reuses the box root.
- **Next:** bring Frank the A3 choices, with the FF dtype now affordable either way.

## A3 (2026-10-19)

- **Exchange approved.** `mybalance` on 2026-10-19: che260157-gpu has 1,485.0 GPU-h, 6.9 used, **1,478.1 left**.
- **Frank approved every recommendation (2026-10-19):** `batch_float32`; h = 1e-5 for the FD check; P6(b)
  relative, which A2 item 2 already defines; a non-finite RESID on a finite draw is recorded, not failed.
  The text is in protocol A3.
- **Code:**
  - `n3_classical.py fdcheck` defaults to h = 1e-5, and the shard name now carries h.
  - `n3_mcf_run.resid_checks` makes the RESID rule testable; 2 new tests.
  - N3 suite: 146 passed, 10 skipped.
- **ffconfig** has to run on an A100, because every FF* reply is checked against the resolved state it
  records. It is submitted twice: 21525052 (gpu-debug) and 21525266 (gpu). Keep whichever starts first and
  cancel the other.
  - Scripts: `results/n3/private/anvil_jobs/anvil_ffconfig*.sh`.
  - Both ran 10-19 (backfill, ~1.5 min each); the file is from 21525266, sha256 8e49a547...5702, copied to the laptop and gitignored.
  - The step-5 pack needs the json locally. It is a toy-cell probe with no CSD content.
- **Next:**
  1. Commit and push A3 (Frank OK 10-19, incl. the 700 GPU-h cap and the RESID rule). The push freezes it.
  2. `n3_box_manifest.py pack --step 5`.
  3. The upload to Anvil needs manual-approval mode.
  4. Run the step-5 job: FD at h = 1e-5 → Gate C → training → VALSEL.
- **2026-10-19, after the push (a25a444):**
  - `pack --step 5`: deployable, 209 files, 1,307.9 MB. The scan found 0 TEST-B/DEV-TEST refcode hits; its
    positive control found all 1,200.
  - Archive sha256 `75d73aa00fb18347...`, at `results/n3/private/uploads/step5/`.
  - **The upload waits for manual-approval mode.** Step 5 reuses the step-4 box root `$SCRATCH/n3_box`;
    unpack overwrites it with the new code.
  - Step 5 splits at the H2 form. 5a: FD h = 1e-5 → Gate C → training → VALSEL truths + anchor. Then pull the
    anchor back and push the H2 form from the laptop. 5b: candidates, VALSEL sampling, scoring, diagnostic (a).

## Step 5a on Anvil (2026-10-20/21)

- **Upload (manual mode):** step-5 tar sha256 OK. On unpack, `verify` FAILED only on the 11 step-4 wandb pointer
  symlinks under `n3_mcf/_timing/wandb`. They were identical to `pullback_step4anvil.skipped_symlinks.json`,
  so only the links were deleted (list in `$SCRATCH/n3_ops/removed_wandb_symlinks_step5_2026-10-20.txt`;
  box-B precedent). After that, verify passed: 209 files, 0 missing, 0 hash mismatches, 0 forbidden.
- **A3 item 5 in code:** `n3_mcf_run.py train` had no P6 switch. `--p6 on|off` was added, with on as the
  default; the timing gate's default is unchanged (adfea31). The tar was re-packed (sha256
  `3a9bcdbf0d992b20...`), the pending jobs were held, the tar was re-uploaded and unpacked, verify passed,
  and the jobs were released.
- **Login node, in the image:** Gate C `download` (Zenodo 19673190; md5 OK; best.ckpt sha256 `fd13cbb6...`)
  and `cb_setup` (3c493f8). The MCF-R/A train preflight passed with P6 on. iv-P, classical and score
  import OK.
- **Jobs** (scripts in `results/n3/private/anvil_jobs/anvil_step5*.sh|sbatch` and `step5_{T,G,M,I}.sh`; each
  re-verifies the manifest first):
  - 21571226 `n3-s5T`: FD at h = 1e-5. Must pass, else no truths. Then VALSEL truths, match, anchor.
  - 21571227 `n3-s5G`: Gate C. Exits 0 only if `all_pass`.
  - 21571228 `n3-s5I`: iv-P seed 0 at three learning rates, then `select window`.
  - 21571229 `n3-s5M` (afterok G): MCF-R/A × 3 seeds with P6 on, then train_report and fixed_val_loss.
- Note: the A3 code edits changed `measurement_code_identity`, so `n3_timing_gate.py project` now refuses
  `rates.json` against HEAD. No re-projection is planned. If one is ever needed, it has to be re-timed or the
  substitution recorded.
- **Next:**
  - T done → pull `anchor_valsel.json` (numeric) → publish the H2 form under `results/n3` and push it before
    any candidate FF* minimisation.
  - I done → review the `select window` output → run seeds 1-2.
  - M done → VALSEL sampling (5b).
- **2026-10-21 status:**
  - **T done** (21571226, 9 min). FD at h = 1e-5 on the A100, float64: **PASS**, max relative error 1.29e-6
    over 10 rows.
  - VALSEL truths and anchor: **anchoring 0.99** (396 ANCHORED, 0 DRIFTED, 4 UNRESOLVED; VAL-only 1.00;
    median drift 0.47°), so the **H2 FORM is EXACT**. Published as `results/n3/h2_form_valsel.json` (summary
    only; private anchor sha256 `1b812d06…`).
  - **I failed** (21571228) in the prelude, not in iv-P. verify's step-5 path rule (`/match/`, `labels_`)
    matched T's own VALSEL products. The prelude now excuses only path hits under
    `results/n3/private/{match,classical}/` that name valsel and no held-out set. Tested on the real root,
    with a planted `fake_testB` match file as negative control (it still fails). Resubmitted as **21580267**.
  - **G** (Gate C) is pending, Slurm ETA 10-26; M waits on G.
- **2026-10-21 evening: RCAC emergency power shutdown** (rcac.purdue.edu/news/7936, posted 3:15 pm EDT,
  updated 5:36 pm). Every RCAC system is powered off, Anvil and all storage included. Running jobs are
  stopped and must be resubmitted. **No return-to-service ETA**; RCAC sends updates by email. RCAC also
  plans a **MATH datacenter outage Nov 6-9**.
  - Anvil ssh times out on all three login nodes.
  - Nothing of ours was running: G 21571227, M 21571229 and I 21580267 were all PENDING.
  - **On return, check:**
    1. `squeue` (pending jobs may or may not survive).
    2. `mybalance`.
    3. `$SCRATCH/n3_ops` STEP5_* markers.
    4. `manifest verify --step 5`.
    Then resubmit what is missing.
