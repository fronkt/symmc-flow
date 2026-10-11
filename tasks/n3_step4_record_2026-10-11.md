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
