# N3 / G3: where we stopped, and exactly how to continue (handoff, 2026-09-28)

The spec is `tasks/n3_protocol.md`: v3, frozen at commit 8f7e494, with amendments A1 and A2 in §9.
This file is NOT frozen; it is the working state.

## Done
- **Protocol** pre-registered after three hostile review rounds. The review material is in the Claude
  scratchpad, and the reasoning is recorded in the protocol's provenance section.
- **Fresh TEST-B (1000) and SEL (300)** built from rows 3501-8000 of `data/csd_mol_scale_big`. Refcode
  lists frozen in A1 (05aa3c9): `tasks/n3_sets/{testB,sel}_refcodes.txt`.
  - The pool cache `data/csd_testB/n3_asym_pool.pt` and the parse `data/csd_testB/ds_o3.pt(.parts)` are
    gitignored CSD files. Do NOT delete them.
- **Input-side work** is complete for every set:
  - stratum labels (`scripts/n3_labels.py`);
  - MCF export + Gate F (all pass; `results/n3/gateF.json`);
  - the legacy oracle;
  - Table R (`results/n3/table_r_{devtest,val,testB}.json`). TEST-B Table R: legacy det+1 cannot represent
    649/1000 (604 of 679 centro); the MCF CSP driver cannot produce 634/634 mixed-handedness cells under
    the strict matcher, while a homochiral approximant matches 323/634 at stol 0.5.
- **Power table** (`results/n3/power_testB.json`): 0.83 at Δ = 2 pp for a 3-seed arm; iii-s 0.61.
- **Arm code, round 1** (implemented → adversarially reviewed → fixed; CPU smoke on VAL/TRAIN only):
  - MCF: `scripts/n3_mcf_{export,fork,smoke,lib,run,resid,gauge,sg,diag,gateC}.py`,
    `patches/mcf_n3.patch` (fork at `external/mcf`, gitignored; rebuild with n3_mcf_fork.py).
  - OLD iv-P: `scripts/n3_old_symmc.py`.
  - CLASSICAL / FF*: `scripts/n3_classical.py`, `scripts/n3_ffstar.py`. UMA s-1p1 weights are in
    `external/uma/` (gated; the SHA is in A1). The local CPU venv is `external/venv_fairchem`
    (fairchem-core 2.23.0).
  - OURS + scorer: `scripts/n3_ours.py`, `scripts/n3_score.py`, `scripts/n3_harness.py`,
    `scripts/n3_tpool.py`, `tests/test_n3_score.py`.
  - Labels / Table R: `scripts/n3_labels.py`, `n3_table_r.py`, `n3_legacy_oracle.py`.
  - The unit reports (commands, deviations, timings) are summarised into A2.
- **A2** (the implementation clarifications) is written in §9. Commit it with the code before any box
  step.

## NOT done: continue here, in this order
1. **Finish implementation round 2.** It was stopped mid-run so work could pause. Its drafts exist but
   are UNREVIEWED:
   - `scripts/n3_select.py`, `tests/test_n3_select.py`;
   - the BASIN-reference edits inside `scripts/n3_score.py`;
   - `scripts/n3_make_set_caches.py`, `n3_box_manifest.py`, `n3_box_setup.sh`, `n3_box_cleanup.sh`,
     `n3_box_constraints_{mcf,ff,symmc}.txt`, `n3_timing_gate.py`.
   The workflow script with both unit specs is saved at
   `~/.claude/projects/C--Users-frank-symmc-flow/40207194-28ba-4c06-bcbf-fb5cb9ac6dff/workflows/scripts/n3-implement-2-wf_cb8207d0-af5.js`.
   Re-run it as review + fix of the existing drafts, not from scratch. What it must deliver:
   - (a) n3_score BASIN reference matching (h2_basin JSON; P1 steric basin vs p1ref; rule (iii));
   - (b) n3_select.py assembling `results/n3/n3_selection.json` with a completeness check;
   - (c) per-set caches, so DEV-TEST/TEST-B never reach the box before §7 step 7. This needs a small
     change to `scripts/n3_common.py`: load_set prefers `data/n3_sets/<set>.pt`, and `data/n3_sets/`
     must be added to .gitignore;
   - (d) box env setup for three conda envs: mcf (torch 2.7.1+cu128 + PyG), ff (fairchem-core 2.23.0,
     torch 2.13), symmc;
   - (e) upload manifests: step 5 = TRAIN/VAL/SEL only; step 7 adds TEST-B/DEV-TEST;
   - (f) the timing gate + cost projector;
   - (g) the cleanup script with find-verification;
   - (h) `tasks/n3_runbook.md`.
2. **Commit and push** round-2 code + A2. Never commit CSD files; stage explicit paths only.
3. **§7 step 4: timing gate on a rented single-GPU Vast box.** This NEEDS FRANK:
   - he must switch Claude Code to manual-approval mode (Shift+Tab) for the CSD upload;
   - the Vast balance is shared with other projects (mlip-dynstab).
   The gate measures MCF/iv-P steps, FF* evals ('batch' vs 'default', float32 vs float64), matching and
   press rates on TRAIN/VAL only. It fixes the FF* configuration (`n3_classical.py ffconfig`), iv-P
   `--pack`, and P6 on/off in a dated step-4 amendment (A3), plus any degrade steps. Then **send Frank
   the projected GPU-h / CPU-h / $** and get approval before anything bigger.
   - Prior estimate: MCF 6 runs × 1-4 GPU-h; FF* up to ~16M UMA evaluations (8-15 GPU-h only if
     batched at ≤ 3.5 ms/eval); iv-P 4.4-20 GPU-h; CPU matching ~250-500 CPU-h for TEST-B + DEV-TEST.
     Expect roughly $15-40, more than the current Vast credit (~$17.8).
4. **§7 step 5** (box): Gate C (C-a Thurlemann reproduction, C-b, C-c) → train MCF-R ×3, MCF-A ×3, iv-P
   (seed 0 at 3 lrs, then seeds 1-2) → FF* truths on the 400 VALSEL → push the H2 FORM → sample and score
   VALSEL for every candidate cell → MCF diagnostic (a).
5. **§7 step 6:** write + commit + PUSH `results/n3/n3_selection.json` BEFORE any TEST-B/DEV-TEST input
   reaches the box.
6. **§7 steps 7-9:** upload TEST-B/DEV-TEST; run exactly the listed runs (sampling, FF* jobs, RESID). Then
   pull back, delete the CSD files, destroy the box, and verify. Score TEST-B once (`n3_score.py --step9`),
   write `results/n3/n3_results.json`, commit and push.
7. Then Table 1/2 and the outcome → claim map (§4) feed the paper rewrite (G4/G5 in
   `omega_rebuild_plan.md`). Resubmission window closes 2027-01-22.

## Rules that must survive
- **Blinding:** never match arm OUTPUTS on SEL/TEST-B/DEV-TEST before the step they are allowed (VALSEL
  at step 5; TEST-B/DEV-TEST only at step 9 with `--step9`). Smoke tests on VAL/TRAIN only.
- **CSD hygiene:** CIFs, pickles, exports, checkpoints and draws stay local or on the box, and are all
  gitignored (`data/csd_*`, `external/`, `n3_mcf/`, `results/n3/private/`). The box is cleaned and
  destroyed at the end.
- **No change to a frozen item** except by a dated §9 amendment that states whether any N3 output
  existed.
- Orphaned idle multiprocessing workers from 2026-09-27 23:05/23:24 (parents dead) sit on the laptop,
  ~120 MB. They are harmless and were left alone.
