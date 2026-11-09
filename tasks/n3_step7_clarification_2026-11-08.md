# N3 step-7 clarification, 2026-11-08 (Frank's OK, 2026-11-08; recorded before any TEST-B or DEV-TEST input reaches the box)

This is kept out of `tasks/n3_protocol.md` on purpose. The step-6 freeze (`results/n3/n3_selection.json`, 772a36e)
records the protocol's SHA-256 (`9a483a41…`), and `n3_score.py` refuses a freeze whose protocol file has
changed. Editing the protocol now would invalidate the freeze, so this clarification lives in its own file.

It concerns execution only. It changes no run, seed, menu, endpoint, hypothesis, never-reduced item or
outcome-to-claim rule, and no listed run is added or removed.

## What "the box dies" means on Anvil (protocol §7 Failures; runbook Step 7)

The protocol and runbook say: if the box dies, or the approved cap forces a stop during step 7, discard every
partial TEST output unscored and re-run the complete frozen list with the same seeds on a new box. They were
written for a rented single box. On Anvil the box is the persistent scratch root `$SCRATCH/n3_box`, and step 7
runs as several Slurm jobs.

1. **One Slurm job ending early is not a box death.** This covers timeout, node failure, preemption and OOM.
   - Runs that finished and were checksummed stay.
   - The job's unfinished runs continue with their committed seeds and commands.
2. **How each unfinished run continues:**
   - Per-crystal resumable runs continue from their complete per-crystal records. Each record is checked
     against the run's configuration, and a run with a different configuration is refused. These are
     `n3_classical` haar, press, screen, search, truths, ie, basin and p1basin, and the `n3_ours sample` part
     file.
   - Single-call runs delete their partial outputs and re-run from the start with the committed seed. These
     are MCF infer (which refuses a run directory holding partial outputs), iv-P sample, the MCF-A prior
     floor, the gauge scan, ii-S and iv-S.
3. **Every stop and restart goes in the run record** (`tasks/n3_step4_record_2026-10-11.md`): job, cause, the
   runs affected and how each continued. Nothing is matched, anchored, reported or inspected against the truth
   before step 9.
4. **A real box death still means the full discard and the full re-run as written.** That includes:
   - the box root lost or corrupted, or the step-7 verify failing;
   - a scratch purge;
   - loss of the account or allocation;
   - a stop forced by the approved cap.
