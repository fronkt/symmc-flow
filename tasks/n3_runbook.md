# N3 execution runbook - 2026-10-11

The scientific specification is [n3_protocol.md](n3_protocol.md), frozen v3 plus A1/A2/A2a.
This runbook implements its section 7 steps 4-9. The decision record permits an ACS Omega
resubmission through **2027-01-22**; completing this benchmark precedes the evidence-based rewrite.
No new N3 arm output on SEL, TEST-B or DEV-TEST was generated during the October implementation review.

## Local preparation and publication

Run from the repository root with the project's Python 3.12 environment. On Frank's laptop the
interpreter is `C:/Users/frank/AppData/Local/Programs/Python/Python312/python.exe`.

```powershell
python -m pytest tests/test_n3_caches.py tests/test_n3_select.py tests/test_n3_score.py tests/test_n3_provenance.py tests/test_n3_diagnostics.py tests/test_n3_operations.py -o addopts= -q
python scripts/n3_make_set_caches.py build
python scripts/n3_make_set_caches.py check
python scripts/n3_make_set_caches.py simulate --patched scripts/n3_common.py --out results/n3/private/cache_simulation.json
python scripts/n3_box_manifest.py wheels
python scripts/n3_box_manifest.py list --step 4
python scripts/n3_box_manifest.py pack --step 4 --out results/n3/private/uploads/timing
```

Publish the reviewed code before transferring anything. Stage specific files; keep CSD inputs,
exports, model weights, checkpoints, draws, timing scratch and archives out of git. Preserve the
existing manuscript/workshop changes. `data/n3_sets` is the portable cache directory; each `.pt`
requires its matching `.json`. The loader checks file and canonical payload hashes, counts, ordered
refcodes, source identity and frozen list hashes. `N3_SET_CACHE_ONLY=1` forbids full-cache fallback.

`pack` refuses incomplete or diagnostic-only inventories. The step-4 archive contains TRAIN/VAL
inputs only. The step-5 archive also contains SEL/VALSEL. Neither contains TEST-B/DEV-TEST inputs,
the full dev/pool caches, or their exports/draws. Do not use `--groups`, `--allow-missing`,
`--allow-unpatched` or `--no-scan` for a transfer.

## Step 4: measured timing and cost

Before renting or uploading, present the actual offer's hourly/storage charges, CPU/GPU/RAM/disk,
the timing archive's size/hash, and a timing-only spending limit to Frank. The protocol explicitly
requires CSD uploads in manual-approval mode and approval of the measured projection before any
larger run. The old handoff's prices and shared balance are historical estimates, not current quotes.

Use a dedicated single-GPU Linux box with a CUDA >=13.0 driver and the resources checked by setup.
Keep upload archives and manifests in distinct directories outside the private repository root.
The following are **box** commands after the approved transfer of the archive, its checksum, and
`scripts/n3_box_manifest.py` to `/root/n3_upload`:

```bash
export N3_BOX_ROOT=/root/n3_box
export N3_UPLOAD_DIR=/root/n3_upload
export N3_MANIFEST_DIR=/root/n3_manifests
cd "$N3_UPLOAD_DIR"
sha256sum -c n3_step4.tar.sha256
python3 n3_box_manifest.py unpack --tar n3_step4.tar --root "$N3_BOX_ROOT"
cd "$N3_BOX_ROOT"
bash scripts/n3_box_setup.sh check
bash scripts/n3_box_setup.sh all
source .n3_box/env.sh
bash scripts/n3_box_setup.sh data
```

Setup creates separate pinned MCF, FF and SymMC environments and records their installed versions.
MCF/SymMC use cu128; the pinned FF Torch 2.13 requires cu130. The read-only `wheels` preflight
checks official distribution indexes before any rental; an unavailable wheel blocks the paid setup.
Production runs must import the actual PyG extensions and use no laptop shims. Hugging Face access
is offline; the archive carries the licensed model weights. Temporary files and wandb/HF caches
are confined to the private box root. A dry setup plan can be printed with `N3_DRY_RUN=1` before setup.

Choose `W` and concurrency to equal the intended production scheduling. These example scheduling
values must be replaced if the selected box or approved plan differs:

```bash
W=4
G=1   # FF worker concurrency; retain the value measured/approved for production
"$PY_SYMMC" scripts/n3_timing_gate.py plan --workers "$W" --concurrency 1 --ivp-cpb 1 8
"$PY_SYMMC" scripts/n3_timing_gate.py run --workers "$W" --concurrency 1 --ivp-cpb 1 8
"$PY_SYMMC" scripts/n3_timing_gate.py rates
```

The gate measures both MCF objectives with P6 on/off, iv-P encoders and batch sizes, 16-draw/50-step
inference, FF settings/dtypes and truth minimisation, LJ screens, press and both matcher endpoints.
FF costs include model startup/restart; a predict-only throughput is insufficient. Failed, changed
or incomplete measurements cannot be used as a completed rate. Nothing from these timing samples
is a VALSEL or confirmatory result. If CPU scoring will use another machine, repeat the `cpu`
measurement there at its intended worker count rather than transplanting a laptop rate.
Failed measurements cannot simply be rerun over resumable unit shards: use a fresh approved timing
box, or inspect and clear only that measurement's verified scratch outputs before removing its
failed timing record. Retain the failure evidence. Do not mix rates from different host/configuration
fingerprints. Complete the protocol's FF float64 finite-difference and P6/packed-encoder equivalence
checks before choosing those performance settings for A3.

Create full-set input statistics **on the laptop**, where all input caches are permitted:

```powershell
python scripts/n3_timing_gate.py stats
```

After checksummed timing pullback, project each candidate measured configuration **on the laptop**.
Set the offer's
actual rate/cores and the separate CPU machine's actual rate/cores; neither default price is a quote:

```powershell
python scripts/n3_timing_gate.py project --rates results/n3/private/timing/rates.json --stats results/n3/private/timing/set_stats.json --usd-per-hour $GPU_BOX_RATE --cores $GPU_BOX_CORES --cpubox-usd-per-hour $CPU_BOX_RATE --cpubox-cores $CPU_BOX_CORES --p6 off --ivp-encoder pack --ivp-cpb 8 --ff batch_float32 --tier2 --cap-usd $APPROVED_CAP --out results/n3/private/timing/projection.json
```

The configuration flags above are examples, not frozen choices. Present measured GPU-hours,
CPU-hours, box-hours and dollars separately for training, VALSEL inference/scoring, test inference,
FF, press, LJ and matching. Freeze FF settings/dtype with `n3_classical.py ffconfig`; record the
chosen P6/iv-P settings and the fewest degrade items needed as dated **A3**, stating that no N3
output exists. Commit/push A3 before step 5. If degrade item 5 is contemplated, A3 must explicitly
record retained DEV H2 support (iii-s, FF truth and all three OURS/OURS-N2 BASIN candidates); these
support the stored DEV rows and the never-reduced H2 requirements. Do not drop them silently.
No degrade item may be newly chosen after this gate. Obtain approval of the measured full plan.

If the timing box will be destroyed here, use the same verified pullback/cleanup procedure as step 8.
Keeping it idle while waiting consumes the offer's charges and must fit the timing cap.

## Step 5: TRAIN + VALSEL only

After approval, pack/upload/unpack the step-5 inventory, verify it and re-source the box environment:

```powershell
python scripts/n3_box_manifest.py pack --step 5 --out results/n3/private/uploads/step5
```

```bash
python3 "$N3_UPLOAD_DIR/n3_box_manifest.py" unpack --tar "$N3_UPLOAD_DIR/n3_step5.tar" --root "$N3_BOX_ROOT"
source "$N3_BOX_ROOT/.n3_box/env.sh"
"$PY_SYMMC" scripts/n3_box_manifest.py verify --step 5 --root "$N3_BOX_ROOT"
bash scripts/n3_box_setup.sh data
```

First run Gate C, then train. Gate C-a scorer agreement, C-b and C-c must pass. A reproduction-rate
failure alone requires the protocol's dated diagnosis and later harness wording; it is not permission
to skip scorer agreement. Use the gate CLI's printed paths and retain its component records:

```bash
"$PY_MCF" scripts/n3_mcf_gateC.py download
"$PY_MCF" scripts/n3_mcf_gateC.py ca_infer --gpu 0 --run
"$PY_MCF" scripts/n3_mcf_gateC.py ca_match --num_cpus "$W"
"$PY_MCF" scripts/n3_mcf_gateC.py cb_setup
"$PY_MCF" scripts/n3_mcf_gateC.py cb --device cpu
"$PY_MCF" scripts/n3_mcf_gateC.py cb --device cuda
"$PY_MCF" scripts/n3_mcf_gateC.py cc_infer --gpu 0 --run
"$PY_MCF" scripts/n3_mcf_gateC.py cc
```

Apply the approved A3 settings when invoking training; defaults must not override A3. Train MCF-R
three seeds and MCF-A the permitted seeds, retaining top-three/last checkpoints, configs, validation
history and fixed-draw VAL loss. Designate top-three/lowest-loss snapshots from the raw training
history before VALSEL scoring; the assembler validates configured candidates but does not independently
reconstruct that designation from the training log. For iv-P, train seed 0 at all three frozen LRs, run `select window`,
then run its printed seed-1/2 commands at the selected LR and re-run `select window`. The final
printed commands sample its checkpoint candidates, select iv-P, freeze its sampling configuration
and select iv-S. Never cap epochs/steps or use smoke modes for reported data.

**H2 form first:** minimise all 400 VALSEL truths, then compute anchoring:

```bash
"$PY_SYMMC" scripts/n3_classical.py truths --set valsel --workers "$G"
"$PY_SYMMC" scripts/n3_score.py match --bundles results/n3/private/draws/fftruth_s-1_valsel.pt \
  --matcher primary --draws all --workers "$W"
"$PY_SYMMC" scripts/n3_classical.py anchor --set valsel
```

Pull the numeric anchoring record to the laptop, publish a numeric copy under `results/n3`, and
commit/push the H2 form **before any arm's candidate FF minimisation**. EXACT is selected if
anchoring >=50%; otherwise BASIN. Retain the original private reference bundle and its hash.

Then sample every VALSEL candidate required by protocol sections 2.3 and 3, with full seed coverage.
Convert stored OURS/N2 VAL draws, sample SEL, and concatenate in VAL-then-SEL order. Generate the
classical selection configs using `n3_select.py --write-configs`. MCF configs must enumerate:

- R and R+G Stage A: all five s_uR values, the full selector menu and three lowest-loss checkpoints;
  Stage B: each seed's top-three/last at its selected Stage-A setting, with fixed VAL loss ties.
- A stages: checkpoint at (9,3), the permitted 3x3 knob grid and RANDOM/LJ/RESID selector, at native
  matcher settings. The step-4 degrade record controls reductions.
- ii-S and P1: their separate permitted selector choices; P1 is compared against the pressed truth.

Use `n3_score.py match` on the candidate bundles to make current score rows, and then
`n3_score.py select --config <candidate-config.json>` to produce each record expected by
`n3_select.py --dry-run`. Its output names every missing role. Bundle paths in these configs must
be repository-relative so pulled results remain portable. Record every VALSEL cell, including
losing choices. The selector validator recomputes the winners from the recorded aggregate scores;
it cannot turn an empty/missing record into a freeze.

Run diagnostic (a) for every selected MCF checkpoint, plus the released-checkpoint provenance repeat,
in the actual MCF environment without shims. Unresolved diagnostics are recorded FAILED at step 6;
they are never retried after seeing test outcomes. A failed diagnostic removes (b)-(d) and the
associated invariance statement. All matching on SEL occurs inside VALSEL, not an isolated SEL bundle.

## Step 6: pull back the choices and publish the freeze

Checksummed pullback must include the selection records, candidate bundles and selected model files,
their training configs, diagnostic/gate records, FF config and private iv-P frozen record. Assemble
the freeze **on the laptop**; box-absolute MCF checkpoint metadata is normalized against the locally
present `n3_mcf/runs` files. The exact files must exist and match their recorded SHA-256.

```powershell
python scripts/n3_select.py --dry-run
python scripts/n3_select.py
git add -- results/n3/n3_selection.json
git commit -m "Freeze N3 VALSEL choices and permitted test runs before held-out upload"
git push origin HEAD
```

Do not use `--allow-absent-bundles` to manufacture a complete freeze. Review `complete`,
`report_configs`, `step7_inputs`, `runs.step7`, `runs.step9`, `runs.step9_scoring`, H2 form and the degrade
record before publishing. Confirm the remote has the commit. No TEST-B/DEV-TEST input can reach
the box before this publication; its selection file becomes the input for step 7.

## Step 7: exactly the frozen jobs, without matching

```powershell
python scripts/n3_box_manifest.py pack --step 7 --out results/n3/private/uploads/step7
# For a fresh replacement box, use --new-box as well.
```

After the approved transfer, unpack/verify the step-7 manifest. It adds only the permitted test
inputs and hashed `step7_inputs` carryover. Copy the selection's commands to a private checklist;
replace only the documented scheduling placeholders `{W}`, `{G}`, `{GPU}`. Run MCF scripts with
`PY_MCF` and OURS/OLD/classical/scorer scripts with `PY_SYMMC`; classical starts FF through
`N3_FF_PYTHON`. Honor the list's `where` field for CPU adapters deferred until after pullback.

Complete/checksum every box job: full draws and selector inputs, MCF RESID/check records, iii-s,
all test truth minima, i-E where permitted, and three-seed OURS/N2 LJ-picked BASIN candidates.
Run both H2 forms regardless of the selected primary form. Verify each expected output exists;
each MCF `check.json` must pass and every production bundle has the full frozen refcode order.
Do not call match, anchor, report or a truth-based diagnostic on test outputs while on this box.

If the box dies or the approved cap forces a stop during step 7, discard **all** partial test
outputs unscored and rerun the complete frozen list on a new box with the same seeds. Do not patch
the runlist, reduce seeds or keep a convenient completed subset. No FF test job may occur after step 8.

## Step 8: verified return, explicit cleanup, destruction

On the box:

```bash
"$PY_SYMMC" scripts/n3_box_manifest.py pullback --root "$N3_BOX_ROOT" \
  --out "$N3_UPLOAD_DIR/return" --label step7
```

Transfer the printed archive and `.sha256` to `results/n3/private/returns/` on the laptop, retaining
the box archive hash. On the laptop, set `N3_MANIFEST_DIR` outside the repository (for example
`C:/Users/frank/n3_manifests`) and unpack into this repository:

```powershell
$env:N3_MANIFEST_DIR='C:/Users/frank/n3_manifests'
python scripts/n3_box_manifest.py unpack --tar results/n3/private/returns/n3_pullback_step7.tar --root .
```

Copy the resulting `pullback_step7.unpack_receipt.json` from the laptop manifest directory back to
the box's external manifest directory. Normal cleanup refuses to delete private outputs without
this successful receipt and its matching archive hash:

```bash
export N3_PULLBACK_RECEIPT="$N3_MANIFEST_DIR/pullback_step7.unpack_receipt.json"
N3_CLEANUP_PLAN_ONLY=1 bash scripts/n3_box_cleanup.sh
bash scripts/n3_box_cleanup.sh
```

Review the explicit canonical targets before entering DELETE. Cleanup retains manifests and a
timestamped receipt outside the deleted root, checks for active jobs, searches for private names
and inventoried hashes and fails if any search/hash fails. Explicit task-specific extra files may
be listed with `N3_EXTRA_DELETE_FILE`; broad HOME/tmp deletion is refused. `N3_DISCARD_OUTPUTS=1`
is for the protocol's unscored stop/restart, not a successful run. Retain the PASS cleanup receipt,
destroy the specific rented instance, and retain the provider's destruction evidence.

## Step 9: one fixed analysis and the manuscript rewrite

Complete the frozen CPU adapters on returned samples, then execute `runs.step9` in its recorded
order, followed by `runs.step9_scoring`, from the pushed selection. These include primary and secondary matches, reference/BASIN reports,
and permitted MCF diagnostics (b)-(d) with the matched three-seed Haar floor. Each held-out match
requires `--step9` and the committed, unchanged, pushed selection. A CPU box without git also
requires `--selection-sha256` with the laptop-verified published freeze hash.

Write each report config directly from `selection.report_configs[set]` to
`results/n3/report_<set>.json`, as named by the frozen report commands, then invoke:

```powershell
python scripts/n3_score.py report --config results/n3/report_testB.json --step9
python scripts/n3_score.py report --config results/n3/report_devtest.json --step9
```

Only the output destination may change; arm identities, seeds, selectors, H2 form, references and
10,000 bootstrap draws are frozen. Stale rows from a changed bundle, truth, reference, scorer,
adapter, package version or selection are refused. Legacy rows lacking provenance are also refused;
do not silently reuse them or redo held-out choices. Finite unconverged FF references remain usable;
missing/nonfinite references force all-arm H2 misses with paired difference zero.

Run commands that consume existing matches in the same recorded scoring environment, including the
actual torch build and pymatgen/scipy/numpy versions. Pulling CUDA-box rows to the laptop does not
make its default Python an equivalent scoring environment. Assemble the freeze from the checked
numeric records locally, but keep score consumption/reporting on the environment that made those
rows; for step 9 choose the final CPU environment before the first match and use it throughout.

Preserve both reports in `results/n3/n3_results.json`, including secondary DEV disclosure,
Gate-C/diagnostic/degrade provenance and the protocol's outcome-to-claim map. Commit/push the numeric
results. Then rebuild Table 1/2, the abstract, methods, results, limitations and ACS response letter
from those outcomes. The scope is orientation completion conditioned on known lattice, centroids
and space-group operations; new full-CSP or acceptance claims cannot be inferred from this benchmark.
