# N3 round-2 implementation review - 2026-10-11

ACS Omega decision on record: ao-2026-09173w rejected 2026-09-24; resubmission permitted through
2027-01-22. Work resumed from the unfinished September round-2 drafts. The frozen scientific
protocol remains v3 plus A1/A2; dated A2a records packaging/integrity clarifications and the state
before new N3 arm outputs. No scientific result or outcome-to-claim rule was changed.

## Changes and their purpose

- Validate complete VALSEL cell tables and recorded winners, selectors/ties, OLD LR/window/sampling,
  FF anchoring and diagnostic evidence. Missing candidate/reference/checkpoint/config artifacts
  prevent a complete freeze. Selected model files travel as checksummed `step7_inputs`.
- Pin report templates to the published choice of H2 form, seeds, selectors, reference direction
  and 10,000 bootstrap draws. Recompute BASIN outcomes from checked existing rows; finite
  unconverged references remain usable, while absent/nonfinite references force paired misses.
- Match provenance binds canonical input content, candidate/reference files, adapters, scorer,
  protocol, actual libraries and selection. Orphan/legacy/stale rows cannot acquire fresh metadata.
  Haar-floor and diagnostic ceiling dependencies are verified before numeric consumption.
- Load only isolated, integrity-checked per-set inputs on the box. Strict mode refuses full-cache
  fallback even if full DEV inputs were already memoized. Composition validates VAL followed by SEL.
  File-identity memoization avoids rehashing/re-walking 400 crystals on every fit.
- Package TRAIN/VAL only for timing; require measured FF configuration before step 5; preserve the
  held-out boundary until the pushed step-6 freeze. Archive extraction checks every listed member
  and checksum before writing. Cleanup preserves manifests/receipts and requires verified return.
- Publish rates only from current measured code/configuration/host records. Measured projections
  also reject source/config changes after rate publication; assumed priors are ineligible for approval.
  Include FF startup/recompile costs and retained H2 support in the runlist/projector.
- Keep pinned package versions but correct FF Torch 2.13 to cu130 and driver >=13.0. Pin exact PyG
  builds, check official wheel availability before rental, and retain installed-version/CUDA smoke
  checks for the actual box. Preserve LF line endings for deployment shell scripts.

## Verification

Final combined suite: **137 passed, 3 skipped in 87.11 s**:

```text
python -m pytest tests/test_n3_caches.py tests/test_n3_select.py tests/test_n3_score.py tests/test_n3_provenance.py tests/test_n3_diagnostics.py tests/test_n3_operations.py -o addopts= -q
```

Three heavy stored-result checks skip without `N3_HEAVY=1`; those skips are not evidence of GPU
execution or a finished benchmark.
Each implementation track was independently reviewed against the frozen protocol, with fixes followed
by regression tests. No real held-out arm output was matched during review.

Input-only checks on the actual corpus:

- All six portable caches built: TRAIN 1687, VAL 100, SEL 300, VALSEL 400, TEST-B 1000, DEV-TEST 200.
- The simulation passed all 12 checks: identical cache/source content and order; full sources not
  opened by strict/cache-preferred loads; step-5 held-out/full-source refusal; tamper refusal.
- Warm repeated strict cache access measured about 0.85 ms for VAL and 1.22 ms for VALSEL on the
  laptop (200 calls each). These are loader checks, not benchmark compute projections.
- The final TRAIN/VAL-only timing archive contains 189 files, 1278.1 MB. SHA-256:
  `280c00bcd49a5c083545d944ba5442694ac20b424f8e3f3279ee69df88e9c544`.
  Packing found no prohibited path/hash or held-out refcode data hits (33 forbidden files checked;
  all three forbidden-cache positive controls detected). Actual local extraction checked all 192
  members before writing, then verified all 189 payload files with no mismatches or boundary hits.
  A fresh process in the extracted root loaded TRAIN 1687 and VAL 100, rejected SEL/VALSEL/TEST-B/
  DEV-TEST, and constructed scoring provenance including the protocol without performing any fits.
  The archive and retained extraction/loader receipts are under gitignored `results/n3/private/`.
  Automatic approval review blocked deleting the temporary extraction (`blocked by policy`, with no
  further reason). The verified duplicate remains in `transfer_check_2026-10-11`; original data,
  archive and receipts were preserved.
- Read-only official wheel/metadata preflight passed for Linux CPython 3.12 x86_64 at 05:01:58 UTC:
  MCF Torch/torchvision, exact torch-scatter/torch-cluster builds, SymMC Torch, and FF Torch/cu130
  plus fairchem-core 2.23.0's `torch~=2.13.0` requirement. No GPU installation was performed.
- Python compilation and both deployment-shell syntax checks passed. Independent review found
  and closed the timing import-closure gap and the scorer's rigid-structure adapter hash gap;
  regressions reject source drift before consuming rates, cached rows or diagnostic ceilings.

## Remaining execution

No rental, CSD upload, GPU training, new N3 sampling, paid cleanup/destruction, H2-form decision,
complete selection freeze or benchmark result exists from this continuation. Actual GPU installation,
FF finite differences, P6/packed-encoder equivalence and measured scheduling/cost remain step-4 gates.
The preflight verifies critical binary distributions/metadata; it is not a full transitive resolver
or proof that GPU imports/performance succeed.

The locally prepared timing-only proposal is in the gitignored
`results/n3/private/timing_offer_proposal_2026-10-11.json`: single RTX 4090, >=8 effective CPU cores,
>=32 GB RAM, 100 GiB priced disk and CUDA >=13.0; proposed $5 total cap and at most eight running
hours, with reserve for verified pullback/cleanup. Offers are transient and must be rechecked before
rental. No offer was rented. This is not approval of the full benchmark.

Protocol section 7 requires manual approval of CSD transfer and approval of the measured projection
before larger work. After step 4, freeze/push A3; after VALSEL selection, freeze/push the complete
runlist before test inputs. Once all fixed outcomes exist, use Table 1/2 and the claim map for the
manuscript and ACS response rewrite.

## Publication

Code commit `783a1ab2020a4fd2601d043196e49f64a8381ba8` was pushed to `origin/paper-draft`.
`git ls-remote origin refs/heads/paper-draft` returned that exact SHA after the push.
Only the 24 explicit implementation/test/handoff paths were staged; unrelated manuscript/workshop
edits remain unstaged, and all CSD caches/archives/receipts/proposals remain gitignored.
