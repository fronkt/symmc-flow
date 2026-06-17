# DiffCSP head-to-head runbook (apples-to-apples match@k)

**Why.** Our current tables compare *our* `match@20` to *DiffCSP's* reported `match@1` — an
apples-to-oranges comparison. DiffCSP papers headline `match@1` because their evaluation default is
one sample per reference; their own `match@20` is rarely tabulated. To make the comparison fair we
run DiffCSP's *released* models ourselves with `num_evals=20` and report DiffCSP's `match@1` AND
`match@20` + RMSE under the **same matcher** we use.

This is a GPU-box task. **No code is added to symmc-flow** — DiffCSP's own evaluation already uses
the CSP-standard matcher, identical to ours:

- DiffCSP `scripts/compute_metrics.py` builds `StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)`
  and scores a match as `get_rms_dist(...) is not None` (RMSE = the returned distance,
  `break_on_match=False`). That is exactly `symmc_flow`'s `match_rate` / `match_rate_topk`
  (see `RESULTS.md` "Matcher reconciliation"). So no re-scoring through our harness is needed —
  DiffCSP's numbers are directly comparable as long as we set the same `num_evals`.

## Steps (vast.ai RTX 5090, Blackwell → cu128 wheels)

```bash
# 1. clone + env (DiffCSP pins older torch/pyg; use a fresh venv, not the symmc-flow one)
git clone https://github.com/jiaor17/DiffCSP.git && cd DiffCSP
# follow their README env; on Blackwell install torch via the cu128 index, then torch-geometric.
# set the project paths their hydra configs expect:
export PROJECT_ROOT=$PWD HYDRA_JOBS=$PWD/hydra WABDB_DIR=$PWD/wandb

# 2. get the released checkpoints (their README / Google Drive links): mp_20 and carbon_24.
#    place each under a model dir, e.g. checkpoints/mp20 and checkpoints/carbon24.

# 3. generate + reconstruct. num_evals=20 draws 20 candidates per reference (their best-of-k).
python scripts/evaluate.py --model_path checkpoints/mp20    --dataset mp_20    --num_evals 20
python scripts/evaluate.py --model_path checkpoints/carbon24 --dataset carbon_24 --num_evals 20

# 4. metrics — match@1 and match@20 + RMSE, same matcher as ours.
python scripts/compute_metrics.py --root_path checkpoints/mp20    --tasks csp --multi_eval
python scripts/compute_metrics.py --root_path checkpoints/carbon24 --tasks csp --multi_eval
```

Notes:
- `--multi_eval` is what makes `compute_metrics.py` report the best-of-`num_evals` (match@k) rate;
  without it you only get match@1. Confirm the flag name against the cloned revision's
  `compute_metrics.py` (DiffCSP/DiffCSP++ have renamed it across versions — `--multi_eval` vs a
  `num_evals` arg). The invariant to preserve: **same `num_evals=20` and same StructureMatcher
  tolerances** as our run.
- Use the **same val split size** we evaluate (256) if you want a strictly matched N; otherwise note
  the N difference. DiffCSP's default evaluates the full test set — record whichever you use.
- carbon-24: DiffCSP conditions on composition+count only (as we do), so its carbon match@k is the
  honest head-to-head for our carbon numbers.

## Recording

Add DiffCSP's `match@1` / `match@20` / RMSE to the comparison tables in `RESULTS.md` (MP-20 and
carbon-24 sections), replacing the "~51% (match@1)" / "~17%" placeholders that were lifted from the
papers. Keep the provenance line: "DiffCSP numbers from our run of the released checkpoint,
num_evals=20, get_rms_dist, N=___".
