"""E4 -- MolCrystalFlow metric alignment.

Report OUR match@10 in MolCrystalFlow's currency so a JCIM reviewer can read the
comparison in the same units. MolCrystalFlow (arXiv:2602.16020v3) evaluates with
pymatgen StructureMatcher(ltol=0.3, angle_tol=10) and a SWEEP of site tolerances
stol in {0.5 ... 1.2}, best-of-10 samples per test compound (match@10). It reports
~6.8% at stol=0.8 and ~8% at stol=1.0. Crucially MolCrystalFlow and MOFFlow are
symmetry-FREE (de-novo); Genarris-3 -- which MCF is only "competitive" with -- IS
space-group conditioned. That split is exactly our thesis.

So this script produces, on the SAME held-out crystals / corpus / split, a paired:
  * coset-OFF (de-novo, no symmetry template)  -> head-to-head with MCF / MOFFlow
  * coset-ON  (deployable symmetry-op template) -> our symmetry-conditioned number
both scored at match@10 across stol in {0.5, 0.8, 1.0, 1.2} (their protocol), same
ltol/angle_tol. It is a same-units, comparable-task alignment -- NOT a strict
head-to-head (our corpus is a smaller CSD-derived molecular set, not Thurlemann /
OMC25). That caveat is printed and belongs in the paper.

Two robustness choices, both to avoid the pool.map-no-timeout StructureMatcher hang
that cost the E2 box 2h15m:
  1. GENERATE ONCE, save refs + both draw sets to a pickle; the stol sweep then
     re-scores for free (and future re-scores need no GPU/CPU regeneration).
  2. PER-CRYSTAL TIMEOUT on the matcher (fresh pool per stol, terminate stragglers):
     a single pathological fit that never returns is counted as a miss, not a hang.

    python scripts/eval_e4_molcrystalflow.py --ckpt checkpoints/coset_deploy_s0.pt \
        --cache data/csd_mol/ds.pt --match-k 10 --workers 6
    # re-score only (skip generation) once the pickle exists:
    python scripts/eval_e4_molcrystalflow.py --score-only
"""
import argparse
import math
import os
import pickle
import sys
import time
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from symmc_flow.config import ModelConfig
from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item,
                                    rigid_to_structure, species_multiplicity,
                                    assign_symmetry_cosets)
from symmc_flow.model import SymMCFlow
from symmc_flow.train import resolve_device, move_batch
from symmc_flow.data import collate, batch_to_state
from symmc_flow.flow import sample_prior
from symmc_flow.sampler import rk4_sample
from symmc_flow import manifolds as M

RAD2DEG = 180.0 / 3.141592653589793


def _match_one(args, ltol, stol, angle_tol):
    """(ref_dict, [k gen_dicts]) -> 1 if any gen matches ref under the tolerances."""
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    ref_d, gen_ds = args
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    ref = Structure.from_dict(ref_d)
    for gd in gen_ds:
        if sm.fit(Structure.from_dict(gd), ref):
            return 1
    return 0


def cell_volume(struct_dict):
    """|det(lattice matrix)| from a pymatgen Structure.as_dict() -- no object build."""
    import numpy as np
    return abs(float(np.linalg.det(np.array(struct_dict["lattice"]["matrix"]))))


def rmad(refs, gens):
    """MolCrystalFlow's RMAD: relative mean-abs deviation of unit-cell VOLUME over ALL
    k samples (not matched-only). Returns (mean%, std%) over every crystal x sample."""
    import numpy as np
    devs = []
    for ref_d, gen_ds in zip(refs, gens):
        vref = cell_volume(ref_d)
        if vref <= 0:
            continue
        for gd in gen_ds:
            devs.append(abs(cell_volume(gd) - vref) / vref)
    devs = np.array(devs) if devs else np.array([float("nan")])
    return 100.0 * float(devs.mean()), 100.0 * float(devs.std())


def _cell_angles(struct_dict):
    """(alpha,beta,gamma) degrees from a Structure.as_dict() lattice matrix."""
    import numpy as np
    Lm = np.array(struct_dict["lattice"]["matrix"])
    a, b, c = np.linalg.norm(Lm, axis=1)
    def ang(u, v):
        return np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)
                                                            + 1e-12), -1, 1)))
    return ang(Lm[1], Lm[2]), ang(Lm[0], Lm[2]), ang(Lm[0], Lm[1])


def angle_spike(structs, tol=2.0):
    """Fraction (%) of cell angles within `tol` deg of 90 -- the crystal-family gate metric.
    `structs`: list of per-crystal single dicts OR lists of dicts (all samples pooled)."""
    import numpy as np
    angs = []
    for entry in structs:
        for sd in (entry if isinstance(entry, list) else [entry]):
            angs.extend(_cell_angles(sd))
    angs = np.array(angs) if angs else np.array([0.0])
    return 100.0 * float(np.mean(np.abs(angs - 90.0) < tol))


def wilson(mr, n):
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    denom = 1 + z * z / n
    centre = (mr + z * z / (2 * n)) / denom
    half = z * math.sqrt(mr * (1 - mr) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def score_with_timeout(work, ltol, stol, angle_tol, workers, per_task_timeout):
    """best-of-k match rate at one stol, with a per-crystal timeout (pathological
    StructureMatcher fits are counted as misses, never allowed to hang the sweep)."""
    import multiprocessing as mp
    fn = partial(_match_one, ltol=ltol, stol=stol, angle_tol=angle_tol)
    if workers <= 1:
        # serial fallback: still time-bounded via a single-worker pool below
        workers = 1
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers)
    asyncs = [pool.apply_async(fn, (w,)) for w in work]
    hits, n_timeout = [], 0
    for a in asyncs:
        try:
            hits.append(a.get(timeout=per_task_timeout))
        except mp.TimeoutError:
            hits.append(0)
            n_timeout += 1
    pool.terminate()   # kill any straggler still grinding on a pathological fit
    pool.join()
    return hits, n_timeout


@torch.no_grad()
def generate(args):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vpa = ck["vol_per_atom"]
    val_idx = ck["val_idx"]
    mcfg = ModelConfig(**ck["model_cfg"])
    device = resolve_device("auto")
    has_coset = mcfg.n_cosets > 0
    lat_repr = getattr(mcfg, "lattice_repr", "shape10")
    fam_mask = getattr(mcfg, "lattice_family_mask", False)
    lvstd = ck.get("prior_logvol_std", 0.3)
    dvstd = ck.get("prior_dev_std", 0.3)

    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    keep = [i for i, it in enumerate(rel) if species_multiplicity(it) >= 2]
    full.items = [rel[i] for i in keep]
    if has_coset:
        assign_symmetry_cosets(full.items)   # ids must match the trained model
    val_items = [full.items[i] for i in val_idx]
    if args.limit:
        val_items = val_items[:args.limit]

    print(f"[E4 gen] corpus {len(rel)} -> {len(keep)} multi-copy; val {len(val_items)}; "
          f"device {device}; k={args.match_k} steps={args.sampler_steps}; "
          f"n_cosets(model)={mcfg.n_cosets}", flush=True)
    if not has_coset:
        print("  NOTE: checkpoint has n_cosets=0 -> coset-ON pass is unavailable "
              "(de-novo only).", flush=True)

    model = SymMCFlow(mcfg).to(device).eval()
    model.load_state_dict(ck["model"])

    refs, gens_on, gens_off = [], [], []
    ori_on, ori_off = [], []
    torch.manual_seed(args.seed)
    t0 = time.time()
    for s in range(0, len(val_items), args.batch_size):
        chunk = val_items[s:s + args.batch_size]
        batch = move_batch(collate([{**it, "idx": torch.tensor(0)} for it in chunk]), device)
        z1 = batch_to_state(batch)
        B = z1.lattice.shape[0]
        mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        coset = batch.get("coset") if has_coset else None
        for b in range(B):
            refs.append(rigid_to_structure(
                z1.lattice[b], batch["Z"][b], batch["local"][b], z1.centroid[b],
                z1.orient[b], batch["atom_mask"][b], batch["mol_mask"][b]).as_dict())
        draws_on = [[] for _ in range(B)]
        draws_off = [[] for _ in range(B)]
        b_ori_on = torch.full((B,), 1e9); b_ori_off = torch.full((B,), 1e9)
        m = z1.mask.float()
        for _ in range(args.match_k):
            z0 = sample_prior(z1, vol_per_atom=vpa, sg=batch["sg"], lattice_repr=lat_repr,
                              family_mask=fam_mask, logvol_std=lvstd, dev_std=dvstd)
            # SAME prior draw feeds both passes
            # de-novo (no symmetry template) -- the MolCrystalFlow / MOFFlow regime
            samp0 = rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps,
                               coset=None)
            for b in range(B):
                draws_off[b].append(rigid_to_structure(
                    samp0.lattice[b], batch["Z"][b], batch["local"][b], samp0.centroid[b],
                    samp0.orient[b], batch["atom_mask"][b], batch["mol_mask"][b]).as_dict())
            oe0 = (M.so3_angle(samp0.orient, z1.orient) * m).sum(-1) / m.sum(-1).clamp_min(1)
            b_ori_off = torch.minimum(b_ori_off, (oe0 * RAD2DEG).cpu())
            if has_coset:
                samp1 = rk4_sample(model, mol_emb, z0, batch["sg"],
                                   steps=args.sampler_steps, coset=coset)
                for b in range(B):
                    draws_on[b].append(rigid_to_structure(
                        samp1.lattice[b], batch["Z"][b], batch["local"][b], samp1.centroid[b],
                        samp1.orient[b], batch["atom_mask"][b], batch["mol_mask"][b]).as_dict())
                oe1 = (M.so3_angle(samp1.orient, z1.orient) * m).sum(-1) / m.sum(-1).clamp_min(1)
                b_ori_on = torch.minimum(b_ori_on, (oe1 * RAD2DEG).cpu())
        gens_off.extend(draws_off)
        ori_off += b_ori_off.tolist()
        if has_coset:
            gens_on.extend(draws_on)
            ori_on += b_ori_on.tolist()
        print(f"  sampled {len(refs)}/{len(val_items)}  ({time.time()-t0:.0f}s)", flush=True)

    return {"refs": refs, "gens_off": gens_off, "gens_on": gens_on,
            "ori_off": ori_off, "ori_on": ori_on, "has_coset": has_coset,
            "match_k": args.match_k, "n_cosets": mcfg.n_cosets,
            "cache": args.cache, "ckpt": args.ckpt}


def report(blob, args):
    refs = blob["refs"]
    n = len(refs)
    stols = [float(x) for x in args.stols.split(",")]
    print(f"\n==== E4: MolCrystalFlow-aligned match@{blob['match_k']} "
          f"(ltol={args.ltol}, angle_tol={args.angle_tol}; n={n}) ====", flush=True)
    print("  protocol: pymatgen StructureMatcher, best-of-k, stol sweep -- mirrors "
          "MolCrystalFlow arXiv:2602.16020v3", flush=True)
    print("  CAVEAT: same matcher + same match@10 as MCF, but OUR corpus is a smaller "
          "CSD-derived\n          molecular set (not Thurlemann/OMC25) -> comparable-task, "
          "not strict head-to-head.", flush=True)
    print("  reference points (MCF paper): MolCrystalFlow ~6.8%@0.8 / ~8%@1.0; "
          "MOFFlow ~3.4%@0.8;\n          both symmetry-FREE. Genarris-3 (symmetry-conditioned) "
          "competitive with MCF.\n")

    conds = [("de-novo (coset OFF)  [vs MCF/MOFFlow]", blob["gens_off"], blob["ori_off"])]
    if blob["has_coset"]:
        conds.append(("conditioned (coset ON) [symmetry template]", blob["gens_on"], blob["ori_on"]))

    header = "  {:<42s}".format("condition") + "".join(f"  stol={s:<4g}" for s in stols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    results = {}
    for name, gens, ori in conds:
        work = list(zip(refs, gens))
        cells, cis = [], []
        for stol in stols:
            hits, n_to = score_with_timeout(work, args.ltol, stol, args.angle_tol,
                                            args.workers, args.timeout)
            mr = sum(hits) / max(len(hits), 1)
            lo, hi = wilson(mr, len(hits))
            cells.append(mr); cis.append((lo, hi))
            results[(name, stol)] = (mr, lo, hi, n_to)
            if n_to:
                print(f"    [stol={stol:g}] {n_to} crystal(s) hit the {args.timeout}s "
                      f"matcher timeout -> counted as misses", flush=True)
        row = "  {:<42s}".format(name) + "".join(f"  {100*c:6.1f}%" for c in cells)
        print(row)
        med_ori = sorted(ori)[len(ori) // 2] if ori else float("nan")
        print("  {:<42s}".format(f"    (median best-of-k orient err {med_ori:.1f} deg)"))

    # --- MCF's second axis: RMAD of unit-cell volume over all k samples ---------
    print("\n  -- cell-volume RMAD (over all k samples; MCF 3.86%, MOFFlow 18.8%, "
          "Genarris-3 raw 59%) --")
    rmads = {}
    for name, gens, ori in conds:
        m, sd = rmad(refs, gens)
        rmads[name] = (m, sd)
        print(f"  {name:<42s}  RMAD {m:6.2f} +/- {sd:.2f}%")

    # --- Phase F gate: crystal-family angle spike (reference ~72%, unmasked ~3%) ------------
    print("\n  -- cell-angle spike: fraction within 2deg of 90 (ref real ~72%; unmasked ~3%) --")
    ref_spike = angle_spike(refs)
    print(f"  {'reference (real crystals)':<42s}  {ref_spike:5.1f}%")
    spikes = {}
    for name, gens, ori in conds:
        sp = angle_spike(gens)
        spikes[name] = sp
        print(f"  {name:<42s}  {sp:5.1f}%")

    print()
    for name, (m, sd) in rmads.items():
        tag = "coset_on" if "ON" in name else "coset_off"
        print(f"  TAG e4 rmad {tag} {m:.2f} std {sd:.2f}")
        print(f"  TAG e4 anglespike {tag} {spikes[name]:.1f} ref {ref_spike:.1f}")
    for (name, stol), (mr, lo, hi, n_to) in results.items():
        print(f"  TAG e4 {('coset_on' if 'ON' in name else 'coset_off')} "
              f"stol {stol:g} match {100*mr:.2f} ci {100*lo:.2f}-{100*hi:.2f}"
              + (f" timeouts {n_to}" if n_to else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/coset_deploy_s0.pt")
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--match-k", type=int, default=10)          # MolCrystalFlow's k
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ltol", type=float, default=0.3)          # MCF ltol
    ap.add_argument("--angle-tol", type=float, default=10.0)    # MCF atol
    ap.add_argument("--stols", default="0.5,0.8,1.0,1.2")       # MCF site-tol sweep
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-crystal matcher timeout (s); a fit exceeding it is a miss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="gpu_results/phaseE4")
    ap.add_argument("--score-only", action="store_true",
                    help="skip generation; re-score the saved pickle across the stol sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    blob_path = os.path.join(args.out, "e4_draws.pkl")

    if args.score_only:
        with open(blob_path, "rb") as f:
            blob = pickle.load(f)
        print(f"[E4] re-scoring saved draws from {blob_path} "
              f"(n={len(blob['refs'])}, k={blob['match_k']})", flush=True)
    else:
        blob = generate(args)
        with open(blob_path, "wb") as f:
            pickle.dump(blob, f)
        print(f"[E4] saved {len(blob['refs'])} refs + draws -> {blob_path}", flush=True)

    report(blob, args)


if __name__ == "__main__":
    main()
