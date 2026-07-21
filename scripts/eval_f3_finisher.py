"""Phase F3a: does the rigid-press finisher unblock exact match?

Two modes, so the expensive GPU generation runs once and the CPU finisher can be tuned for free:

  # 1. GPU box: generate the family-mask model's draws and save the raw State tensors (tiny pickle)
  python scripts/eval_f3_finisher.py --gen-only --ckpt checkpoints/coset_fammask_s0.pt \
      --match-k 5 --out gpu_results/phaseF3

  # 2. anywhere (CPU): finish each draw + score RAW vs FINISHED, paired, no model needed
  python scripts/eval_f3_finisher.py --from-tensors gpu_results/phaseF3/f3_tensors.pkl \
      --finish-steps 60 [--no-relax-cell] [--limit 20]

Scores best-of-k match@k over the stol sweep, cell-volume RMAD, crystal-family angle-spike, and the
median best-of-k min-atom-RMSD to the reference. The finisher never sees the reference -- only the
generated draw + the space-group template that conditioned the sampler.

Pre-registered gate (tasks/phaseF3_spec.md): promote to F3b iff FINISHED coset-ON match@k at
stol <= 1.0 moves off 0 (CI-separated from RAW), OR the median min-atom-RMSD drops >= 30%.
"""
import argparse
import os
import pickle
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import eval_e4_molcrystalflow as E4   # cell_volume, rmad, angle_spike, wilson

STOLS = (0.5, 0.8, 1.0, 1.2)


# ------------------------------------------------------------------ generation (needs the model/GPU)
def generate(args):
    from symmc_flow.config import ModelConfig
    from symmc_flow.molcrystal import (MolCrystalDataset, relative_gauge_item, rigid_to_structure,
                                        species_multiplicity, assign_symmetry_cosets, _species_groups)
    from symmc_flow.model import SymMCFlow
    from symmc_flow.train import resolve_device, move_batch
    from symmc_flow.data import collate, batch_to_state
    from symmc_flow.flow import sample_prior
    from symmc_flow.sampler import rk4_sample
    from symmc_flow.rigid_press import symmetry_op_indices

    device = resolve_device("auto")
    ck = torch.load(args.ckpt, map_location=device)
    mcfg = ModelConfig(**ck["model_cfg"])
    lat_repr = getattr(mcfg, "lattice_repr", "shape10")
    fam_mask = getattr(mcfg, "lattice_family_mask", False)
    vpa = ck["vol_per_atom"]; lvstd = ck.get("prior_logvol_std", 0.3); dvstd = ck.get("prior_dev_std", 0.3)
    has_coset = mcfg.n_cosets > 0

    full = MolCrystalDataset(cache_path=args.cache)
    rel = [relative_gauge_item(full.items[i]) for i in range(len(full))]
    keep = [i for i, it in enumerate(rel) if species_multiplicity(it) >= 2]
    full.items = [rel[i] for i in keep]
    if has_coset:
        assign_symmetry_cosets(full.items)
    val_items = [full.items[i] for i in ck["val_idx"]]
    if args.limit:
        val_items = val_items[:args.limit]
    print(f"[F3 gen-only] val {len(val_items)}; device {device}; k={args.match_k}; "
          f"repr={lat_repr} fam_mask={fam_mask} coset={has_coset}", flush=True)

    model = SymMCFlow(mcfg).to(device).eval(); model.load_state_dict(ck["model"])
    torch.manual_seed(args.seed)
    crystals = []
    t0 = time.time()
    for s in range(0, len(val_items), args.batch_size):
        chunk = val_items[s:s + args.batch_size]
        batch = move_batch(collate([{**it, "idx": torch.tensor(0)} for it in chunk]), device)
        z1 = batch_to_state(batch); B = z1.lattice.shape[0]
        mol_emb = model.encode_molecules(batch["Z"], batch["local"], batch["atom_mask"])
        coset = batch.get("coset") if has_coset else None
        rec = []
        for b in range(B):
            gr = [[m for m in g if bool(batch["mol_mask"][b][m])] for g in _species_groups(chunk[b])]
            gr = [g for g in gr if g]
            oi = symmetry_op_indices(z1.centroid[b], int(batch["sg"][b]), gr)
            ref = rigid_to_structure(z1.lattice[b], batch["Z"][b], batch["local"][b],
                                     z1.centroid[b], z1.orient[b], batch["atom_mask"][b],
                                     batch["mol_mask"][b]).as_dict()
            rec.append({"ref": ref, "sg": int(batch["sg"][b]), "groups": gr, "op_idx": oi,
                        "local": batch["local"][b].cpu(), "Z": batch["Z"][b].cpu(),
                        "atom_mask": batch["atom_mask"][b].cpu(), "mol_mask": batch["mol_mask"][b].cpu(),
                        "draws": []})
        for _ in range(args.match_k):
            z0 = sample_prior(z1, vol_per_atom=vpa, sg=batch["sg"], lattice_repr=lat_repr,
                              family_mask=fam_mask, logvol_std=lvstd, dev_std=dvstd)
            samp = rk4_sample(model, mol_emb, z0, batch["sg"], steps=args.sampler_steps, coset=coset)
            for b in range(B):
                rec[b]["draws"].append((samp.lattice[b].cpu(), samp.centroid[b].cpu(),
                                        samp.orient[b].cpu()))
        crystals.extend(rec)
        print(f"  sampled {len(crystals)}/{len(val_items)}  ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "f3_tensors.pkl")
    torch.save({"crystals": crystals, "match_k": args.match_k, "lat_repr": lat_repr,
                "fam_mask": fam_mask, "vol_per_atom": float(vpa), "ckpt": args.ckpt}, path)
    print(f"[F3 gen-only] saved {len(crystals)} crystals x {args.match_k} draws -> {path}", flush=True)


# ------------------------------------------------------------------ finish + score (CPU, no model)
def _min_rms_S(sm, refS, draw_dicts):
    from pymatgen.core import Structure
    best = None
    for gd in draw_dicts:
        try:
            rd = sm.get_rms_dist(Structure.from_dict(gd), refS)
        except Exception:
            rd = None
        if rd is not None:
            best = rd[0] if best is None else min(best, rd[0])
    return best


def _finish_score_one(payload):
    """One crystal end-to-end (finish k draws + score vs ref). Runs in a worker process."""
    torch.set_num_threads(1)                               # avoid thread oversubscription in the pool
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from symmc_flow.molcrystal import rigid_to_structure
    from symmc_flow.rigid_press import finish_structure
    c, fk = payload
    Z, loc, am, mm = c["Z"], c["local"], c["atom_mask"], c["mol_mask"]
    raws, fins = [], []
    for (L, cen, ori) in c["draws"]:
        raws.append(rigid_to_structure(L, Z, loc, cen, ori, am, mm).as_dict())
        Lf, cf, Rf = finish_structure(L, cen, ori, loc, Z, am, mm, c["sg"],
                                      groups=c["groups"], op_idx=c["op_idx"], **fk)
        fins.append(rigid_to_structure(Lf, Z, loc, cf, Rf, am, mm).as_dict())
    refS = Structure.from_dict(c["ref"])
    res = {"ref": c["ref"], "raws": raws, "fins": fins}
    for stol in STOLS:
        sm = StructureMatcher(ltol=0.3, stol=stol, angle_tol=10)
        res[f"raw_{stol}"] = int(any(sm.fit(Structure.from_dict(g), refS) for g in raws))
        res[f"fin_{stol}"] = int(any(sm.fit(Structure.from_dict(g), refS) for g in fins))
    sm2 = StructureMatcher(ltol=0.5, stol=2.0, angle_tol=20)
    res["raw_rms"] = _min_rms_S(sm2, refS, raws)
    res["fin_rms"] = _min_rms_S(sm2, refS, fins)
    return res


def finish_and_score(args):
    import multiprocessing as mp
    blob = torch.load(args.from_tensors, map_location="cpu", weights_only=False)
    crystals = blob["crystals"]; k = blob["match_k"]; vpa = blob["vol_per_atom"]
    fammask_lat = blob["lat_repr"] == "logmetric6"
    if args.limit:
        crystals = crystals[:args.limit]
    relax_cell = not args.no_relax_cell
    fk = dict(steps=args.finish_steps, contact=args.contact, cutoff=args.cutoff,
              relax_cell=relax_cell, family_mask=fammask_lat, vol_per_atom=vpa)
    workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)
    print(f"[F3 finish] {len(crystals)} crystals x {k} draws; steps={args.finish_steps} "
          f"relax_cell={relax_cell} contact={args.contact} cutoff={args.cutoff}; workers={workers}",
          flush=True)

    payloads = [(c, fk) for c in crystals]
    t0 = time.time(); results = []
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, r in enumerate(pool.imap(_finish_score_one, payloads)):
                results.append(r)
                if (i + 1) % 10 == 0 or i + 1 == len(payloads):
                    print(f"  finished {i+1}/{len(payloads)}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        for i, p in enumerate(payloads):
            results.append(_finish_score_one(p))
            if (i + 1) % 10 == 0 or i + 1 == len(payloads):
                print(f"  finished {i+1}/{len(payloads)}  ({time.time()-t0:.0f}s)", flush=True)

    refs = [r["ref"] for r in results]
    raw_on = [r["raws"] for r in results]; fin_on = [r["fins"] for r in results]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "f3_scored.pkl"), "wb") as f:
        pickle.dump({"refs": refs, "raw_on": raw_on, "fin_on": fin_on, "match_k": k}, f)
    _report_agg(results, k)


def _report_agg(results, k):
    """Aggregate per-crystal worker results into the paired RAW-vs-FINISHED report."""
    n = len(results)
    raw_on = [r["raws"] for r in results]; fin_on = [r["fins"] for r in results]
    refs = [r["ref"] for r in results]
    print(f"\n==== F3a: RAW vs FINISHED (coset ON), n={n}, best-of-{k} ====")
    print(f"  {'stol':>6} {'RAW match':>11} {'FIN match':>11}   FIN 95% CI")
    for stol in STOLS:
        mr = sum(r[f"raw_{stol}"] for r in results) / max(n, 1)
        mf = sum(r[f"fin_{stol}"] for r in results) / max(n, 1)
        clo, chi = E4.wilson(mf, n)
        print(f"  {stol:>6} {100*mr:>9.1f}% {100*mf:>9.1f}%   {100*clo:.2f}-{100*chi:.2f}")
        print(f"  TAG f3 match stol {stol} raw {100*mr:.2f} fin {100*mf:.2f} ci {100*clo:.2f}-{100*chi:.2f}")
    r_raw, r_fin = E4.rmad(refs, raw_on), E4.rmad(refs, fin_on)
    as_raw, as_fin, as_ref = E4.angle_spike(raw_on), E4.angle_spike(fin_on), E4.angle_spike(refs)
    rr = [r["raw_rms"] for r in results if r["raw_rms"] is not None]
    rf = [r["fin_rms"] for r in results if r["fin_rms"] is not None]
    mr_ = statistics.median(rr) if rr else float("nan")
    mf_ = statistics.median(rf) if rf else float("nan")
    drop = f"{100*(1-mf_/mr_):+.1f}%" if rr and rf and mr_ else "n/a"
    print(f"\n  cell-vol RMAD   RAW {r_raw[0]:.2f}%   FIN {r_fin[0]:.2f}%")
    print(f"  angle-spike     RAW {as_raw:.1f}%   FIN {as_fin:.1f}%   (ref {as_ref:.1f}%)")
    print(f"  median min-RMSD RAW {mr_:.3f} (n={len(rr)})   FIN {mf_:.3f} (n={len(rf)})   -> {drop}")
    print(f"  TAG f3 rmad raw {r_raw[0]:.2f} fin {r_fin[0]:.2f}")
    print(f"  TAG f3 anglespike raw {as_raw:.1f} fin {as_fin:.1f} ref {as_ref:.1f}")
    print(f"  TAG f3 minrmsd raw {mr_:.3f} fin {mf_:.3f} drop {drop}")
    print("\nGATE: FIN match@stol<=1.0 off 0 (CI-separated) OR median min-RMSD drop >=30% -> F3b; else F4.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/coset_fammask_s0.pt")
    ap.add_argument("--cache", default="data/csd_mol/ds.pt")
    ap.add_argument("--gen-only", action="store_true", help="generate + save raw tensors, no finish")
    ap.add_argument("--from-tensors", default="", help="finish + score from a saved f3_tensors.pkl")
    ap.add_argument("--match-k", type=int, default=5)
    ap.add_argument("--sampler-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--finish-steps", type=int, default=60)
    ap.add_argument("--contact", type=float, default=0.90)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--no-relax-cell", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="finish+score processes (0 = cpu_count-1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gpu_results/phaseF3")
    args = ap.parse_args()
    if args.from_tensors:
        finish_and_score(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
