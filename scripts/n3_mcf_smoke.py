"""N3 / G3: smoke test of the MolCrystalFlow fork on exported pickles (protocol tasks/n3_protocol.md §3.2, §7 step 3).

Random-init weights; nothing here is a reported number, and no output is matched against a truth structure
(the P3 check compares output lattice/centroids with the INPUT truth tensors). Run on VAL or TRAIN only.
  1. configs: ours_molcrystal_{R,A} and ours_inference_{R,A} compose; num_atom_types 16, refit lattice prior,
     MCF-R overrides, step budget / patience, P6 keys off, gt_lattice_trans; per-call keys raise when unset;
     training without experiment.seed raises; the inference.py merge (launch config OVER checkpoint config)
     keeps our values and records the training seed / run name as null (they live in <ckpt_dir>/config.yaml)
  2. MCF's MCDataset loads the exported pickle (|set| entries); MCF's datamodule collates --n crystals
  3. FlowModule(cfg R and A): corrupt_batch (R exercises P2: trans.corrupt=False) + model_step -> finite loss
     and finite gradients after backward
  4. FlowModule.forward (the sampler) under ours_inference_R: P3 ON -> lattices / pred_trans equal the input
     lattice_1 / trans_1 bitwise; P3 OFF (control) -> they move
  5. --p6: P6 equivalence (box gate uses the same code without shims): the vectorised prior is bit-identical
     to _symmetric_so3 under a fixed numpy seed; scatter pooling vs the loop on 3 real TRAIN batches (1e-5)

    python scripts/n3_mcf_smoke.py --set val --n 4 --shims              # laptop (torch_scatter/cluster shims)
    python scripts/n3_mcf_smoke.py --set train --p6 --p6_batch 128 --device cuda   # box: P6 gate, no shims
"""
import argparse
import copy
import json
import os
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import torch

MCF = os.environ.get("N3_MCF", os.path.join(REPO, "external", "mcf"))
CFG_DIR = os.path.join(MCF, "molcrystalflow", "configs")
SHIMS = os.path.join(REPO, "external", "shims")
INFER_KEYS = ["inference.ckpt_path=/nonexistent/last.ckpt", "inference.output_dir=/nonexistent/out",
              "inference.inference_dir=/nonexistent/inf", "inference.seed=100000"]


def compose(name, overrides=()):
    from hydra import compose as hcompose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        return hcompose(config_name=name, overrides=list(overrides))


def check_configs(set_name):
    from omegaconf import OmegaConf
    from omegaconf.errors import MissingMandatoryValue, InterpolationToMissingValueError
    import yaml
    lat = yaml.safe_load(open(os.path.join(CFG_DIR, "ours_lattice.yaml")))["lattice"]["lognormal"]
    out = {}
    trained = {}
    for arm in ("R", "A"):
        c = compose(f"ours_molcrystal_{arm}", ["experiment.seed=0", f"experiment.wandb.name=n3_mcf{arm}_s0"])
        trained[arm] = c
        exp = c.experiment
        chk = {
            "num_atom_types_16": c.model.bb_embedder.num_atom_types == 16,
            "lattice_prior_refit": list(c.interpolant.lattice.lognormal.loc) == lat["loc"]
            and list(c.interpolant.lattice.lognormal.scale) == lat["scale"],
            "max_epochs_1715": exp.trainer.max_epochs == 1715,
            "patience_61": exp.lr_scheduler.patience == 61,
            "released_lr_sched": exp.lr_scheduler.factor == 0.6 and exp.lr_scheduler.min_lr == 1e-6
            and exp.optimizer.lr == 1e-4 and exp.checkpointer.save_top_k == 3 and exp.checkpointer.save_last,
            "p6_off": c.model.bb_embedder.p6_scatter_pooling is False
            and c.interpolant.rots.p6_vectorized_prior is False,
            "cache_dir": os.path.normpath(c.data.cache_dir) == os.path.normpath(os.path.join(REPO, "n3_mcf", "trainval")),
            "wandb_name": exp.wandb.name == f"n3_mcf{arm}_s0",
            "debug_off": exp.debug is False,
        }
        if arm == "R":
            chk["R_overrides"] = (c.interpolant.trans.corrupt is False and c.interpolant.lattice.corrupt is False
                                  and c.interpolant.rots.corrupt is True
                                  and exp.training.translation_loss_weight == 0 and exp.training.cell_loss_weight == 0)
        else:
            chk["A_released_objective"] = (c.interpolant.trans.corrupt and c.interpolant.lattice.corrupt
                                           and c.interpolant.rots.corrupt
                                           and exp.training.translation_loss_weight == 2.0
                                           and exp.training.cell_loss_weight == 0.1)
        # inference: launch config merged OVER the checkpoint config, exactly as inference.py:37-40
        knobs = ["data.cache_dir=" + os.path.join(REPO, "n3_mcf", "valsel_as_test").replace("\\", "/"),
                 "interpolant.rots.exp_rate=3"] + (["interpolant.trans.scaling=9"] if arm == "A" else []) \
            + (["++inference.gt_lattice_trans=true"] if arm == "R" else [])
        li = compose(f"ours_inference_{arm}", INFER_KEYS + knobs)
        ck = OmegaConf.create(OmegaConf.to_container(c))                    # the saved config.yaml
        OmegaConf.set_struct(li, False)
        m = OmegaConf.merge(ck, li)
        chk["infer_merge"] = (m.model.bb_embedder.num_atom_types == 16
                              and list(m.interpolant.lattice.lognormal.loc) == lat["loc"]
                              and m.data.cache_dir.endswith("valsel_as_test") and m.interpolant.rots.exp_rate == 3
                              and m.interpolant.trans.scaling == 9 and m.interpolant.sampling.num_timesteps == 50
                              and m.interpolant.rots.sample_schedule == "exp" and m.inference.num_samples == 16
                              and m.inference.save_trajectories is False and m.data.loader.num_workers == 4
                              and m.inference.gt_lattice_trans is (arm == "R"))
        bare = OmegaConf.merge(ck, compose(f"ours_inference_{arm}"))
        raised = 0
        for key in ("inference.ckpt_path", "data.cache_dir", "interpolant.rots.exp_rate", "inference.seed"):
            try:
                OmegaConf.select(bare, key, throw_on_missing=True)
            except (MissingMandatoryValue, InterpolationToMissingValueError):
                raised += 1
        chk["unset_keys_raise"] = raised == 4
        # experiment.seed is mandatory for training (the released default 123 would train silently as s123)
        bare_t = compose(f"ours_molcrystal_{arm}")
        raised = 0
        for key in ("experiment.seed", "experiment.wandb.name"):
            try:
                OmegaConf.select(bare_t, key, throw_on_missing=True)
            except (MissingMandatoryValue, InterpolationToMissingValueError):
                raised += 1
        chk["train_seed_mandatory"] = raised == 2
        # the merged inference config (saved as inference.yaml) never carries a stale training seed / run name:
        # both are null there, and the checkpoint's config.yaml keeps its own (seed 2 with the default run name)
        ck2 = OmegaConf.create(OmegaConf.to_container(compose(f"ours_molcrystal_{arm}", ["experiment.seed=2"])))
        m2 = OmegaConf.merge(ck2, li)
        chk["infer_no_stale_train_seed"] = (m.experiment.seed is None and m.experiment.wandb.name is None
                                            and m2.experiment.seed is None and m2.experiment.wandb.name is None
                                            and ck2.experiment.seed == 2
                                            and ck2.experiment.wandb.name == f"n3_mcf{arm}_s2")
        out[arm] = chk
    return out, trained


def batch_of(cfg, set_name, n):
    from molcrystalflow.data.dataloader import MCDatamodule
    from molcrystalflow.data.dataset import MCDataset
    split = {"train": "train", "val": "val"}[set_name]
    ds = MCDataset(cache_path=os.path.join(cfg.data.cache_dir, f"{split}.pt"), dataset_cfg=cfg.data,
                   is_training=False)
    c = copy.deepcopy(cfg)
    c.data.loader.num_workers = 0
    c.data.loader.batch_size.train = n
    dl = MCDatamodule(data_cfg=c.data, train_dataset=ds, valid_dataset=None).train_dataloader(shuffle=False)
    return ds, dl


def train_step(cfg, batch, seed=0):
    from pytorch_lightning import seed_everything
    from molcrystalflow.models.molcrystalflow import FlowModule
    seed_everything(seed, verbose=False)
    mod = FlowModule(cfg)
    mod.interpolant.set_device(batch.trans_1.device)
    noisy = mod.interpolant.corrupt_batch(batch)
    losses = mod.model_step(noisy)
    total = torch.mean(losses["se3_vf_loss"]) + torch.mean(losses["cell_loss"])
    total.backward()
    g = [p.grad for p in mod.model.parameters() if p.grad is not None]
    return mod, {"loss": float(total), "loss_finite": bool(torch.isfinite(total)),
                 "grads_finite": bool(all(torch.isfinite(x).all() for x in g)), "n_grad_tensors": len(g),
                 "b_trans_zero": bool((noisy["b_trans"] == 0).all())}


def sample(cfg_inf, state, batch, gt, steps=4, samples=2):
    from molcrystalflow.models.molcrystalflow import FlowModule
    from omegaconf import OmegaConf
    c = copy.deepcopy(cfg_inf)
    OmegaConf.set_struct(c, False)
    c.inference.gt_lattice_trans = gt
    c.inference.num_samples = samples
    c.interpolant.sampling.num_timesteps = steps
    mod = FlowModule(c)
    mod.load_state_dict(state)
    mod.eval()
    np.random.seed(7)
    torch.manual_seed(7)
    with torch.no_grad():
        r = mod.forward(copy.deepcopy(batch))
    L1, T1 = batch.lattice_1, batch.trans_1
    R = r["pred_rotmats"].double()
    orth = (R @ R.transpose(-1, -2) - torch.eye(3, dtype=torch.float64)).abs().max().item()
    return {"lattice_eq_truth": all(torch.equal(r["lattices"][s], L1) for s in range(samples)),
            "trans_eq_truth": all(torch.equal(r["pred_trans"][s], T1) for s in range(samples)),
            "max_lattice_move": float((r["lattices"] - L1).abs().max()),
            "rotmats_finite": bool(torch.isfinite(R).all()), "rotmats_orth_err": orth,
            "cart_shape": list(r["cart_coords"].shape)}


def p6_check(cfg, set_name, batch_size, device, n_batches=3):
    from molcrystalflow.data.interpolant import _symmetric_so3, _symmetric_so3_vec
    from molcrystalflow.models.molcrystalflow import FlowModule
    from pytorch_lightning import seed_everything
    ds, dl = batch_of(cfg, set_name, batch_size)
    seed_everything(0, verbose=False)
    mod = FlowModule(cfg).to(device)
    pool = mod.model.bb_embedder.attention_pooling
    grabbed = {}
    h = pool.register_forward_pre_hook(lambda m, args: grabbed.update(args=[a.detach().clone() for a in args]))
    res = {"prior_bit_identical": [], "pool_out_maxabs": [], "pool_grad_h_maxabs": [], "pool_grad_x_maxabs": []}
    for b, batch in enumerate(dl):
        if b == n_batches:
            break
        batch = batch.to(device)
        for seed in (0, 1, 2):
            np.random.seed(seed)
            a = _symmetric_so3(int(batch.num_bbs.sum()), batch.num_bbs, device)
            np.random.seed(seed)
            v = _symmetric_so3_vec(int(batch.num_bbs.sum()), batch.num_bbs, device)
            res["prior_bit_identical"].append(bool(torch.equal(a, v)))
        with torch.no_grad():
            mod.model.bb_embedder(batch)
        hf, xc, bi = grabbed["args"]
        w = torch.randn(int(bi.max()) + 1, hf.shape[1], device=device, generator=None)
        outs, grads = [], []
        for flag in (False, True):
            pool.p6_scatter = flag
            h_ = hf.clone().requires_grad_(True)
            x_ = xc.clone().requires_grad_(True)
            o = pool(h_, x_, bi)
            (o * w).sum().backward()
            outs.append(o.detach())
            grads.append((h_.grad.detach(), x_.grad.detach()))
        pool.p6_scatter = False
        res["pool_out_maxabs"].append(float((outs[0] - outs[1]).abs().max()))
        res["pool_grad_h_maxabs"].append(float((grads[0][0] - grads[1][0]).abs().max()))
        res["pool_grad_x_maxabs"].append(float((grads[0][1] - grads[1][1]).abs().max()))
        # relative to the reference's scale: float32 sums of |out| ~ 1e2 cannot agree to 1e-5 absolute
        res.setdefault("pool_out_maxrel", []).append(float((outs[0] - outs[1]).abs().max() / outs[0].abs().max()))
        res.setdefault("pool_grad_h_maxrel", []).append(
            float((grads[0][0] - grads[1][0]).abs().max() / grads[0][0].abs().max()))
        res.setdefault("pool_grad_x_maxrel", []).append(
            float((grads[0][1] - grads[1][1]).abs().max() / grads[0][1].abs().max()))
        res.setdefault("pool_out_absmax", []).append(float(outs[0].abs().max()))
    h.remove()
    res["prior_pass"] = all(res["prior_bit_identical"])
    res["pool_pass_abs_1e-5"] = max(res["pool_out_maxabs"] + res["pool_grad_h_maxabs"] + res["pool_grad_x_maxabs"]) <= 1e-5
    res["pool_pass_rel_1e-5"] = max(res["pool_out_maxrel"] + res["pool_grad_h_maxrel"] + res["pool_grad_x_maxrel"]) <= 1e-5
    res["batch_size"], res["device"] = batch_size, str(device)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="val", choices=["val", "train"])
    ap.add_argument("--n", type=int, default=4, help="crystals per smoke batch (2-4)")
    ap.add_argument("--shims", action="store_true", help="laptop only: shim torch_scatter / torch_cluster")
    ap.add_argument("--p6", action="store_true")
    ap.add_argument("--p6_batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.environ.setdefault("N3_ROOT", REPO)
    sys.path.insert(0, MCF)
    t0 = time.time()
    rep = {"set": args.set, "shims": []}
    if args.shims:   # load by path: SHIMS joins sys.path only if torch_scatter is really missing
        import importlib.util
        spec = importlib.util.spec_from_file_location("n3_mcf_shims", os.path.join(SHIMS, "n3_mcf_shims.py"))
        shims = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shims)
        rep["shims"] = shims.install()
    rep["configs"], cfgs = check_configs(args.set)
    ds, dl = batch_of(cfgs["R"], args.set, args.n)
    batch = next(iter(dl))
    rep["dataset"] = {"len": len(ds), "batch_crystals": int(batch.num_graphs),
                      "batch_copies": batch.num_bbs.tolist(), "batch_atoms": int(batch.local_coords.shape[0]),
                      "atom_types_max": int(batch.atom_types.max()),
                      "features": {k: list(getattr(batch, k).shape) for k in
                                   ("basic_features", "chemical_features", "geometric_features")}}
    modR, rep["train_step_R"] = train_step(cfgs["R"], copy.deepcopy(batch))
    _, rep["train_step_A"] = train_step(cfgs["A"], copy.deepcopy(batch))
    from omegaconf import OmegaConf
    infR = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(cfgs["R"])), compose("ours_inference_R", INFER_KEYS + [
        "data.cache_dir=" + cfgs["R"].data.cache_dir.replace("\\", "/"), "interpolant.rots.exp_rate=3",
        "++inference.gt_lattice_trans=true"]))           # launch config OVER the checkpoint config, as inference.py
    state = modR.state_dict()
    rep["sample_R_P3_on"] = sample(infR, state, batch, True)
    rep["sample_R_P3_off"] = sample(infR, state, batch, False)
    if args.p6:
        rep["p6"] = p6_check(cfgs["R"], args.set, args.p6_batch, torch.device(args.device))
    ok = (all(all(v.values()) for v in rep["configs"].values())
          and rep["dataset"]["len"] == {"val": 100, "train": 1687}[args.set]
          and rep["train_step_R"]["loss_finite"] and rep["train_step_R"]["grads_finite"]
          and rep["train_step_R"]["b_trans_zero"]
          and rep["train_step_A"]["loss_finite"] and rep["train_step_A"]["grads_finite"]
          and rep["sample_R_P3_on"]["lattice_eq_truth"] and rep["sample_R_P3_on"]["trans_eq_truth"]
          and not rep["sample_R_P3_off"]["lattice_eq_truth"] and rep["sample_R_P3_on"]["rotmats_finite"])
    rep["pass"] = bool(ok)
    rep["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(rep, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
