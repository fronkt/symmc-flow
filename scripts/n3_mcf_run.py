"""N3 / G3: MolCrystalFlow run harness (tasks/n3_protocol.md §3.2 (ii-R), (ii-A), "Sampling harness",
"Fixed-draw VAL loss", §2.3 RESID for MCF, §7 steps 5 and 7 for MCF).

Subcommands (box = Linux GPU box of §7 step 5/7; the harness prints every command, --run executes it):
  train           MCF-R / MCF-A training on the fixed TRAIN/VAL export: our fork's train.py with
                  --config-name=ours_molcrystal_{R,A} and exactly the protocol overrides on the command line
                  (R: trans/lattice corrupt=False, translation/cell loss weight 0; both: max_epochs=1715,
                  lr_scheduler.patience=61, experiment.seed=s, experiment.wandb.name=n3_mcf{R,A}_s{s}),
                  WANDB_MODE=offline, one visible GPU. Pre-flight (also without --run): the composed config
                  holds every value; the TRAIN / VAL pickle SHA-256 equal their sidecars; the run's ckpt dir holds
                  no checkpoint yet (a box death restarts the run from scratch, §7 Failures).
  train_report    LR trace, valid/loss curve and the step of the 9th LR reduction (floor 1.008e-6) from the run's
                  offline wandb directory (§3.2 "Reported per seed"). Only the NEWEST offline run with the name
                  counts (a box death restarts from scratch, §7 Failures); older ones are listed as discarded.
                  Per checkpoint in the run's ckpt dir: epoch and global_step (read from the file) and the logged
                  valid/loss of that validation epoch (also for last.ckpt, whose name carries no loss).
  fixed_val_loss  the tie-break loss: model_step over VAL x 8 draws inside torch.random.fork_rng() with
                  torch.manual_seed(7) and np.random.seed(7) (numpy state saved and restored), per checkpoint.
  infer           one sampling call (§3.2 "Sampling harness"): --config-name=ours_inference_{R,A}, ckpt,
                  data.cache_dir, num_samples=16, num_timesteps=50, sample_schedule=exp, exp_rate=<s_uR>,
                  trans.scaling=9 (R) or <s_uF> (A), ++inference.gt_lattice_trans=true (R only), unique
                  output_dir / inference_dir, save_trajectories=False, data.loader.num_workers=4, inference.seed
                  = 100000 (R) / 200000 (A) + 1000 s + split (split 0 VALSEL, 1 TEST-B, 2 DEV-TEST; s = the
                  training seed read from <ckpt_dir>/config.yaml). --run then runs the RESID pass
                  (scripts/n3_mcf_resid.py) and `check` in the same environment. A run dir that already holds
                  predictions is refused; --discard moves it to n3_mcf/infer/_discarded (never scored).
                  GENERATOR ERRORS (§1.3: a MISS for that crystal only; inference.py samples the whole set in one
                  call, so one crystal's exception, e.g. openfold's eigh on a NaN rotation, aborts every crystal):
                  a failed call is retried once in a fresh process at the same seed; if it fails again the set is
                  sampled in chunks of 25 crystals (sidecar order; chunk j at inference.seed + 1,000,000 (j+1),
                  same overrides, its own data/output dirs under <run>/fallback/), and a chunk that fails twice is
                  sampled one crystal at a time (crystal q of chunk j at inference.seed + 1,000,000 (j+1) +
                  1,000 (q+1), two attempts). A crystal that fails both attempts is a generator error: every draw
                  is NaN in the merged predictions_<S>.pt (arm failure, MISS). If no chunk or crystal call
                  succeeds the run stops (a harness failure, not generator errors). Recorded in n3_run.json and
                  <run>/fallback/fallback.json.
  check           assertions on a predictions file and its resid file before scoring: S draws, |set| crystals
                  in sidecar order (lattice_1, num_atoms, num_bbs equal the pickle's; frozen order, or on the box
                  the committed lists and Gate F's recorded sidecar SHA-256), clamp exact for MCF-R on every finite
                  draw (lattice == lattice_1, trans == trans_1 mod 1, atol 1e-5), the resolved inference.yaml (and
                  every fallback call's) holds the harness values, resid_<S>.pt is [S, |set|], finite on every
                  finite draw and NaN on the non-finite ones (arm failures), recomputes from its saved rotations
                  -> <run>/check.json.
  to_bundle       predictions + resid + sidecar -> n3_common draw bundle, kind 'cells' (whole molecules, species
                  through the extended inverse map), sel 'lj' (steric LJ via the Gate-F5 adapter; MCF-A / floor
                  cells whose generated cell L_gen, before the F5 supercell, has a height < 2.5 A get +inf) and
                  'resid'. valid = finite lattice and coordinates (non-finite draws and generator errors are arm
                  failures).
  prior_floor     MCF-A prior floor (§3.2 ii-A): refit lognormal lattice + uniform centroids + symmetric-prior
                  rotations, 16 draws per crystal, torch/np seed 300000 + 1000 s + split; same file layout.
  smoke_init      laptop smoke only: a VAL subset exported as test_* (+ sidecar) and a random-init MCF-R
                  checkpoint dir (config.yaml + last.ckpt) under n3_mcf/_smoke.

    python scripts/n3_mcf_run.py train --arm R --seed 0                    # print (and pre-flight)
    python scripts/n3_mcf_run.py train --arm R --seed 0 --run --gpu 0      # box
    python scripts/n3_mcf_run.py train_report --arm R --seed 0
    python scripts/n3_mcf_run.py fixed_val_loss --ckpt n3_mcf/runs/n3_mcfR_s0/ckpt/*.ckpt --device cuda
    python scripts/n3_mcf_run.py infer --ckpt <ckpt> --set valsel --s_uR 3 --run --gpu 0
    python scripts/n3_mcf_run.py infer --ckpt <ckpt> --set valsel --s_uR 3 --s_uF 9 --run --gpu 0   # MCF-A
    python scripts/n3_mcf_run.py check --run_dir n3_mcf/infer/mcfR/valsel/s0_last_u3
    python scripts/n3_mcf_run.py to_bundle --run_dir n3_mcf/infer/mcfR/valsel/s0_last_u3 --workers 8
    python scripts/n3_mcf_run.py prior_floor --seed 0 --set valsel
    # laptop smoke (shims; VAL only; 2 crystals x 2 draws x 5 steps; CPU emulation of inference.py)
    python scripts/n3_mcf_run.py smoke_init --shims
    python scripts/n3_mcf_run.py infer --ckpt n3_mcf/_smoke/ckpt_R/last.ckpt --set n3_mcf/_smoke/val2 --s_uR 3 \
        --smoke --num_samples 2 --num_timesteps 5 --num_workers 1 --run --cpu_emulate --shims
    # generator-error fallback smoke: the emulated call raises for any set holding that (VAL) refcode
    N3_SMOKE_FAIL_REFCODE=<refcode of val2 #1> python scripts/n3_mcf_run.py infer <same as above> --label_extra fbtest
"""
import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import torch

import n3_common as nc
import n3_mcf_lib as ML

PROTOCOL_INFER = {"num_samples": 16, "num_timesteps": 50, "num_workers": 4}
TRAIN_EPOCHS, TRAIN_PATIENCE = 1715, 61
LR_FLOOR_STEPS = 9            # 1e-4 * 0.6^9 = 1.008e-6: the 9th reduction reaches the effective floor
SMOKE_ROOT = os.path.join(REPO, "n3_mcf", "_smoke")


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def env_for(gpu=None):
    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"
    env["N3_ROOT"] = REPO
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def require_one_gpu(env):
    v = env.get("CUDA_VISIBLE_DEVICES")
    if v is None or v.strip() == "" or "," in v:
        raise SystemExit("exactly one visible GPU is required (--gpu N or CUDA_VISIBLE_DEVICES=N): "
                         "inference.py's Trainer(strategy='ddp', devices='auto') would shard the set")


def shell_line(env_keys, env, cmd):
    return " ".join(f"{k}={env[k]}" for k in env_keys if k in env) + " " + " ".join(cmd)


# ============================================================================================ train
def train_overrides(arm, seed):
    ov = []
    if arm == "R":
        ov += ["interpolant.trans.corrupt=False", "interpolant.lattice.corrupt=False",
               "experiment.training.translation_loss_weight=0", "experiment.training.cell_loss_weight=0"]
    ov += [f"experiment.seed={seed}", f"experiment.trainer.max_epochs={TRAIN_EPOCHS}",
           f"experiment.lr_scheduler.patience={TRAIN_PATIENCE}", f"experiment.wandb.name=n3_mcf{arm}_s{seed}"]
    return ov


def preflight_train(arm, seed, ov):
    """The composed training config holds the protocol values (§3.2 ii-R / ii-A)."""
    import yaml
    c = ML.compose(f"ours_molcrystal_{arm}", ov)
    e = c.experiment
    lat = yaml.safe_load(open(os.path.join(ML.CFG_DIR, "ours_lattice.yaml")))["lattice"]["lognormal"]
    chk = {"max_epochs_1715": e.trainer.max_epochs == TRAIN_EPOCHS,
           "patience_61": e.lr_scheduler.patience == TRAIN_PATIENCE,
           "released_scheduler": e.lr_scheduler.factor == 0.6 and e.lr_scheduler.min_lr == 1e-6
           and e.use_lr_scheduler is True and e.optimizer.lr == 1e-4,
           "released_trainer": e.trainer.check_val_every_n_epoch == 1 and e.trainer.gradient_clip_val == 0.5
           and e.trainer.accumulate_grad_batches == 1,
           "checkpointer_top3_last": e.checkpointer.save_top_k == 3 and e.checkpointer.save_last is True
           and e.checkpointer.monitor == "valid/loss",
           "batch_128": c.data.loader.batch_size.train == 128,
           "debug_off": e.debug is False, "seed": e.seed == seed,
           "wandb_name": e.wandb.name == f"n3_mcf{arm}_s{seed}",
           "num_atom_types_16": c.model.bb_embedder.num_atom_types == 16,
           "lattice_prior_refit": list(c.interpolant.lattice.lognormal.loc) == lat["loc"]
           and list(c.interpolant.lattice.lognormal.scale) == lat["scale"],
           "cache_dir_trainval": os.path.normpath(c.data.cache_dir) == os.path.normpath(
               os.path.join(REPO, "n3_mcf", "trainval")),
           "p6_off": c.model.bb_embedder.p6_scatter_pooling is False and c.interpolant.rots.p6_vectorized_prior is False}
    if arm == "R":
        chk["R_overrides"] = (c.interpolant.trans.corrupt is False and c.interpolant.lattice.corrupt is False
                              and c.interpolant.rots.corrupt is True and e.training.translation_loss_weight == 0
                              and e.training.cell_loss_weight == 0)
    else:
        chk["A_released_objective"] = (c.interpolant.trans.corrupt is True and c.interpolant.lattice.corrupt is True
                                       and c.interpolant.rots.corrupt is True
                                       and e.training.translation_loss_weight == 2.0 and e.training.cell_loss_weight == 0.1)
    return c, chk


def cmd_train(args):
    import n3_mcf_export as X
    ML.setup_mcf(args.shims)
    ov = train_overrides(args.arm, args.seed)
    cfg, chk = preflight_train(args.arm, args.seed, ov)
    for s in ("train", "val"):
        X.load_export(s)                               # asserts pickle SHA-256 == sidecar
        chk[f"{s}_pickle_sha_ok"] = True
    ckpt_dir = cfg.paths.ckpt_dir
    existing = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    chk["ckpt_dir_empty"] = not existing
    env = env_for(args.gpu)
    cmd = [sys.executable, "molcrystalflow/experiments/train.py", f"--config-name=ours_molcrystal_{args.arm}"] + ov
    print(json.dumps(chk))
    print(f"cd {fwd(ML.MCF)}\n" + shell_line(["WANDB_MODE", "N3_ROOT", "CUDA_VISIBLE_DEVICES"], env,
                                             ["python"] + cmd[1:]))
    bad = [k for k, v in chk.items() if not v and k != "ckpt_dir_empty"]
    if bad:
        raise SystemExit(f"pre-flight failed: {bad}")
    if not args.run:
        return
    if existing and not args.force:
        raise SystemExit(f"{ckpt_dir} already holds {len(existing)} checkpoints: a restarted run starts from scratch "
                         "(move the old run away first, or --force)")
    require_one_gpu(env)
    log = os.path.join(REPO, "n3_mcf", "runs", f"n3_mcf{args.arm}_s{args.seed}_train.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rec = {"cmd": cmd, "overrides": ov, "preflight": chk, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
           "versions": ML.lib_versions(), "ckpt_dir": ckpt_dir}
    ML.atomic_json(rec, log[:-4] + ".json", indent=1)
    t0 = time.time()
    with open(log, "a") as fh:
        r = subprocess.run(cmd, cwd=ML.MCF, env=env, stdout=fh, stderr=subprocess.STDOUT)
    rec.update(returncode=r.returncode, hours=round((time.time() - t0) / 3600, 3),
               checkpoints=sorted(os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, "*.ckpt"))))
    ML.atomic_json(rec, log[:-4] + ".json", indent=1)
    print(json.dumps({k: rec[k] for k in ("returncode", "hours", "checkpoints")}))
    if r.returncode:
        raise SystemExit(r.returncode)


# ============================================================================================ train_report
def read_wandb_offline(run_dir):
    """(run display names, history rows) of every offline wandb run file (.wandb) under run_dir, read with
    wandb's own datastore (offline runs keep their name and config only inside the .wandb records)."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore
    names, rows = set(), []
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "*.wandb"), recursive=True)):
        ds = datastore.DataStore()
        ds.open_for_scan(path)
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = wandb_internal_pb2.Record()
            rec.ParseFromString(data)
            kind = rec.WhichOneof("record_type")
            if kind == "run" and rec.run.display_name:
                names.add(rec.run.display_name)
            if kind != "history":
                continue
            row = {}
            for it in rec.history.item:
                key = it.key if it.key else "/".join(it.nested_key)
                try:
                    row[key] = json.loads(it.value_json)
                except ValueError:
                    row[key] = it.value_json
            rows.append(row)
    return names, rows


def ckpt_epoch_step(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return int(ck.get("epoch", -1)), int(ck.get("global_step", -1))


def filename_loss(path):
    base = os.path.basename(path)
    if "loss_" not in base:
        return None
    try:
        return float(base.split("loss_")[1].rsplit(".ckpt", 1)[0])
    except ValueError:
        return None


def cmd_train_report(args):
    name = f"n3_mcf{args.arm}_s{args.seed}"
    wdir = args.wandb_dir or os.path.join(REPO, "n3_mcf", "runs")
    runs = sorted(glob.glob(os.path.join(wdir, "wandb", "offline-run-*")) + glob.glob(os.path.join(wdir, "offline-run-*")),
                  key=lambda d: os.path.basename(d))           # offline-run-YYYYMMDD_HHMMSS-<id>: time order
    mine = [d for d in runs if name in read_wandb_offline(d)[0]]
    if args.run_dirs:
        mine = args.run_dirs
    if not mine:
        raise SystemExit(f"no offline wandb run of {name} under {wdir}")
    used, discarded = mine[-1], mine[:-1]                        # a restart from scratch supersedes older runs
    rows = read_wandb_offline(used)[1]
    lr_key = next((k for r in rows for k in r if k.startswith("lr-")), None)
    lr = [(int(r["trainer/global_step"]), float(r[lr_key])) for r in rows if lr_key in r and "trainer/global_step" in r]
    vl = [(int(r["trainer/global_step"]), int(r.get("epoch", -1)), float(r["valid/loss"]))
          for r in rows if "valid/loss" in r and "trainer/global_step" in r]
    lr.sort()
    vl.sort()
    red, prev = [], None
    for step, v in lr:
        if prev is not None and v < prev * (1 - 1e-9):
            red.append({"step": step, "lr": v})
        prev = v
    rel = lambda d: os.path.relpath(d, REPO).replace("\\", "/")
    ckdir = args.ckpt_dir or os.path.join(REPO, "n3_mcf", "runs", name, "ckpt")
    cks = []
    for p in sorted(glob.glob(os.path.join(ckdir, "*.ckpt"))):
        ep, gs = ckpt_epoch_step(p)
        at = [(s_, v) for s_, e, v in vl if e == ep]
        logged = at[-1][1] if at else None
        fl = filename_loss(p)
        cks.append({"ckpt": os.path.basename(p), "ckpt_sha256": ML.sha256(p), "epoch": ep, "global_step": gs,
                    "logged_valid_loss": logged, "logged_at_step": at[-1][0] if at else None,
                    "logged_step_equals_ckpt_step": bool(at and at[-1][0] == gs),
                    "filename_loss": fl, "filename_loss_agrees": None if fl is None or logged is None
                    else abs(fl - logged) <= 5.1e-5})
    out = {"run": name, "wandb_dir": rel(used), "discarded_wandb_dirs": [rel(d) for d in discarded],
           "lr_key": lr_key, "lr_trace": lr, "valid_loss": vl, "lr_reductions": red,
           "step_9th_reduction": red[LR_FLOOR_STEPS - 1]["step"] if len(red) >= LR_FLOOR_STEPS else "not reached",
           "n_history_rows": len(rows), "ckpt_dir": rel(ckdir), "checkpoints": cks}
    path = args.out or os.path.join(ML.PRIV, "train_report", f"{name}.json")
    ML.atomic_json(out, path, indent=1)
    print(json.dumps({k: out[k] for k in ("run", "lr_key", "n_history_rows", "step_9th_reduction")}),
          f"reductions {len(red)}, valid/loss points {len(vl)}, discarded runs {len(discarded)}, "
          f"checkpoints {len(cks)} -> {path}")


# ============================================================================================ fixed_val_loss
def fixed_val_loss(ckpt, device="cpu", draws=8, seed=7, limit=None):
    """§3.2: model_step over VAL x draws, inside fork_rng with torch.manual_seed(7) / np.random.seed(7); the
    numpy state is saved and restored. Batches are MCF's val_dataloader batches (unshuffled, batch_size.valid);
    the loss is valid/loss = mean(se3_vf_loss) + mean(cell_loss) per batch, averaged weighted by batch size
    (Lightning's epoch reduction) over all draws."""
    from molcrystalflow.data.dataset import MCDataset
    from torch_geometric.loader import DataLoader
    cfg = ML.ckpt_config(ckpt)
    with torch.random.fork_rng():   # module construction (random init, then overwritten) and the loader draw RNG
        mod = ML.load_flow_module(ckpt, cfg, device)
        ds = MCDataset(cache_path=os.path.join(cfg.data.cache_dir, "val.pt"), dataset_cfg=cfg.data, is_training=False)
        if limit:                               # laptop smoke only (the shimmed radius_graph is dense O(N^2))
            ds = [ds[i] for i in range(limit)]
        batches = list(DataLoader(ds, batch_size=cfg.data.loader.batch_size.valid, shuffle=False, num_workers=0))
    mod.interpolant.set_device(torch.device(device))
    devs = [torch.device(device)] if str(device).startswith("cuda") else []
    tot, n, per_draw = 0.0, 0, []
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(seed)
            np.random.seed(seed)
            with torch.no_grad():
                for d in range(draws):
                    dt, dn = 0.0, 0
                    for b in batches:
                        b = b.to(device)
                        noisy = mod.interpolant.corrupt_batch(b)
                        losses = mod.model_step(noisy)
                        v = float(torch.mean(losses["se3_vf_loss"]) + torch.mean(losses["cell_loss"]))
                        dt += v * int(b.num_graphs)
                        dn += int(b.num_graphs)
                    per_draw.append(dt / dn)
                    tot += dt
                    n += dn
    finally:
        np.random.set_state(np_state)
    return {"fixed_val_loss": tot / n, "per_draw": per_draw, "n_val": len(ds), "draws": draws, "seed": seed,
            "val_pickle": os.path.join(cfg.data.cache_dir, "val_molcrystal_normalized.pkl.gz")}


def cmd_fixed_val_loss(args):
    ML.setup_mcf(args.shims)
    out = args.out or os.path.join(ML.SMOKE_DIR if (args.shims or args.limit) else ML.PRIV, "fixed_val_loss.jsonl")
    import n3_harness as H
    done = {(r["ckpt_sha256"], r["device"]) for r in H.read_jsonl(out)}
    for ck in args.ckpt:
        info = ML.ckpt_info(ck)
        if (info["ckpt_sha256"], args.device) in done and not args.force:
            print(f"{ck}: already in {out}")
            continue
        t0 = time.time()
        r = fixed_val_loss(ck, args.device, limit=args.limit)
        if args.limit:
            r["smoke_limit"] = args.limit
        logged = filename_loss(ck)          # None for last.ckpt: train_report gives it by (epoch, global_step)
        ep, gs = ckpt_epoch_step(ck)
        row = {**info, **r, "logged_valid_loss": logged, "epoch": ep, "global_step": gs, "device": args.device,
               "torch_threads": torch.get_num_threads(), "shims": ML._STATE["shims"], "sec": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{info['arm']} s{info['seed']} {info['label']}: fixed-draw VAL loss {r['fixed_val_loss']:.6f} "
              f"(logged {logged})  [{row['sec']}s]")


# ============================================================================================ infer
def infer_plan(args):
    info = ML.ckpt_info(args.ckpt)
    arm = info["arm"]
    smoke = bool(args.smoke)
    if arm == "A" and args.s_uF is None:
        raise SystemExit("MCF-A needs --s_uF (interpolant.trans.scaling)")
    if arm == "R" and args.s_uF not in (None, 9, 9.0):
        raise SystemExit("MCF-R samples with interpolant.trans.scaling=9 (protocol); --s_uF is MCF-A only")
    if info["seed"] is None:
        raise SystemExit(f"{args.ckpt}: training seed missing from <ckpt_dir>/config.yaml")
    if not smoke and info["run_name"] != f"n3_mcf{arm}_s{info['seed']}":
        raise SystemExit(f"{args.ckpt}: run name {info['run_name']!r} is not n3_mcf{arm}_s{info['seed']}")
    if smoke:
        set_name, split, cache = os.path.abspath(args.set), 0, ML.cache_dir(args.set)
    else:
        if args.set not in ML.SPLIT:
            raise SystemExit(f"--set must be one of {list(ML.SPLIT)}")
        set_name, split, cache = args.set, ML.SPLIT[args.set], ML.cache_dir(args.set)
    S = args.num_samples if smoke and args.num_samples else PROTOCOL_INFER["num_samples"]
    T = args.num_timesteps if smoke and args.num_timesteps else PROTOCOL_INFER["num_timesteps"]
    W = args.num_workers if smoke and args.num_workers else PROTOCOL_INFER["num_workers"]
    seed = ML.inference_seed(arm, info["seed"], split)
    s_uF = 9 if arm == "R" else args.s_uF
    label = ML.run_label(info, "smoke" if smoke else set_name, args.s_uR, None if arm == "R" else s_uF,
                         extra=args.label_extra)
    run_dir = os.path.join(ML.INFER_ROOT if not smoke else os.path.join(SMOKE_ROOT, "infer"), ML.ARM_NAME[arm],
                           "smoke" if smoke else set_name, label)
    rp = ML.run_paths(run_dir, S)
    ov = [f"inference.ckpt_path={fwd(args.ckpt)}", f"data.cache_dir={fwd(cache)}", f"inference.num_samples={S}",
          f"interpolant.sampling.num_timesteps={T}", "interpolant.rots.sample_schedule=exp",
          f"interpolant.rots.exp_rate={args.s_uR:g}", f"interpolant.trans.scaling={s_uF:g}"]
    if arm == "R":
        ov.append("++inference.gt_lattice_trans=true")
    ov += [f"inference.output_dir={fwd(rp['out'])}", f"inference.inference_dir={fwd(rp['hydra'])}",
           "inference.save_trajectories=False", f"data.loader.num_workers={W}", f"inference.seed={seed}"]
    cmd = [sys.executable, "molcrystalflow/experiments/inference.py", f"--config-name=ours_inference_{arm}"] + ov
    man = {"arm": arm, "arm_name": ML.ARM_NAME[arm], "set": set_name, "split": split, "train_seed": info["seed"],
           "run_name": info["run_name"], "ckpt": info["ckpt"], "ckpt_sha256": info["ckpt_sha256"],
           "ckpt_label": info["label"], "s_uR": args.s_uR, "s_uF": s_uF, "inference_seed": seed,
           "num_samples": S, "num_timesteps": T, "num_workers": W, "cache_dir": fwd(cache), "overrides": ov,
           "config_name": f"ours_inference_{arm}", "smoke": smoke, "label": label,
           "protocol": "tasks/n3_protocol.md §3.2 Sampling harness"}
    return man, rp, cmd


def emulate_inference(config_name, overrides):
    """LAPTOP SMOKE ONLY: inference.py's EvalRunner (same Hydra composition, config merge, seeding, dataset,
    dataloader, predict and _combine_predictions) with Lightning on CPU instead of Trainer(gpu, ddp) and the
    module on 'cpu' instead of device 0. Never used for a reported run."""
    import pytorch_lightning as pl
    import molcrystalflow.experiments.inference as inf
    from molcrystalflow.models.molcrystalflow import FlowModule

    class _CPUFlowModule(FlowModule):
        def to(self, *a, **k):
            if a and isinstance(a[0], int):
                a = ("cpu",) + a[1:]
            return super().to(*a, **k)

    def _cpu_trainer(**_):   # LightningEnvironment: skip cluster detection (a broken local mpi4py raises there)
        from pytorch_lightning.plugins.environments import LightningEnvironment
        return pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                          enable_model_summary=False, plugins=[LightningEnvironment()])
    saved = inf.Trainer, inf.FlowModule
    inf.Trainer, inf.FlowModule = _cpu_trainer, _CPUFlowModule
    try:
        cfg = ML.compose(config_name, overrides)
        runner = inf.EvalRunner(cfg)                     # saves hydra/inference.yaml, as on the box
        fail = os.environ.get("N3_SMOKE_FAIL_REFCODE")   # smoke test of the generator-error fallback only
        if fail:
            for fn in ("test_sidecar.json", "n3_fallback_part.json"):
                p = os.path.join(cfg.data.cache_dir, fn)
                if os.path.exists(p) and fail in json.load(open(p))["refcodes"]:
                    raise RuntimeError(f"injected generator error on {fail} (smoke test of the fallback)")
        runner.run_sampling()
    finally:
        inf.Trainer, inf.FlowModule = saved


# ---------------------------------------------------------------------------------- generator-error fallback
FALLBACK_CHUNK = 25
FALLBACK_SEED_CHUNK, FALLBACK_SEED_CRYSTAL = 1_000_000, 1_000
PRED_KEYS = ("cart_coords", "atom_types", "lattices", "num_atoms", "pred_trans", "pred_rotmats", "num_bbs")
GT_KEYS = ("gt_coords", "atom_types", "lattice_1", "trans_1", "rotmats_1", "num_atoms")


def _log_tail(path, n=12):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])[-1500:]
    except OSError:
        return None


def call_inference(config_name, overrides, env, log_path, emulate):
    """One sampling call: inference.py in a fresh process (box), or EvalRunner in-process (CPU smoke).
    Returns (ok, error text)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if emulate:
        import traceback
        try:
            emulate_inference(config_name, overrides)
            return True, None
        except Exception as ex:  # noqa: BLE001 -- a generator error of the smoke model
            with open(log_path, "a") as fh:
                fh.write(traceback.format_exc())
            return False, f"{type(ex).__name__}: {ex}"[:500]
    cmd = [sys.executable, "molcrystalflow/experiments/inference.py", f"--config-name={config_name}"] + overrides
    with open(log_path, "a") as fh:
        r = subprocess.run(cmd, cwd=ML.MCF, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode == 0, (None if r.returncode == 0 else f"returncode {r.returncode}:\n{_log_tail(log_path)}")


def with_overrides(ov, **repl):
    """The call's overrides with data.cache_dir / inference.output_dir / inference.inference_dir /
    inference.seed replaced (every other harness value unchanged)."""
    keys = {"cache_dir": "data.cache_dir", "output_dir": "inference.output_dir",
            "inference_dir": "inference.inference_dir", "seed": "inference.seed"}
    out = list(ov)
    for k, v in repl.items():
        pre = keys[k] + "="
        hit = [j for j, o in enumerate(out) if o.startswith(pre)]
        assert len(hit) == 1, (k, hit)
        out[hit[0]] = pre + (fwd(v) if k != "seed" else str(int(v)))
    return out


def fallback_sampling(man, rp, entries, side, env, emulate, chunk=FALLBACK_CHUNK):
    """Chunked, then per-crystal sampling after a twice-failed full call (module docstring); writes the merged
    predictions_<S>.pt (generator-error crystals as NaN draws) and <run>/fallback/fallback.json."""
    import n3_mcf_export as X
    fb = os.path.join(rp["dir"], "fallback")
    seed0, S, n = int(man["inference_seed"]), man["num_samples"], len(entries)
    parts = []

    def run_part(name, idx, seed):
        d = os.path.join(fb, name)
        data = os.path.join(d, "data")
        os.makedirs(data, exist_ok=True)
        pk = os.path.join(data, "test_molcrystal_normalized.pkl.gz")
        X.save_pkl_gz([entries[i] for i in idx], pk)
        ML.atomic_json({"refcodes": [side["refcodes"][i] for i in idx], "indices": idx, "parent_pickle_sha256":
                        man["pickle_sha256"], "sha256": ML.sha256(pk)}, os.path.join(data, "n3_fallback_part.json"))
        ov = with_overrides(man["overrides"], cache_dir=data, output_dir=os.path.join(d, "out"),
                            inference_dir=os.path.join(d, "hydra"), seed=seed)
        rec = {"name": name, "indices": idx, "seed": seed, "cache_dir": fwd(data), "overrides": ov,
               "pickle_sha256": ML.sha256(pk), "attempts": []}
        for _ in range(2):
            ok, err = call_inference(man["config_name"], ov, env, os.path.join(d, "run.log"), emulate)
            rec["attempts"].append({"ok": ok, "error": err})
            if ok:
                break
        rec["ok"] = rec["attempts"][-1]["ok"]
        pred = os.path.join(d, "out", f"predictions_{S}.pt")
        rec["predictions_sha256"] = ML.sha256(pred) if rec["ok"] else None
        print(f"  fallback {name} ({len(idx)} crystals, seed {seed}): {'ok' if rec['ok'] else 'FAILED twice'}",
              flush=True)
        return rec
    for j, lo in enumerate(range(0, n, chunk)):
        idx = list(range(lo, min(lo + chunk, n)))
        base = seed0 + FALLBACK_SEED_CHUNK * (j + 1)
        rec = run_part(f"c{j:03d}", idx, base)
        parts.append(rec)
        if not rec["ok"]:
            rec["split"] = True
            for q, i in enumerate(idx):
                parts.append(run_part(f"c{j:03d}_i{i:04d}", [i], base + FALLBACK_SEED_CRYSTAL * (q + 1)))
    used = [p for p in parts if p["ok"]]
    if not used:
        ML.atomic_json({"parts": parts, "status": "harness failure: no call succeeded"},
                       os.path.join(fb, "fallback.json"), indent=1)
        raise SystemExit("every fallback call failed: a harness failure, not generator errors (see fallback/*/run.log)")
    errors = sorted(p["indices"][0] for p in parts if not p["ok"] and not p.get("split"))
    merged = merge_parts(man, rp, parts, errors)
    ML.atomic_torch(merged, rp["pred"])
    rec = {"rule": f"chunks of {chunk} crystals in sidecar order at inference.seed + {FALLBACK_SEED_CHUNK} (j+1); "
                   f"a chunk failing twice -> per crystal at + {FALLBACK_SEED_CRYSTAL} (q+1); a crystal failing "
                   "twice = generator error (NaN draws, MISS)", "chunk": chunk, "parts": parts,
           "used_parts": [p["name"] for p in parts if p["ok"]],
           "generator_error_indices": errors, "generator_error_refcodes": [side["refcodes"][i] for i in errors],
           "merged_predictions_sha256": ML.sha256(rp["pred"])}
    ML.atomic_json(rec, os.path.join(fb, "fallback.json"), indent=1)
    return rec


def merge_parts(man, rp, parts, errors):
    """predictions_<S>.pt in sidecar order from the successful parts; a generator-error crystal gets NaN
    coordinates, lattice, centroids and rotations with its own atom types / counts and gt fields (MCDataset of
    its one-crystal pickle, as inference.py would batch it)."""
    from omegaconf import OmegaConf
    from molcrystalflow.data.dataset import MCDataset
    from torch_geometric.data import Batch
    S = man["num_samples"]
    cfg = OmegaConf.load(rp["config"])                 # the full call's merged config (written before it failed)
    pieces = []
    for p in parts:
        if p["ok"]:
            pieces.append((p["indices"][0], ML.load_predictions(os.path.join(rp["dir"], "fallback", p["name"], "out",
                                                                             f"predictions_{S}.pt"))))
    ref = pieces[0][1]
    for i in errors:
        name = next(p["name"] for p in parts if p["indices"] == [i])
        data = os.path.join(rp["dir"], "fallback", name, "data")
        b = Batch.from_data_list([MCDataset(cache_path=os.path.join(data, "test.pt"), dataset_cfg=cfg.data,
                                            is_training=False)[0]])
        na, nb = int(b.num_atoms.sum()), int(b.num_bbs.sum())
        nan = lambda shape, like: torch.full(shape, float("nan"), dtype=like.dtype)
        rep = lambda v, like: v.reshape(1, -1).expand(S, -1).clone().to(like.dtype)
        pl = {"cart_coords": nan((S, na, 3), ref["cart_coords"]), "atom_types": rep(b.atom_types, ref["atom_types"]),
              "lattices": nan((S, 1, 3, 3), ref["lattices"]), "num_atoms": rep(b.num_atoms, ref["num_atoms"]),
              "pred_trans": nan((S, nb, 3), ref["pred_trans"]), "pred_rotmats": nan((S, nb, 3, 3), ref["pred_rotmats"]),
              "num_bbs": rep(b.num_bbs, ref["num_bbs"]),
              "gt_data_batch": {k: getattr(b, k).to(ref["gt_data_batch"][k].dtype) for k in GT_KEYS}}
        pieces.append((i, pl))
    pieces.sort(key=lambda t: t[0])
    out = {k: torch.cat([p[k].cpu() for _, p in pieces], dim=1) for k in PRED_KEYS}
    out["gt_data_batch"] = {k: torch.cat([p["gt_data_batch"][k].cpu() for _, p in pieces], dim=0) for k in GT_KEYS}
    out["mc_trajs"] = None
    return out


def cmd_infer(args):
    ML.setup_mcf(args.shims)
    man, rp, cmd = infer_plan(args)
    env = env_for(args.gpu)
    print(f"cd {fwd(ML.MCF)}\n" + shell_line(["CUDA_VISIBLE_DEVICES", "N3_ROOT", "WANDB_MODE"], env,
                                             ["python"] + cmd[1:]))
    print(f"python scripts/n3_mcf_resid.py --run_dir {fwd(rp['dir'])} --device {'cpu' if args.cpu_emulate else 'cuda'}")
    print(f"python scripts/n3_mcf_run.py check --run_dir {fwd(rp['dir'])}")
    if not args.run:
        return
    if os.path.exists(rp["dir"]):
        if os.path.exists(rp["pred"]) or os.path.exists(rp["resid"]) or os.path.exists(rp["hydra"]):
            if not args.discard:
                raise SystemExit(f"{rp['dir']} already holds a run (a failing pair is discarded and regenerated, "
                                 "never scored): --discard moves it to n3_mcf/infer/_discarded")
            dst = os.path.join(os.path.dirname(ML.INFER_ROOT) if not man["smoke"] else SMOKE_ROOT, "infer",
                               "_discarded", f"{man['arm_name']}_{os.path.basename(str(man['set']))}_{man['label']}_"
                               + time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(rp["dir"], dst)
            print(f"discarded the previous run -> {dst}")
    os.makedirs(rp["dir"], exist_ok=True)
    entries, side = ML.export_set(man["set"])           # pickle SHA-256 == sidecar (§3.2 Integrity), before sampling
    man.update(pickle_sha256=side["sha256"], n=len(entries), sidecar_refcodes_sha256=_sha_list(side["refcodes"]),
               command=cmd, versions=ML.lib_versions(), shims=ML._STATE["shims"], cpu_emulate=bool(args.cpu_emulate),
               start=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.cpu_emulate and not man["smoke"]:
        raise SystemExit("--cpu_emulate is for the laptop smoke only (--smoke)")
    ML.atomic_json(man, rp["manifest"], indent=1)
    t0 = time.time()
    if not args.cpu_emulate:
        require_one_gpu(env)
    man["attempts"] = []
    for _ in range(2):                          # the full call; a failure is retried once in a fresh process
        ok, err = call_inference(man["config_name"], man["overrides"], env, rp["log"], args.cpu_emulate)
        man["attempts"].append({"call": "full", "ok": ok, "error": err})
        ML.atomic_json(man, rp["manifest"], indent=1)
        if ok:
            break
        print(f"inference call failed: {(err or '').splitlines()[0] if err else ''}", flush=True)
    if not ok:
        if not os.path.exists(rp["config"]):
            raise SystemExit(f"inference.py failed before writing its config (not a generator error); see {rp['log']}")
        print("the full call failed twice: generator-error fallback (chunks, then single crystals)", flush=True)
        fbr = fallback_sampling(man, rp, entries, side, env, args.cpu_emulate,
                                chunk=args.fallback_chunk if man["smoke"] and args.fallback_chunk else FALLBACK_CHUNK)
        man["fallback"] = {k: fbr[k] for k in ("rule", "chunk", "used_parts", "generator_error_indices",
                                                "generator_error_refcodes", "merged_predictions_sha256")}
    man["sampling_sec"] = round(time.time() - t0, 1)
    t1 = time.time()
    if args.cpu_emulate:
        import n3_mcf_resid
        n3_mcf_resid.resid_pass(rp["dir"], device="cpu")
    else:
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "n3_mcf_resid.py"), "--run_dir", rp["dir"],
                            "--device", "cuda"], env=env)
        if r.returncode:
            raise SystemExit(f"RESID pass failed ({r.returncode})")
    man["resid_sec"] = round(time.time() - t1, 1)
    ML.atomic_json(man, rp["manifest"], indent=1)
    res = run_check(rp["dir"])
    print(json.dumps({"run_dir": fwd(rp["dir"]), "check_pass": res["pass"], "sampling_sec": man["sampling_sec"],
                      "resid_sec": man["resid_sec"]}))
    if not res["pass"]:
        raise SystemExit(f"check FAILED: {res['failed']} (discard and regenerate; never score)")


def _sha_list(xs):
    import hashlib
    return hashlib.sha256("\n".join(xs).encode()).hexdigest()


# ============================================================================================ check
def _close(a, b, atol):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return a.shape == b.shape and bool(((a - b).abs() <= atol).all())


def config_checks(cfg, man, S, cache_dir, seed):
    """The resolved inference.yaml of one sampling call holds the harness values."""
    import yaml
    arm = man["arm"]
    lat = yaml.safe_load(open(os.path.join(ML.CFG_DIR, "ours_lattice.yaml")))["lattice"]["lognormal"]
    want_scaling = 9.0 if arm == "R" else float(man["s_uF"])
    cc = {"num_samples": cfg.inference.num_samples == S,
          "num_timesteps": cfg.interpolant.sampling.num_timesteps == man["num_timesteps"],
          "sample_schedule_exp": cfg.interpolant.rots.sample_schedule == "exp",
          "exp_rate": float(cfg.interpolant.rots.exp_rate) == float(man["s_uR"]),
          "trans_scaling": float(cfg.interpolant.trans.scaling) == want_scaling,
          "gt_lattice_trans": bool(cfg.inference.get("gt_lattice_trans", False)) is (arm == "R"),
          "save_trajectories_off": cfg.inference.save_trajectories is False,
          "num_workers": cfg.data.loader.num_workers == man["num_workers"],
          "inference_seed": cfg.inference.seed == seed,
          "cache_dir": fwd(cfg.data.cache_dir) == fwd(cache_dir),
          "ckpt_path": fwd(cfg.inference.ckpt_path) == fwd(man["ckpt"]),
          "num_atom_types_16": cfg.model.bb_embedder.num_atom_types == 16,
          "lattice_prior_refit": list(cfg.interpolant.lattice.lognormal.loc) == lat["loc"]
          and list(cfg.interpolant.lattice.lognormal.scale) == lat["scale"],
          "rots_prior_symmetric": cfg.interpolant.rots.prior_type == "symmetric"}
    if not man.get("smoke"):
        cc["protocol_values"] = (S == PROTOCOL_INFER["num_samples"] and man["num_timesteps"] == 50
                                 and man["num_workers"] == 4)
    return cc


def run_check(run_dir):
    """Every §3.2 assertion on (predictions, resid, resolved config) of one run -> <run>/check.json. Non-finite
    draws (generator errors of the fallback included) are arm failures (§1.3): the clamp and RESID finiteness
    are asserted on every finite draw, and RESID must be NaN on the others."""
    from omegaconf import OmegaConf
    import n3_mcf_resid as RS
    man, rp = ML.load_manifest(run_dir)
    S, arm = man["num_samples"], man["arm"]
    c = {}
    entries, side = ML.export_set(man["set"])
    c["pickle_sha_equals_sidecar_and_manifest"] = side["sha256"] == man["pickle_sha256"]
    n = len(entries)
    c["n_equals_sidecar"] = n == side["n"] == len(side["refcodes"])
    if man["set"] in ML.SPLIT and not man.get("smoke"):
        c.update(ML.frozen_order_checks(man["set"], side))
    c["ckpt_sha_unchanged"] = ML.sha256(man["ckpt"]) == man["ckpt_sha256"]
    pred = ML.load_predictions(rp["pred"])
    na, nb = pred["num_atoms"], pred["num_bbs"]
    c["S_draws"] = tuple(pred["cart_coords"].shape[:1]) == (S,) and tuple(na.shape) == (S, n) and tuple(nb.shape) == (S, n)
    g = pred["gt_data_batch"]
    L1 = torch.stack([e["lattice_1"] for e in entries])
    T1 = torch.cat([e["trans_1"] for e in entries])
    c["lattice_1_equals_pickle"] = _close(g["lattice_1"], L1, 1e-6)
    c["num_atoms_equals_pickle"] = all(int(na[s][b]) == int(entries[b]["local_coords"].shape[0])
                                       for s in range(S) for b in range(n)) and _close(g["num_atoms"], na[0], 0)
    c["num_bbs_equals_pickle"] = all(int(nb[s][b]) == int(entries[b]["bb_num_vec"].shape[0])
                                     for s in range(S) for b in range(n))
    c["trans_1_equals_pickle"] = _close(g["trans_1"], T1, 1e-6)
    c["atom_types_in_0_16"] = bool(((pred["atom_types"] >= 0) & (pred["atom_types"] < 16)).all())
    fin = ML.draw_finite(pred)                                         # [S, n]
    miss = [side["refcodes"][i] for i in range(n) if not bool(fin[:, i].any())]
    info = {"nonfinite_draws": int((~fin).sum()), "crystals_without_finite_draw": len(miss),
            "arm_miss_refcodes": miss}
    fb = man.get("fallback")
    if fb:
        info["generator_error_refcodes"] = fb["generator_error_refcodes"]
        c["generator_errors_are_nonfinite"] = all(not bool(fin[:, i].any()) for i in fb["generator_error_indices"])
    if arm == "R":   # clamp exact (P3), on every finite draw
        bb_c = torch.repeat_interleave(torch.arange(n), nb[0].long())
        dl = max((float((pred["lattices"][s][fin[s]].double() - L1[fin[s]].double()).abs().max())
                  for s in range(S) if bool(fin[s].any())), default=0.0)
        dt = max((float((((pred["pred_trans"][s][fin[s][bb_c]].double() - T1[fin[s][bb_c]].double()) + 0.5) % 1.0
                          - 0.5).abs().max()) for s in range(S) if bool(fin[s].any())), default=0.0)
        info.update(max_lattice_dev=dl, max_trans_dev_mod1=dt)
        c["clamp_lattice_atol_1e-5"] = dl <= 1e-5
        c["clamp_trans_mod1_atol_1e-5"] = dt <= 1e-5
    # resolved config (the merged inference.yaml inference.py saved; and every fallback call's)
    if arm in ("R", "A"):
        cfg = OmegaConf.load(rp["config"])
        cc = config_checks(cfg, man, S, man["cache_dir"], man["inference_seed"])
        if fb:
            fbj = json.load(open(os.path.join(run_dir, "fallback", "fallback.json")))
            c["fallback_merged_predictions_sha"] = fbj["merged_predictions_sha256"] == ML.sha256(rp["pred"])
            for p in fbj["parts"]:
                if not p["ok"]:
                    continue
                d = os.path.join(run_dir, "fallback", p["name"])
                pc = config_checks(OmegaConf.load(os.path.join(d, "hydra", "inference.yaml")), man, S,
                                   p["cache_dir"], p["seed"])
                pred_p = os.path.join(d, "out", f"predictions_{S}.pt")
                pc["predictions_unchanged"] = ML.sha256(pred_p) == p["predictions_sha256"]
                bad = [k for k, v in pc.items() if not v]
                c[f"fallback_{p['name']}_config"] = not bad
        info["p6"] = {"p6_vectorized_prior": bool(cfg.interpolant.rots.get("p6_vectorized_prior", False)),
                      "p6_scatter_pooling": bool(cfg.model.bb_embedder.get("p6_scatter_pooling", False))}
        c.update({f"config_{k}": v for k, v in cc.items()})
        rs = torch.load(rp["resid"], map_location="cpu", weights_only=False)
        r = rs["resid"]
        c["resid_shape"] = tuple(r.shape) == (S, n)
        c["resid_state_finite_equals_predictions"] = "state_finite" in rs and torch.equal(rs["state_finite"], fin)
        c["resid_finite_on_finite_draws"] = bool(torch.isfinite(r[fin]).all())
        c["resid_nan_on_nonfinite_draws"] = bool(torch.isnan(r[~fin]).all())
        rec = torch.stack([RS.resid_from_rotmats(rs["pred_rotmats_resid"][s], pred["pred_rotmats"][s], nb[s])
                           for s in range(S)]) if "pred_rotmats_resid" in rs else None
        c["resid_recomputes_from_saved_rotmats"] = rec is not None and bool(
            ((rec[fin] - r[fin]).abs() <= 1e-9).all())
        c["resid_matches_predictions"] = rs.get("predictions_sha256") == ML.sha256(rp["pred"])
        info["branch_cut_draws"] = int((rs["branch_cut_edges"] > 0).sum()) if "branch_cut_edges" in rs else None
    c = {k: bool(v) for k, v in c.items()}
    res = {"run_dir": fwd(run_dir), "pass": all(c.values()), "failed": [k for k, v in c.items() if not v],
           "checks": c, "info": info, "predictions_sha256": ML.sha256(rp["pred"]),
           "resid_sha256": ML.sha256(rp["resid"]) if os.path.exists(rp["resid"]) else None,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    ML.atomic_json(res, rp["check"], indent=1)
    return res


def cmd_check(args):
    ML.setup_mcf(args.shims)
    res = run_check(args.run_dir)
    print(json.dumps({k: res[k] for k in ("pass", "failed", "info")}))
    if not res["pass"]:
        raise SystemExit(1)


# ============================================================================================ to_bundle
def bundle_arm(man):
    if man["arm"] == "floor":
        return "mcfAfloor"
    base = f"{man['arm_name']}_{man['ckpt_label']}_u{man['s_uR']:g}"
    return base + (f"_f{man['s_uF']:g}" if man["arm"] == "A" else "")


def smoke_subset(man):
    if not man.get("smoke"):
        return None, man["set"]
    side = json.load(open(os.path.join(man["cache_dir"], "test_sidecar.json")))
    return side["val_indices"], side["n3_set"]


def cmd_to_bundle(args):
    import n3_mcf_export as X
    man, rp = ML.load_manifest(args.run_dir)
    chk = (run_check_floor if man["arm"] == "floor" else run_check)(args.run_dir)
    if not chk["pass"]:
        raise SystemExit(f"check FAILED {chk['failed']}: this run is discarded and regenerated, never scored")
    entries, side = ML.export_set(man["set"])
    pred = ML.load_predictions(rp["pred"])
    S, n = man["num_samples"], len(entries)
    resid = (torch.load(rp["resid"], map_location="cpu", weights_only=False)["resid"].double()
             if os.path.exists(rp["resid"]) else torch.full((S, n), float("nan"), dtype=torch.float64))
    cells = [[None] * S for _ in range(n)]
    for s in range(S):
        for b, c in enumerate(X.predictions_cells(pred, s)):
            assert (c["species"] >= 0).all(), "atom type outside the extended inverse map"
            cells[b][s] = c
    valid = torch.tensor([[bool(torch.isfinite(cells[i][s]["lattice"]).all() and torch.isfinite(cells[i][s]["cart"]).all())
                           for s in range(S)] for i in range(n)])
    generated = man["arm"] in ("A", "floor")
    Ms = [it["M"] for it in side["items"]]
    tasks = [((i, s), (ML.cell_payload(cells[i][s]), Ms[i], generated)) for i in range(n) for s in range(S)
             if valid[i, s]]
    t0 = time.time()
    rows = ML.run_jobs(ML.lj_task, tasks, rp["lj"], workers=args.workers, row=lambda r: {"lj": r[0], "why": r[1]},
                       log_path=rp["lj"][:-6] + ".events.jsonl", label="LJ (F5 adapter)",
                       params={"predictions_sha256": chk["predictions_sha256"], "pickle_sha256": side["sha256"],
                               "generated_cell_height_rule": generated})
    lj = torch.full((n, S), float("nan"), dtype=torch.float64)
    counts = {"ok": 0, "spacing_inf": 0, "det_fail": 0, "nonfinite": 0}
    for (i, s), _ in tasks:
        r = rows[(i, s)]
        if r["status"] == "ok":
            lj[i, s] = float(r["lj"])
            counts["spacing_inf" if r["why"] == "spacing" else ("nonfinite" if r["why"] == "nonfinite" else "ok")] += 1
        else:
            lj[i, s] = float("inf")
            counts["det_fail"] += 1
    sub, set_name = smoke_subset(man)
    arm = args.arm or bundle_arm(man)
    seed = man["train_seed"]
    refcodes = side["refcodes"]
    meta = {"source": "n3_mcf_run.py to_bundle", "run": {k: man[k] for k in man if k not in ("command",)},
            "check": {"pass": chk["pass"], "predictions_sha256": chk["predictions_sha256"],
                      "resid_sha256": chk["resid_sha256"]},
            "pickle_sha256": side["sha256"], "lj_counts": counts, "lj_evaluations_per_crystal": S,
            "lj_rule": "g2_asym_baselines.energy via n3_mcf_export.mcf_cells_to_supercell (Gate F5 adapter)"
                       + ("; generated cell L_gen (before the F5 supercell) with a height V/|a_j x a_k| < 2.5 A -> +inf"
                          if generated else ""),
            "resid": f"geodesic(pred_rotmats at t={ML.RESID_T} forward, returned pred_rotmats), mean over copies; "
                     f"angle {ML.GEODESIC}; NaN on non-finite draws",
            "arm_failures": {"nonfinite_draws": int((~valid).sum()),
                             "crystals_without_valid_draw": [refcode for refcode, v in zip(side["refcodes"], valid)
                                                             if not bool(v.any())],
                             "generator_error_refcodes": (man.get("fallback") or {}).get("generator_error_refcodes", [])},
            "z_MCF": [it["z_MCF"] for it in side["items"]], "M": Ms,
            "lj_sec": round(time.time() - t0, 1), "versions": ML.lib_versions()}
    if sub is not None:
        meta["smoke_subset"] = sub
    out = args.out or (nc.bundle_path(arm, seed, set_name) if sub is None
                       else os.path.join(ML.SMOKE_DIR, "draws", f"{arm}_s{seed}_{set_name}.pt"))
    ML.save_bundle(out, arm=arm, seed=seed, set_name=set_name, refcodes=refcodes, kind="cells", valid=valid,
                   sel={"lj": lj, "resid": resid.T.contiguous()}, meta=meta, cells=cells, subset=sub is not None)
    print(json.dumps({"bundle": fwd(out), "arm": arm, "seed": seed, "set": set_name, "n": n, "S": S,
                      "valid": int(valid.sum()), "lj_counts": counts, "sec": meta["lj_sec"]}))


# ============================================================================================ prior_floor
def cmd_prior_floor(args):
    """MCF-A prior floor: FlowModule.forward's prior draws (per test batch, per sample: trans_0 uniform,
    rotmats_0 symmetric, lattice lengths lognormal (refit) + angles uniform), assembled with MCF's own
    _assemble_coords, no model; seeds torch.manual_seed / np.random.seed(300000 + 1000 s + split)."""
    ML.setup_mcf(args.shims)
    from molcrystalflow.data.dataset import MCDataset
    from molcrystalflow.data.interpolant import Interpolant, _symmetric_so3, _uniform_cell_trans
    from molcrystalflow.data.utils import lattice6_to_mat33
    smoke = bool(args.smoke)
    set_name = os.path.abspath(args.set) if smoke else args.set
    split = 0 if smoke else ML.SPLIT[args.set]
    S = args.num_samples if smoke and args.num_samples else 16
    cfg = ML.compose("ours_molcrystal_A", [f"experiment.seed={args.seed}"])
    cache = ML.cache_dir(set_name)
    run_dir = os.path.join(ML.INFER_ROOT if not smoke else os.path.join(SMOKE_ROOT, "infer"), "mcfA_floor",
                           "smoke" if smoke else set_name, f"s{args.seed}")
    rp = ML.run_paths(run_dir, S)
    if os.path.exists(rp["pred"]) and not args.force:
        raise SystemExit(f"{rp['pred']} exists")
    entries, side = ML.export_set(set_name)
    seed = ML.inference_seed("floor", args.seed, split)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ip = Interpolant(cfg.interpolant)
    c = copy.deepcopy(cfg)
    c.data.cache_dir = cache
    c.data.loader.num_workers = 0
    ds = MCDataset(cache_path=os.path.join(cache, "test.pt"), dataset_cfg=c.data, is_training=False)
    from torch_geometric.loader import DataLoader
    dl = DataLoader(ds, batch_size=c.data.loader.batch_size.test, shuffle=False, num_workers=0)
    per = {k: [[] for _ in range(S)] for k in ("cart_coords", "atom_types", "lattices", "num_atoms", "pred_trans",
                                                "pred_rotmats", "num_bbs")}
    gt = {k: [] for k in ("gt_coords", "atom_types", "lattice_1", "trans_1", "rotmats_1", "num_atoms")}
    for batch in dl:
        nbb, bv, B = batch.num_bbs, batch.batch, int(batch.num_graphs)
        for s in range(S):
            trans_0 = _uniform_cell_trans(len(bv), bv, nbb, "cpu")
            rot_0 = _symmetric_so3(len(bv), nbb, "cpu")
            lengths_0 = ip._lognormal.sample((B,))
            angles_0 = ip._uniform.sample((B, 3))
            lat_0 = lattice6_to_mat33(torch.cat([lengths_0, angles_0], dim=-1))
            tc = torch.bmm(trans_0.unsqueeze(1), lat_0[bv]).squeeze(1)
            cart = ip._assemble_coords(batch.local_coords, rot_0, tc, batch.bb_num_vec)
            for k, v in (("cart_coords", cart), ("atom_types", batch.atom_types), ("lattices", lat_0),
                         ("num_atoms", batch.num_atoms), ("pred_trans", trans_0), ("pred_rotmats", rot_0),
                         ("num_bbs", nbb)):
                per[k][s].append(v)
        for k, v in (("gt_coords", batch.gt_coords), ("atom_types", batch.atom_types), ("lattice_1", batch.lattice_1),
                     ("trans_1", batch.trans_1), ("rotmats_1", batch.rotmats_1), ("num_atoms", batch.num_atoms)):
            gt[k].append(v)
    pred = {k: torch.stack([torch.cat(v[s]) for s in range(S)]) for k, v in per.items()}
    pred["gt_data_batch"] = {k: torch.cat(v) for k, v in gt.items()}
    os.makedirs(rp["out"], exist_ok=True)
    ML.atomic_torch(pred, rp["pred"])
    man = {"arm": "floor", "arm_name": "mcfA_floor", "set": set_name, "split": split, "train_seed": args.seed,
           "floor_seed": seed, "num_samples": S, "cache_dir": fwd(cache), "pickle_sha256": side["sha256"],
           "smoke": smoke, "label": f"s{args.seed}", "lattice_prior": "ours_lattice.yaml (refit)",
           "ckpt": None, "ckpt_sha256": None, "protocol": "tasks/n3_protocol.md §3.2 ii-A prior floor"}
    ML.atomic_json(man, rp["manifest"], indent=1)
    print(json.dumps({"run_dir": fwd(run_dir), "S": S, "n": len(entries), "seed": seed}))


def run_check_floor(run_dir):
    man, rp = ML.load_manifest(run_dir)
    entries, side = ML.export_set(man["set"])
    pred = ML.load_predictions(rp["pred"])
    n, S = len(entries), man["num_samples"]
    c = {"S_n": tuple(pred["num_atoms"].shape) == (S, n), "pickle_sha": side["sha256"] == man["pickle_sha256"],
         "num_atoms": all(int(pred["num_atoms"][0][b]) == int(entries[b]["local_coords"].shape[0]) for b in range(n))}
    res = {"run_dir": fwd(run_dir), "pass": all(c.values()), "failed": [k for k, v in c.items() if not v], "checks": c,
           "predictions_sha256": ML.sha256(rp["pred"]), "resid_sha256": None}
    ML.atomic_json(res, rp["check"], indent=1)
    return res


# ============================================================================================ smoke_init
def cmd_smoke_init(args):
    """VAL subset (test_* + a sidecar naming the VAL indices) and a random-init MCF-R checkpoint (Lightning
    format: state_dict + hyper_parameters) with the training config.yaml train.py would save."""
    import n3_mcf_export as X
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    ML.setup_mcf(args.shims)
    from molcrystalflow.models.molcrystalflow import FlowModule
    entries, side = X.load_export("val")
    idx = [int(x) for x in args.val_indices.split(",")]
    d = os.path.join(SMOKE_ROOT, args.name)
    os.makedirs(d, exist_ok=True)
    pk = os.path.join(d, "test_molcrystal_normalized.pkl.gz")
    X.save_pkl_gz([entries[i] for i in idx], pk)
    sc = {"set": "smoke", "n3_set": "val", "val_indices": idx, "sha256": ML.sha256(pk), "n": len(idx),
          "refcodes": [side["refcodes"][i] for i in idx], "items": [side["items"][i] for i in idx],
          "source_pickle_sha256": side["sha256"], "note": "laptop smoke only (VAL subset); never a reported input"}
    ML.atomic_json(sc, os.path.join(d, "test_sidecar.json"))
    ck = os.path.join(SMOKE_ROOT, f"ckpt_{args.arm}")
    os.makedirs(ck, exist_ok=True)
    cfg = ML.compose(f"ours_molcrystal_{args.arm}", [f"experiment.seed={args.seed}",
                                                     f"experiment.wandb.name=n3_mcf{args.arm}_s{args.seed}"])
    OmegaConf.save(config=cfg, f=os.path.join(ck, "config.yaml"))
    pl.seed_everything(args.seed, verbose=False)
    mod = FlowModule(cfg)
    torch.save({"state_dict": mod.state_dict(), "hyper_parameters": dict(mod.hparams),
                "pytorch-lightning_version": pl.__version__, "epoch": 0, "global_step": 0},
               os.path.join(ck, "last.ckpt"))
    print(json.dumps({"subset_dir": fwd(d), "val_indices": idx, "z_MCF": [it["z_MCF"] for it in sc["items"]],
                      "mult": [it["mult"] for it in sc["items"]], "ckpt": fwd(os.path.join(ck, "last.ckpt"))}))


# ============================================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--shims", action="store_true", help="laptop smoke only: torch_scatter/cluster/GPUtil stand-ins")
        return p
    p = common(sub.add_parser("train"))
    p.add_argument("--arm", choices=["R", "A"], required=True)
    p.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--gpu", default=None)
    p.add_argument("--run", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("train_report")
    p.add_argument("--arm", choices=["R", "A"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--wandb_dir", default=None)
    p.add_argument("--run_dirs", nargs="*", default=None, help="explicit offline run dirs (the last one is used)")
    p.add_argument("--ckpt_dir", default=None, help="default: n3_mcf/runs/n3_mcf<arm>_s<seed>/ckpt")
    p.add_argument("--out", default=None)
    p = common(sub.add_parser("fixed_val_loss"))
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="laptop smoke only: first N VAL crystals")
    p = common(sub.add_parser("infer"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--set", required=True, help="valsel | testB | devtest (or a smoke subset dir with --smoke)")
    p.add_argument("--s_uR", type=float, required=True)
    p.add_argument("--s_uF", type=float, default=None, help="MCF-A only (interpolant.trans.scaling)")
    p.add_argument("--gpu", default=None)
    p.add_argument("--run", action="store_true")
    p.add_argument("--discard", action="store_true")
    p.add_argument("--label_extra", default="")
    p.add_argument("--smoke", action="store_true", help="laptop smoke: allows the knobs below and a subset dir")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--num_timesteps", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--cpu_emulate", action="store_true", help="smoke only: EvalRunner on CPU in this process")
    p.add_argument("--fallback_chunk", type=int, default=None, help="smoke only: generator-error fallback chunk size")
    p = common(sub.add_parser("check"))
    p.add_argument("--run_dir", required=True)
    p = common(sub.add_parser("to_bundle"))
    p.add_argument("--run_dir", required=True)
    p.add_argument("--arm", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int, default=2)
    p = common(sub.add_parser("prior_floor"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p = common(sub.add_parser("smoke_init"))
    p.add_argument("--val_indices", default="1,44", help="VAL positions (default: z_MCF 2 det Q -1; centred z_MCF 4)")
    p.add_argument("--name", default="val2")
    p.add_argument("--arm", default="R", choices=["R", "A"])
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    {"train": cmd_train, "train_report": cmd_train_report, "fixed_val_loss": cmd_fixed_val_loss,
     "infer": cmd_infer, "check": cmd_check, "to_bundle": cmd_to_bundle, "prior_floor": cmd_prior_floor,
     "smoke_init": cmd_smoke_init}[args.cmd](args)


if __name__ == "__main__":
    main()
