"""N3 §7 step 4: timing gate (box) and budget projector (anywhere).

Every measurement uses TRAIN / VAL only and the per-unit production commands; nothing it produces is matched
against a truth. The one pre-registered exception is TIMING StructureMatcher().fit and the secondary get_rms_dist
on Haar candidates of VAL crystals (§7 step 4): the booleans / RMS values are computed and discarded unread.

GPU box (each measurement is a subprocess in its own env: PY_MCF / PY_SYMMC / N3_FF_PYTHON from .n3_box/env.sh)
  mcf_train_{R,A}_p{0,1}    MCF train.py with the N3 overrides (n3_mcf_run.train_overrides; P6 off / on), runs of
                            4 and 22 epochs into n3_mcf/_timing; epoch time = (T22 - T4) / 18 = steps 57-308
                            (14 steps / epoch, per-epoch validation and checkpointing included). --concurrency k runs
                            k copies at once (the step-5 plan if trainings will share the GPU).
  mcf_infer_R_n{10,50}, mcf_infer_A_n50
                            n3_mcf_run.py infer --smoke on a random-init checkpoint (smoke_init) over the first 10 /
                            50 VAL crystals at the production knobs (16 draws, 50 steps, 4 loader workers), RESID pass
                            and check included -> per-call overhead c0 and per-crystal cost c1.
  ivp_train_{trim,pack}     n3_old_symmc.py train, 3 epochs (318 steps): epochs 2-3 from hist.jsonl.
  ivp_sample_{enc}_b{N}     n3_old_symmc.py sample on 50 VAL crystals x 16 draws x 50 RK4 steps (+ LJ workers).
  ff_{batch,default}_{float32,float64}   n3_classical.py fftime: 20 VAL crystals (K=2 x7, 4 x7, >=8 x6),
                            single calls and batched 64.
  ff_min_truth_{float32,float64}          n3_classical.py ffmin --truth, 20 VAL truths (evaluations used,
                            convergence), 'batch' settings. Truth + FF* only; no pose is matched.
CPU (the box's worker count W, one thread per worker = the production concurrency)
  cpu                       energy(), press(80 steps), StructureMatcher().fit, get_rms_dist at stol 0.5 / 0.8 / 1.0
                            on 16 Haar candidates (timing seed 990000 + j, outside every protocol seed range) of the
                            same 20 VAL crystals.
  lj_screen                 n3_classical.py lbgate, then screen --set val --limit 20 (batched LJ, 4,608 orientations).
  ours_sample               n3_ours.py sample --set val --smoke --seed-base 40000 on the first 2W VAL crystals.

    python scripts/n3_timing_gate.py plan --workers W                     # print every command; run nothing
    python scripts/n3_timing_gate.py run  --workers W [--only NAME ...]   # box; finished measurements are kept
    python scripts/n3_timing_gate.py cpu  --workers W                     # the CPU measurement alone
    python scripts/n3_timing_gate.py rates                                # timing/*.json -> timing/rates.json
    python scripts/n3_timing_gate.py stats                                # laptop: per-crystal atoms, K, z_MCF
    python scripts/n3_timing_gate.py reported                             # prior rates: units' reports + assumptions
    python scripts/n3_timing_gate.py project --rates R --usd-per-hour X --cores N [--cap-usd C]
Outputs: results/n3/private/timing/ (gitignored; the rates and projection hold numbers only). Scratch runs of the
gate live in n3_mcf/_timing, n3_mcf/_smoke and results/n3/private/timing and are deleted with the box.
"""
import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = os.path.join(REPO, "results", "n3", "private", "timing")
MCF_TIMING = os.path.join(REPO, "n3_mcf", "_timing")
S_DRAWS = 16
TIMING_SEED = 990000
MCF_EPOCHS, MCF_STEPS_PER_EPOCH, IVP_EPOCHS, IVP_RUNS = 1715, 14, 60, 5


def py(env):
    return os.environ.get({"mcf": "PY_MCF", "symmc": "PY_SYMMC", "ff": "N3_FF_PYTHON"}[env], sys.executable)


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


# ================================================================================================ measurements
def mcf_train_overrides(arm):
    """The production overrides of n3_mcf_run.train_overrides (read from the mcf env, never re-typed here)."""
    code = ("import sys, json; sys.path.insert(0, 'scripts'); import n3_mcf_run as R; "
            f"print(json.dumps(R.train_overrides('{arm}', 0)))")
    try:
        r = subprocess.run([py("mcf"), "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def measurements(args):
    """name -> spec {env, gpu, cmds [(cwd, argv)], parse, prep [(cwd, argv)]}."""
    W, g = args.workers, str(args.gpu)
    M = {}
    for arm in ("R", "A"):
        base = mcf_train_overrides(arm) if not args.dry else None
        err = None
        if base is None:
            err = None if args.dry else f"could not read n3_mcf_run.train_overrides('{arm}', 0) with PY_MCF"
            base = [f"<n3_mcf_run.train_overrides('{arm}', 0)>"]
        for p6 in (0, 1):
            runs = []
            for E in (4, 22):
                for c in range(args.concurrency):
                    name = f"n3_timing_mcf{arm}_p{p6}_e{E}_c{c}"
                    ov = [o for o in base if not o.startswith(("experiment.trainer.max_epochs=",
                                                               "experiment.wandb.name="))]
                    ov += [f"experiment.trainer.max_epochs={E}", f"experiment.wandb.name={name}",
                           f"paths.log_dir={fwd(os.path.join(MCF_TIMING, 'runs', name))}",
                           f"paths.wandb_dir={fwd(os.path.join(MCF_TIMING, 'wandb'))}"]
                    if p6:
                        ov += ["interpolant.rots.p6_vectorized_prior=true", "model.bb_embedder.p6_scatter_pooling=true"]
                    runs.append((E, c, (os.path.join(REPO, "external", "mcf"),
                                        [py("mcf"), "molcrystalflow/experiments/train.py",
                                         f"--config-name=ours_molcrystal_{arm}"] + ov)))
            M[f"mcf_train_{arm}_p{p6}"] = {"env": "mcf", "gpu": True, "kind": "mcf_train", "runs": runs, "error": err}
    sub = {10: os.path.join(REPO, "n3_mcf", "_smoke", "val10"), 50: os.path.join(REPO, "n3_mcf", "_smoke", "val50")}
    for arm, n in (("R", 10), ("R", 50), ("A", 50)):
        prep = [(REPO, [py("mcf"), "scripts/n3_mcf_run.py", "smoke_init", "--name", f"val{n}", "--arm", arm,
                        "--val_indices", ",".join(str(i) for i in range(n))])]
        cmd = [py("mcf"), "scripts/n3_mcf_run.py", "infer", "--ckpt", fwd(f"{REPO}/n3_mcf/_smoke/ckpt_{arm}/last.ckpt"),
               "--set", fwd(sub[n]), "--s_uR", "3", "--smoke", "--run", "--gpu", g, "--discard",
               "--label_extra", f"timing_n{n}"] + (["--s_uF", "9"] if arm == "A" else [])
        M[f"mcf_infer_{arm}_n{n}"] = {"env": "mcf", "gpu": True, "kind": "mcf_infer", "n": n, "prep": prep,
                                      "cmds": [(REPO, cmd)]}
    for enc in ("trim", "pack"):
        root = os.path.join(OUT, f"ivp_{enc}")
        M[f"ivp_train_{enc}"] = {"env": "symmc", "gpu": True, "kind": "ivp_train", "root": root, "cmds": [(REPO, [
            py("symmc"), "scripts/n3_old_symmc.py", "train", "--lr", "3e-4", "--seed", "0", "--epochs", "3",
            "--runs-root", fwd(root), "--device", "cuda"] + (["--pack"] if enc == "pack" else []))]}
        for b in args.ivp_cpb:
            M[f"ivp_sample_{enc}_b{b}"] = {"env": "symmc", "gpu": True, "kind": "ivp_sample", "n": 50,
                                           "after": f"ivp_train_{enc}", "root": root, "cmds": [(REPO, [
                py("symmc"), "scripts/n3_old_symmc.py", "sample", "--run", "<run dir of ivp_train>", "--epoch", "3",
                "--set", "val", "--limit", "50", "--workers", str(min(W, 8)), "--crystals-per-batch", str(b),
                "--device", "cuda", "--arm", f"ivPtiming{enc}b{b}"] + (["--pack"] if enc == "pack" else []))]}
    for st in ("batch", "default"):
        for dt in ("float32", "float64"):
            M[f"ff_{st}_{dt}"] = {"env": "symmc", "gpu": True, "kind": "ff_time", "settings": st, "dtype": dt,
                                  "cmds": [(REPO, [py("symmc"), "scripts/n3_classical.py", "fftime", "--set", "val",
                                                   "--device", "cuda", "--batch", "64", "--ff-settings", st,
                                                   "--ff-dtype", dt])]}
    for dt in ("float32", "float64"):
        M[f"ff_min_truth_{dt}"] = {"env": "symmc", "gpu": True, "kind": "ff_min", "dtype": dt, "cmds": [(REPO, [
            py("symmc"), "scripts/n3_classical.py", "ffmin", "--set", "val", "--truth", "--max-evals", "1000",
            "--limit", "20", "--device", "cuda", "--ff-settings", "batch", "--ff-dtype", dt])]}
    M["cpu"] = {"env": "symmc", "gpu": False, "kind": "cpu", "cmds": [(REPO, [
        py("symmc"), "scripts/n3_timing_gate.py", "cpu", "--workers", str(W)])]}
    M["lj_screen"] = {"env": "symmc", "gpu": False, "kind": "screen", "cmds": [
        (REPO, [py("symmc"), "scripts/n3_classical.py", "lbgate", "--workers", str(W)]),
        (REPO, [py("symmc"), "scripts/n3_classical.py", "screen", "--set", "val", "--limit", "20",
                "--workers", str(W)])]}
    n_ours = min(100, 2 * W)
    M["ours_sample"] = {"env": "symmc", "gpu": False, "kind": "ours", "n": n_ours, "workers": W, "cmds": [(REPO, [
        py("symmc"), "scripts/n3_ours.py", "sample", "--arm", "ours", "--set", "val", "--smoke", "--seed-base",
        "40000", "--limit", str(n_ours), "--workers", str(W)])]}
    return M


class GPUSampler(threading.Thread):
    """nvidia-smi every 2 s: max memory and mean utilisation of GPU 0 during a measurement."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                                    "-i", "0"], capture_output=True, text=True, timeout=10).stdout.strip()
                u, m = (float(x) for x in o.split(","))
                self.rows.append((u, m))
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(2.0)

    def summary(self):
        if not self.rows:
            return {}
        return {"gpu_util_mean": round(sum(r[0] for r in self.rows) / len(self.rows), 1),
                "gpu_mem_max_MiB": max(r[1] for r in self.rows), "samples": len(self.rows)}


def run_cmd(cwd, argv, log, env):
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write(f"\n$ (cd {cwd}) {' '.join(argv)}\n")
        fh.flush()
        r = subprocess.run(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def run_concurrent(items, log, env):
    """[(cwd, argv)] started together; returns (max returncode, wall of the slowest)."""
    t0 = time.time()
    procs = []
    for k, (cwd, argv) in enumerate(items):
        fh = open(f"{log}.{k}", "a")
        fh.write(f"$ (cd {cwd}) {' '.join(argv)}\n")
        fh.flush()
        procs.append((subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    rc = 0
    for p, fh in procs:
        rc = max(rc, p.wait())
        fh.close()
    return rc, time.time() - t0


def tail_json(log):
    """Last JSON object line of a log (n3_mcf_run infer prints one)."""
    for line in reversed(open(log).read().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def measure(name, spec, args):
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    log = os.path.join(OUT, "logs", f"{name}.log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="offline", N3_ROOT=REPO)
    res = {"name": name, "kind": spec["kind"], "env": spec["env"], "host": platform.node(), "cpus": os.cpu_count(),
           "workers": args.workers, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    samp = GPUSampler() if spec["gpu"] else None
    if samp:
        samp.start()
    try:
        if spec.get("error"):
            raise RuntimeError(spec["error"])
        for cwd, argv in spec.get("prep", []):
            rc, _ = run_cmd(cwd, argv, log, env)
            if rc:
                raise RuntimeError(f"prep failed ({rc}); see {log}")
        k = spec["kind"]
        if k == "mcf_train":
            walls = {}
            for E in (4, 22):
                items = [c for (e, _, c) in spec["runs"] if e == E]
                rc, wall = run_concurrent(items, f"{log}.e{E}", env)
                if rc:
                    raise RuntimeError(f"train E={E} failed ({rc}); see {log}.e{E}.*")
                walls[E] = wall
            res.update(wall_e4=round(walls[4], 1), wall_e22=round(walls[22], 1), concurrency=args.concurrency,
                       epoch_s=round((walls[22] - walls[4]) / 18, 3),
                       step_s=round((walls[22] - walls[4]) / (18 * MCF_STEPS_PER_EPOCH), 4),
                       note="per-epoch wall of ONE run (all `concurrency` copies ran at once), epochs 5-22")
        elif k == "ivp_sample":
            runs = [d for d in os.listdir(spec["root"]) if os.path.isdir(os.path.join(spec["root"], d))]
            if len(runs) != 1:
                raise RuntimeError(f"expected one iv-P timing run under {spec['root']}, found {runs}")
            cwd, argv = spec["cmds"][0]
            argv = [fwd(os.path.join(spec["root"], runs[0])) if a == "<run dir of ivp_train>" else a for a in argv]
            rc, wall = run_cmd(cwd, argv, log, env)
            if rc:
                raise RuntimeError(f"failed ({rc}); see {log}")
            m = re.findall(r"\(([\d.]+) s sampling, ([\d.]+) s LJ, ([\d.]+) s total", open(log).read())
            if not m:
                raise RuntimeError(f"no timing line in {log}")
            smp, lj, tot = (float(x) for x in m[-1])
            b = os.path.join(REPO, "results", "n3", "private", "old", "smoke",
                             f"{argv[argv.index('--arm') + 1]}_s0_val_limit{spec['n']}.pt")
            try:                                          # the bundle meta keeps unrounded seconds
                import torch
                meta = torch.load(b, weights_only=False)["meta"]
                smp, lj = float(meta["sample_sec"]), float(meta["lj_sec"])
            except Exception:  # noqa: BLE001 -- fall back to the rounded log line
                pass
            res.update(wall=round(wall, 1), n=spec["n"], sample_s=smp, lj_s=lj,
                       gpu_s_per_crystal=round(smp / spec["n"], 3), lj_cpu_s_per_crystal=round(lj / spec["n"], 3))
        else:
            wall = 0.0
            for cwd, argv in spec["cmds"]:
                rc, w = run_cmd(cwd, argv, log, env)
                wall += w
                if rc:
                    raise RuntimeError(f"failed ({rc}); see {log}")
            res["wall"] = round(wall, 1)
            if k == "mcf_infer":
                j = tail_json(log)
                if not j.get("check_pass"):
                    raise RuntimeError(f"infer check did not pass: {j}")
                res.update(n=spec["n"], sampling_s=j["sampling_sec"], resid_s=j["resid_sec"],
                           call_s=round(j["sampling_sec"] + j["resid_sec"], 1))
            elif k == "ivp_train":
                hist = [os.path.join(d, "hist.jsonl") for d, _, fs in os.walk(spec["root"]) if "hist.jsonl" in fs]
                rows = [json.loads(line) for line in open(hist[0]) if line.strip()]
                ep = [r["sec"] for r in rows if r["epoch"] >= 2]
                res.update(epoch_s=round(sum(ep) / len(ep), 2), steps_per_epoch=rows[-1]["steps"],
                           epochs=[{"epoch": r["epoch"], "sec": r["sec"], "steps": r["steps"]} for r in rows])
            elif k in ("ff_time", "ff_min"):
                pat = "fftime_val_" if k == "ff_time" else "ffmin_val_t1_h0_e1000_"
                tag = f"cuda_{spec.get('settings', 'batch')}_{spec['dtype']}"
                d = os.path.join(REPO, "results", "n3", "private", "classical")
                shards = [f for f in os.listdir(d) if f.startswith(pat) and tag in f and f.endswith(".jsonl")]
                if not shards:
                    raise RuntimeError(f"no {pat}*{tag}*.jsonl shard in {d}")
                rows = [json.loads(line) for line in open(os.path.join(d, sorted(shards)[-1])) if line.strip()]
                if k == "ff_time":
                    res["rows"] = [{x: r[x] for x in ("K", "atoms", "warm_sec", "sec_single", "sec_single_predict",
                                                      "sec_per_struct_batched", "sec_per_struct_batched_predict")}
                                   for r in rows]
                else:
                    res["rows"] = [{"K": r["K"], "atoms": r["atoms"], "sec": r["sec"],
                                    "nev": [x["nev"] for x in r["res"]], "conv": [x["conv"] for x in r["res"]],
                                    "why": [x["why"] for x in r["res"]]} for r in rows]
            elif k == "screen":
                p = os.path.join(REPO, "results", "n3", "private", "classical", "screen_val.smoke.jsonl")
                rows = [json.loads(line) for line in open(p) if line.strip()]
                res["rows"] = [{"refcode_idx": i, "evaluator": r["evaluator"], "n_eval": r["n_eval"], "sec": r["sec"]}
                               for i, r in enumerate(rows)]
                res["note"] = "per-crystal sec of the batched screen; atoms joined in `rates` from the VAL set cache"
            elif k == "ours":
                res.update(n=spec["n"], workers=spec["workers"],
                           cpu_s_per_crystal=round(res["wall"] * min(spec["workers"], spec["n"]) / spec["n"], 1),
                           note="3 checkpoints x 16 draws x 40 steps + e_lj + torque_end per crystal")
            elif k == "cpu":
                res.update(json.load(open(os.path.join(OUT, "cpu_rows.json"))))
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, error=str(e)[:500])
    finally:
        if samp:
            samp.stop.set()
            samp.join(timeout=5)
            res.update(samp.summary())
    res["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "epochs")}), flush=True)
    return res


def cmd_plan(args):
    args.dry = True
    for name, spec in measurements(args).items():
        print(f"# {name}  [{spec['env']}{', GPU' if spec['gpu'] else ', CPU'}]")
        for cwd, argv in spec.get("prep", []):
            print(f"  (cd {fwd(cwd)}) {' '.join(argv)}")
        for item in spec.get("runs", []):
            E, c, (cwd, argv) = item
            print(f"  [E={E} copy {c}] (cd {fwd(cwd)}) {' '.join(argv)}")
        for cwd, argv in spec.get("cmds", []):
            print(f"  (cd {fwd(cwd)}) {' '.join(argv)}")


def cmd_run(args):
    args.dry = False
    if os.environ.get("N3_SET_CACHE_ONLY") != "1":
        print("warning: N3_SET_CACHE_ONLY is not 1 (source .n3_box/env.sh on the box)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    M = measurements(args)
    names = args.only or list(M)
    for name in names:
        p = os.path.join(OUT, f"{name}.json")
        if os.path.exists(p) and json.load(open(p)).get("ok") and not args.redo:
            print(f"{name}: done (kept)")
            continue
        after = M[name].get("after")
        if after and not (os.path.exists(os.path.join(OUT, f"{after}.json"))
                          and json.load(open(os.path.join(OUT, f"{after}.json"))).get("ok")):
            print(f"{name}: needs {after} first; skipped")
            continue
        measure(name, M[name], args)


# ================================================================================================ CPU part
def val20():
    """The fftime crystal choice: the first 7 K=2, 7 K=4 and 6 K>=8 VAL crystals, in VAL order."""
    import n3_common as C
    items = C.load_set("val")
    idx = ([j for j, a in enumerate(items) if a["K"] == 2][:7] + [j for j, a in enumerate(items) if a["K"] == 4][:7]
           + [j for j, a in enumerate(items) if a["K"] >= 8][:6])
    return idx


def haar(S, gen):
    import torch
    A = torch.randn(S, 3, 3, generator=gen, dtype=torch.float64)
    Q, Rm = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rm, dim1=-2, dim2=-1)).unsqueeze(-2)
    det = torch.linalg.det(Q)
    Q[..., :, 0] = Q[..., :, 0] * det.unsqueeze(-1)
    return Q


def _cpu_task(t):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    import n3_common as C
    from g2_asym_baselines import energy, expand, press
    from pymatgen.analysis.structure_matcher import StructureMatcher
    j, S, n_press, n_fit = t
    a = C.load_set("val")[j]
    R = haar(S, torch.Generator().manual_seed(TIMING_SEED + j))
    out = {"j": j, "K": int(a["K"]), "atoms": int(a["K"] * a["local"].shape[0])}
    t0 = time.perf_counter()
    for s in range(S):
        energy(*expand(a, R[s]))
    out["energy_s"] = (time.perf_counter() - t0) / S
    t0 = time.perf_counter()
    for s in range(n_press):
        energy(*press(a, R[s], 80))
    out["press_s"] = (time.perf_counter() - t0) / n_press
    truth = C.truth_structure(a)
    cands = [C.rasym_structure(a, R[s]) for s in range(n_fit)]
    t0 = time.perf_counter()
    sm = StructureMatcher()
    for c in cands:
        sm.fit(truth, c)                          # outcome discarded unread (timing only, §7 step 4)
    out["fit_primary_s"] = (time.perf_counter() - t0) / n_fit
    for name, stol in (("mcf05", 0.5), ("mcf08", 0.8), ("mcf10", 1.0)):
        sm = StructureMatcher(ltol=0.3, angle_tol=10, scale=False, stol=stol)
        t0 = time.perf_counter()
        for c in cands:
            sm.get_rms_dist(truth, c)             # value discarded unread
        out[f"rms_{name}_s"] = (time.perf_counter() - t0) / n_fit
    return out


def cmd_cpu(args):
    from multiprocessing import Pool
    idx = val20()[:args.n_crystals] if args.n_crystals else val20()
    tasks = [(j, args.draws, args.press, args.fits) for j in idx]
    t0 = time.time()
    if args.workers <= 1:
        rows = [_cpu_task(t) for t in tasks]
    else:
        with Pool(args.workers) as pool:
            rows = pool.map(_cpu_task, tasks, chunksize=1)
    rec = {"rows": rows, "wall_s": round(time.time() - t0, 1), "workers": args.workers, "draws": args.draws,
           "press_draws": args.press, "fit_draws": args.fits, "timing_seed": f"{TIMING_SEED} + VAL index",
           "smoke": bool(args.n_crystals and args.n_crystals < 20),
           "note": "per-operation seconds inside one single-thread worker, W workers busy at once; outcomes discarded"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, "cpu_rows.json"), "w"), indent=1)
    keys = ("energy_s", "press_s", "fit_primary_s", "rms_mcf05_s", "rms_mcf08_s", "rms_mcf10_s")
    print(json.dumps({k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}), f"({rec['wall_s']} s)")


# ================================================================================================ set stats
def cmd_stats(args):
    """Per-crystal atoms (K x asym atoms), K and z_MCF of every set the projection needs (input-side counts)."""
    import n3_common as C
    out = {}
    for s in ("val", "sel", "valsel", "testB", "devtest"):
        items = C.load_set(s)
        side = {"val": "n3_mcf/trainval/val_sidecar.json", "valsel": "n3_mcf/valsel_as_test/test_sidecar.json",
                "testB": "n3_mcf/test/test_sidecar.json", "devtest": "n3_mcf/devtest/test_sidecar.json"}.get(s)
        z = None
        if side and os.path.exists(os.path.join(REPO, side)):
            sc = json.load(open(os.path.join(REPO, side)))
            assert sc["refcodes"] == [a["refcode"] for a in items], f"{side}: order differs"
            z = [int(it["z_MCF"]) for it in sc["items"]]
        out[s] = {"n": len(items), "atoms": [int(a["K"] * a["local"].shape[0]) for a in items],
                  "K": [int(a["K"]) for a in items], "z_MCF": z}
    out["sel"]["z_MCF"] = out["valsel"]["z_MCF"][100:] if out["valsel"]["z_MCF"] else None
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "set_stats.json")
    json.dump(out, open(p, "w"))
    print({s: {"n": v["n"], "mean_atoms": round(sum(v["atoms"]) / v["n"], 1)} for s, v in out.items()}, "->", p)


# ================================================================================================ rates
def fit_power(xs, ys):
    """y = a * x^p by least squares in log-log (p clipped to [0, 4]); const if the x's do not vary."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2 or max(p[0] for p in pts) - min(p[0] for p in pts) < 1e-6:
        return {"type": "const", "v": sum(ys) / len(ys)}
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    p_ = max(0.0, min(4.0, sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx))
    return {"type": "power", "a": math.exp(my - p_ * mx), "p": p_, "n": len(pts),
            "x_range": [min(xs), max(xs)]}


def model_eval(m, x):
    if m["type"] == "const":
        return m["v"]
    if m["type"] == "power":
        return m["a"] * x ** m["p"]
    return m["a"] + m["b"] * x


def cmd_rates(args):
    """Collect the measurement JSONs into one rates file (numbers only)."""
    J = {}
    for f in os.listdir(OUT):
        if f.endswith(".json") and f not in ("rates.json", "set_stats.json", "cpu_rows.json") and \
                not f.startswith(("projection", "rates_")):
            r = json.load(open(os.path.join(OUT, f)))
            if r.get("ok"):
                J[r["name"]] = r
    rates = {"gpu": {}, "cpu": {}, "sources": {}, "measured": sorted(J), "host": platform.node(),
             "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    g, c, src = rates["gpu"], rates["cpu"], rates["sources"]
    for arm in ("R", "A"):
        for p6 in (0, 1):
            r = J.get(f"mcf_train_{arm}_p{p6}")
            if r:
                g.setdefault("mcf_train_epoch_s", {}).setdefault(arm, {})[f"p6_{'on' if p6 else 'off'}"] = r["epoch_s"]
                src[f"mcf_train_{arm}_p{p6}"] = f"measured, concurrency {r['concurrency']}"
    iR10, iR50, iA50 = J.get("mcf_infer_R_n10"), J.get("mcf_infer_R_n50"), J.get("mcf_infer_A_n50")
    if iR10 and iR50:
        c1 = (iR50["call_s"] - iR10["call_s"]) / 40
        c0 = max(0.0, iR10["call_s"] - 10 * c1)
        g["mcf_infer"] = {"R": {"c0": c0, "c1": c1}}
        if iA50:
            g["mcf_infer"]["A"] = {"c0": c0, "c1": max(0.0, (iA50["call_s"] - c0) / 50)}
        src["mcf_infer"] = "measured: call = c0 + c1 x crystals (16 draws, 50 steps, RESID + check)"
    for enc in ("trim", "pack"):
        r = J.get(f"ivp_train_{enc}")
        if r:
            g.setdefault("ivp_train_epoch_s", {})[enc] = r["epoch_s"]
        for b in (1, 2, 4, 8, 16, 32):
            r = J.get(f"ivp_sample_{enc}_b{b}")
            if r:
                g.setdefault("ivp_sample_gpu_s_per_crystal", {}).setdefault(enc, {})[str(b)] = r["gpu_s_per_crystal"]
    for st in ("batch", "default"):
        for dt in ("float32", "float64"):
            r = J.get(f"ff_{st}_{dt}")
            if r:
                rows = r["rows"]
                g.setdefault("ff", {})[f"{st}_{dt}"] = {
                    "batched": fit_power([x["atoms"] for x in rows], [x["sec_per_struct_batched_predict"] for x in rows]),
                    "single": fit_power([x["atoms"] for x in rows], [x["sec_single_predict"] for x in rows]),
                    "warm_s_mean": sum(x["warm_sec"] for x in rows) / len(rows)}
    for dt in ("float32", "float64"):
        r = J.get(f"ff_min_truth_{dt}")
        if r:
            nev = [n for x in r["rows"] for n in x["nev"]]
            conv = [cv for x in r["rows"] for cv in x["conv"]]
            g.setdefault("ff_truth_evals_mean", {})[dt] = sum(nev) / len(nev)
            g.setdefault("ff_truth_converged_frac", {})[dt] = sum(bool(v) for v in conv) / len(conv)
    cr = J.get("cpu")
    if cr:
        rows = cr["rows"]
        for k in ("energy_s", "press_s", "fit_primary_s", "rms_mcf05_s", "rms_mcf08_s", "rms_mcf10_s"):
            c[k] = fit_power([r["atoms"] for r in rows], [r[k] for r in rows])
        src["cpu"] = f"measured, {cr['workers']} workers x 1 thread"
    sc = J.get("lj_screen")
    if sc and os.path.exists(os.path.join(OUT, "set_stats.json")):
        at = json.load(open(os.path.join(OUT, "set_stats.json")))["val"]["atoms"]
        c["screen_s"] = fit_power([at[r["refcode_idx"]] for r in sc["rows"]], [r["sec"] for r in sc["rows"]])
    elif sc:
        c["screen_s"] = {"type": "const", "v": sum(r["sec"] for r in sc["rows"]) / len(sc["rows"])}
    o = J.get("ours_sample")
    if o:
        c["ours_sample_s"] = {"type": "const", "v": o["cpu_s_per_crystal"]}
    p = os.path.join(OUT, "rates.json")
    json.dump(rates, open(p, "w"), indent=1)
    print(f"{len(J)} measurements -> {p}")


def cmd_reported(args):
    """A rates file from what the first-round units REPORTED (laptop CPU, 2 workers, loaded machine) plus the
    protocol's §7 prior estimates for everything GPU (labelled 'assumed'): a projector test and a pre-box prior,
    never the step-4 projection itself."""
    st = json.load(open(os.path.join(OUT, "set_stats.json")))["valsel"]
    z2 = sorted(a for a, z in zip(st["atoms"], st["z_MCF"]) if z == 2)
    small = z2[len(z2) // 2]              # median z_MCF = 2 VALSEL cell: where +G's 3.2 CPU-h per cell was spent
    two = lambda y_small, y_big, big=368: fit_power([small, big], [y_small, y_big])
    rates = {
        "gpu": {
            "mcf_train_epoch_s": {a: {"p6_off": 2.5 * 3600 / MCF_EPOCHS, "p6_on": 2.5 * 3600 / MCF_EPOCHS}
                                  for a in ("R", "A")},
            "mcf_infer": {a: {"c0": 60.0, "c1": 0.10} for a in ("R", "A")},
            "ivp_train_epoch_s": {"pack": 4.4 * 3600 / (IVP_RUNS * IVP_EPOCHS),
                                  "trim": 9.0 * 3600 / (IVP_RUNS * IVP_EPOCHS)},
            "ivp_sample_gpu_s_per_crystal": {e: {"1": 2.1, "8": 2.1} for e in ("pack", "trim")},
            "ff": {"batch_float32": {"batched": {"type": "const", "v": 0.003},
                                     "single": {"type": "const", "v": 0.05}, "warm_s_mean": 20.0}},
            "ff_truth_evals_mean": {"float32": 7.0}},
        "cpu": {
            "energy_s": two(0.020, 0.60), "press_s": {"type": "const", "v": 1.0},
            "fit_primary_s": {"type": "const", "v": 0.36},
            "rms_mcf05_s": {"type": "const", "v": 0.35}, "rms_mcf08_s": {"type": "const", "v": 0.35},
            "rms_mcf10_s": {"type": "const", "v": 0.35}, "screen_s": two(6.5, 90.0),
            "ours_sample_s": {"type": "const", "v": 110.0}},
        "sources": {
            "energy_s": f"reported: MCF harness +G 3.2 CPU-h per VALSEL cell (~578k evaluations, mostly z_MCF = 2) "
                        f"-> 0.020 s at the median z_MCF = 2 cell ({small} atoms); classical: energy() up to 46 min per "
                        "4,608-orientation crystal -> 0.60 s at 368 atoms; power law between the two",
            "press_s": "reported (classical): 0.45-1.7 s per draw -> 1.0",
            "fit_primary_s": "reported (scorer): 0.36 CPU-s per primary fit",
            "rms": "reported (scorer, gate-ca): 0.3-0.4 CPU-s per MCF-settings fit -> 0.35",
            "screen_s": f"reported (classical): batched screen 6-7 s per crystal for small cells (taken at {small} "
                        "atoms), 66-115 s at 352-384 atoms",
            "ours_sample_s": "reported (scorer): OURS sampling 70-150 s per crystal for 3 checkpoints -> 110",
            "gpu": "ASSUMED, not measured: MCF 2.5 GPU-h per training run (protocol 1-4); inference c0 60 s + 0.10 s "
                   "per crystal (16 draws); iv-P 4.4 (packed) / 9 (trimmed) GPU-h for 5 runs (old-arm unit "
                   "projection); iv-P sampling 2.1 s per crystal (the LAPTOP CPU value, an upper bound); FF* 3 ms per "
                   "structure-evaluation batched (protocol 2-3.5 ms), 50 ms single; 7 evaluations per truth "
                   "(classical smoke, 2 VAL truths)"},
        "measured": [], "made": time.strftime("%Y-%m-%d %H:%M:%S"), "note": cmd_reported.__doc__}
    p = os.path.join(OUT, "rates_reported.json")
    json.dump(rates, open(p, "w"), indent=1)
    print("->", p)


# ================================================================================================ projection
def build_jobs(R, ST, cfg):
    """Every job of steps 4-9 with its count formula -> [dict(name, cat, step, where, h, note)].
    where: gpu (box GPU-h), cpu (box CPU-h), local (step-9 CPU-h, laptop or CPU box). cfg.d = degrade items taken."""
    d = cfg["degrade"]
    g, c = R["gpu"], R["cpu"]
    S = S_DRAWS
    V, T, D, VAL, SEL = (ST[k] for k in ("valsel", "testB", "devtest", "val", "sel"))
    test_sets = [T] if 5 in d else [T, D]          # d5: DEV-TEST rows of the new arms dropped
    both = [T, D]                                  # OURS / OURS-N2 / FF* truths / N2 picks keep DEV-TEST
    nA = 1 if 3 in d else 3
    stage2 = 0 if 4 in d else 8
    nA_calls_V = nA * (4 + stage2)
    ie_seeds = 0 if 7 in d else (1 if 6 in d else 3)
    p1_seeds = 1 if 6 in d else 3
    mcfA_native = ["mcf08", "mcf10"] + ([] if 2 in d else ["mcf05"])

    def tot(m, st, per=1.0, sel=None):
        """per x sum over the set's crystals of model(atoms) (sel: crystal filter / weight on (atoms, K, z))."""
        s, zz = 0.0, st["z_MCF"] or [4] * st["n"]
        for i, x in enumerate(st["atoms"]):
            w = 1.0 if sel is None else sel(x, st["K"][i], zz[i])
            if w:
                s += w * model_eval(m, x)
        return per * s

    def infer(arm, n, S_=S):
        m = g["mcf_infer"][arm]
        return m["c0"] + m["c1"] * n * S_ / S
    epR = g["mcf_train_epoch_s"]["R"]["p6_" + cfg["p6"]]
    epA = g["mcf_train_epoch_s"]["A"]["p6_" + cfg["p6"]]
    ep_ivp = g["ivp_train_epoch_s"][cfg["ivp_encoder"]]
    ivp_s = g["ivp_sample_gpu_s_per_crystal"][cfg["ivp_encoder"]][str(cfg["ivp_cpb"])]
    ff = g["ff"][cfg["ff"]]
    frac = cfg["ff_eval_frac"]
    tev = cfg["truth_evals"] if cfg["truth_evals"] else g.get("ff_truth_evals_mean", {}).get(cfg["ff"].split("_")[1], 1000)
    e, pr, fit = c["energy_s"], c["press_s"], c["fit_primary_s"]
    rms = {k: c[f"rms_{k}_s"] for k in ("mcf05", "mcf08", "mcf10")}
    gz = lambda x, K, z: 360 if z == 2 else (1 if z == 1 else 4)          # +G LJ evaluations per draw
    J = []

    def add(name, cat, step, where, sec, note=""):
        J.append({"name": name, "cat": cat, "step": step, "where": where, "h": sec / 3600.0, "note": note})

    # ---------------------------------------------------------------- training + Gate C (step 5, GPU)
    add("MCF-R training", "training", 5, "gpu", 3 * MCF_EPOCHS * epR, "3 seeds x 1715 epochs")
    add("MCF-A training", "training", 5, "gpu", nA * MCF_EPOCHS * epA, f"{nA} seed(s) x 1715 epochs")
    add("iv-P training", "training", 5, "gpu", IVP_RUNS * IVP_EPOCHS * ep_ivp, "5 runs x 60 epochs")
    add("MCF fixed-draw VAL losses", "training", 5, "gpu", 4 * (3 + nA) * infer("R", 100, 8), "4 ckpts per run, VAL x 8")
    add("Gate C-a/C-c/C-b inference", "gateC", 5, "gpu",
        10 * infer("A", 744, 10) + infer("R", 750) + 2 * infer("R", 750, 1),
        "Thurlemann TEST 750 / VAL 738, 5 runs x 10 draws each; C-c 16 draws; C-b 3 batches x 2 trees (approx)")
    add("Gate C-a matching (n3_score, --mcf-reading, MCF scorer)", "matching", 5, "cpu",
        3 * 2 * 5 * 10 * 3 * 744 * model_eval(rms["mcf10"], sum(V["atoms"]) / V["n"]),
        "2 labels x 5 runs x 10 draws x 3 stols; n3 + mcf-reading + MCF scorer at the stol-1.0 rate; VALSEL mean size")
    # ---------------------------------------------------------------- VALSEL sampling + scoring (step 5)
    add("MCF-R VALSEL sampling (Stage A 15 + Stage B 9 calls)", "valsel_sampling", 5, "gpu", 24 * infer("R", V["n"]))
    if cfg["gplus_separate"]:
        add("MCF-R+G separate Stage-B VALSEL sampling (contingency)", "valsel_sampling", 5, "gpu", 12 * infer("R", V["n"]))
    add("MCF-A VALSEL sampling", "valsel_sampling", 5, "gpu", nA_calls_V * infer("A", V["n"]),
        f"{nA} seed(s) x (4 stage-1 + {stage2} stage-2) calls")
    add("iv-P VALSEL sampling (6 calls)", "valsel_sampling", 5, "gpu", 6 * V["n"] * ivp_s)
    add("OURS + OURS-N2 SEL sampling", "valsel_sampling", 5, "cpu", 2 * tot(c["ours_sample_s"], SEL))
    add("MCF LJ selector inputs VALSEL", "valsel_sampling", 5, "cpu", (24 + nA_calls_V + nA) * S * tot(e, V))
    add("iii-u VALSEL (3 bases)", "valsel_sampling", 5, "cpu", 3 * S * tot(e, V))
    add("ii-S VALSEL (3 frozen +G cells)", "valsel_sampling", 5, "cpu", 3 * S * tot(e, V, sel=lambda x, K, z: K))
    add("iv-P LJ + iv-S VALSEL", "valsel_sampling", 5, "cpu", 2 * 6 * S * tot(e, V))
    add("MCF diagnostic (a), 6 checkpoints + Thurlemann", "valsel_sampling", 5, "cpu", 7 * 54.0, "reported 54 s each")
    add("+G gauge scan VALSEL (24 cells)", "lj_screens", 5, "cpu", 24 * S * tot(e, V, sel=gz))
    add("iii-s screen VAL (sanity row)", "lj_screens", 5, "cpu", tot(c["screen_s"], VAL))
    add("iii-p press VALSEL", "press", 5, "cpu", 3 * S * tot(pr, V))
    add("steric BASIN press VALSEL (P1 OURS + p1ref)", "press", 5, "cpu", (p1_seeds * S + 1) * tot(pr, V))
    add("FF* ceiling: 400 VALSEL truths", "ffstar", 5, "gpu", tot(ff["single"], V, per=tev),
        f"{tev:.0f} evaluations per truth (cap 1000)")
    add("iii-s VAL sanity row", "ffstar", 5, "gpu", tot(ff["batched"], VAL, per=64 * 100 * frac))
    nb_V = 3 + 3 + 24 + 24 + 3 + nA_calls_V + nA + 6 + 6 + 3 + 3 + p1_seeds + 3
    add(f"VALSEL primary matching ({nb_V} bundles x 16 draws)", "matching", 5, "cpu",
        nb_V * S * tot(fit, V) + S * tot(fit, VAL) + tot(fit, V))
    add("MCF-A VALSEL stol 0.8 / 1.0 on all draws", "matching", 5, "cpu",
        (nA_calls_V + nA) * S * (tot(rms["mcf08"], V) + tot(rms["mcf10"], V)))
    # ---------------------------------------------------------------- TEST-B / DEV-TEST sampling (step 7)
    for st, lab in ((T, "TEST-B"), (D, "DEV-TEST")):
        if st is D and 5 in d:
            continue
        add(f"MCF-R {lab} (3 calls)", "testdev_sampling", 7, "gpu", 3 * infer("R", st["n"]))
        if cfg["gplus_separate"]:
            add(f"MCF-R+G {lab} separate (contingency)", "testdev_sampling", 7, "gpu", 3 * infer("R", st["n"]))
        add(f"MCF-A {lab} ({nA} calls)", "testdev_sampling", 7, "gpu", nA * infer("A", st["n"]))
        add(f"iv-P {lab} (3 calls)", "testdev_sampling", 7, "gpu", 3 * st["n"] * ivp_s)
    add("OURS + OURS-N2 TEST-B sampling", "testdev_sampling", 7, "cpu", 2 * tot(c["ours_sample_s"], T))
    add("MCF LJ + A floor + iii-u + ii-S + iv-P/iv-S LJ (TEST-B/DEV-TEST)", "testdev_sampling", 7, "cpu",
        sum((3 + 2 * nA + 3 + 6) * S * tot(e, st) + 3 * S * tot(e, st, sel=lambda x, K, z: K) for st in test_sets))
    add("+G gauge scan TEST-B/DEV-TEST", "lj_screens", 7, "cpu", sum(3 * S * tot(e, st, sel=gz) for st in test_sets))
    add("iii-s screen TEST-B/DEV-TEST", "lj_screens", 7, "cpu", sum(tot(c["screen_s"], st) for st in test_sets))
    add("iii-p press TEST-B/DEV-TEST", "press", 7, "cpu", sum(3 * S * tot(pr, st) for st in test_sets))
    add("steric BASIN press TEST-B/DEV-TEST", "press", 7, "cpu", sum((p1_seeds * S + 1) * tot(pr, st) for st in test_sets))
    add("iii-s FF* minimisations (64 starts x <=100 evaluations)", "ffstar", 7, "gpu",
        sum(tot(ff["batched"], st, per=64 * 100 * frac) for st in test_sets))
    add("FF* truths TEST-B + DEV-TEST (labels, BASIN reference)", "ffstar", 7, "gpu",
        sum(tot(ff["single"], st, per=tev) for st in both))
    if ie_seeds:
        add(f"i-E ({ie_seeds} seed(s) x 16 draws x <=100)", "ffstar", 7, "gpu",
            sum(tot(ff["batched"], st, per=ie_seeds * S * 100 * frac) for st in test_sets))
    else:
        add("H2 BASIN of the OURS LJ picks (i-E dropped)", "ffstar", 7, "gpu",
            sum(tot(ff["single"], st, per=3 * 100 * frac) for st in both))
    add("OURS-N2 LJ picks, H2 BASIN", "ffstar", 7, "gpu", sum(tot(ff["single"], st, per=3 * 100 * frac) for st in both))
    # ---------------------------------------------------------------- step-9 scoring (laptop or CPU box)
    nb_new = 3 + 3 + 3 + nA + nA + 3 + 3 + 3 + 3 + 1 + ie_seeds + p1_seeds + 3
    add(f"TEST-B/DEV-TEST primary matching (OURS/N2 6 + new {nb_new} bundles x 16)", "matching", 9, "local",
        sum(6 * S * tot(fit, st) for st in both) + sum(nb_new * S * tot(fit, st) for st in test_sets))
    add(f"MCF-A native sweep {mcfA_native} + floor, all draws", "matching", 9, "local",
        sum(2 * nA * S * sum(tot(rms[m], st) for m in mcfA_native) for st in test_sets))
    add("iii-u Haar floor secondaries (stol 0.5 / 0.8), all draws", "matching", 9, "local",
        sum(3 * S * (tot(rms["mcf05"], st) + tot(rms["mcf08"], st)) for st in test_sets))
    pick3 = lambda st: 4 * (tot(rms["mcf05"], st) + tot(rms["mcf08"], st) + tot(rms["mcf10"], st))
    add("secondaries + continuous RMS on picks (14 rows x 4 picks x 3)", "matching", 9, "local",
        sum(2 * pick3(st) for st in both) + sum(12 * pick3(st) for st in test_sets))
    add("H2 BASIN + FF* truth labels", "matching", 9, "local", sum(8 * tot(fit, st) for st in both))
    add("MCF diagnostics (b) ceiling + (c) oracle-aligned", "matching", 9, "local",
        sum(3 * tot(fit, st, sel=lambda x, K, z: 164 if z == 2 else (4 if z >= 3 else 0))
            + 3 * 3 * S * tot(fit, st, sel=lambda x, K, z: 1 if z >= 2 else 0) for st in test_sets))
    # ---------------------------------------------------------------- tier-2 (§8), only when budgeted
    if cfg["tier2"] and 1 not in d:
        add("tier-2 data-size curve: MCF-R 422 / 843 + OURS G2 x2", "tier2", 7, "gpu",
            (4 + 7) / 14 * MCF_EPOCHS * epR + 2 * infer("R", VAL["n"]) + 0.75 * 2 * cfg["g2_train_h"] * 3600,
            "G2 training time is the --g2-train-h assumption")
        add("tier-2 data-size curve: OURS VAL draws + matching", "tier2", 7, "cpu",
            2 * tot(c["ours_sample_s"], VAL) / 3 + 4 * S * tot(fit, VAL))
        sub = {"atoms": T["atoms"][:200], "K": T["K"][:200], "z_MCF": (T["z_MCF"] or [4] * T["n"])[:200], "n": 200}
        add("tier-2 D-c0 (200 TEST-B x 3 sigma): OURS + iii-u + matching", "tier2", 9, "local",
            3 * (tot(c["ours_sample_s"], sub) + 3 * S * tot(e, sub) + 2 * 6 * S * tot(fit, sub)))
        add("tier-2 MCF conventional-cell export (85 centred TEST-B)", "tier2", 7, "gpu", 3 * infer("R", 85))
    return J


CAT_ORDER = ["training", "gateC", "valsel_sampling", "testdev_sampling", "ffstar", "press", "lj_screens", "matching",
             "tier2"]
DEGRADE = {1: "drop the §8 tier-2 diagnostics", 2: "MCF-A native sweep reduced to {0.8, 1.0}",
           3: "MCF-A seed 0 only", 4: "MCF-A without its knob grid", 5: "drop the DEV-TEST rows for new arms",
           6: "i-E and the steric BASIN on seed 0 only", 7: "drop i-E (H2 BASIN of the OURS LJ picks still runs)"}


def totals(J, cfg):
    """GPU-h, box CPU-h, local CPU-h; box hours per step as [concurrent lower bound, serial upper bound]."""
    ce = max(1, cfg["cores"] - cfg["gpu_job_cores"])
    t = {"gpu_h": 0.0, "cpu_h": 0.0, "local_cpu_h": 0.0, "box_h_lo": 0.0, "box_h_hi": 0.0, "steps": {}}
    for st in (5, 7):
        gh = sum(j["h"] for j in J if j["step"] == st and j["where"] == "gpu")
        ch = sum(j["h"] for j in J if j["step"] == st and j["where"] == "cpu")
        lo, hi = max(gh, ch / ce), gh + ch / ce
        t["steps"][st] = {"gpu_h": gh, "cpu_h": ch, "box_h_lo": lo * cfg["overhead"], "box_h_hi": hi * cfg["overhead"]}
        t["gpu_h"] += gh
        t["cpu_h"] += ch
        t["box_h_lo"] += lo * cfg["overhead"]
        t["box_h_hi"] += hi * cfg["overhead"]
    t["local_cpu_h"] = sum(j["h"] for j in J if j["where"] == "local")
    t["usd_lo"], t["usd_hi"] = t["box_h_lo"] * cfg["usd_per_hour"], t["box_h_hi"] * cfg["usd_per_hour"]
    t["step9_local_wall_h"] = t["local_cpu_h"] / max(1, cfg["local_cores"])
    t["step9_cpubox_usd"] = t["local_cpu_h"] / max(1, cfg["cpubox_cores"]) * cfg["cpubox_usd_per_hour"]
    return t


def cmd_project(args):
    R = json.load(open(args.rates))
    ST = json.load(open(args.stats))
    if args.testB and args.testB != ST["testB"]["n"]:
        raise SystemExit("--testB differs from the frozen TEST-B size in set_stats.json")
    cfg = {"p6": args.p6, "ivp_encoder": args.ivp_encoder, "ivp_cpb": args.ivp_cpb, "ff": args.ff,
           "ff_eval_frac": args.ff_eval_frac, "truth_evals": args.truth_evals, "gplus_separate": args.gplus_separate,
           "tier2": args.tier2, "g2_train_h": args.g2_train_h, "cores": args.cores, "gpu_job_cores": args.gpu_job_cores,
           "overhead": args.overhead, "usd_per_hour": args.usd_per_hour, "local_cores": args.local_cores,
           "cpubox_cores": args.cpubox_cores, "cpubox_usd_per_hour": args.cpubox_usd_per_hour, "degrade": set()}
    J = build_jobs(R, ST, cfg)
    T0 = totals(J, cfg)
    ce = max(1, args.cores - args.gpu_job_cores)
    rate = args.usd_per_hour
    print(f"N3 budget projection ({args.rates}; {args.cores} cores, {args.gpu_job_cores} kept for GPU jobs, "
          f"${rate}/h, x{args.overhead} idle overhead; P6 {args.p6}, iv-P {args.ivp_encoder} b{args.ivp_cpb}, "
          f"FF* {args.ff}, FF* evaluations at {args.ff_eval_frac:.0%} of the caps)")
    if R.get("sources"):
        print("rate sources:", json.dumps(R["sources"]))
    print(f"\n{'job':74s} {'step':>4s} {'where':>5s} {'GPU-h':>8s} {'CPU-h':>8s} {'box-h':>7s} {'$':>8s}")
    by = {}
    for j in J:
        gh = j["h"] if j["where"] == "gpu" else 0.0
        ch = j["h"] if j["where"] != "gpu" else 0.0
        bh = gh + (ch / ce if j["where"] == "cpu" else 0.0)
        b = by.setdefault(j["cat"], [0.0, 0.0, 0.0, 0.0])
        b[0] += gh; b[1] += ch if j["where"] == "cpu" else 0.0; b[2] += bh; b[3] += ch if j["where"] == "local" else 0.0
        print(f"{j['name'][:74]:74s} {j['step']:>4d} {j['where']:>5s} {gh:8.2f} {ch:8.2f} {bh:7.2f} {bh * rate:8.2f}")
    print(f"\nby category (§7 step 4 list; box-h and $ as if run alone on the box; 'local' = step-9 CPU-h)")
    print(f"{'category':20s} {'GPU-h':>8s} {'box CPU-h':>10s} {'box-h':>7s} {'$':>8s} {'local CPU-h':>12s}")
    for cat in CAT_ORDER:
        if cat in by:
            gh, ch, bh, lh = by[cat]
            print(f"{cat:20s} {gh:8.2f} {ch:10.2f} {bh:7.2f} {bh * rate:8.2f} {lh:12.2f}")
    for st in (5, 7):
        s = T0["steps"][st]
        print(f"step {st} on the box: GPU {s['gpu_h']:.1f} h, CPU {s['cpu_h']:.1f} h -> box {s['box_h_lo']:.1f}-"
              f"{s['box_h_hi']:.1f} h = ${s['box_h_lo'] * rate:.0f}-{s['box_h_hi'] * rate:.0f}"
              + ("  (critical path before step 6)" if st == 5 else ""))
    print(f"TOTAL box: GPU {T0['gpu_h']:.1f} h, CPU {T0['cpu_h']:.1f} h, box {T0['box_h_lo']:.1f}-{T0['box_h_hi']:.1f} h"
          f" = ${T0['usd_lo']:.0f}-{T0['usd_hi']:.0f}; step-9 matching {T0['local_cpu_h']:.1f} CPU-h = "
          f"{T0['step9_local_wall_h']:.1f} h on {args.local_cores} laptop cores, or ${T0['step9_cpubox_usd']:.0f} on a "
          f"{args.cpubox_cores}-core CPU box at ${args.cpubox_usd_per_hour}/h")
    # degrade order (§7): cumulative, in the fixed order; the fewest items that fit a cap
    print("\ndegrade order (cumulative; never-reduced items untouched):")
    rows, taken = [], set()
    prev = T0
    fit_at = 0 if args.cap_usd and T0["usd_hi"] <= args.cap_usd else None
    for k in range(1, 8):
        taken.add(k)
        cfg["degrade"] = set(taken)
        Tk = totals(build_jobs(R, ST, cfg), cfg)
        rows.append((k, Tk))
        print(f"  +{k} {DEGRADE[k]:58s} saves GPU {prev['gpu_h'] - Tk['gpu_h']:6.1f} h, box CPU "
              f"{prev['cpu_h'] - Tk['cpu_h']:6.1f} h, step-9 CPU {prev['local_cpu_h'] - Tk['local_cpu_h']:6.1f} h, "
              f"${prev['usd_hi'] - Tk['usd_hi']:5.0f} -> box {Tk['box_h_lo']:.1f}-{Tk['box_h_hi']:.1f} h, "
              f"${Tk['usd_lo']:.0f}-{Tk['usd_hi']:.0f}")
        if args.cap_usd and fit_at is None and Tk["usd_hi"] <= args.cap_usd:
            fit_at = k
        prev = Tk
    if args.cap_usd:
        if fit_at == 0:
            print(f"cap ${args.cap_usd}: fits with no degrade item (serial upper bound ${T0['usd_hi']:.0f})")
        elif fit_at:
            print(f"cap ${args.cap_usd}: take degrade items 1-{fit_at} (serial upper bound "
                  f"${rows[fit_at - 1][1]['usd_hi']:.0f})")
        else:
            print(f"cap ${args.cap_usd}: does NOT fit after item 7 (${rows[-1][1]['usd_hi']:.0f}): Frank approves the "
                  "extra cost, or the run stops before any TEST-B input reaches the box (§7)")
    if args.out:
        json.dump({"cfg": {k: (sorted(v) if isinstance(v, set) else v) for k, v in cfg.items() if k != "degrade"},
                   "jobs": J, "totals": T0, "degrade": [{"item": k, "text": DEGRADE[k], **v} for k, v in rows],
                   "rates": args.rates}, open(args.out, "w"), indent=1, default=str)
        print("->", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
        p.add_argument("--gpu", default="0")
        p.add_argument("--concurrency", type=int, default=1, help="MCF training copies run at once")
        p.add_argument("--ivp-cpb", type=int, nargs="+", default=[1, 8], help="iv-P --crystals-per-batch values")
        if name == "run":
            p.add_argument("--only", nargs="+", default=None)
            p.add_argument("--redo", action="store_true")
    p = sub.add_parser("cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--n-crystals", type=int, default=0, help="smoke: first N of the 20 (0 = all 20)")
    p.add_argument("--draws", type=int, default=S_DRAWS)
    p.add_argument("--press", type=int, default=4)
    p.add_argument("--fits", type=int, default=8)
    sub.add_parser("rates")
    sub.add_parser("stats")
    sub.add_parser("reported")
    p = sub.add_parser("project")
    p.add_argument("--rates", default=os.path.join(OUT, "rates.json"))
    p.add_argument("--stats", default=os.path.join(OUT, "set_stats.json"))
    p.add_argument("--usd-per-hour", type=float, required=True)
    p.add_argument("--cores", type=int, required=True, help="box CPU cores")
    p.add_argument("--gpu-job-cores", type=int, default=4, help="cores kept for the GPU jobs' drivers / loaders")
    p.add_argument("--overhead", type=float, default=1.15, help="box-hour multiplier for idle gaps and set-up")
    p.add_argument("--p6", choices=["off", "on"], default="off")
    p.add_argument("--ivp-encoder", choices=["trim", "pack"], default="pack")
    p.add_argument("--ivp-cpb", type=int, default=8)
    p.add_argument("--ff", default="batch_float32", help="<settings>_<dtype> of the FF* rates")
    p.add_argument("--ff-eval-frac", type=float, default=1.0, help="fraction of the evaluation caps used (1 = caps)")
    p.add_argument("--truth-evals", type=float, default=0, help="evaluations per truth (0 = measured mean, else 1000)")
    p.add_argument("--gplus-separate", action="store_true", help="add +G's contingency re-sampling")
    p.add_argument("--tier2", action="store_true", help="budget the §8 tier-2 diagnostics (degrade item 1 drops them)")
    p.add_argument("--g2-train-h", type=float, default=0.5, help="GPU-h of one G2 training run (tier-2 only)")
    p.add_argument("--testB", type=int, default=1000)
    p.add_argument("--local-cores", type=int, default=6)
    p.add_argument("--cpubox-cores", type=int, default=32)
    p.add_argument("--cpubox-usd-per-hour", type=float, default=0.30)
    p.add_argument("--cap-usd", type=float, default=0)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "cpu": cmd_cpu, "rates": cmd_rates, "stats": cmd_stats,
     "reported": cmd_reported,
     "project": cmd_project}[args.cmd](args)


if __name__ == "__main__":
    main()
