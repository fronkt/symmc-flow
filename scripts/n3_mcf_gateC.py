"""N3 / G3: Gate C (competence and patch equivalence) for the MolCrystalFlow fork, on the box before any N3 MCF
training (tasks/n3_protocol.md §3.2 "Gate C"). Released checkpoint and data are used ONLY here (§1.2).

  download   Zenodo 19673190 (model-checkpoints.zip, thurlemann23.zip) -> external/mcf_release, md5 checked
             against the Zenodo record; model-checkpoints/thurlemann23/{best.ckpt, config.yaml} and
             thurlemann23/preprocessed/normalized/*; the released VAL pickle is copied into its own dir as
             test_molcrystal_normalized.pkl.gz (inference.py hard-codes test.pt).
  ca_infer   C-a sampling: our patched code, flag OFF, stock Thurlemann config (--config-name=inference: model.yaml
             num_atom_types 7, molcrystal_lattice prior), released best.ckpt; 50 steps, s_uF 9, s_uR 3, 5 runs x
             10 samples (inference.seed 0-4) on the released TEST pickle and on VAL-as-test; one visible GPU.
  ca_match   MCF's own scorer once per stol (run_structure_matching.py --num_samples 10 --stol {0.8|0.9|1.0}
             --ltol 0.3 --angle_tol 10) on every run, then n3_score.py gate-ca (same predictions files, MCF's
             settings) per label -> agreement (i) >= 99% of (crystal, stol) outcomes and 5-run medians vs Fig. 3a
             (ii: stol 0.9 / 1.0 within +-5 pp of 30% / 66%; stol 0.8 reported). If only VAL agrees, C-a passes
             and this is disclosed.
  cb_setup   an unpatched checkout at 3c493f8 (n3_mcf/_gateC/mcf_unpatched; clean, HEAD asserted).
  cb         C-b: FlowModule(cfg) built in the patched and in the unpatched tree (separate processes: the two trees
             are the same package), ONE shared state_dict (released best.ckpt), flag OFF, P6 off, stock Thurlemann
             config (the released config.yaml); per released TEST batch (the unshuffled loader at --batch_size,
             default 250: the 750 released TEST structures in exactly 3 batches -- the released batch_size.test
             of 512 gives only 2) random/numpy/torch seeded identically before each call: FlowModule.forward (the
             sampler) and one corrupt_batch + model_step. CPU: every tensor bitwise equal. --device cuda: the
             patched-vs-unpatched max difference is no larger than unpatched-vs-unpatched at the same seed.
             C-b passes only when BOTH the cpu and the cuda halves have run and passed.
  cc_infer   C-c sampling: released checkpoint, stock config, P3 ON (++inference.gt_lattice_trans=true), 16 samples,
             released TEST pickle.
  cc         C-c: every output lattice and centroid of that predictions_16.pt equals the input truth bitwise
             (lattices == lattice_1, pred_trans == trans_1, against gt_data_batch and the pickle).
Results (pass/fail and counts only): results/n3/gateC.json; details under results/n3/private/gateC/.

    python scripts/n3_mcf_gateC.py download
    python scripts/n3_mcf_gateC.py ca_infer --gpu 0 --run
    python scripts/n3_mcf_gateC.py ca_match --num_cpus 16
    python scripts/n3_mcf_gateC.py cb_setup --src https://github.com/Liu-Group-UF/MolCrystalFlow
    python scripts/n3_mcf_gateC.py cb --device cpu
    python scripts/n3_mcf_gateC.py cb --device cuda
    python scripts/n3_mcf_gateC.py cc_infer --gpu 0 --run
    python scripts/n3_mcf_gateC.py cc
"""
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import torch

import n3_mcf_lib as ML

ZENODO = "19673190"
RELEASE = os.path.join(REPO, "external", "mcf_release")
CKPT = os.path.join(RELEASE, "model-checkpoints", "thurlemann23", "best.ckpt")
NORM = os.path.join(RELEASE, "thurlemann23", "preprocessed", "normalized")
VAL_AS_TEST = os.path.join(RELEASE, "val_as_test")
GC_RUNS = os.path.join(REPO, "n3_mcf", "_gateC")
UNPATCHED = os.path.join(GC_RUNS, "mcf_unpatched")
GATE_JSON = os.path.join(REPO, "results", "n3", "gateC.json")
PRIV = os.path.join(ML.nc.PRIVATE, "gateC")
STOLS = (0.8, 0.9, 1.0)
CA_KNOBS = ["interpolant.sampling.num_timesteps=50", "interpolant.trans.scaling=9",
            "interpolant.rots.exp_rate=3", "inference.num_samples=10"]
LABELS = {"test": NORM, "val": VAL_AS_TEST}


def fwd(p):
    return os.path.abspath(p).replace("\\", "/")


def update_gate(section, rec):
    g = json.load(open(GATE_JSON)) if os.path.exists(GATE_JSON) else {"protocol": "tasks/n3_protocol.md §3.2 Gate C"}
    g[section] = rec
    parts = [g.get(k, {}).get("pass") for k in ("C-a", "C-b", "C-c")]
    g["all_pass"] = all(p is True for p in parts)
    ML.atomic_json(g, GATE_JSON, indent=1)
    return g


# ============================================================================================ download
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_download(args):
    import urllib.request
    import zipfile
    os.makedirs(RELEASE, exist_ok=True)
    rec = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO}"))
    files = {f["key"]: f for f in rec["files"]}
    got = {}
    for key in ("model-checkpoints.zip", "thurlemann23.zip"):
        f = files[key]
        dst = os.path.join(RELEASE, key)
        want = f["checksum"].split(":", 1)[1]
        if not (os.path.exists(dst) and md5(dst) == want):
            url = f["links"]["self"]
            print(f"downloading {key} ({f['size'] / 1e6:.0f} MB) from {url}", flush=True)
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
        assert md5(dst) == want, f"{key}: md5 mismatch"
        with zipfile.ZipFile(dst) as z:
            z.extractall(RELEASE)
        got[key] = {"md5": want, "sha256": ML.sha256(dst), "size": os.path.getsize(dst)}
    assert os.path.exists(CKPT) and os.path.exists(os.path.join(os.path.dirname(CKPT), "config.yaml")), CKPT
    os.makedirs(VAL_AS_TEST, exist_ok=True)
    shutil.copyfile(os.path.join(NORM, "val_molcrystal_normalized.pkl.gz"),
                    os.path.join(VAL_AS_TEST, "test_molcrystal_normalized.pkl.gz"))
    out = {"zenodo": ZENODO, "files": got, "best_ckpt_sha256": ML.sha256(CKPT),
           "test_pickle_sha256": ML.sha256(os.path.join(NORM, "test_molcrystal_normalized.pkl.gz")),
           "val_pickle_sha256": ML.sha256(os.path.join(VAL_AS_TEST, "test_molcrystal_normalized.pkl.gz"))}
    ML.atomic_json(out, os.path.join(PRIV, "download.json"), indent=1)
    print(json.dumps(out))


# ============================================================================================ inference calls
def released_call(cache, run_dir, seed, num_samples, p3=False):
    ov = [f"inference.ckpt_path={fwd(CKPT)}", f"data.cache_dir={fwd(cache)}"] + \
         [k for k in CA_KNOBS if not k.startswith("inference.num_samples")] + \
         [f"inference.num_samples={num_samples}", f"inference.seed={seed}",
          f"inference.output_dir={fwd(os.path.join(run_dir, 'out'))}",
          f"inference.inference_dir={fwd(os.path.join(run_dir, 'hydra'))}",
          "inference.save_trajectories=False", "data.loader.num_workers=4"]
    if p3:
        ov.append("++inference.gt_lattice_trans=true")
    return [sys.executable, "molcrystalflow/experiments/inference.py", "--config-name=inference"] + ov


def run_calls(calls, args):
    import n3_mcf_run as RUN
    env = RUN.env_for(args.gpu)
    for run_dir, cmd, S in calls:
        pred = os.path.join(run_dir, "out", f"predictions_{S}.pt")
        print(f"cd {fwd(ML.MCF)} && CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<gpu>')} python "
              + " ".join(cmd[1:]))
        if not args.run:
            continue
        if os.path.exists(pred):
            print(f"  exists: {pred}")
            continue
        RUN.require_one_gpu(env)
        os.makedirs(run_dir, exist_ok=True)
        t0 = time.time()
        with open(os.path.join(run_dir, "run.log"), "a") as fh:
            r = subprocess.run(cmd, cwd=ML.MCF, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode:
            raise SystemExit(f"inference failed ({r.returncode}): {run_dir}/run.log")
        ML.atomic_json({"cmd": cmd, "sec": round(time.time() - t0, 1), "predictions_sha256": ML.sha256(pred),
                        "ckpt_sha256": ML.sha256(CKPT)}, os.path.join(run_dir, "n3_gateC_run.json"), indent=1)


def cmd_ca_infer(args):
    calls = [(os.path.join(GC_RUNS, "ca", lab, f"r{r}"), released_call(cache, os.path.join(GC_RUNS, "ca", lab, f"r{r}"),
                                                                        r, 10), 10)
             for lab, cache in LABELS.items() for r in range(5)]
    run_calls(calls, args)


def cmd_ca_match(args):
    recs = {}
    for lab, cache in LABELS.items():
        runs = [os.path.join(GC_RUNS, "ca", lab, f"r{r}", "out") for r in range(5)]
        for d in runs:
            pt = os.path.join(d, "predictions_10.pt")
            for stol in STOLS:
                csv = os.path.join(d, "results", f"all_atom_rmsd_results_10_stol{stol}.csv")
                cmd = [sys.executable, "molcrystalflow/experiments/run_structure_matching.py", "--pt_file", fwd(pt),
                       "--num_samples", "10", "--stol", str(stol), "--ltol", "0.3", "--angle_tol", "10",
                       "--num_cpus", str(args.num_cpus)]
                print(" ".join(["python"] + cmd[1:]))
                if os.path.exists(csv):
                    continue
                r = subprocess.run(cmd, cwd=ML.MCF, stdout=open(os.path.join(d, f"match_stol{stol}.log"), "a"),
                                   stderr=subprocess.STDOUT)
                if r.returncode:
                    raise SystemExit(f"run_structure_matching failed: {d} stol {stol}")
        cmd = [sys.executable, os.path.join(REPO, "scripts", "n3_score.py"), "gate-ca", "--label", lab,
               "--pickle", os.path.join(cache, "test_molcrystal_normalized.pkl.gz"),
               "--predictions", *[os.path.join(d, "predictions_10.pt") for d in runs],
               "--mcf-results", *runs, "--workers", str(args.num_cpus)]
        print(" ".join(["python"] + cmd[1:]))
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode:
            raise SystemExit(f"n3_score.py gate-ca failed for {lab}")
        s = json.load(open(os.path.join(ML.nc.PRIVATE, f"gateCa_{lab}.json")))
        recs[lab] = {"n": s["n"], "median_any_pct": s["median_any_pct"], "per_run_any_pct":
                     {m: [e["any_pct"][m] for e in s["runs"]] for m in s["median_any_pct"]},
                     "fig3a_ref_pct": s["fig3a_ref_pct"], "ii_within_5pp": s["ii_within_5pp"], "ii_pass": s["ii_pass"],
                     "agreement": None if s["agreement"] is None else
                     {k: s["agreement"][k] for k in ("outcomes", "agree", "fraction")}, "i_pass": s.get("i_pass")}
    i_pass = all(v["i_pass"] for v in recs.values())
    ii_test, ii_val = recs["test"]["ii_pass"], recs["val"]["ii_pass"]
    rec = {"labels": recs, "i_pass": i_pass, "ii_pass_test": ii_test, "ii_pass_val": ii_val,
           "pass": bool(i_pass and (ii_test or ii_val)),
           "disclose_only_val_agrees": bool(i_pass and ii_val and not ii_test),
           "on_ii_failure": "record the diagnosis and the TEST / VAL 5-run medians at stol 0.9 / 1.0 in a dated §9 "
                            "amendment before any N3 MCF training; H1 sentences carry the harness clause"}
    g = update_gate("C-a", rec)
    print(json.dumps({"C-a": {k: rec[k] for k in ("i_pass", "ii_pass_test", "ii_pass_val", "pass")},
                      "all_pass": g["all_pass"]}))


# ============================================================================================ C-b
def cb_pass(cb):
    """C-b passes only when both halves (CPU bitwise; GPU no worse than unpatched-vs-unpatched) ran and passed."""
    return all(isinstance(cb.get(d), dict) and cb[d].get("pass") is True for d in ("cpu", "cuda"))


def cmd_cb_setup(args):
    import n3_mcf_fork as F
    if os.path.exists(UNPATCHED):
        raise SystemExit(f"{UNPATCHED} exists")
    F.checkout(args.src, UNPATCHED)
    head = F.git("rev-parse", "HEAD", cwd=UNPATCHED).stdout.strip()
    dirty = F.git("status", "--porcelain", cwd=UNPATCHED).stdout.strip()
    assert head.startswith(F.BASE) and not dirty, (head, dirty)
    print(f"unpatched {F.BASE} at {UNPATCHED}")


def cb_worker(args):
    """One tree (its own process): the §3.2 C-b calls, every output tensor saved for the comparison."""
    tree = os.path.abspath(args.tree)
    sys.path.insert(0, tree)
    if args.shims:
        import importlib.util
        if importlib.util.find_spec("torch_scatter") is not None:
            raise SystemExit("--shims is for the laptop smoke only (torch_scatter is installed here)")
        sys.path.insert(1, ML.SHIMS)
    import molcrystalflow
    assert os.path.abspath(molcrystalflow.__file__).startswith(tree), molcrystalflow.__file__
    if args.shims:
        import importlib.util
        spec = importlib.util.spec_from_file_location("n3_mcf_shims", os.path.join(ML.SHIMS, "n3_mcf_shims.py"))
        sh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sh)
        sh.install()
    from omegaconf import OmegaConf
    from molcrystalflow.data.dataset import MCDataset
    from molcrystalflow.models.molcrystalflow import FlowModule
    from torch_geometric.loader import DataLoader
    torch.set_num_threads(args.threads)
    cfg = OmegaConf.load(args.config)
    dev = torch.device(args.device)
    sd = torch.load(args.state, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    mod = FlowModule(cfg)
    mod.load_state_dict(sd)
    mod.eval().to(dev)
    ds = MCDataset(cache_path=args.pickle.replace("_molcrystal_normalized.pkl.gz", ".pt"), dataset_cfg=cfg.data,
                   is_training=False)
    out = {}
    n_batches, n_crystals = 0, 0
    for k, batch in enumerate(DataLoader(ds, batch_size=args.batch_size, shuffle=False)):
        if k == args.batches:
            break
        n_batches, n_crystals = n_batches + 1, n_crystals + int(batch.num_graphs)
        batch = batch.to(dev)

        def seed(s):
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(s)
        seed(1000 + k)
        mod.interpolant.set_device(dev)
        with torch.no_grad():
            r = mod.forward(copy.deepcopy(batch), num_timesteps=args.steps)
        for key in ("cart_coords", "lattices", "pred_trans", "pred_rotmats"):
            out[f"b{k}/forward/{key}"] = r[key].detach().cpu()
        seed(2000 + k)
        with torch.no_grad():
            noisy = mod.interpolant.corrupt_batch(batch)
            losses = mod.model_step(noisy)
        for key, v in losses.items():
            out[f"b{k}/model_step/{key}"] = v.detach().cpu()
        for key in ("rotmats_t", "trans_t", "lattice_t", "so3_t"):
            out[f"b{k}/corrupt/{key}"] = noisy[key].detach().cpu()
    torch.save({"tensors": out, "tree": tree, "molcrystalflow": molcrystalflow.__file__, "batches": n_batches,
                "crystals": n_crystals, "dataset": len(ds)}, args.out)


def cmd_cb(args):
    tmp = os.path.join(GC_RUNS, "cb")
    os.makedirs(tmp, exist_ok=True)
    state = args.state or CKPT
    config = args.config or os.path.join(os.path.dirname(CKPT), "config.yaml")
    pickle_ = args.pickle or os.path.join(NORM, "test_molcrystal_normalized.pkl.gz")
    unpatched = args.unpatched or UNPATCHED

    def run(tree, tag):
        o = os.path.join(tmp, f"{tag}_{args.device}.pt")
        cmd = [sys.executable, os.path.abspath(__file__), "_cb_worker", "--tree", tree, "--state", state,
               "--config", config, "--pickle", pickle_, "--out", o, "--device", args.device, "--batches",
               str(args.batches), "--batch_size", str(args.batch_size), "--threads", str(args.threads),
               "--steps", str(args.steps)] + (["--shims"] if args.shims else [])
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        r = subprocess.run(cmd, env=env)
        if r.returncode:
            raise SystemExit(f"C-b worker failed in {tree}")
        w = torch.load(o, weights_only=False)
        cover.append({k: w.get(k) for k in ("batches", "crystals", "dataset")})
        return w["tensors"]
    cover = []
    P = run(ML.MCF, "patched")
    U = run(unpatched, "unpatched")
    keys = sorted(set(P) | set(U))
    same = {k: k in P and k in U and P[k].shape == U[k].shape and torch.equal(P[k], U[k]) for k in keys}
    rec = {"device": args.device, "batches": args.batches, "batch_size": args.batch_size, "steps": args.steps,
           "threads": args.threads, "state_dict_sha256": ML.sha256(state), "config": config,
           "pickle_sha256": ML.sha256(pickle_), "tensors": len(keys), "shims": bool(args.shims),
           "coverage": cover[0], "all_batches_run": all(c["batches"] == args.batches for c in cover)}
    if args.device == "cpu":
        rec["bitwise_equal"] = sum(same.values())
        rec["differing"] = [k for k, v in same.items() if not v]
        rec["pass"] = all(same.values()) and len(keys) > 0 and rec["all_batches_run"]
    else:
        U2 = run(unpatched, "unpatched2")
        rec["all_batches_run"] = all(c["batches"] == args.batches for c in cover)

        def dmax(a, b):
            if a is None or b is None or a.shape != b.shape:
                return float("inf")
            return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        cmp_ = {k: {"patched_vs_unpatched": dmax(P.get(k), U.get(k)), "unpatched_vs_unpatched": dmax(U2.get(k), U.get(k))}
                for k in sorted(set(keys) | set(U2))}
        rec["max_diffs"] = cmp_
        rec["pass"] = (all(v["patched_vs_unpatched"] <= v["unpatched_vs_unpatched"] < float("inf") for v in cmp_.values())
                       and len(keys) > 0 and rec["all_batches_run"])
    reportable = not args.shims and state == CKPT
    if reportable:
        g = json.load(open(GATE_JSON)) if os.path.exists(GATE_JSON) else {}
        cb = g.get("C-b", {})
        cb[args.device] = rec
        cb["pass"] = cb_pass(cb)
        update_gate("C-b", cb)
    else:
        ML.atomic_json(rec, os.path.join(ML.SMOKE_DIR, f"gateC_cb_smoke_{args.device}.json"), indent=1)
    print(json.dumps({k: rec[k] for k in rec if k not in ("max_diffs",)}))


# ============================================================================================ C-c
def cmd_cc_infer(args):
    rd = os.path.join(GC_RUNS, "cc", "test")
    run_calls([(rd, released_call(NORM, rd, 0, 16, p3=True), 16)], args)


def cmd_cc(args):
    import gzip
    import pickle
    pred_path = args.predictions or os.path.join(GC_RUNS, "cc", "test", "out", "predictions_16.pt")
    pk = args.pickle or os.path.join(NORM, "test_molcrystal_normalized.pkl.gz")
    pred = torch.load(pred_path, map_location="cpu", weights_only=False)
    with gzip.open(pk, "rb") as f:
        entries = pickle.load(f)
    g = pred["gt_data_batch"]
    S = int(pred["lattices"].shape[0])
    L1 = torch.stack([e["lattice_1"] for e in entries])
    T1 = torch.cat([e["trans_1"] for e in entries])
    c = {"S": S, "n": len(entries),
         "gt_equals_pickle": bool(torch.equal(g["lattice_1"], L1) and torch.equal(g["trans_1"], T1)),
         "lattice_bitwise": sum(bool(torch.equal(pred["lattices"][s], g["lattice_1"])) for s in range(S)),
         "trans_bitwise": sum(bool(torch.equal(pred["pred_trans"][s], g["trans_1"])) for s in range(S)),
         "max_lattice_dev": float((pred["lattices"].double() - g["lattice_1"].double()).abs().max()),
         "max_trans_dev": float((pred["pred_trans"].double() - g["trans_1"].double()).abs().max())}
    c["pass"] = bool(c["gt_equals_pickle"] and c["lattice_bitwise"] == S and c["trans_bitwise"] == S)
    c["predictions_sha256"] = ML.sha256(pred_path)
    if args.predictions is None:
        c["ckpt_sha256"] = ML.sha256(CKPT)
        update_gate("C-c", c)
    else:
        ML.atomic_json(c, os.path.join(ML.SMOKE_DIR, "gateC_cc_smoke.json"), indent=1)
    print(json.dumps(c))


# ============================================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    for name in ("ca_infer", "cc_infer"):
        p = sub.add_parser(name)
        p.add_argument("--gpu", default=None)
        p.add_argument("--run", action="store_true")
    p = sub.add_parser("ca_match")
    p.add_argument("--num_cpus", type=int, default=8)
    p = sub.add_parser("cb_setup")
    p.add_argument("--src", default="https://github.com/Liu-Group-UF/MolCrystalFlow")
    p = sub.add_parser("cb")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=250, help="250: the 750 released TEST structures in 3 batches")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--state", default=None, help="default: the released best.ckpt (smoke: any Lightning ckpt)")
    p.add_argument("--config", default=None, help="default: the released config.yaml")
    p.add_argument("--pickle", default=None, help="default: the released TEST pickle")
    p.add_argument("--unpatched", default=None)
    p.add_argument("--shims", action="store_true")
    p = sub.add_parser("_cb_worker")
    for k in ("--tree", "--state", "--config", "--pickle", "--out", "--device"):
        p.add_argument(k, required=True)
    for k, d in (("--batches", 3), ("--batch_size", 16), ("--threads", 1), ("--steps", 50)):
        p.add_argument(k, type=int, default=d)
    p.add_argument("--shims", action="store_true")
    p = sub.add_parser("cc")
    p.add_argument("--predictions", default=None, help="smoke: any P3-ON predictions file (not the gate)")
    p.add_argument("--pickle", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.cmd == "_cb_worker":
        return cb_worker(args)
    {"download": cmd_download, "ca_infer": cmd_ca_infer, "ca_match": cmd_ca_match, "cb_setup": cmd_cb_setup,
     "cb": cmd_cb, "cc_infer": cmd_cc_infer, "cc": cmd_cc}[args.cmd](args)


if __name__ == "__main__":
    main()
