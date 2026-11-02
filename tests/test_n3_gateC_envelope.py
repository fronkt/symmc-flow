"""Gate C-b GPU half, A4 item 2 option (b): cb_envelope on synthetic GPU-like outputs. No MCF code, data or GPU.

Noise model: each run's tensor = base + run-level scale x iid noise, the scale shared by every tensor of a batch
(one sampler trajectory per batch), lognormal so some runs diverge more than others; the corrupt tensors are
bitwise identical in every run, as on the box on 2026-10-22.
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import n3_mcf_gateC as G  # noqa: E402

NOISY = ("forward/cart_coords", "forward/lattices", "forward/pred_rotmats", "forward/pred_trans",
         "model_step/cell_loss", "model_step/rots_vf_loss", "model_step/se3_vf_loss", "model_step/trans_loss")
DET = ("corrupt/lattice_t", "corrupt/rotmats_t", "corrupt/so3_t", "corrupt/trans_t")
KEYS = [f"b{b}/{k}" for b in range(3) for k in NOISY + DET]


def make_runs(rng, n, shift=None, det_change=None, drop=None):
    """n runs; runs[0] plays the patched one. shift = (key, size): a systematic change of the patched run."""
    base = {k: torch.from_numpy(rng.normal(size=40)) for k in KEYS}
    runs = []
    for i in range(n):
        scale = {b: float(np.exp(rng.normal())) for b in range(3)}
        r = {}
        for k in KEYS:
            if k.split("/")[1] == "corrupt":
                r[k] = base[k].clone()
            else:
                r[k] = base[k] + 1e-3 * scale[int(k[1])] * torch.from_numpy(rng.normal(size=40))
        runs.append(r)
    if shift:
        runs[0][shift[0]] = runs[0][shift[0]] + shift[1]
    if det_change:
        runs[0][det_change] = runs[0][det_change] + 1e-7
    if drop:
        del runs[0][drop]
    return runs


def passes(runs):
    T, _ = G.cb_envelope(runs, sorted(set().union(*runs)))
    return bool(T[0] <= T[1:].max())


def old_rule_passes(runs):
    P, U, U2 = runs[0], runs[1], runs[2]
    return all(G.cb_dmax(P[k], U[k]) <= G.cb_dmax(U2[k], U[k]) for k in KEYS)


def test_false_failure_rate_when_patch_changes_nothing():
    rng = np.random.default_rng(0)
    trials = 300
    fails = sum(not passes(make_runs(rng, 20)) for _ in range(trials))
    assert fails / trials <= 0.09, fails          # bound 1/20 = 0.05; 0.09 is ~3 binomial SE above it


def test_old_rule_fails_a_perfect_patch_most_of_the_time():
    rng = np.random.default_rng(1)
    trials = 200
    fails = sum(not old_rule_passes(make_runs(rng, 3)) for _ in range(trials))
    assert fails / trials > 0.9, fails


def test_systematic_change_fails():
    """Power under this deliberately heavy-tailed noise (median pair noise ~0.0028 on the shifted tensor), measured
    2026-11-02 over 100 trials: shift 0.02 -> 32%, 0.05 -> 83%, 0.1 -> 97%, 0.25 -> 100% detected. The envelope only
    flags changes larger than the noisiest unpatched run; the exact check is the CPU half (bitwise)."""
    rng = np.random.default_rng(2)
    trials = 100
    fails = sum(not passes(make_runs(rng, 20, shift=("b1/forward/lattices", 0.1))) for _ in range(trials))
    assert fails / trials >= 0.9, fails


def test_change_to_a_bitwise_deterministic_tensor_fails():
    rng = np.random.default_rng(3)
    assert not passes(make_runs(rng, 20, det_change="b2/corrupt/trans_t"))


def test_missing_tensor_in_patched_run_fails():
    rng = np.random.default_rng(4)
    assert not passes(make_runs(rng, 20, drop="b0/forward/pred_trans"))


def test_cmd_cb_cuda_wiring_and_resume(tmp_path, monkeypatch):
    """cmd_cb --device cuda with a fake worker: 1 + noise_runs runs, the A4 record fields, and a second call that
    reuses every finished run file (no worker started). Writes only under tmp_path."""
    import json
    import types
    monkeypatch.setattr(G, "GC_RUNS", str(tmp_path / "gc"))
    monkeypatch.setattr(G.ML, "SMOKE_DIR", str(tmp_path / "smoke"))
    os.makedirs(tmp_path / "smoke")
    for f in ("state.ckpt", "config.yaml", "test_molcrystal_normalized.pkl.gz"):
        (tmp_path / f).write_bytes(b"x")
    rng = np.random.default_rng(6)
    base = make_runs(rng, 1)[0]
    calls = []

    def fake_run(cmd, env=None):
        calls.append(cmd)
        out, tree = cmd[cmd.index("--out") + 1], cmd[cmd.index("--tree") + 1]
        t = {k: v + 1e-3 * torch.from_numpy(rng.normal(size=v.shape)) if "corrupt" not in k else v.clone()
             for k, v in base.items()}
        torch.save({"tensors": t, "tree": os.path.abspath(tree), "batches": 3, "crystals": 750, "dataset": 750}, out)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(G.subprocess, "run", fake_run)
    args = types.SimpleNamespace(device="cuda", batches=3, batch_size=250, steps=50, threads=1,
                                 state=str(tmp_path / "state.ckpt"), config=str(tmp_path / "config.yaml"),
                                 pickle=str(tmp_path / "test_molcrystal_normalized.pkl.gz"),
                                 unpatched=str(tmp_path / "unpatched"), noise_runs=19, shims=False)
    G.cmd_cb(args)
    assert len(calls) == 20
    rec = json.load(open(tmp_path / "smoke" / "gateC_cb_smoke_cuda.json"))
    assert rec["runs"] == 20 and rec["false_fail_bound"] == 0.05 and len(rec["stat_unpatched"]) == 19
    assert rec["tensors"] == len(KEYS) and len(rec["run_files_sha256"]) == 20 and isinstance(rec["pass"], bool)
    G.cmd_cb(args)                                    # resume: every run file exists, so no worker starts
    assert len(calls) == 20
    assert json.load(open(tmp_path / "smoke" / "gateC_cb_smoke_cuda.json"))["stat_patched"] == rec["stat_patched"]


def test_patched_identical_to_an_unpatched_run_passes():
    rng = np.random.default_rng(5)
    runs = make_runs(rng, 20)
    runs[0] = {k: v.clone() for k, v in runs[7].items()}
    assert passes(runs)
