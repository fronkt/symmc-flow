"""N3 FF* (protocol §3.3): UMA s-1p1, OMC task, through fairchem-core in its own environment.

fairchem-core pins its own torch, so FF* runs in a separate venv (external/venv_fairchem locally; any python
with fairchem-core on the box, set with N3_FF_PYTHON or --ff-python). This file is both ends of a pipe:
  server  `<venv python> scripts/n3_ffstar.py serve ...` loads the predict unit once,
          load_predict_unit(ckpt, inference_settings, device=..., atom_refs=OmegaConf.load(refs)) (the call
          fairchem's own get_predict_unit makes), and answers pickled requests on stdin/stdout. One request is
          one batched predict call: task 'omc', pbc, positions wrapped into the cell, whole-cell energy (eV)
          and forces (eV/A) per structure. Nothing but frames is written to the frame stream.
  client  FFStar (imported by scripts/n3_classical.py in the main env) spawns the server and applies the §3.3
          failure rule: a call that raises, or returns a non-finite energy or force, is retried once in a
          FRESH process, one structure per call; a second failure is DETERMINISTIC (result None). Every
          retry is appended to <log_dir>/ff_retries.jsonl. A server that cannot start raises (environment
          error, never an FF* failure).
Every reply carries the RESOLVED inference state of the predict unit after that call (merge_mole, compile,
execution_mode, backbone backend class, tf32, activation checkpointing, dtype, graph generation). fairchem
resolves it lazily: on CUDA merge_mole selects the umas_fast_gpu backend, and a composition change under
merge_mole falls back for good to the unmerged, uncompiled GENERAL path. The client requires every reply to
equal `expect["state"]` (or, without one, the first reply it saw) and a server's fairchem-core version to equal
`expect["fairchem_core"]`; a mismatch raises FFConfigError (environment error, not an FF* failure).
--settings is fairchem's inference_settings name ("default" = merge_mole + torch.compile, which needs a C++
compiler; "batch" = unmerged, uncompiled); --dtype float64 recasts that preset's base_precision_dtype (used for
the finite-difference check). fairchem-core 2.23.0 / torch 2.13.0 when this was written.
    <venv python> scripts/n3_ffstar.py info                  # versions + SHA-256 of the weights and refs
    python scripts/n3_ffstar.py selftest --device cpu         # client -> server round trip, toy cell
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UMA_CKPT = os.path.join(REPO, "external", "uma", "uma-s-1p1.pt")
UMA_REFS = os.path.join(REPO, "external", "uma", "iso_atom_elem_refs.yaml")
TASK = "omc"


def default_python():
    if os.environ.get("N3_FF_PYTHON"):
        return os.environ["N3_FF_PYTHON"]
    venv = os.path.join(REPO, "external", "venv_fairchem")
    return os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------ server (fairchem venv)
def _load(args):
    import copy
    import torch
    from omegaconf import OmegaConf
    from importlib.metadata import version
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
    torch.set_num_threads(args.threads)
    settings = args.settings
    if args.dtype != "float32":                      # non-default precision: the named preset, recast
        settings = copy.deepcopy(guess_inference_settings(args.settings))
        settings.base_precision_dtype = getattr(torch, args.dtype)
    t0 = time.time()
    unit = load_predict_unit(args.ckpt, inference_settings=settings, device=args.device,
                             atom_refs=OmegaConf.load(args.refs))
    meta = {"fairchem_core": version("fairchem-core"), "torch": torch.__version__, "device": args.device,
            "settings": args.settings, "dtype": args.dtype, "threads": args.threads,
            "load_sec": round(time.time() - t0, 1), "pid": os.getpid()}
    return unit, meta


def _state(unit):
    """The predict unit's resolved inference state (plain types: the client env has no fairchem)."""
    s = unit.inference_settings
    em = s.execution_mode
    try:
        backend = type(unit.model.module.backbone.backend).__name__
    except Exception:
        backend = None
    return {"merge_mole": bool(s.merge_mole), "compile": bool(s.compile),
            "execution_mode": None if em is None else str(getattr(em, "value", em)), "backend": backend,
            "tf32": bool(s.tf32), "activation_checkpointing": bool(s.activation_checkpointing),
            "dtype": str(s.base_precision_dtype).replace("torch.", ""),
            "graph_gen": s.internal_graph_gen_version, "external_graph_gen": bool(s.external_graph_gen)}


def _predict(unit, structs):
    import torch
    from ase import Atoms
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    data, nat = [], []
    for Z, X, cell in structs:
        at = Atoms(numbers=np.asarray(Z), positions=np.asarray(X, float), cell=np.asarray(cell, float), pbc=True)
        at.wrap()
        unit.validate_atoms_data(at, TASK)
        data.append(AtomicData.from_ase(at, task_name=TASK, r_edges=False, radius=6.0, max_neigh=None,
                                        target_dtype=unit.inference_settings.base_precision_dtype))
        nat.append(len(at))
    t0 = time.time()
    pred = unit.predict(atomicdata_list_to_batch(data))
    E = pred["energy"].detach().to(torch.float64).cpu().numpy().reshape(-1)
    F = pred["forces"].detach().to(torch.float64).cpu().numpy()
    cuts = np.cumsum([0] + nat)
    return {"energy": E, "forces": [F[cuts[i]:cuts[i + 1]] for i in range(len(nat))],
            "sec": time.time() - t0, "state": _state(unit)}


def _send(out, obj):
    pickle.dump(obj, out, protocol=pickle.HIGHEST_PROTOCOL)
    out.flush()


def cmd_serve(args):
    out = os.fdopen(os.dup(1), "wb")      # frame stream = the original stdout
    os.dup2(2, 1)                          # anything fairchem prints goes to stderr
    sys.stdout = sys.stderr
    inp = sys.stdin.buffer
    try:
        unit, meta = _load(args)
        _send(out, dict(ok=True, ready=True, **meta))
    except Exception:
        _send(out, {"ok": False, "error": traceback.format_exc()})
        return
    while True:
        try:
            req = pickle.load(inp)
        except EOFError:
            return
        if req.get("op") == "quit":
            return
        try:
            _send(out, dict(ok=True, **_predict(unit, req["structs"])))
        except Exception:
            _send(out, {"ok": False, "error": traceback.format_exc()})


def cmd_info(args):
    import torch
    from importlib.metadata import version
    print(json.dumps({"fairchem_core": version("fairchem-core"), "torch": torch.__version__,
                      "python": sys.version.split()[0], "ckpt": args.ckpt, "ckpt_sha256": sha256(args.ckpt),
                      "refs": args.refs, "refs_sha256": sha256(args.refs)}, indent=1))


# ------------------------------------------------------------------------------ client (main env)
class FFDied(RuntimeError):
    pass


class FFConfigError(RuntimeError):
    """The FF* environment is not the fixed one (version, resolved inference state): stop, never retry."""


def toy_structs():
    """A synthetic 8-atom periodic cell (not CSD data): probe and self-test input."""
    cell = np.diag([6.0, 6.5, 7.0])
    Z = np.array([6, 6, 1, 1, 1, 1, 8, 1])
    X = np.array([[0, 0, 0], [1.33, 0, 0], [-0.55, 0.93, 0], [-0.55, -0.93, 0], [1.88, 0.93, 0],
                  [1.88, -0.93, 0], [3.2, 2.5, 2.0], [3.9, 2.5, 2.6]], float) + 1.0
    return Z, X, cell


class FFStar:
    """FF* evaluator. evaluate(structs) -> per structure (E, F) or None (deterministic failure).
    expect = {"fairchem_core": str, "state": dict} (either optional) pins the environment; `state` is the
    resolved inference state every reply so far has shown."""

    def __init__(self, python=None, ckpt=UMA_CKPT, refs=UMA_REFS, device="cpu", threads=2, settings="default",
                 dtype="float32", max_atoms=None, log_dir=None, expect=None):
        self.cmd = [python or default_python(), os.path.abspath(__file__), "serve", "--ckpt", ckpt, "--refs", refs,
                    "--device", device, "--threads", str(threads), "--settings", settings, "--dtype", dtype]
        self.log_dir = log_dir or os.path.join(REPO, "results", "n3", "private", "classical", "ff_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.max_atoms = max_atoms        # None = one predict call per request (protocol §3.3)
        self.expect = dict(expect or {})
        self.state = None
        self.proc = None
        self.meta = None
        self.n_spawn = 0
        self.stats = {"calls": 0, "structs": 0, "sec_predict": 0.0, "sec_spawn": 0.0, "retries": 0,
                      "deterministic": 0}

    def _check_state(self, state):
        ref = self.expect.get("state") or self.state
        if ref is not None and state != ref:
            raise FFConfigError(f"FF* resolved inference state changed: {state} (expected {ref})")
        self.state = state

    def _spawn(self):
        t0 = time.time()
        try:
            return self._spawn_inner()
        finally:
            self.stats["sec_spawn"] += time.time() - t0

    def _spawn_inner(self):
        self.n_spawn += 1
        log = open(os.path.join(self.log_dir, f"ff_server_{os.getpid()}_{self.n_spawn}.log"), "ab")
        p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        p._n3_log = log
        r = self._recv(p)
        if not r.get("ok"):
            self._kill(p)
            raise RuntimeError("FF* server failed to start:\n" + r.get("error", "?"))
        self.meta = {k: v for k, v in r.items() if k not in ("ok", "ready")}
        want = self.expect.get("fairchem_core")
        if want and self.meta.get("fairchem_core") != want:
            self._kill(p)
            raise FFConfigError(f"fairchem-core {self.meta.get('fairchem_core')} != the fixed {want}")
        return p

    @staticmethod
    def _recv(p):
        try:
            return pickle.load(p.stdout)
        except (EOFError, pickle.UnpicklingError, OSError) as e:
            raise FFDied(f"FF* server exited (code {p.poll()}): {e!r}")

    def _ask(self, p, structs):
        try:
            pickle.dump({"op": "eval", "structs": structs}, p.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            p.stdin.flush()
        except OSError as e:
            raise FFDied(f"FF* server pipe closed: {e!r}")
        r = self._recv(p)
        if not r["ok"]:
            raise RuntimeError(r["error"])
        return r

    @staticmethod
    def _kill(p):
        if p is None:
            return
        try:
            p.kill(); p.wait(timeout=30)
        except Exception:
            pass
        try:
            p._n3_log.close()
        except Exception:
            pass

    @staticmethod
    def _finite(E, F):
        return bool(np.isfinite(E)) and bool(np.isfinite(F).all())

    def evaluate(self, structs, tag=""):
        """One batched predict call (or, with max_atoms set, consecutive sub-batches of <= max_atoms atoms).
        A server that cannot START is an environment error and raises; only calls on a running server
        count as FF* failures."""
        out = [None] * len(structs)
        chunks, cur, n_at = [], [], 0
        for i, st in enumerate(structs):
            if cur and self.max_atoms and n_at + len(st[0]) > self.max_atoms:
                chunks.append(cur); cur, n_at = [], 0
            cur.append(i); n_at += len(st[0])
        if cur:
            chunks.append(cur)
        for ch in chunks:
            sub = [structs[i] for i in ch]
            for i, r in zip(ch, self._evaluate_call(sub, tag)):
                out[i] = r
        return out

    def _evaluate_call(self, structs, tag):
        out = [None] * len(structs)
        bad, why = [], None
        if self.proc is None:
            self.proc = self._spawn()                        # outside the try: a start failure raises
        try:
            r = self._ask(self.proc, structs)
        except Exception as e:
            bad, why = list(range(len(structs))), f"{type(e).__name__}: {str(e)[-400:]}"
            self._kill(self.proc); self.proc = None          # never reuse a process that failed
        else:
            self._check_state(r["state"])                    # FFConfigError propagates (not an FF* failure)
            self.stats["calls"] += 1; self.stats["structs"] += len(structs); self.stats["sec_predict"] += r["sec"]
            for i in range(len(structs)):
                if self._finite(r["energy"][i], r["forces"][i]):
                    out[i] = (float(r["energy"][i]), r["forces"][i])
                else:
                    bad.append(i); why = "non-finite"
        if bad:
            self._retry(structs, bad, out, tag, why)
        return out

    def _retry(self, structs, bad, out, tag, why):
        fresh = None
        try:
            with open(os.path.join(self.log_dir, "ff_retries.jsonl"), "a") as fh:
                for i in bad:
                    self.stats["retries"] += 1
                    ok, err = False, None
                    if fresh is None:
                        fresh = self._spawn()
                    try:
                        r = self._ask(fresh, [structs[i]])
                    except Exception as e:
                        r, err = None, f"{type(e).__name__}: {str(e)[-400:]}"
                        self._kill(fresh); fresh = None
                    if r is not None:
                        self._check_state(r["state"])
                        if self._finite(r["energy"][0], r["forces"][0]):
                            out[i] = (float(r["energy"][0]), r["forces"][0]); ok = True
                        else:
                            err = "non-finite"
                    if not ok:
                        self.stats["deterministic"] += 1
                    fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "index": i,
                                         "batch": len(structs), "atoms": int(len(structs[i][0])), "first": why,
                                         "retry_ok": ok, "retry_error": err}) + "\n")
        finally:
            if fresh is not None:
                self._close(fresh)

    def _close(self, p):
        try:
            pickle.dump({"op": "quit"}, p.stdin); p.stdin.flush(); p.wait(timeout=60)
        except Exception:
            pass
        self._kill(p)

    def close(self):
        if self.proc is not None:
            self._close(self.proc)
            self.proc = None


def cmd_selftest(args):
    """Toy periodic cell (not CSD data): round trip, translation invariance, force-energy consistency."""
    ff = FFStar(python=args.ff_python, device=args.device, threads=args.threads, settings=args.settings,
                dtype=args.dtype)
    rng = np.random.default_rng(0)
    Z, X, cell = toy_structs()
    t0 = time.time()
    r = ff.evaluate([(Z, X, cell), (Z, X + cell[0], cell), (Z, X + 0.01 * rng.standard_normal(X.shape), cell)])
    print("server", ff.meta, "first call incl. load %.1fs" % (time.time() - t0))
    print("resolved state", ff.state)
    (E0, F0), (E1, F1), (E2, F2) = r
    print("E", E0, "lattice-translated dE", E1 - E0, "max dF", float(np.abs(F1 - F0).max()))
    # energy change vs -F.dx along a small displacement
    d = 1e-3 * rng.standard_normal(X.shape)
    (Ea, _), (Eb, _) = ff.evaluate([(Z, X + d, cell), (Z, X - d, cell)])
    print("central dE", (Ea - Eb) / 2, "vs -F.d", float(-(F0 * d).sum()))
    ff.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "info", "selftest"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", default=UMA_CKPT)
        s.add_argument("--refs", default=UMA_REFS)
        s.add_argument("--device", default="cpu")
        s.add_argument("--threads", type=int, default=2)
        s.add_argument("--settings", default="default")
        s.add_argument("--dtype", default="float32", choices=["float32", "float64"])
        s.add_argument("--ff-python", default=None)
    args = ap.parse_args()
    {"serve": cmd_serve, "info": cmd_info, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
