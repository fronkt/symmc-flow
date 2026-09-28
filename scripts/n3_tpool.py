"""N3 helpers for the input-side label / Table R scripts (§2.6, §5): timed process pool, JSONL resume with the
§1.3 re-run rule, provenance.

run(): multiprocessing.Pool cannot stop one slow task. Here each worker runs one task at a time over its own
pipe; a worker that exceeds the limit is terminated and replaced, and its task is reported as "timeout".
Workers call `init(*init_args)` once and report ready before they receive a task, so start-up never counts
against a task. Results arrive in completion order as (task, status, result, seconds), status in
{"ok", "error", "timeout", "crash"}; "error" carries the exception text as the result.

rerun_once() / resolve_rerun(): §1.3 harness rule. An "error" or "crash" is re-run once, alone, in a fresh
process (same per-task limit); the same failure again is DETERMINISTIC (the caller scores it as a
non-match). Timeouts are their own count (§5) and are not re-run.

read_jsonl(): resume rows; a truncated last line (process killed mid-write) is dropped and trimmed off.
provenance(): SHA-256 of the scripts + library versions, recorded in every summary.

    from n3_tpool import run
    for task, status, res, dt in run(fn, tasks, init=load, init_args=("devtest",), workers=2, timeout=600):
        ...
fn(state, task) and init(...) must be module-level functions (spawn start method, Windows and Linux).
"""
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from multiprocessing.connection import wait

FAILED = ("error", "crash")


def read_jsonl(path, trim=True):
    """Rows of a JSONL file ([] if absent). An unparsable LAST line is a write cut short: it is dropped and, with
    trim (only for the caller's own resume file), the file is truncated before it so later appends stay
    line-aligned. An unparsable earlier line raises."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    rows, pos = [], 0
    lines = data.split(b"\n")
    for k, line in enumerate(lines):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if any(l.strip() for l in lines[k + 1:]):
                    raise SystemExit(f"{path}: unparsable line {k + 1} is not the last line")
                if trim:
                    with open(path, "r+b") as f:
                        f.truncate(pos)
                print(f"{path}: dropped a truncated last line ({len(line)} bytes)", flush=True)
                break
        pos += len(line) + 1
    return rows


def rerun_once(fn, task, init=None, init_args=(), timeout=600.0):
    """One task alone in a fresh worker process -> (status, result, seconds)."""
    for _, st, res, dt in run(fn, [task], init=init, init_args=init_args, workers=1, timeout=timeout):
        return st, res, dt
    raise RuntimeError("n3_tpool.rerun_once: no result")


def resolve_rerun(first, st, res):
    """Final status of a re-run, given the first attempt's row: ok stays ok; the same failure again (same
    exception text, or a second crash) is 'deterministic'; any other outcome keeps its own status."""
    if st == "ok":
        return "ok"
    if st == first["status"] and (st == "crash" or res == first.get("error")):
        return "deterministic"
    return st


def provenance(scripts):
    """SHA-256 of the given repo files and the versions of the libraries the constructions depend on."""
    from importlib import metadata
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = {}
    for rel in scripts:
        p = os.path.join(repo, rel)
        if os.path.exists(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read())
            sha[rel] = h.hexdigest()
        else:
            sha[rel] = None
    ver = {}
    for pkg in ("numpy", "scipy", "torch", "pymatgen", "spglib", "gemmi", "networkx", "ase", "rdkit"):
        try:
            ver[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            ver[pkg] = None
    return {"scripts_sha256": sha, "versions": ver, "python": sys.version.split()[0],
            "platform": platform.platform()}


def _loop(init, init_args, fn, conn):
    try:
        state = init(*init_args) if init is not None else None
    except Exception:
        conn.send(("init_error", None, traceback.format_exc(), 0.0))
        return
    conn.send(("ready", None, None, 0.0))
    while True:
        task = conn.recv()
        if task is None:
            return
        t0 = time.time()
        try:
            res, st = fn(state, task), "ok"
        except Exception as e:
            res, st = f"{type(e).__name__}: {e}", "error"
        conn.send((st, task, res, time.time() - t0))


def run(fn, tasks, init=None, init_args=(), workers=2, timeout=600.0, poll=0.5):
    ctx = mp.get_context("spawn")
    todo = list(tasks)[::-1]
    pending = len(todo)
    pool = {}                                   # conn -> dict(proc, task, t0, ready)

    def spawn():
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_loop, args=(init, init_args, fn, child), daemon=True)
        p.start()
        child.close()
        pool[parent] = {"proc": p, "task": None, "t0": None, "ready": False}

    def drop(conn):
        w = pool.pop(conn)
        if w["proc"].is_alive():
            w["proc"].terminate()
        w["proc"].join()
        conn.close()

    try:
        for _ in range(min(workers, pending)):
            spawn()
        while pending:
            for conn, w in pool.items():
                if w["ready"] and w["task"] is None and todo:
                    w["task"], w["t0"] = todo.pop(), time.time()
                    conn.send(w["task"])
            for conn in wait(list(pool), timeout=poll):
                w = pool[conn]
                try:
                    st, task, res, dt = conn.recv()
                except (EOFError, OSError):          # worker died (killed, out of memory, segfault)
                    task, dt = w["task"], time.time() - (w["t0"] or time.time())
                    drop(conn)
                    if task is None:
                        raise RuntimeError("n3_tpool: worker died during start-up")
                    pending -= 1
                    yield task, "crash", None, dt
                    if todo:
                        spawn()
                    continue
                if st == "init_error":
                    raise RuntimeError(f"n3_tpool: worker init failed:\n{res}")
                if st == "ready":
                    w["ready"] = True
                    continue
                w["task"] = None
                pending -= 1
                yield task, st, res, dt
            for conn in [c for c, w in pool.items() if not w["proc"].is_alive()]:
                # a worker that died before unpickling its pipe end never signals EOF on Windows
                try:
                    if conn.poll():
                        continue                        # its last message is read on the next pass
                except (EOFError, OSError):
                    pass
                task, t0 = pool[conn]["task"], pool[conn]["t0"]
                drop(conn)
                if task is None:
                    raise RuntimeError("n3_tpool: worker died during start-up")
                pending -= 1
                yield task, "crash", None, time.time() - t0
                if todo:
                    spawn()
            now = time.time()
            for conn in [c for c, w in pool.items() if w["task"] is not None and now - w["t0"] > timeout]:
                task, dt = pool[conn]["task"], now - pool[conn]["t0"]
                drop(conn)
                pending -= 1
                yield task, "timeout", None, dt
                if todo:
                    spawn()
    finally:
        for conn, w in list(pool.items()):
            try:
                conn.send(None)
            except (OSError, BrokenPipeError):
                pass
        for conn, w in list(pool.items()):
            w["proc"].join(timeout=5)
            if w["proc"].is_alive():
                w["proc"].terminate()
                w["proc"].join()
            conn.close()
        pool.clear()
