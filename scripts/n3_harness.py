"""N3 harness: run CPU jobs (matcher fits, energies, sampling) under the protocol §1.3 failure rule.

Every task first runs in a pool of long-lived worker processes. A task whose worker dies, that raises, or
that runs longer than `slow` seconds (None = never) is a HARNESS failure: its worker is killed and replaced,
and the task is re-run ALONE in a fresh process with no wall-clock limit. A failure that recurs identically
in a fresh process is DETERMINISTIC; a different failure gets one more fresh-process run, after which any
failure is deterministic. No task is dropped: each yields exactly one outcome, and every failure and re-run
is passed to `log`.

    import n3_harness as H
    for key, status, result, info in H.run(fn, [(key, task), ...], workers=2, slow=1200.0, log=print):
        ...   # status "ok" (result = fn(task)) or "det_fail" (result None; info["fails"] lists each failure)

`fn` and `init` must be module-level functions (picklable by reference; the platform's default start method is
used, spawn on Windows, fork on Linux). `prior` = {key: [failure text]} sends those tasks straight to a fresh
process, for failures that happened outside this harness (e.g. an energy call inside a sampling worker); such
texts must be cut with err_text() so an identical recurrence compares equal.

Resumable JSONL outputs: read_jsonl() skips an unfinished final line (a kill mid-write), repair_jsonl() cuts it
off before a run appends again.
"""
import collections
import json
import multiprocessing as mp
import os
import time
from multiprocessing.connection import wait

ERR_LEN = 400


def err_text(e):
    """The failure text of an exception, as the harness records it."""
    return f"{type(e).__name__}: {e}"[:ERR_LEN]


def read_jsonl(path):
    """Rows of a resumable JSONL file. An unparseable FINAL line without its newline (a kill mid-write) is
    skipped; any other unparseable line is corruption and raises."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    rows = []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if k == len(lines) - 1:                       # no newline after it: unfinished final write
                print(f"{path}: skipping an unfinished final line ({len(line)} bytes)", flush=True)
                continue
            raise ValueError(f"{path}: corrupt line {k + 1}")
    return rows


def repair_jsonl(path):
    """Cut an unfinished final line (no trailing newline) off a JSONL file before appending; returns the
    number of bytes removed."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0
        f.seek(0)
        data = f.read()
        keep = data.rfind(b"\n") + 1
        f.truncate(keep)
    print(f"{path}: removed an unfinished final line ({len(data) - keep} bytes) before resuming", flush=True)
    return len(data) - keep


def _loop(conn, fn, init, initargs):
    if init is not None:
        init(*initargs)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        key, task = msg
        conn.send((key, "start", None, 0.0))            # the slow clock starts here, not at process start
        t0 = time.time()
        try:
            res = fn(task)
            conn.send((key, "ok", res, time.time() - t0))
        except Exception as e:  # noqa: BLE001 -- any exception is a harness failure of this task
            conn.send((key, "exc", err_text(e), time.time() - t0))


class _Slot:
    def __init__(self, ctx, fn, init, initargs, isolated):
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_loop, args=(child, fn, init, initargs), daemon=True)
        self.proc.start()
        child.close()
        self.isolated = isolated
        self.key = self.task = self.t0 = self.started = None

    def send(self, key, task):
        self.key, self.task, self.t0, self.started = key, task, time.time(), None
        self.conn.send((key, task))

    def kill(self):
        try:
            self.proc.kill()
        except Exception:  # noqa: BLE001
            pass
        self.proc.join()
        self.conn.close()

    def close(self):
        try:
            self.conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.kill()
        else:
            self.conn.close()


def run(fn, tasks, workers=2, slow=None, init=None, initargs=(), log=None, prior=None, poll=2.0):
    """Generator over (key, status, result, info); see the module docstring."""
    log = log or (lambda ev: None)
    ctx = mp.get_context()
    fails = {k: list(v) for k, v in (prior or {}).items()}
    queue = collections.deque()
    retry = collections.deque()
    for key, task in tasks:
        (retry if fails.get(key) else queue).append((key, task))
    pool, iso = [], []                                   # long-lived pool slots; one-shot isolated slots
    out = []

    def fail(slot, kind):
        key, task = slot.key, slot.task
        fl = fails.setdefault(key, [])
        fl.append(kind)
        ev = {"key": key, "event": "harness_failure", "kind": kind, "attempt": len(fl),
              "isolated": slot.isolated, "sec": round(time.time() - (slot.started or slot.t0), 2)}
        if len(fl) >= 2 and slot.isolated and (fl[-1] == fl[-2] or len(fl) >= 3):
            ev["event"] = "deterministic_failure"
            out.append((key, "det_fail", None, {"fails": fl, "attempts": len(fl)}))
        else:
            retry.append((key, task))
        log(ev)

    try:
        while queue or retry or any(s.key is not None for s in pool + iso):
            busy = [s for s in pool + iso if s.key is not None]
            while len(busy) < workers and (retry or queue):
                if retry:
                    key, task = retry.popleft()
                    s = _Slot(ctx, fn, init, initargs, isolated=True)
                    iso.append(s)
                    log({"key": key, "event": "rerun_isolated", "prior": fails.get(key, [])})
                else:
                    key, task = queue.popleft()
                    free = [s for s in pool if s.key is None]
                    if free:
                        s = free[0]
                    else:
                        s = _Slot(ctx, fn, init, initargs, isolated=False)
                        pool.append(s)
                s.send(key, task)
                busy.append(s)
            if not queue:                                # no more pool work: release idle pool workers
                for s in [s for s in pool if s.key is None]:
                    s.close()
                    pool.remove(s)
            ready = wait([s.conn for s in busy] + [s.proc.sentinel for s in busy], timeout=poll)
            now = time.time()
            for s in busy:
                got = None
                if s.conn in ready or s.proc.sentinel in ready:
                    try:
                        while got is None and s.conn.poll():
                            msg = s.conn.recv()
                            if msg[1] == "start":
                                s.started = time.time()
                            else:
                                got = msg
                    except (EOFError, OSError):
                        got = None
                if got is not None:
                    key, st, res, sec = got
                    if st == "ok":
                        info = {"sec": sec, "attempts": len(fails.get(key, [])) + 1}
                        if fails.get(key):
                            info["fails"] = fails[key]
                            log({"key": key, "event": "rerun_ok", "prior": fails[key], "sec": round(sec, 2)})
                        out.append((key, "ok", res, info))
                        s.key = s.task = None
                    else:
                        fail(s, res)
                        s.key = s.task = None
                    if s.isolated:
                        s.close()
                        iso.remove(s)
                    continue
                if not s.proc.is_alive():
                    fail(s, f"process died (exitcode {s.proc.exitcode})")
                    s.kill()
                    (iso if s.isolated else pool).remove(s)
                elif slow is not None and not s.isolated and s.started and now - s.started > slow:
                    fail(s, f"slow (> {slow:.0f} s)")
                    s.kill()
                    pool.remove(s)
            while out:
                yield out.pop(0)
    finally:
        for s in pool + iso:
            if s.key is None:
                s.close()
            else:
                s.kill()
