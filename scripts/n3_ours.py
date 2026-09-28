"""N3 arm (i) OURS and the OURS-N2 sensitivity row (protocol §3.1, §2.2): learned R_asym draw bundles.

sample   16 draws per crystal from each checkpoint of the arm, 40 Euler steps, the batched sampler
         (g2_rank.sample_batch). Per crystal i the RNG is seeded ONCE with base + i (TEST-B 80000, SEL 90000)
         and the arm's three checkpoints are then sampled in seed order from that one stream, exactly as
         g2_rank.cmd_sample did for the stored N1/N2 draws. Each draw records R, e_lj (unrelaxed steric LJ of
         expand(a, R), g2_asym_baselines.energy) and torque_end (|field| at t = 0.975). No matcher runs and
         there is no per-crystal timeout (§1.3). Finished crystals are appended to a JSONL part file (resume
         = re-run the same command); the three bundles (n3_common format, kind 'rasym', sel {'lj', 'resid'})
         are written when every crystal is done. A resumed run must have the part file's checkpoints
         (SHA-256), seed base, S and steps (checked against its first 'start' event); an unfinished final
         line from a killed run is cut off before appending.
           arm failures (§1.3): a sampler exception or non-finite R -> valid False for those draws.
           harness failures: an energy() exception is re-run alone in a fresh process; if it recurs
           identically it is DETERMINISTIC and e_lj = NaN (ranks last, §2.3). A crash of a whole crystal
           task is re-run in a fresh process (n3_harness); if that recurs, the crystal is re-run once more
           WITHOUT energy() (generation + torque only, same seed): if that succeeds the draws stay valid and
           every energy goes through the energy harness above (a hard crash of energy() is a harness
           failure, never an arm MISS); only a crash of generation itself is an arm failure.
           Every event goes to <part>.events.jsonl.
convert  stored N1 / N2 draws (results/vast_n12/{n1,n2}_{val,test}.jsonl) -> VAL / DEV-TEST bundles from R,
         e_lj and torque_end only; the stored 'exact' flags are never read (§1.3 Blinding).
concat   VAL bundle + SEL bundle of one arm and seed (same checkpoint SHA-256) -> its VALSEL bundle (i = 0..399).
Every writer refuses to overwrite a bundle that already has match results (--force to redo): a rewrite changes
the bundle's SHA-256 (its meta records the command) and n3_score would then refuse the old match files.

    python scripts/n3_ours.py sample --arm ours --set testB --workers 8
    python scripts/n3_ours.py sample --arm ours_n2 --set sel --workers 8
    python scripts/n3_ours.py convert --arm ours --set val
    python scripts/n3_ours.py concat --arm ours
    # smoke only (writes a part file, never a bundle): VAL with the N1 seed base must reproduce N1's R
    python scripts/n3_ours.py sample --arm ours --set val --smoke --seed-base 40000 --limit 4 --workers 2
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n3_common as C

import torch

S_DRAWS, STEPS, T_RESID = 16, 40, 0.975
SEED_BASE = {"testB": 80_000, "sel": 90_000}
R_ = lambda *p: os.path.join(C.REPO, *p)


def arm_ckpts(arm):
    """Checkpoint paths of an arm in seed order (0, 1, 2)."""
    if arm == "ours":
        return [R_("results", "vast_g2", f"eval_s{s}.pt") for s in range(3)]
    if arm == "ours_n2":                                   # the three N2 checkpoints chosen on VAL
        chosen = json.load(open(R_("results", "n2_selection.json")))["chosen"]
        assert len(chosen) == 3 and [c.split("_")[1] for c in chosen] == ["s0", "s1", "s2"], chosen
        return [R_("results", "vast_n12", c) for c in chosen]
    raise ValueError(arm)


STORED = {("ours", "val"): "results/vast_n12/n1_val.jsonl", ("ours", "devtest"): "results/vast_n12/n1_test.jsonl",
          ("ours_n2", "val"): "results/vast_n12/n2_val.jsonl",
          ("ours_n2", "devtest"): "results/vast_n12/n2_test.jsonl"}


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", C.REPO, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 -- a box without git
        return "unknown"


# ------------------------------------------------------------------------------------------- workers
_M = {}


def _init(ckpts, set_name):
    """Worker setup; ckpts None = energy-only worker (no model load)."""
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    _M["set"] = set_name
    _M["models"] = []
    if ckpts is None:
        return
    from g2_learned_flow import TorqueField
    for p in ckpts:
        m = TorqueField()
        ck = torch.load(p, weights_only=False)
        m.load_state_dict(ck.get("best_state", ck))
        m.eval()
        _M["models"].append(m)


def _item(set_name, i):
    if _M.get("items_set") != set_name:
        _M["items"], _M["items_set"] = C.load_set(set_name), set_name
    return _M["items"][i]


def _sample_task(task):
    """One crystal: seed once, then every checkpoint in seed order (g2_rank._work without the matcher).
    task = (i, seed) or (i, seed, skip_energy_reason): with a reason, energy() is not called and every valid
    draw carries the reason as its energy failure text (the energy harness re-runs them one by one)."""
    from g2_asym_baselines import energy, expand
    from g2_learned_flow import precompute
    from g2_rank import sample_batch
    import n3_harness as H
    i, seed = task[:2]
    skip_energy = task[2] if len(task) > 2 else None
    a = _item(_M["set"], i)
    torch.manual_seed(seed)
    g = precompute(a, 8.0)
    out = []
    for m in _M["models"]:
        t0 = time.time()
        rec = {"R": None, "e_lj": [None] * S_DRAWS, "torque_end": [None] * S_DRAWS,
               "valid": [False] * S_DRAWS, "arm_fail": None, "e_err": {}}
        try:
            R = sample_batch(m, g, S_DRAWS, steps=STEPS)
            with torch.no_grad():
                tq = m.forward_batch(g, R, torch.full((S_DRAWS,), T_RESID)).norm(dim=-1)
        except Exception as e:  # noqa: BLE001 -- generator error = arm failure (§1.3)
            rec["arm_fail"] = "generator error: " + H.err_text(e)
            rec["R"] = [[float("nan")] * 9] * S_DRAWS
            rec["sec"] = time.time() - t0
            out.append(rec)
            continue
        Rd = R.double()
        rec["R"] = Rd.reshape(S_DRAWS, 9).tolist()
        rec["torque_end"] = [float(x) for x in tq]
        for j in range(S_DRAWS):
            if not bool(torch.isfinite(Rd[j]).all()):
                rec["valid"][j] = False                   # non-finite coordinates = arm failure
                continue
            rec["valid"][j] = True
            if skip_energy is not None:
                rec["e_err"][str(j)] = skip_energy
                continue
            try:
                rec["e_lj"][j] = float(energy(*expand(a, Rd[j])))
            except Exception as e:  # noqa: BLE001 -- energy crash = harness failure, re-run outside
                rec["e_err"][str(j)] = H.err_text(e)
        rec["sec"] = time.time() - t0
        out.append(rec)
    return {"i": i, "refcode": a["refcode"], "seed": seed, "models": out}


def _energy_task(task):
    from g2_asym_baselines import energy, expand
    i, R9 = task
    return float(energy(*expand(_item(_M["set"], i), torch.tensor(R9, dtype=torch.float64).reshape(3, 3))))


# ------------------------------------------------------------------------------------------- commands
def part_path(arm, set_name):
    return os.path.join(C.PRIVATE, "draws", "parts", f"{arm}_{set_name}.jsonl")


def _read_part(path):
    import n3_harness as H
    return {r["i"]: r for r in H.read_jsonl(path)}


HEAD_KEYS = ("arm", "set", "seed_base", "ckpt_sha256", "S", "steps")


def _check_resume(part, rows, head):
    """A resumed part file must come from the same checkpoints, seed base, S and steps (its first 'start'
    event); rows without a recorded start are refused."""
    if not rows:
        return
    import n3_harness as H
    starts = [e for e in H.read_jsonl(part + ".events.jsonl") if e.get("event") == "start"]
    if not starts:
        raise SystemExit(f"{part}: {len(rows)} rows but no 'start' event to check them against; refusing to mix")
    first = starts[0]
    bad = [k for k in HEAD_KEYS if first.get(k) != head[k]]
    if bad:
        raise SystemExit(f"{part}: the part file was started with different {bad}; refusing to resume")


def _sample_without_energy(i, seed, refcode, info, ckpts, set_name, log):
    """The crystal task failed deterministically. Re-run generation + torque only (same seed; energy() does not
    touch the RNG) in a fresh process: on success the draws stay valid and their energies carry the task's
    failure text, so the energy harness re-runs each alone (identical recurrence -> NaN, ranks last). Only a
    failure of generation itself is an arm failure (§1.3)."""
    import n3_harness as H
    why = info["fails"][-1]
    log({"event": "rerun_without_energy", "key": i, "prior": info["fails"]})
    got = list(H.run(_sample_task, [(i, (i, seed, why))], workers=1, slow=None, init=_init,
                     initargs=(ckpts, set_name), log=log))
    _, status, res, info2 = got[0]
    if status == "ok":
        res["task_fails"] = info["fails"]
        return res
    return {"i": i, "refcode": refcode, "seed": seed, "task_fails": info["fails"] + info2["fails"], "models": [
        {"R": [[float("nan")] * 9] * S_DRAWS, "e_lj": [None] * S_DRAWS, "torque_end": [None] * S_DRAWS,
         "valid": [False] * S_DRAWS, "e_err": {}, "sec": None,
         "arm_fail": f"generator error (deterministic, generation-only re-run): {info2['fails']}"}
        for _ in ckpts]}


def cmd_sample(args):
    import n3_harness as H
    warnings.filterwarnings("ignore")
    if args.set in SEED_BASE:
        if args.smoke or args.seed_base is not None:
            raise SystemExit("TEST-B / SEL use the frozen seed bases (80000 / 90000); no --smoke/--seed-base")
        base, arm_out = SEED_BASE[args.set], args.arm
    else:                                                  # VAL / TRAIN: smoke only, never a bundle
        if not args.smoke or args.seed_base is None:
            raise SystemExit(f"{args.set}: OURS reuses the stored N1/N2 draws there (convert); sampling it is "
                             "a smoke test only: pass --smoke --seed-base B")
        base, arm_out = args.seed_base, args.arm + "_smoke"
    ckpts = arm_ckpts(args.arm)
    items = C.load_set(args.set)
    n = len(items)
    part = part_path(arm_out, args.set)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    head = {"arm": arm_out, "set": args.set, "seed_base": base, "ckpts": ckpts,
            "ckpt_sha256": [C.sha256(p) for p in ckpts], "S": S_DRAWS, "steps": STEPS, "cmd": sys.argv,
            "git": git_rev()}
    H.repair_jsonl(part)
    rows = _read_part(part)
    _check_resume(part, rows, head)
    events = open(part + ".events.jsonl", "a")
    log = lambda ev: (events.write(json.dumps({"t": round(time.time(), 1), **ev}) + "\n"), events.flush())
    log({"event": "start", **head})
    todo = [i for i in range(n) if i not in rows]
    if args.limit:
        todo = [i for i in todo if i < args.limit]
    print(f"{arm_out}/{args.set}: {len(rows)} done, {len(todo)} to go", flush=True)
    tasks = [(i, (i, base + i)) for i in todo]
    t0 = time.time()
    with open(part, "a") as fh:
        for k, (i, status, res, info) in enumerate(H.run(_sample_task, tasks, workers=args.workers, slow=None,
                                                         init=_init, initargs=(ckpts, args.set), log=log)):
            if status != "ok":                             # the whole crystal task failed twice in a fresh process
                res = _sample_without_energy(i, base + i, items[i]["refcode"], info, ckpts, args.set, log)
            res["attempts"] = info.get("attempts", 1)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if (k + 1) % 20 == 0:
                print(f"{len(rows) + k + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    rows = _read_part(part)
    # energy() harness failures: re-run alone in a fresh process; identical recurrence -> NaN (ranks last)
    redo = {(i, mi, int(j)): err for i, r in rows.items() for mi, m in enumerate(r["models"])
            for j, err in m["e_err"].items() if m["e_lj"][int(j)] is None}
    if redo:
        tasks = [(key, (key[0], rows[key[0]]["models"][key[1]]["R"][key[2]])) for key in redo]
        prior = {key: [err] for key, err in redo.items()}
        fixed = {}
        for key, status, res, info in H.run(_energy_task, tasks, workers=args.workers, init=_init,
                                            initargs=(None, args.set), log=log, prior=prior):
            fixed[key] = res if status == "ok" else float("nan")
            log({"event": "energy_" + status, "key": list(key), "e_lj": fixed[key]})
        for (i, mi, j), e in fixed.items():
            rows[i]["models"][mi]["e_lj"][j] = e
            rows[i]["models"][mi].setdefault("e_rerun", {})[str(j)] = e
        with open(part + ".tmp", "w") as fh:
            for i in sorted(rows):
                fh.write(json.dumps(rows[i]) + "\n")
        os.replace(part + ".tmp", part)
    events.close()
    if len(rows) < n or args.smoke:
        print(f"part file {part}: {len(rows)}/{n} crystals" + (" (smoke: no bundle)" if args.smoke else ""))
        return
    write_bundles(args.arm, args.set, rows, head)


def _writable(path, force=False):
    """A bundle with match results is never rewritten silently: its SHA-256 would change and n3_score would
    refuse its match files as stale."""
    import glob
    stem = os.path.splitext(os.path.basename(path))[0]
    done = [p for p in glob.glob(os.path.join(C.PRIVATE, "match", stem + "__*.jsonl"))
            if not p.endswith(".events.jsonl")]
    if done and not force:
        raise SystemExit(f"{path} already has match results ({len(done)} files, e.g. {os.path.basename(done[0])}); "
                         "not rewriting it (pass --force and delete those match files to redo)")


def write_bundles(arm, set_name, rows, head, force=False):
    for s in range(len(head["ckpts"])):
        _writable(C.bundle_path(arm, s, set_name), force)
    items = C.load_set(set_name)
    n = len(items)
    assert sorted(rows) == list(range(n))
    for s, ck in enumerate(head["ckpts"]):
        R = torch.empty(n, S_DRAWS, 3, 3, dtype=torch.float64)
        valid = torch.zeros(n, S_DRAWS, dtype=torch.bool)
        lj = torch.full((n, S_DRAWS), float("nan"), dtype=torch.float64)
        resid = torch.full((n, S_DRAWS), float("nan"), dtype=torch.float64)
        sec, arm_fail, det_e = [], [], []
        for i in range(n):
            r = rows[i]
            assert r["refcode"] == items[i]["refcode"] and r["seed"] == head["seed_base"] + i
            m = r["models"][s]
            R[i] = torch.tensor(m["R"], dtype=torch.float64).reshape(S_DRAWS, 3, 3)
            valid[i] = torch.tensor(m["valid"]) & torch.isfinite(R[i]).reshape(S_DRAWS, -1).all(-1)
            lj[i] = torch.tensor([float("nan") if e is None else e for e in m["e_lj"]], dtype=torch.float64)
            resid[i] = torch.tensor([float("nan") if e is None else e for e in m["torque_end"]],
                                    dtype=torch.float64)
            sec.append(m["sec"])
            if m.get("arm_fail") or not bool(valid[i].any()):
                arm_fail.append(r["refcode"])
            det_e += [r["refcode"]] * sum(1 for e in m.get("e_rerun", {}).values() if not math.isfinite(e))
        meta = {**head, "ckpt": ck, "ckpt_sha256": head["ckpt_sha256"][s], "seed_index": s,
                "sec_per_crystal": sec, "n_lj_per_crystal": S_DRAWS, "arm_fail_refcodes": arm_fail,
                "energy_det_fail_refcodes": det_e,
                "note": "seed_base + i seeds one RNG stream per crystal; checkpoints sampled in seed order"}
        path = C.bundle_path(arm, s, set_name)
        C.save_bundle(path, arm=arm, seed=s, set_name=set_name, refcodes=[a["refcode"] for a in items],
                      kind="rasym", valid=valid, sel={"lj": lj, "resid": resid}, meta=meta, R=R)
        print(f"wrote {path}  (arm failures {len(arm_fail)}, deterministic energy failures {len(det_e)})")


def cmd_convert(args):
    src = args.src or R_(*STORED[(args.arm, args.set)].split("/"))
    ckpts = arm_ckpts(args.arm)
    keys = [os.path.basename(p) for p in ckpts]
    items = C.load_set(args.set)
    rows = {}
    for line in open(src):
        r = json.loads(line)
        rows[r["refcode"]] = r
    order = [a["refcode"] for a in items]
    assert set(rows) == set(order), f"{src}: refcodes differ from the frozen {args.set} list"
    timeouts = [r for r in order if rows[r].get("timeout")]
    n = len(order)
    for s in range(len(keys)):
        _writable(C.bundle_path(args.arm, s, args.set), args.force)
    for s, key in enumerate(keys):
        R = torch.full((n, S_DRAWS, 3, 3), float("nan"), dtype=torch.float64)
        valid = torch.zeros(n, S_DRAWS, dtype=torch.bool)
        lj = torch.full((n, S_DRAWS), float("nan"), dtype=torch.float64)
        resid = torch.full((n, S_DRAWS), float("nan"), dtype=torch.float64)
        for i, ref in enumerate(order):
            if rows[ref].get("timeout"):
                continue                                   # stored timeout row = no draws (arm failure)
            d = rows[ref][key]
            assert len(d) == S_DRAWS
            R[i] = torch.tensor([x["R"] for x in d], dtype=torch.float64).reshape(S_DRAWS, 3, 3)
            lj[i] = torch.tensor([x["e_lj"] for x in d], dtype=torch.float64)
            resid[i] = torch.tensor([x["torque_end"] for x in d], dtype=torch.float64)
            valid[i] = torch.isfinite(R[i]).reshape(S_DRAWS, -1).all(-1)
        meta = {"source": os.path.relpath(src, C.REPO).replace("\\", "/"), "source_sha256": C.sha256(src),
                "source_key": key, "ckpt": ckpts[s], "ckpt_sha256": C.sha256(ckpts[s]), "seed_index": s,
                "seed_base": 50_000 if args.set == "devtest" else 40_000, "S": S_DRAWS, "steps": STEPS,
                "n_lj_per_crystal": S_DRAWS, "stored_exact_ignored": True, "timeout_refcodes": timeouts, "cmd": sys.argv, "git": git_rev(),
                "note": "R rounded to 6 dp, e_lj to 3 dp, torque_end to 5 dp by g2_rank at storage"}
        path = C.bundle_path(args.arm, s, args.set)
        C.save_bundle(path, arm=args.arm, seed=s, set_name=args.set, refcodes=order, kind="rasym",
                      valid=valid, sel={"lj": lj, "resid": resid}, meta=meta, R=R)
        print(f"wrote {path}  ({key}; stored timeouts {len(timeouts)})")
    if args.check_lj:
        _check_lj(args, items, rows, keys)


def _check_lj(args, items, rows, keys):
    """Recompute e_lj from the stored (rounded) R on the first --check-lj crystals: the stored value is the
    selector input, this only checks it belongs to the structure that is scored."""
    from g2_asym_baselines import energy, expand
    worst, flips = 0.0, 0
    for i, a in enumerate(items[:args.check_lj]):
        for key in keys:
            d = rows[a["refcode"]][key]
            e = [energy(*expand(a, torch.tensor(x["R"], dtype=torch.float64).reshape(3, 3))) for x in d]
            st = [x["e_lj"] for x in d]
            worst = max(worst, max(abs(u - v) / max(1.0, abs(v)) for u, v in zip(e, st)))
            flips += int(min(range(S_DRAWS), key=lambda j: (e[j], j)) != min(range(S_DRAWS), key=lambda j: (st[j], j)))
    print(f"check-lj: {args.check_lj} crystals x {len(keys)} seeds: max rel |E(R_stored) - e_lj| = {worst:.2e}, "
          f"LJ-pick changes {flips}")


def cmd_concat(args):
    for s in range(3):
        v = C.load_bundle(C.bundle_path(args.arm, s, "val"))
        e = C.load_bundle(C.bundle_path(args.arm, s, "sel"))
        assert v["kind"] == e["kind"] == "rasym" and set(v["sel"]) == set(e["sel"])
        for k in ("ckpt_sha256", "seed_index", "S", "steps"):            # one checkpoint per seed (§3.1)
            if v["meta"].get(k) != e["meta"].get(k):
                raise SystemExit(f"seed {s}: VAL and SEL bundles differ in {k}; not concatenating")
        path = C.bundle_path(args.arm, s, "valsel")
        _writable(path, args.force)
        C.save_bundle(path, arm=args.arm, seed=s, set_name="valsel", refcodes=v["refcodes"] + e["refcodes"],
                      kind="rasym", valid=torch.cat([v["valid"], e["valid"]]),
                      sel={k: torch.cat([v["sel"][k], e["sel"][k]]) for k in v["sel"]},
                      meta={"parts": {"val": v["meta"], "sel": e["meta"]}, "cmd": sys.argv},
                      R=torch.cat([v["R"], e["R"]]))
        print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--arm", choices=["ours", "ours_n2"], required=True)
    s.add_argument("--set", choices=["testB", "sel", "val", "train"], required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--smoke", action="store_true", help="VAL/TRAIN smoke run: part file only, arm '<arm>_smoke'")
    s.add_argument("--seed-base", type=int, default=None, help="smoke only")
    s.add_argument("--limit", type=int, default=0, help="only crystals i < LIMIT (bundle written only when all done)")
    c = sub.add_parser("convert")
    c.add_argument("--arm", choices=["ours", "ours_n2"], required=True)
    c.add_argument("--set", choices=["val", "devtest"], required=True)
    c.add_argument("--src", default=None, help="stored g2_rank JSONL (default: results/vast_n12/{n1,n2}_*)")
    c.add_argument("--check-lj", type=int, default=0, help="recompute e_lj on the first N crystals")
    c.add_argument("--force", action="store_true", help="rewrite a bundle that already has match results")
    k = sub.add_parser("concat")
    k.add_argument("--arm", choices=["ours", "ours_n2"], required=True)
    k.add_argument("--force", action="store_true", help="rewrite a bundle that already has match results")
    args = ap.parse_args()
    {"sample": cmd_sample, "convert": cmd_convert, "concat": cmd_concat}[args.cmd](args)


if __name__ == "__main__":
    main()
