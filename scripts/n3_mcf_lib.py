"""N3 / G3: helpers shared by the MolCrystalFlow run harness (tasks/n3_protocol.md §3.2 (ii-R), (ii-R+G), (ii-S),
(ii-A), MCF diagnostics, Gate C). Imported by scripts/n3_mcf_{run,resid,gauge,sg,diag,gateC}.py; not a CLI.

  - setup_mcf(shims): the patched fork (external/mcf or $N3_MCF) on sys.path; --shims (laptop smoke only) adds
    external/shims (torch_scatter, torch_cluster radius_graph, GPUtil, p_tqdm stand-ins). No reported number
    comes from a shimmed run (§7 step 3); every output records which shims were active.
  - run directories: one per MCF sampling call, n3_mcf/infer/<arm>/<set>/<label>/ with n3_run.json (manifest),
    hydra/inference.yaml (the merged config inference.py saved), out/predictions_<S>.pt, out/resid_<S>.pt,
    check.json. Unique by construction (arm, set, training seed, checkpoint, knobs).
  - MCF frame algebra: Rz, the prior's C2s, commutant members h applied as h R_k about each copy's rigid origin
    (lattice and centroids fixed), geodesic angles (atan2 form, accurate at every angle on float32 inputs), the per-crystal
    count of relative-rotation axes on the azimuth branch cut of MCF's gen_edges, per-draw finiteness.
  - per-draw CPU jobs (LJ through the Gate-F5 adapter, gauge scan, SG back-projection) through n3_harness (§1.3
    failure rule), resumable per draw in JSONL; each cache records the inputs it was computed from
    (<jsonl>.params.json) and is refused if they differ.
  - frozen-order checks of an export's sidecar, with the box-side fallback through Gate F's recorded SHA-256.
"""
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import numpy as np
import torch

import n3_common as nc

MCF = os.environ.get("N3_MCF", os.path.join(REPO, "external", "mcf"))
MCF_BASE = "3c493f8"
CFG_DIR = os.path.join(MCF, "molcrystalflow", "configs")
SHIMS = os.path.join(REPO, "external", "shims")
INFER_ROOT = os.path.join(REPO, "n3_mcf", "infer")
PRIV = os.path.join(nc.PRIVATE, "mcf")
SMOKE_DIR = os.path.join(nc.PRIVATE, "smoke")

# §2.2 split codes and the export directories inference.py reads (<cache_dir>/test_molcrystal_normalized.pkl.gz)
SPLIT = {"valsel": 0, "testB": 1, "devtest": 2}
BLIND_SETS = ("sel", "testB", "valsel")       # §1.1: only the listed input-side operations before sampling
ARM_NAME = {"R": "mcfR", "A": "mcfA"}
SEED_BASE = {"R": 100000, "A": 200000, "floor": 300000}   # inference.seed = base + 1000 s + split
RESID_T = 0.975                                             # §2.3 RESID time (torque_end convention)
MIN_SPACING_A = 2.5                                         # §3.2 ii-A: generated cells below -> e_lj = +inf

# the prior's mirrors (interpolant._symmetric_so3: copy i conjugated by g_(i mod 4)) in MCF's standardized frame
C2Z = np.diag([-1.0, -1.0, 1.0])
C2Y = np.diag([-1.0, 1.0, -1.0])
C2X = np.diag([1.0, -1.0, -1.0])
D2 = (np.eye(3), C2X, C2Y, C2Z)
D2_NAMES = ("I", "C2x", "C2y", "C2z")

_STATE = {"shims": None}


# ---------------------------------------------------------------------------------------------------- setup
def setup_mcf(shims=False):
    """Put the fork on sys.path (first), set N3_ROOT, optionally install the smoke shims. Returns shim list."""
    if _STATE["shims"] is not None:
        return _STATE["shims"]
    os.environ.setdefault("N3_ROOT", REPO)
    if MCF not in sys.path:
        sys.path.insert(0, MCF)
    done = []
    if shims:
        import importlib.util
        if importlib.util.find_spec("torch_scatter") is not None:
            raise SystemExit("--shims is for the laptop smoke only: torch_scatter is installed here, and the shim "
                             "directory must never shadow real code (drop --shims)")
        if SHIMS not in sys.path:
            sys.path.insert(1, SHIMS)     # torch_scatter / GPUtil / p_tqdm stand-ins, only where really missing
        for mod in ("torch_scatter", "GPUtil", "p_tqdm"):
            spec = importlib.util.find_spec(mod)
            if spec is not None and os.path.abspath(spec.origin or "").startswith(os.path.abspath(SHIMS)):
                done.append(mod)
        spec = importlib.util.spec_from_file_location("n3_mcf_shims", os.path.join(SHIMS, "n3_mcf_shims.py"))
        sh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sh)
        done += [d for d in sh.install() if d not in done]
    import molcrystalflow
    assert os.path.abspath(molcrystalflow.__file__).startswith(os.path.abspath(MCF)), \
        f"molcrystalflow imported from {molcrystalflow.__file__}, not the fork {MCF}"
    _STATE["shims"] = done
    return done


def compose(name, overrides=()):
    """Hydra composition exactly as a --config-name=<name> <overrides> command line would build it."""
    from hydra import compose as hcompose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        return hcompose(config_name=name, overrides=list(overrides))


def sha256(path):
    return nc.sha256(path)


def atomic_json(obj, path, **kw):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, **kw)
    os.replace(path + ".tmp", path)


def atomic_torch(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def lib_versions():
    import platform
    out = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    for m in ("pytorch_lightning", "torch_geometric", "hydra", "omegaconf", "pymatgen", "scipy"):
        try:
            import importlib.metadata as md
            out[m] = md.version({"hydra": "hydra-core"}.get(m, m))
        except Exception:  # noqa: BLE001
            out[m] = None
    out["cuda"] = torch.version.cuda
    return out


# ---------------------------------------------------------------------------------------------------- exports
def export_set(set_name):
    """(entries, sidecar) of the export an MCF run reads; the pickle SHA-256 is re-checked (§3.2 Integrity).
    set_name is an N3 set (valsel/testB/devtest) or a smoke cache dir holding test_* + test_sidecar.json."""
    import n3_mcf_export as X
    if set_name in X.TARGET:
        return X.load_export(set_name)
    side = json.load(open(os.path.join(set_name, "test_sidecar.json")))
    pk = os.path.join(set_name, "test_molcrystal_normalized.pkl.gz")
    assert sha256(pk) == side["sha256"], f"{pk}: SHA-256 differs from its sidecar"
    return X.load_pkl_gz(pk), side


def cache_dir(set_name):
    import n3_mcf_export as X
    return X.paths(set_name)["dir"] if set_name in X.TARGET else os.path.abspath(set_name)


def frozen_order_checks(set_name, side):
    """§3.2 '|set| crystals in sidecar order'. Where the set's asym items are on this machine: the sidecar's
    refcodes equal the frozen order. Elsewhere (the box): the committed lists (TEST-B; the SEL part of VALSEL)
    and, for every set, the sidecar's SHA-256 equals the one Gate F recorded when it checked the frozen order
    locally (results/n3/gateF_<set>.json: sidecar_sha256 with F1.sidecar_order_ok)."""
    import n3_mcf_export as X
    c = {}
    try:
        c["sidecar_frozen_order"] = side["refcodes"] == [a["refcode"] for a in nc.load_set(set_name)]
        return c
    except Exception:  # noqa: BLE001 -- inputs of the set not on this machine (box)
        pass
    if set_name == "testB":
        c["sidecar_frozen_order_list"] = side["refcodes"] == nc.frozen_refcodes("testB")
    elif set_name == "valsel":
        c["sidecar_frozen_order_sel_part"] = side["refcodes"][100:] == nc.frozen_refcodes("sel")
    gp = os.path.join(REPO, "results", "n3", f"gateF_{set_name}.json")
    g = json.load(open(gp)) if os.path.exists(gp) else {}
    c["sidecar_sha_equals_gateF_order_checked"] = bool(
        g.get("sidecar_sha256") == sha256(X.paths(set_name)["sidecar"]) and g.get("F1", {}).get("sidecar_order_ok"))
    return c


# ---------------------------------------------------------------------------------------------------- checkpoints
def ckpt_config(ckpt_path):
    from omegaconf import OmegaConf
    return OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.yaml"))


def ckpt_info(ckpt_path):
    """Arm, training seed and run name of a checkpoint, from <ckpt_dir>/config.yaml (the merged inference.yaml
    records both as null by design)."""
    c = ckpt_config(ckpt_path)
    arm = "R" if (c.interpolant.trans.corrupt is False and c.interpolant.lattice.corrupt is False) else "A"
    seed = c.experiment.seed
    name = c.experiment.wandb.name if "n3_set_on_command_line" not in str(c.experiment.wandb.get("name", "")) else None
    base = os.path.basename(ckpt_path)
    m = re.match(r"epoch_(\d+)-step_(\d+)", base)
    label = "last" if base == "last.ckpt" else (f"ep{int(m.group(1))}_st{int(m.group(2))}" if m else
                                                 re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(base)[0])[:40])
    return {"arm": arm, "seed": None if seed is None else int(seed), "run_name": name, "label": label,
            "ckpt": os.path.abspath(ckpt_path), "ckpt_sha256": sha256(ckpt_path)}


def inference_seed(arm, s, split):
    return SEED_BASE[arm] + 1000 * int(s) + int(split)


def load_flow_module(ckpt_path, cfg, device="cpu"):
    from molcrystalflow.models.molcrystalflow import FlowModule
    mod = FlowModule.load_from_checkpoint(checkpoint_path=ckpt_path, cfg=cfg, map_location="cpu",
                                          weights_only=False)
    mod.eval()
    return mod.to(device)


# ---------------------------------------------------------------------------------------------------- run dirs
def run_label(info, set_name, s_uR, s_uF=None, extra=""):
    lab = f"s{info['seed']}_{info['label']}_u{s_uR:g}" + ("" if s_uF is None else f"_f{s_uF:g}")
    return lab + (f"_{extra}" if extra else "")


def run_paths(run_dir, S=16):
    return {"dir": run_dir, "manifest": os.path.join(run_dir, "n3_run.json"),
            "hydra": os.path.join(run_dir, "hydra"), "out": os.path.join(run_dir, "out"),
            "config": os.path.join(run_dir, "hydra", "inference.yaml"),
            "pred": os.path.join(run_dir, "out", f"predictions_{S}.pt"),
            "resid": os.path.join(run_dir, "out", f"resid_{S}.pt"),
            "check": os.path.join(run_dir, "check.json"), "log": os.path.join(run_dir, "run.log"),
            "lj": os.path.join(run_dir, f"lj_{S}.jsonl")}


def load_manifest(run_dir):
    man = json.load(open(os.path.join(run_dir, "n3_run.json")))
    return man, run_paths(run_dir, man["num_samples"])


def load_predictions(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------------------------------- frames
def Rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def z2_members(step_deg=2.0):
    """z_MCF = 2 commutant on a grid: both cosets {Rz(t)} and {Rz(t) C2x}, t = 0, step, ..., < 360 degrees."""
    n = int(round(360.0 / step_deg))
    out = []
    for coset in ("rot", "flip"):
        for q in range(n):
            th = math.radians(q * step_deg)
            out.append(((coset, q * step_deg), Rz(th) if coset == "rot" else Rz(th) @ C2X))
    return out


def commutant_members(z, step_deg=2.0):
    """The §3.2 +G scan set H by z_MCF: z = 2 both cosets on the grid, z >= 3 D2, z = 1 identity only."""
    if z == 2:
        return z2_members(step_deg)
    if z >= 3:
        return [((n, 0.0), g) for n, g in zip(D2_NAMES, D2)]
    return [(("I", 0.0), np.eye(3))]


def origins(cell_rot_trans):
    """Cartesian rigid origins T_k = trans_k @ lattice (MCF's assembly: X_k = R_k local_k + T_k)."""
    trans, lattice = cell_rot_trans
    return np.asarray(trans, dtype=np.float64) @ np.asarray(lattice, dtype=np.float64)


def apply_common_rotation(cart, mol, T, h):
    """h R_k for every copy with lattice and centroids fixed: X'_k = (X_k - T_k) h^T + T_k (exact for MCF's
    assembly, whatever the local-frame origin). h = I returns cart unchanged (bitwise)."""
    if np.array_equal(h, np.eye(3)):
        return cart
    cart = np.asarray(cart, dtype=np.float64)
    Tm = T[np.asarray(mol)]
    return (cart - Tm) @ h.T + Tm


GEODESIC = "atan2(|vee(M - M^T)| / 2, (tr M - 1) / 2), M = Ra^T Rb, float64"


def geodesic(Ra, Rb):
    """Angle (rad, float64) between rotation stacks [..., 3, 3] as atan2(sin, cos) of M = Ra^T Rb: sin from its
    antisymmetric part, cos from its trace. Accurate at every angle: identical matrices give exactly 0 and float32
    inputs are within ~4e-8 rad of the angle between them, where arccos((tr M - 1) / 2) has a float32 floor of
    ~4e-4 rad near 0 (the lowest-RESID pick lives there) and 2 asin(|Ra - Rb|_F / 2 sqrt 2) loses ~1e-4 near pi."""
    M = torch.as_tensor(Ra).double().transpose(-1, -2) @ torch.as_tensor(Rb).double()
    c = (torch.diagonal(M, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    v = torch.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1)
    return torch.atan2(v.norm(dim=-1) / 2.0, c)


def branch_cut_counts(R, num_bbs):
    """Per crystal, the edges of MCF's fully connected copy graph (gen_edges: rotmat_to_rotvec of R_i^T R_j, then
    azimuth atan2(axis_y, axis_x), used raw in rot_unit_dots) whose relative-rotation axis lies on the azimuth
    branch cut (|axis_y| <= 1e-12 |axis|, axis_x < 0, angle > 1e-6): there the feature is +-pi by the sign of a
    rounding zero, so a common rotation of all copies can flip it. Exact C2 conjugates (the symmetric prior at
    t = 0) sit there. Returns (edges [n] long, on_cut [n] long); needs the fork on sys.path."""
    from molcrystalflow.data.so3_utils import rotmat_to_rotvec
    nbb = [int(v) for v in num_bbs]
    R = torch.as_tensor(R)
    src, dst, crys = [], [], []
    off = 0
    for c, k in enumerate(nbb):
        ii, jj = torch.meshgrid(torch.arange(k), torch.arange(k), indexing="ij")
        src.append(ii.reshape(-1) + off)
        dst.append(jj.reshape(-1) + off)
        crys.append(torch.full((k * k,), c, dtype=torch.long))
        off += k
    src, dst, crys = torch.cat(src), torch.cat(dst), torch.cat(crys)
    v = rotmat_to_rotvec(R[src].transpose(-1, -2) @ R[dst])
    a = v.norm(dim=-1)
    on = (a > 1e-6) & (v[:, 1].abs() <= 1e-12 * a) & (v[:, 0] < 0)
    n = len(nbb)
    return (torch.bincount(crys, minlength=n), torch.bincount(crys[on], minlength=n))


def draw_finite(pred):
    """bool [S, n]: draw s of crystal i has finite lattice, centroids, rotations and coordinates (a non-finite
    draw is an arm failure, §1.3)."""
    S, n = pred["num_bbs"].shape
    fin = torch.ones(S, n, dtype=torch.bool)
    for s in range(S):
        nb, na = pred["num_bbs"][s].long(), pred["num_atoms"][s].long()
        bb_c = torch.repeat_interleave(torch.arange(n), nb)
        at_c = torch.repeat_interleave(torch.arange(n), na)
        bad = ~torch.isfinite(pred["lattices"][s].reshape(n, -1)).all(1)
        for t, idx in ((pred["pred_rotmats"][s].reshape(len(bb_c), -1), bb_c),
                       (pred["pred_trans"][s].reshape(len(bb_c), -1), bb_c),
                       (pred["cart_coords"][s].reshape(len(at_c), -1), at_c)):
            nb_bad = (~torch.isfinite(t).all(1)).long()
            bad |= torch.zeros(n, dtype=torch.long).index_add_(0, idx, nb_bad) > 0
        fin[s] = ~bad
    return fin


def min_heights(L):
    """The three cell heights V / |a_j x a_k| (spacings of the generated cell's (100), (010), (001) planes). The
    §3.2 ii-A rule is applied to L_gen as sampled (before the F5 supercell; energy() runs on L_c = M L_gen, whose
    heights differ for centred crystals)."""
    L = np.asarray(L, dtype=np.float64)
    V = abs(np.linalg.det(L))
    return min(V / np.linalg.norm(np.cross(L[(i + 1) % 3], L[(i + 2) % 3])) for i in range(3))


# ---------------------------------------------------------------------------------------------------- LJ jobs
def _winit():
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)


def cell_payload(c):
    return {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in c.items()}


def payload_cell(p):
    return {k: torch.from_numpy(np.asarray(v)) for k, v in p.items()}


def lj_of_payload(p, M, generated=False):
    """Unrelaxed steric LJ of one MCF 'cells' payload through the Gate-F5 adapter (energy() unchanged).
    generated: MCF-A / prior-floor cells with a cell height < 2.5 A get +inf (§3.2 ii-A)."""
    import n3_mcf_export as X
    if not (np.isfinite(p["lattice"]).all() and np.isfinite(p["cart"]).all()):
        return float("nan"), "nonfinite"
    if generated and min_heights(p["lattice"]) < MIN_SPACING_A:
        return float("inf"), "spacing"
    return float(X.mcf_energy(payload_cell(p), M)), "ok"


def lj_task(task):
    p, M, generated = task
    return lj_of_payload(p, M, generated)


def check_params(out_jsonl, params):
    """A resumable JSONL cache is valid only for the inputs it was computed from: <out_jsonl>.params.json holds
    them. A cache with rows but other (or unrecorded) params is refused, never silently reused."""
    import n3_harness as H
    if params is None:
        return
    pp = out_jsonl + ".params.json"
    want = json.loads(json.dumps(params))
    rows = bool(H.read_jsonl(out_jsonl))
    have = json.load(open(pp)) if os.path.exists(pp) else None
    if rows and have != want:
        raise SystemExit(f"{out_jsonl}: cached rows were computed from other inputs ({have} vs {want}); "
                         "move the cache away, or write to another path / label")
    if have != want:
        atomic_json(want, pp, indent=1)


def tensors_sha256(*objs):
    """Content hash of tensors / arrays / nested lists and dicts of them (dtype, shape, bytes; dict keys sorted).
    Stable across torch.save calls, unlike the file SHA-256 of a re-written bundle."""
    import hashlib
    h = hashlib.sha256()

    def add(o):
        if o is None:
            h.update(b"N")
        elif isinstance(o, dict):
            for k in sorted(o):
                h.update(str(k).encode())
                add(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(f"L{len(o)}".encode())
            for x in o:
                add(x)
        else:
            a = o.detach().cpu().contiguous().numpy() if torch.is_tensor(o) else np.ascontiguousarray(o)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
    for o in objs:
        add(o)
    return h.hexdigest()


def run_jobs(fn, tasks, out_jsonl, workers=2, row=None, log_path=None, label="jobs", params=None):
    """Resumable per-key CPU jobs under the §1.3 harness rule (n3_harness.run): rows {key, status, ...} are
    appended to out_jsonl; keys already there are skipped, and only if `params` (the inputs the rows depend on)
    equal the ones recorded beside the cache. Returns {key(tuple): row}."""
    import time
    import n3_harness as H
    check_params(out_jsonl, params)
    H.repair_jsonl(out_jsonl)
    done = {tuple(r["key"]): r for r in H.read_jsonl(out_jsonl)}
    todo = [(k, t) for k, t in tasks if tuple(k) not in done]
    if todo:
        print(f"  {label}: {len(todo)} tasks ({len(done)} already done), {workers} workers", flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    logf = open(log_path, "a") if log_path else None
    try:
        with open(out_jsonl, "a") as fh:
            def log(ev):
                if logf:
                    k = ev.get("key")
                    logf.write(json.dumps({"t": round(time.time(), 1), **ev,
                                           "key": list(k) if isinstance(k, tuple) else k}) + "\n")
                    logf.flush()
            n = 0
            for key, st, res, info in H.run(fn, todo, workers=workers, slow=None, init=_winit, log=log):
                r = {"key": list(key), "status": st, "attempts": info.get("attempts", 1)}
                r.update(row(res) if (row and st == "ok") else ({} if st == "ok" else {"fails": info.get("fails")}))
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[tuple(key)] = r
                n += 1
                if n % 500 == 0:
                    print(f"    {label} {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    finally:
        if logf:
            logf.close()
    return done


# ---------------------------------------------------------------------------------------------------- bundles
def save_bundle(path, *, arm, seed, set_name, refcodes, kind, valid, sel, meta, R=None, cells=None, subset=False):
    """n3_common.save_bundle for a frozen set; subset=True (smoke only) writes the same format for the first
    len(refcodes) crystals of the frozen order, flagged meta['smoke_subset'] (never a reported bundle)."""
    if not subset:
        return nc.save_bundle(path, arm=arm, seed=seed, set_name=set_name, refcodes=refcodes, kind=kind,
                              valid=valid, sel=sel, meta=meta, R=R, cells=cells)
    expect = [a["refcode"] for a in nc.load_set(set_name)]
    idx = meta["smoke_subset"]
    assert list(refcodes) == [expect[i] for i in idx], "smoke subset refcodes not in frozen order"
    n = len(refcodes)
    assert valid.shape[0] == n and all(v.shape == valid.shape for v in sel.values())
    atomic_torch({"arm": arm, "seed": seed, "set": set_name, "refcodes": list(refcodes), "kind": kind, "R": R,
                  "cells": cells, "valid": valid, "sel": sel, "meta": meta}, path)


def smoke_items(bundle_or_meta, set_name):
    """Asym items of a bundle's crystals (all of the set, or the smoke subset)."""
    items = nc.load_set(set_name)
    sub = (bundle_or_meta.get("meta", bundle_or_meta) or {}).get("smoke_subset")
    return [items[i] for i in sub] if sub is not None else items
