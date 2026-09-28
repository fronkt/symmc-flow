"""N3 / G3 arm iv-P: the OLD symmc-flow architecture on the O(3) asym data (tasks/n3_protocol.md §3.4).

The network, loss and sampler are the published code, unchanged: symmc_flow.model.SymMCFlow,
symmc_flow.train._step_loss (with cond_clean_packing) and eval_orient_matchrate.sample_orient_only.
This script builds the items, runs the training loop (our own, because symmc_flow.train.train returns
only the final model), implements the §3.4 selection rules, and writes n3_common draw bundles.

Published configuration (the paper's coset_deploy_s{0,1,2}: scripts/run_phaseB.sh ->
diag_orient_coset.py --deployable --steps 800 --batch-size 16 --seed s; the model_cfg stored in
checkpoints/coset_deploy_s0.pt):
    ModelConfig defaults: max_z 118, atom_embed_dim 64, egnn_hidden 128, egnn_layers 4, d_model 128,
    n_heads 8, n_attn_layers 4, ffn_mult 4, dropout 0.0, pair_n_freq 2, n_space_groups 230,
    sg_embed_dim 64, time_embed_dim 64, lattice_repr shape10, lattice_family_mask False,
    self_cond False; published n_cosets 238 and lambda_lattice = lambda_centroid = lambda_orient = 1.
    TrainConfig: lr 3e-4, batch 16, 800 steps, AdamW weight_decay 0, grad_clip 1.0, constant lr,
    prior_vol_per_atom = corpus mean, noised packing, relative gauge (relative_gauge_item).
iv-P keeps every ModelConfig default and sets (protocol §3.4): n_cosets = TRAIN key table + 1 reserved
id, lambda_lattice = lambda_centroid = 0, lambda_orient = 1; trains with cond_clean_packing=True in the
ABSOLUTE gauge (no relative_gauge_item, no assign_symmetry_cosets), batch 16, AdamW wd 0, clip 1.0,
constant lr in {1e-4, 3e-4, 1e-3}, 60 epochs over TRAIN.

Items (§3.4). For op k of the asym item (spglib ops, identity first), s_k = [det W_k = -1], D = diag(1,1,-1):
    centroid_k = frac(W_k c0 + t_k),  local_k = D^{s_k} local,  target orient_k = Rc_k R0 D^{s_k} (proper).
Slot k = op k, padded to 16 x 64. coset[0] = 0 (identity copy, like padding); coset[k>0] = id of the key
(sg, W_k as integers) in a table built from TRAIN only (ids 1..N in sorted key order); unseen keys map to
the reserved id N+1, whose embedding row is zeroed at init and never trained.

Batches are trimmed to the batch's real slots / atoms before the forward pass (--no-trim keeps 16 x 64;
--pack also runs the EGNN on real molecules only). Padding slots and atoms are masked everywhere in
SymMCFlow, so outputs on real slots are unchanged up to float rounding (`smoke` measures this); only the RNG
stream of the per-slot training prior differs. EGNN activations per batch-16 step: padded ~16 GB, trimmed
7 GB mean / 15.5 GB max, packed 3.4 / 4.6 GB (fp32, measured on CPU). The encoder (--pack or not) is one
choice for all five runs and every iv-P sampling run; `sample` refuses a checkpoint trained with the other.

Draw-affecting sampling settings (encoder, --crystals-per-batch, device type, draws, RK4 steps) are recorded in
every bundle's meta['sampling'] and in the .part resume key; `select ivs` freezes them from the scored VALSEL
bundles into ivP_frozen.json, and `sample --set testB|devtest` runs only the frozen pick (--arm ivP) in that
configuration (§7 steps 6-7). A sampling exception is retried one crystal, then one draw at a time; a draw that
still fails is an ARM failure (§1.3): R = NaN, RESID = inf, valid False, listed in meta['arm_failures'].
Pool payloads (gate, LJ, iv-S) are numpy arrays, never tensors (no shared-memory fds or mappings).

Subcommands (outputs under results/n3/private/{old,draws,select}/; CSD-derived)
    items    build iv-P items for sets; write counts and the TRAIN coset table
    gate     smoke gate, input side: true orientations rebuild the truth; slot -> op bijection
             (--no-fit: bijection and item-vs-expand only, for SEL / TEST-B)
    smoke    two training steps (finite loss, s/step), padded / trimmed / packed equivalence, a 2-crystal x
             2-draw VAL sample -> bundle round trip, iv-S from it (no output is matched against a truth)
    train    60 epochs; per-epoch checkpoint + fixed-draw VAL loss; resumable (resume.pt)
    select   window: §3.4 steps 1-3 from the loss histories, then the n3_score.py select config for steps
             4-5 (Stage A selector pooled over seeds, Stage B checkpoint per seed) and the runs it needs;
             ivs: freeze iv-P from the scorer's record and write the iv-S select config
             (--runs-root: read runs there and write every select output there; smoke only)
    sample   16 RK4 draws per crystal -> 'cells' bundle (sel lj, resid) + per-copy rotation sidecar
    ivs      iv-S: identity-copy orientation -> R_asym -> 'rasym' bundle (sel lj, resid of the source draw)

    python scripts/n3_old_symmc.py items --sets train val devtest sel testB
    python scripts/n3_old_symmc.py gate --sets val devtest --workers 2
    python scripts/n3_old_symmc.py gate --sets sel testB --no-fit
    python scripts/n3_old_symmc.py smoke --threads 2 --pack
    python scripts/n3_old_symmc.py train --lr 3e-4 --seed 0 --pack          # x {1e-4, 3e-4, 1e-3}, then seeds 1-2
    python scripts/n3_old_symmc.py select window --sample-args "--crystals-per-batch 8 --device cuda"
    python scripts/n3_old_symmc.py sample --run lr3e-4_s0 --epoch 12 --set valsel --pack --crystals-per-batch 8
    python scripts/n3_score.py select --config results/n3/private/select/ivP.json
    python scripts/n3_old_symmc.py select ivs --record results/n3/private/select/ivP_record.json
    python scripts/n3_old_symmc.py ivs --bundle results/n3/private/draws/ivP_lr3e-4_ep12_s0_valsel.pt
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

import n3_common as C
from symmc_flow import manifolds as M

OLD_DIR = os.path.join(C.PRIVATE, "old")
RUNS_DIR = os.path.join(OLD_DIR, "runs")
MAX_MOLS, MAX_ATOMS = 16, 64
LRS = (1e-4, 3e-4, 1e-3)
PUBLISHED_LR = 3e-4
EPOCHS = 60
S_DRAWS = 16
RK4_STEPS = 50
RESID_T = 0.975
VAL_SEED = 12345
D_MIRROR = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
SLOT_KEYS = ("Z", "local", "atom_mask", "mol_mask", "centroid", "orient", "coset")
SAMPLING_KEYS = ("encoder", "crystals_per_batch", "device_type", "draws", "rk4_steps")   # change iv-P draws


def lr_tag(lr):
    return "lr" + f"{lr:.0e}".replace("e-0", "e-")


def run_name(lr, seed):
    return f"{lr_tag(lr)}_s{seed}"


def git_rev():
    try:
        return subprocess.run(["git", "-C", C.REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def to_np(x):
    """Tensors -> numpy arrays, recursively, for Pool task and result payloads. A tensor crossing a Pool is
    pickled by torch's reducer into shared memory, which holds one fd (Linux default) or one shm mapping per
    storage in the parent; a TEST-B sample returns ~64k cell tensors. numpy arrays pickle as plain bytes."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    if type(x) in (list, tuple):
        return type(x)(to_np(v) for v in x)
    return x


def to_pt(x):
    """Inverse of to_np (numpy arrays -> tensors, recursively; dtypes preserved)."""
    import numpy as np
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, dict):
        return {k: to_pt(v) for k, v in x.items()}
    if type(x) in (list, tuple):
        return type(x)(to_pt(v) for v in x)
    return x


def pool_map(fn, tasks, workers, chunksize=4, unordered=False):
    """fn over numpy payloads (to_np) in `workers` processes (1 = in process); results come back via to_pt."""
    tasks = [to_np(t) for t in tasks]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            it = pool.imap_unordered(fn, tasks) if unordered else pool.map(fn, tasks, chunksize=chunksize)
            return [to_pt(r) for r in it]
    return [to_pt(fn(t)) for t in tasks]


# ------------------------------------------------------------------------------------------ items
def op_key(sg, W):
    Wi = torch.round(W)
    assert float((W - Wi).abs().max()) < 1e-6, "non-integer W"
    return (int(sg),) + tuple(int(v) for v in Wi.reshape(-1).tolist())


def key_str(key):
    return f"{key[0]}|" + ",".join(str(v) for v in key[1:])


def build_coset_table():
    """TRAIN-only table (sg, W as integers) -> id 1..N (sorted key order); reserved unseen id N+1."""
    keys = set()
    for a in C.load_set("train"):
        for k in range(1, a["K"]):
            keys.add(op_key(a["sg"], a["W"][k]))
    table = {key_str(k): i + 1 for i, k in enumerate(sorted(keys))}
    return table, len(table) + 1


def orbit_check(a):
    """Build step 2 (i) on the stored copy centroids of `orig`: every orb_k = W_k c0 + t_k within 1e-3 of a
    stored centroid, no two orb_k within 1e-3 (max-abs fractional component after min-image wrapping),
    n_mol = K; plus the nearest-slot map is a permutation. Returns (ok, max residual, min separation)."""
    o = a["orig"]
    orb = torch.einsum("kij,j->ki", a["W"].double(), a["c0"].double()) + a["t"].double()
    cs = o["centroid"][o["mol_mask"]].double()
    d = cs.unsqueeze(0) - orb.unsqueeze(1)
    d = (d - torch.round(d)).abs().amax(-1)                         # (K, n_mol)
    res, slot = d.min(1)
    e = orb.unsqueeze(0) - orb.unsqueeze(1)
    e = (e - torch.round(e)).abs().amax(-1) + torch.eye(len(orb), dtype=torch.float64) * 9.0
    sep = float(e.min()) if len(orb) > 1 else float("inf")
    ok = (cs.shape[0] == a["K"] and float(res.max()) < 1e-3 and sep > 1e-3
          and sorted(slot.tolist()) == list(range(a["K"])))
    return ok, float(res.max()), sep


def build_item(a, table, reserved, idx):
    """asym item -> all-tensor iv-P item (padded 16 x 64) and its per-crystal counts."""
    from g2_asym_baselines import cart_ops
    K, A = a["K"], a["local"].shape[0]
    assert K <= MAX_MOLS and A <= MAX_ATOMS, (a["refcode"], K, A)
    W, t = a["W"].double(), a["t"].double()
    I3 = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(W[0], I3) and float(t[0].abs().max()) < 1e-6, "op 0 is not the identity"
    Rc = cart_ops(a).double()
    s = torch.linalg.det(W) < 0
    cent = torch.einsum("kij,j->ki", W, a["c0"].double()) + t
    cent = cent - torch.floor(cent)
    loc = a["local"].double()
    Z = torch.zeros(MAX_MOLS, MAX_ATOMS, dtype=torch.long)
    local = torch.zeros(MAX_MOLS, MAX_ATOMS, 3)
    am = torch.zeros(MAX_MOLS, MAX_ATOMS, dtype=torch.bool)
    mm = torch.zeros(MAX_MOLS, dtype=torch.bool)
    centroid = torch.zeros(MAX_MOLS, 3)
    orient = torch.eye(3).expand(MAX_MOLS, 3, 3).clone()
    coset = torch.zeros(MAX_MOLS, dtype=torch.long)
    unseen = 0
    for k in range(K):
        Dk = D_MIRROR if bool(s[k]) else I3
        Rk = Rc[k] @ a["R0"].double() @ Dk
        assert abs(float(torch.linalg.det(Rk)) - 1.0) < 1e-6 and \
            float((Rk @ Rk.T - I3).abs().max()) < 1e-6, "target orientation not proper"
        local[k, :A] = (loc @ Dk).float()                           # rows are atoms: (D x)^T = x^T D
        orient[k] = Rk.float()
        Z[k, :A] = a["Z"]
        am[k, :A] = True
        mm[k] = True
        centroid[k] = cent[k].float()
        if k > 0:
            cid = table.get(key_str(op_key(a["sg"], W[k])))
            if cid is None:
                cid, unseen = reserved, unseen + 1
            coset[k] = cid
    item = {"Z": Z, "local": local, "atom_mask": am, "mol_mask": mm,
            "lattice": a["L"].float(), "centroid": centroid, "orient": orient,
            "sg": torch.tensor(int(a["sg"]), dtype=torch.long), "coset": coset,
            "idx": torch.tensor(idx, dtype=torch.long)}
    assert all(torch.is_tensor(v) for v in item.values())
    return item, {"K": K, "improper": int(s.sum()), "unseen": unseen}


def items_path(set_name):
    return os.path.join(OLD_DIR, f"items_{set_name}.pt")


def cmd_items(args):
    import collections
    table, reserved = build_coset_table()
    sha = hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()
    os.makedirs(OLD_DIR, exist_ok=True)
    json.dump({"keys": table, "reserved": reserved, "n_cosets": reserved, "sha256": sha},
              open(os.path.join(OLD_DIR, "coset_table.json"), "w"), indent=0)
    summary = {"train_keys": len(table), "reserved_id": reserved, "n_cosets": reserved, "table_sha256": sha}
    for name in args.sets:
        asym = C.load_set(name)
        items, cnt = [], collections.Counter()
        khist = collections.Counter()
        for i, a in enumerate(asym):
            it, c = build_item(a, table, reserved, i)
            items.append(it)
            khist[c["K"]] += 1
            cnt["improper_copies"] += c["improper"]
            cnt["crystals_with_improper"] += int(c["improper"] > 0)
            cnt["unseen_copies"] += c["unseen"]
            cnt["unseen_crystals"] += int(c["unseen"] > 0)
            cnt["orbit_fail"] += int(not orbit_check(a)[0])
        atomic_save({"set": name, "refcodes": [a["refcode"] for a in asym], "items": items,
                     "table_sha256": sha, "n_cosets": reserved, "reserved": reserved}, items_path(name))
        summary[name] = {"n": len(items), "K_hist": dict(sorted(khist.items())), **cnt}
        print(name, json.dumps(summary[name]), flush=True)
    path = os.path.join(OLD_DIR, "items_counts.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(summary)
    json.dump(prev, open(path, "w"), indent=1)
    print(f"train coset keys {len(table)} (reserved id {reserved}) -> {path}")


def load_items(set_name):
    """iv-P items of a set in its frozen order; valsel = val + sel with idx 0..399."""
    if set_name == "valsel":
        va, se = load_items("val"), load_items("sel")
        assert va["table_sha256"] == se["table_sha256"]
        items = [dict(it, idx=torch.tensor(i, dtype=torch.long)) for i, it in enumerate(va["items"] + se["items"])]
        return dict(va, set="valsel", refcodes=va["refcodes"] + se["refcodes"], items=items,
                    sha256={"val": va["sha256"], "sel": se["sha256"]})
    p = items_path(set_name)
    if not os.path.exists(p):
        sys.exit(f"no {p}; run: python scripts/n3_old_symmc.py items --sets {set_name}")
    blob = torch.load(p, weights_only=False)
    blob["sha256"] = C.sha256(p)
    src = C.DEV_CACHE if set_name in ("train", "val", "devtest") else C.POOL_CACHE
    if os.path.exists(src):
        assert blob["refcodes"] == [a["refcode"] for a in C.load_set(set_name)], "items not in frozen order"
    else:            # a training box holds only the item files (§7 step 5): tie the run to their sha256
        print(f"note: {os.path.relpath(src, C.REPO)} absent; {set_name} items sha256 {blob['sha256'][:16]}",
              flush=True)
    return blob


# ------------------------------------------------------------------------------------------ gate
def _gate_work(task):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from symmc_flow.molcrystal import rigid_to_structure
    i, orig, it = to_pt(task)
    truth = C.truth_structure({"orig": orig})
    cand = rigid_to_structure(it["lattice"], it["Z"], it["local"], it["centroid"], it["orient"],
                              it["atom_mask"], it["mol_mask"])
    t0 = time.time()
    ok = bool(StructureMatcher().fit(truth, cand))
    return i, ok, round(time.time() - t0, 2)


def cart_of(it, R):
    """Whole-molecule Cartesian coords (float64) of the real atoms: centroid_k L + R_k local_k."""
    L = it["lattice"].double()
    X = (it["centroid"].double() @ L).unsqueeze(1) + torch.einsum("mij,maj->mai", R.double(), it["local"].double())
    return X[it["atom_mask"]]


def cmd_gate(args):
    """Input-side gate (§3.4 smoke gate): true orientations rebuild the truth; slot -> op bijection.
    --no-fit skips the StructureMatcher rebuild (§3.4 requires it on VAL/DEV-TEST only)."""
    from g2_asym_baselines import expand
    out = {}
    for name in args.sets:
        asym, blob = C.load_set(name), load_items(name)
        n = len(asym)
        bij = [orbit_check(a) for a in asym]
        dx = []
        for a, it in zip(asym, blob["items"]):
            L, cent, orient, local, Z, am, mm = expand(a, a["R0"].double())
            X_exp = (cent.double() @ L.double()).unsqueeze(1) + torch.einsum("mij,maj->mai", orient.double(),
                                                                             local.double())
            dx.append(float((cart_of(it, it["orient"]) - X_exp.reshape(-1, 3)).abs().max()))
        t0 = time.time()
        out[name] = {"n": n, "bijection_ok": sum(b[0] for b in bij), "orbit_max_resid": max(b[1] for b in bij),
                     "orbit_min_sep": min(b[2] for b in bij), "max_abs_dx_vs_expand_A": max(dx)}
        if args.no_fit:
            out[name].update({"rebuild_match": None, "rebuild": "skipped (--no-fit)"})
        else:
            tasks = [(i, asym[i]["orig"], blob["items"][i]) for i in range(n)]
            fits = {i: (ok, sec) for i, ok, sec in pool_map(_gate_work, tasks, args.workers, unordered=True)}
            fail = [asym[i]["refcode"] for i in range(n) if not fits[i][0]]
            out[name].update({"rebuild_match": n - len(fail), "rebuild_fail_refcodes": fail,
                              "fit_sec_max": max(v[1] for v in fits.values())})
        out[name]["wall_sec"] = round(time.time() - t0, 1)
        print(name, json.dumps(out[name]), flush=True)
    path = os.path.join(OLD_DIR, "gate.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(out)
    json.dump(prev, open(path, "w"), indent=1)
    passed = all(v["rebuild_match"] in (None, v["n"]) and v["bijection_ok"] == v["n"] for v in out.values())
    print("GATE", "PASS" if passed else "FAIL", "->", path)


# ------------------------------------------------------------------------------------------ model / batches
def resolve_device(name):
    from symmc_flow.train import resolve_device as rd
    return rd(name)


def build_model(n_cosets, reserved, device):
    from symmc_flow.config import ModelConfig
    from symmc_flow.model import SymMCFlow
    mcfg = ModelConfig(n_cosets=n_cosets, lambda_lattice=0.0, lambda_centroid=0.0, lambda_orient=1.0)
    model = SymMCFlow(mcfg).to(device)
    with torch.no_grad():
        model.coset_embed.weight[reserved].zero_()                 # unseen-key row: zeroed, never trained
    return model, mcfg


def trim(batch):
    """Cut a collated batch to its real slots and atoms (padding is masked everywhere in SymMCFlow)."""
    mm, am = batch["mol_mask"], batch["atom_mask"]
    Mb = int(mm.sum(-1).max())
    Ab = int(am.sum(-1).max())
    assert not bool(mm[:, Mb:].any()) and not bool(am[..., Ab:].any()), "real slots/atoms not leading"
    out = dict(batch)
    for k in SLOT_KEYS:
        out[k] = batch[k][:, :Mb]
    for k in ("Z", "local", "atom_mask"):
        out[k] = out[k][:, :, :Ab]
    return out


def pack_encoder(model):
    """Opt-in (--pack): run SymMCFlow.encode_molecules' own layers (atom_embed, egnn, mol_proj) on the real
    molecules only. Real rows are identical to the published encoder up to float rounding; padding rows are
    zero instead of the one-fake-atom embedding, and padding tokens are masked everywhere downstream."""
    import types

    def encode(self, Z, local, atom_mask):
        B, Mm, A = Z.shape
        real = atom_mask.reshape(B * Mm, A).any(-1)
        feats = self.atom_embed(Z.reshape(B * Mm, A)[real])
        _, pooled = self.egnn(feats, local.reshape(B * Mm, A, 3)[real], atom_mask.reshape(B * Mm, A)[real])
        emb = self.mol_proj(pooled)
        out = emb.new_zeros(B * Mm, emb.shape[-1])
        out[real] = emb
        return out.reshape(B, Mm, -1)
    model.encode_molecules = types.MethodType(encode, model)
    return model


def to_dev(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def corpus_vol_per_atom(items):
    vols = sum(abs(float(torch.det(it["lattice"].double()))) for it in items)
    return vols / max(sum(int(it["atom_mask"].sum()) for it in items), 1)


def train_step(model, opt, batch, device, vpa, do_trim=True):
    """One step of the published loop body (symmc_flow.train.train): _step_loss with clean packing,
    weights (0, 0, 1), grad clip 1.0, skip on a non-finite loss or gradient norm."""
    from symmc_flow.train import _step_loss
    b = to_dev(trim(batch) if do_trim else batch, device)
    loss, parts = _step_loss(model, b, (0.0, 0.0, 1.0), device, False, vpa, None, None, True)
    opt.zero_grad()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(loss) and torch.isfinite(gnorm):
        opt.step()
        return float(parts["orient"]), True
    opt.zero_grad(set_to_none=True)
    return float("nan"), False


def fixed_val_draws():
    """§3.4: per-slot R0 and t generated ONCE, torch.manual_seed(12345) right before
    random_so3((100,16,16)) and rand((100,16)). CPU, float32 (the model's dtype)."""
    torch.manual_seed(VAL_SEED)
    R0v = M.random_so3((100, S_DRAWS, MAX_MOLS))
    tv = torch.rand((100, S_DRAWS))
    return R0v, tv


@torch.no_grad()
def fixed_val_loss(model, items, R0v, tv, device, do_trim=True, chunk=16):
    """Orientation CFM loss (cfm_loss's orient term, clean packing) over VAL x 16 fixed draws:
    sum over (crystal, draw, real slot) of |v_R - u_R|^2 / number of (crystal, draw, real slot)."""
    from symmc_flow.data import collate
    was = model.training
    model.eval()
    num, den = 0.0, 0
    for s in range(0, len(items), chunk):
        ids = list(range(s, min(s + chunk, len(items))))
        b = collate([items[i] for i in ids])
        b = to_dev(trim(b) if do_trim else b, device)
        Mb = b["mol_mask"].shape[1]
        emb = model.encode_molecules(b["Z"], b["local"], b["atom_mask"])
        m = b["mol_mask"].unsqueeze(-1).float()
        for j in range(S_DRAWS):
            R0 = R0v[ids, j, :Mb].to(device)
            t = tv[ids, j].to(device)
            Rt = M.so3_geodesic(R0, b["orient"], t.view(-1, 1))
            u = M.so3_velocity(R0, b["orient"])
            vR = model(emb, b["lattice"], b["centroid"], Rt, t, b["sg"], b["mol_mask"], coset=b["coset"])[2]
            num += float((((vR - u) ** 2) * m).sum().double())
            den += int(b["mol_mask"].sum())
    model.train(was)
    return num / den


# ------------------------------------------------------------------------------------------ train
def rng_state(device):
    st = {"torch": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st, device):
    torch.set_rng_state(st["torch"])
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def cmd_train(args):
    from torch.utils.data import DataLoader
    from symmc_flow.data import collate
    warnings.filterwarnings("ignore")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    tr, va = load_items("train"), load_items("val")
    assert len(tr["items"]) == 1687 and len(va["items"]) == 100 and tr["table_sha256"] == va["table_sha256"]
    vpa = corpus_vol_per_atom(tr["items"])
    out = os.path.join(args.runs_root or RUNS_DIR, run_name(args.lr, args.seed))
    os.makedirs(out, exist_ok=True)

    R0v, tv = fixed_val_draws()
    torch.manual_seed(args.seed)                                    # as symmc_flow.train.train
    model, mcfg = build_model(tr["n_cosets"], tr["reserved"], device)
    if args.pack:
        pack_encoder(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    dl = DataLoader(tr["items"], batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    hist, start = [], 0
    resume = os.path.join(out, "resume.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        hist, start = ck["hist"], ck["epoch"]
        set_rng_state(ck["rng"], device)
        print(f"resumed after epoch {start}", flush=True)
    meta = {"lr": args.lr, "seed": args.seed, "batch_size": args.batch_size, "epochs": args.epochs,
            "trim": not args.no_trim, "encoder": "packed" if args.pack else "published",
            "steps_per_epoch_cap": args.steps_per_epoch, "vol_per_atom": vpa, "n_cosets": tr["n_cosets"],
            "reserved": tr["reserved"], "table_sha256": tr["table_sha256"],
            "items_sha256": {"train": tr["sha256"], "val": va["sha256"]}, "model_cfg": dict(mcfg.__dict__),
            "git": git_rev(), "torch": torch.__version__, "device": str(device), "argv": sys.argv}
    mpath = os.path.join(out, "meta.json")
    if start == 0:
        json.dump(meta, open(mpath, "w"), indent=1)
    else:
        m0 = json.load(open(mpath))
        for k in ("lr", "seed", "batch_size", "trim", "encoder", "steps_per_epoch_cap", "table_sha256",
                  "items_sha256"):
            assert m0[k] == meta[k], f"resume with a different {k}: {m0[k]} vs {meta[k]}"
    for ep in range(start + 1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for k, batch in enumerate(dl):
            if args.steps_per_epoch and k == args.steps_per_epoch:
                break
            lo, ok = train_step(model, opt, batch, device, vpa, not args.no_trim)
            if ok:
                tot, n = tot + lo, n + 1
            else:
                skipped += 1
        assert float(model.coset_embed.weight[tr["reserved"]].abs().max()) == 0.0, "reserved row moved"
        vl = fixed_val_loss(model, va["items"], R0v, tv, device, not args.no_trim)
        row = {"epoch": ep, "train_orient": tot / max(n, 1), "val": vl, "steps": n, "skipped": skipped,
               "sec": round(time.time() - t0, 1)}
        atomic_save({"model": model.state_dict(), "epoch": ep, "val": vl, **meta},
                    os.path.join(out, f"ep{ep:02d}.pt"))
        hist.append(row)
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "epoch": ep,
                     "rng": rng_state(device)}, resume)
        with open(os.path.join(out, "hist.jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in hist)
        print(json.dumps(row), flush=True)


# ------------------------------------------------------------------------------------------ select
def load_val_hist(run):
    """{epoch: fixed-draw val loss} from a run's resume.pt (authoritative) or hist.jsonl."""
    d = run if os.path.isabs(run) else os.path.join(RUNS_DIR, run)
    rp = os.path.join(d, "resume.pt")
    if os.path.exists(rp):
        rows = torch.load(rp, map_location="cpu", weights_only=False)["hist"]
    else:
        rows = [json.loads(l) for l in open(os.path.join(d, "hist.jsonl"))]
    return {int(r["epoch"]): float(r["val"]) for r in rows}


def improving(v):
    """§3.4 step 1/3 condition: epoch-30 val loss more than 2% below the epoch-25 val loss."""
    return v[30] < 0.98 * v[25]


def best_epoch(v, window):
    """Lowest val loss within epochs 1..window; exact ties go to the earlier epoch."""
    return min(range(1, window + 1), key=lambda e: (v[e], e))


def window_and_lr(seed0):
    """Steps 1-2. seed0: {lr: {epoch: val}} for the three seed-0 runs (all 60 epochs)."""
    for lr, v in seed0.items():
        assert all(e in v for e in range(1, EPOCHS + 1)), f"seed-0 run at lr {lr} incomplete"
    window = 60 if any(improving(v) for v in seed0.values()) else 30
    # exact float ties are not covered by §3.4; they go to the published lr, then the lower lr
    lr = min(seed0, key=lambda x: (min(seed0[x][e] for e in range(1, window + 1)), x != PUBLISHED_LR, x))
    return window, lr


def late_extension(window, others):
    """Step 3. others: {seed: {epoch: val}} of the seed-1/2 runs at the chosen lr."""
    if window == 30 and any(improving(v) for v in others.values()):
        return 60
    return window


def cand_bundle(lr, epoch, seed, arm="ivP", set_name="valsel"):
    """Repo-relative bundle path of one checkpoint's draws (arm ivP / ivS)."""
    p = C.bundle_path(f"{arm}_{lr_tag(lr)}_ep{epoch:02d}", seed, set_name)
    return os.path.relpath(p, C.REPO).replace("\\", "/")


SELECT_DIR = os.path.join(C.PRIVATE, "select")
PY = "python scripts/n3_old_symmc.py"
RUN_SAME = ("encoder", "trim", "batch_size", "steps_per_epoch_cap", "table_sha256", "items_sha256", "n_cosets")


def select_dirs(args):
    """(dir of selection_window.json / ivP_frozen.json, dir of the scorer configs and records). Under --runs-root
    when given (smoke), so a test never leaves selection files at the real paths."""
    if args.runs_root:
        return args.runs_root, os.path.join(args.runs_root, "select")
    return OLD_DIR, SELECT_DIR


def repo_rel(p):
    """Repo-relative '/' path when p is inside the repo, else the absolute path."""
    p = os.path.abspath(p)
    try:
        r = os.path.relpath(p, C.REPO)
    except ValueError:
        return p
    return p if r.startswith("..") else r.replace("\\", "/")


def rel(p):
    return p if os.path.isabs(p) else os.path.join(C.REPO, p)


def run_meta(root, lr, seed):
    p = os.path.join(root, run_name(lr, seed), "meta.json")
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    return json.load(open(p))


def encoder_flag(encoder):
    return " --pack" if encoder == "packed" else ""


def cmd_select_window(args):
    """§3.4 steps 1-3 from the loss histories; once all five runs are complete, write the n3_score.py select
    config for steps 4-5 (Stage A pooled over seeds on each seed's best-in-window checkpoint; Stage B per seed
    over {best in window, last of window}; ties draw@1 -> any@16 -> lower val loss) and list the runs it needs.
    All runs must share the training configuration (encoder, trim, batch, items); the printed sample commands
    carry the runs' encoder plus --sample-args (the step-4 --crystals-per-batch / --device choice)."""
    root = args.runs_root or RUNS_DIR
    odir, sdir = select_dirs(args)
    seed0, metas = {}, {}
    for lr in LRS:
        d = os.path.join(root, run_name(lr, 0))
        if not os.path.isdir(d):
            sys.exit(f"missing seed-0 run {d}")
        seed0[lr] = load_val_hist(d)
        metas[run_name(lr, 0)] = run_meta(root, lr, 0)
    w0, lr = window_and_lr(seed0)
    others = {}
    for s in (1, 2):
        d = os.path.join(root, run_name(lr, s))
        if os.path.isdir(d):
            v = load_val_hist(d)
            if all(e in v for e in range(1, EPOCHS + 1)):
                others[s] = v
                metas[run_name(lr, s)] = run_meta(root, lr, s)
    ref = metas[run_name(lr, 0)]
    for name, m in metas.items():
        diff = [k for k in RUN_SAME if m.get(k) != ref.get(k)]
        assert not diff, f"run {name} differs from {run_name(lr, 0)} in {diff}"
    if not args.runs_root:
        assert ref["steps_per_epoch_cap"] == 0, "real runs must not cap steps per epoch"
    enc = ref["encoder"]
    samp_args = encoder_flag(enc) + (" " + args.sample_args.strip() if args.sample_args.strip() else "")
    out = {"window_step1": w0, "lr": lr,
           "lr_min_val_in_window": {lr_tag(x): min(seed0[x][e] for e in range(1, w0 + 1)) for x in LRS},
           "improving_seed0": {lr_tag(x): improving(seed0[x]) for x in LRS},
           "seeds_complete_at_lr": [0] + sorted(others),
           "train_config": {k: ref.get(k) for k in RUN_SAME}}
    cmds = []
    if len(others) == 2:
        w = late_extension(w0, others)
        runs = {0: seed0[lr], **others}
        cand = {s: {"best": best_epoch(v, w), "last": w} for s, v in runs.items()}
        out.update({"window": w, "late_extension": w != w0,
                    "candidates": {str(s): dict(c, val_best=runs[s][c["best"]], val_last=runs[s][w])
                                   for s, c in cand.items()}})
        cfg = {"name": "ivP", "matcher": "primary", "mode_tiebreak": "resid",
               "stage_a": {"cells": [{"label": f"ivP_{lr_tag(lr)}_best_in_window",
                                      "bundles": [cand_bundle(lr, cand[s]["best"], s) for s in (0, 1, 2)]}],
                           "tie": ["selector"]},
               "stage_b": {"selector": "from_stage_a", "tie": ["draw1", "any16", "val_loss"],
                           "seeds": {str(s): [{"label": f"ep{e:02d}", "bundle": cand_bundle(lr, e, s),
                                               "val_loss": runs[s][e]}
                                              for e in sorted({cand[s]["best"], cand[s]["last"]})]
                                     for s in (0, 1, 2)}},
               "out": repo_rel(os.path.join(sdir, "ivP_record.json"))}
        os.makedirs(sdir, exist_ok=True)
        cpath = os.path.join(sdir, "ivP.json")
        json.dump(cfg, open(cpath, "w"), indent=1)
        out["scorer_config"] = repo_rel(cpath)
        for s in (0, 1, 2):
            for e in sorted({cand[s]["best"], cand[s]["last"]}):
                cmds.append(f"{PY} sample --run {run_name(lr, s)} --epoch {e} --set valsel{samp_args}")
        bundles = sorted({c["bundle"] for sd in cfg["stage_b"]["seeds"].values() for c in sd})
        cmds.append("python scripts/n3_score.py match --matcher primary --bundles " + " ".join(bundles))
        cmds.append(f"python scripts/n3_score.py select --config {out['scorer_config']}")
        cmds.append(f"{PY} select ivs --record {cfg['out']}" + (f" --runs-root {args.runs_root}" if args.runs_root else ""))
    else:
        cmds += [f"{PY} train --lr {lr:g} --seed {s}{encoder_flag(enc)}" for s in (1, 2) if s not in others]
        cmds.append(f"{PY} select window")
    out["next_commands"] = cmds
    os.makedirs(odir, exist_ok=True)
    path = os.path.join(odir, "selection_window.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("->", path)


def sampling_config(meta):
    """The draw-affecting sampling settings recorded in an iv-P bundle's meta (frozen for step 7)."""
    return {k: meta["sampling"][k] for k in SAMPLING_KEYS}


def cmd_select_ivs(args):
    """After n3_score.py select on ivP.json: freeze iv-P (lr, window, Stage-A selector, Stage-B epoch per seed,
    sampling configuration) and write the iv-S select config (its own selector, Stage A on iv-S candidates of
    the same checkpoints). Refuses a record not made from this selection_window's scorer config, scored bundles
    changed since scoring, and iv-P VALSEL bundles sampled in different configurations."""
    odir, sdir = select_dirs(args)
    rec = json.load(open(args.record))
    win = json.load(open(os.path.join(odir, "selection_window.json")))
    assert "scorer_config" in win, "selection_window.json predates the five complete runs; re-run select window"
    assert rec["config"] == json.load(open(rel(win["scorer_config"]))), \
        f"{args.record} was not made from {win['scorer_config']}"
    for p, h in rec["bundles_sha256"].items():
        assert C.sha256(rel(p)) == h, f"{p} changed since it was scored"
    lr = float(win["lr"])
    root = args.runs_root or RUNS_DIR
    epochs = {int(s): int(v["chosen"][2:]) for s, v in rec["stage_b"]["seeds"].items()}
    assert sorted(epochs) == [0, 1, 2]
    for s, e in epochs.items():
        c = win["candidates"][str(s)]
        assert e in (c["best"], c["last"]), f"seed {s}: chosen epoch {e} is not a window candidate {c}"
    samp, samp_rec = None, {}
    for p in sorted(rec["bundles_sha256"]):
        m = C.load_bundle(rel(p))["meta"]
        sc = sampling_config(m)
        assert samp is None or sc == samp, f"{p}: sampling {sc} differs from {samp}"
        samp = sc
        samp_rec[p] = {"torch": m.get("torch"), "device": m.get("device")}
    assert samp["draws"] == S_DRAWS and samp["rk4_steps"] == RK4_STEPS, samp
    flags = (f" --crystals-per-batch {samp['crystals_per_batch']} --device {samp['device_type']}"
             + encoder_flag(samp["encoder"]))
    cfg = {"name": "ivS", "matcher": "primary", "mode_tiebreak": "resid",
           "stage_a": {"cells": [{"label": "ivS", "bundles": [cand_bundle(lr, epochs[s], s, "ivS") for s in (0, 1, 2)]}],
                       "tie": ["selector"]},
           "out": repo_rel(os.path.join(sdir, "ivS_record.json"))}
    os.makedirs(sdir, exist_ok=True)
    cpath = os.path.join(sdir, "ivS.json")
    json.dump(cfg, open(cpath, "w"), indent=1)
    ckpts = {s: os.path.join(root, run_name(lr, s), f"ep{e:02d}.pt") for s, e in epochs.items()}
    frozen = {"arm": "iv-P", "lr": lr, "window": win["window"], "selector": rec["stage_a"]["chosen"]["selector"],
              "epoch_per_seed": {str(s): e for s, e in epochs.items()},
              "checkpoints": {str(s): repo_rel(p) for s, p in ckpts.items()},
              "checkpoint_sha256": {str(s): C.sha256(p) for s, p in ckpts.items()},
              "train_config": win.get("train_config"),
              "sampling": samp,
              "sampling_recorded_valsel": samp_rec,
              "prior_seeds": "torch.manual_seed(700000 + 10000 s + 1000 split + i); split 0 VALSEL, 1 TEST-B, 2 DEV-TEST",
              "ivS_config": repo_rel(cpath),
              "step7_runs": [f"{PY} sample --run {run_name(lr, s)} --epoch {e} --set {st} --arm ivP{flags}"
                             for st in ("testB", "devtest") for s, e in sorted(epochs.items())]
              + [f"{PY} ivs --bundle results/n3/private/draws/ivP_s{s}_{st}.pt"
                 for st in ("testB", "devtest") for s in sorted(epochs)]}
    fpath = os.path.join(odir, "ivP_frozen.json")
    json.dump(frozen, open(fpath, "w"), indent=1)
    cmds = [f"{PY} ivs --bundle {cand_bundle(lr, epochs[s], s)}" for s in (0, 1, 2)]
    cmds.append("python scripts/n3_score.py match --matcher primary --bundles "
                + " ".join(cfg["stage_a"]["cells"][0]["bundles"]))
    cmds.append(f"python scripts/n3_score.py select --config {frozen['ivS_config']}")
    print(json.dumps(frozen, indent=1))
    print("iv-S selection:\n  " + "\n  ".join(cmds))
    print("->", fpath)


def cmd_select(args):
    if args.what == "window":
        cmd_select_window(args)
    else:
        if not args.record:
            sys.exit("select ivs needs --record <n3_score.py select record of ivP.json>")
        cmd_select_ivs(args)


# ------------------------------------------------------------------------------------------ sample
def prior_seed(seed, set_name, i):
    """§2.2: 700000 + 10000 s + 1000 split + i; i is the VALSEL position for val/sel (split code 0)."""
    base = 100 if set_name == "sel" else 0
    return 700_000 + 10_000 * seed + 1_000 * C.SPLIT_CODE[set_name] + base + i


def load_ckpt_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_model(ck["n_cosets"], ck["reserved"], device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def sample_chunk(model, items, R0s, device):
    """RK4 orientation-only sampling (lattice and all centroids frozen at the truth) for several crystals.
    items: list of iv-P items; R0s: list of (S,K,3,3) priors. Returns list of (R (S,K,3,3), resid (S,))."""
    from symmc_flow.data import collate
    from eval_orient_matchrate import sample_orient_only
    b = to_dev(trim(collate(items)), device)
    Bc, Mb = b["mol_mask"].shape
    S = R0s[0].shape[0]
    rep = lambda x: x.repeat_interleave(S, dim=0)
    emb = rep(model.encode_molecules(b["Z"], b["local"], b["atom_mask"]))
    L, x, sg, mask, cs = rep(b["lattice"]), rep(b["centroid"]), rep(b["sg"]), rep(b["mol_mask"]), rep(b["coset"])
    R0 = torch.eye(3).expand(Bc, S, Mb, 3, 3).clone()
    for c, r in enumerate(R0s):
        R0[c, :, :r.shape[1]] = r
    R = sample_orient_only(model, emb, L, x, R0.reshape(Bc * S, Mb, 3, 3).to(device), sg, mask, RK4_STEPS, coset=cs)
    vR = model.forward(emb, L, x, R, torch.full((Bc * S,), RESID_T, device=device), sg, mask, coset=cs)[2]
    m = mask.float()
    resid = (vR.norm(dim=-1) * m).sum(-1) / m.sum(-1)
    R, resid = R.reshape(Bc, S, Mb, 3, 3).cpu(), resid.reshape(Bc, S).cpu()
    return [(R[c, :, :r.shape[1]].clone(), resid[c].clone()) for c, r in enumerate(R0s)]


def is_fatal(e):
    """Resource or device errors abort the run (resumable from .part); they are not generator errors."""
    msg = str(e).lower()
    return (isinstance(e, (MemoryError, torch.cuda.OutOfMemoryError))
            or any(s in msg for s in ("out of memory", "can't allocate memory", "not enough memory",
                                      "cuda error", "cublas", "cudnn")))


def sample_robust(model, items, R0s, device, tags, log):
    """sample_chunk; on an exception, retry one crystal at a time, then one draw at a time (same priors). A draw
    that still fails is an ARM failure (§1.3 generator error): R = NaN, RESID = inf, so valid = False for it,
    logged with its refcode. Retried outputs differ from batched ones only by float rounding."""
    try:
        return sample_chunk(model, items, R0s, device)
    except Exception as e:
        if is_fatal(e):
            raise
        err = f"{type(e).__name__}: {e}"[:300]
        log["retries"].append({"i": [t["i"] for t in tags], "draw": tags[0].get("draw"), "error": err})
    if len(items) > 1:
        return [o for it, r0, tg in zip(items, R0s, tags)
                for o in sample_robust(model, [it], [r0], device, [tg], log)]
    r0, tg = R0s[0], tags[0]
    if r0.shape[0] > 1:
        parts = [sample_robust(model, items, [r0[j:j + 1]], device, [dict(tg, draw=j)], log)[0]
                 for j in range(r0.shape[0])]
        return [(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))]
    log["failures"].append(dict(tg, error=err))
    return [(torch.full_like(r0, float("nan")), torch.full((1,), float("inf")))]


def _lj_work(task):
    """LJ + whole-molecule cells for one crystal's draws (CPU). numpy payload in and out (to_np)."""
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from g2_asym_baselines import energy
    it, R = to_pt(task)
    K = int(it["mol_mask"].sum())
    A = int(it["atom_mask"][0].sum())
    Z, local, am = it["Z"][:K, :A], it["local"][:K, :A], it["atom_mask"][:K, :A]
    cent, L = it["centroid"][:K], it["lattice"]
    mm = torch.ones(K, dtype=torch.bool)
    lj, cells, valid = [], [], []
    for j in range(R.shape[0]):
        Rj = R[j]
        ok = bool(torch.isfinite(Rj).all())
        X = ((cent.double() @ L.double()).unsqueeze(1) + torch.einsum("mij,maj->mai", Rj.double(), local.double()))
        ok = ok and bool(torch.isfinite(X).all())
        e = energy(L, cent, Rj, local, Z, am, mm) if ok else float("inf")
        lj.append(e if math.isfinite(e) else float("inf"))
        valid.append(ok)
        cells.append({"lattice": L.double(), "cart": X[am], "species": Z[am].clone(),
                      "mol": torch.arange(K).unsqueeze(1).expand(K, A)[am].clone()})
    return to_np((lj, cells, valid))


def check_frozen(args, ck, ckpt_sha, samp):
    """§7 step 7: TEST-B / DEV-TEST are sampled only for the frozen iv-P pick (ivP_frozen.json from `select ivs`),
    as arm 'ivP', in the frozen sampling configuration."""
    fpath = args.frozen or os.path.join(OLD_DIR, "ivP_frozen.json")
    if not os.path.exists(fpath):
        sys.exit(f"--set {args.set}: no {fpath}; TEST-B / DEV-TEST are sampled only for the frozen iv-P pick")
    fz = json.load(open(fpath))
    s = str(int(ck["seed"]))
    bad = []
    if args.arm != "ivP":
        bad.append("--arm must be ivP")
    if args.limit:
        bad.append("--limit is not allowed")
    if s not in fz["epoch_per_seed"] or args.epoch != fz["epoch_per_seed"][s] or ckpt_sha != fz["checkpoint_sha256"][s]:
        bad.append(f"checkpoint is not the frozen seed-{s} pick {fz['checkpoints'].get(s)}")
    if samp != fz["sampling"]:
        bad.append(f"sampling {samp} != frozen {fz['sampling']}")
    if bad:
        sys.exit(f"refused by {fpath}: " + "; ".join(bad))
    print(f"frozen iv-P pick (seed {s}, epoch {args.epoch}) in the frozen sampling configuration", flush=True)


def cmd_sample(args):
    warnings.filterwarnings("ignore")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    run = args.run if os.path.isabs(args.run) else os.path.join(RUNS_DIR, args.run)
    ckpt = os.path.join(run, f"ep{args.epoch:02d}.pt")
    model, ck = load_ckpt_model(ckpt, device)
    enc = "packed" if args.pack else "published"
    if ck.get("encoder", enc) != enc:
        sys.exit(f"{ckpt} was trained with the {ck['encoder']} encoder; sample with the same (--pack)")
    if args.pack:
        pack_encoder(model)
    samp = {"encoder": enc, "crystals_per_batch": args.crystals_per_batch, "device_type": device.type,
            "draws": args.draws, "rk4_steps": RK4_STEPS}
    ckpt_sha = C.sha256(ckpt)
    if args.set in ("testB", "devtest"):
        check_frozen(args, ck, ckpt_sha, samp)
    blob = load_items(args.set)
    assert blob["table_sha256"] == ck["table_sha256"], "item coset table differs from the checkpoint's"
    items, refcodes = blob["items"], blob["refcodes"]
    n = len(items) if not args.limit else min(args.limit, len(items))
    seed = int(ck["seed"])
    arm = args.arm or f"ivP_{lr_tag(ck['lr'])}_ep{args.epoch:02d}"
    path = C.bundle_path(arm, seed, args.set)
    if args.limit:
        path = os.path.join(OLD_DIR, "smoke", f"{arm}_s{seed}_{args.set}_limit{n}.pt")
    part = path + ".part.pt"
    key = {"ckpt_sha256": ckpt_sha, "set": args.set, "n": n, **samp}
    done = torch.load(part, weights_only=False) if os.path.exists(part) else \
        {"R": [], "resid": [], "sec": 0.0, "key": key, "failures": [], "retries": []}
    assert done["key"] == key, f"stale {part} (made with {done['key']}); delete it to restart"
    t_start = time.time()
    i = len(done["R"])
    print(f"{arm} s{seed} {args.set}: {i}/{n} sampled, device {device}", flush=True)
    while i < n:
        ids = list(range(i, min(i + args.crystals_per_batch, n)))
        R0s = []
        for c in ids:
            torch.manual_seed(prior_seed(seed, args.set, c))
            R0s.append(M.random_so3((args.draws, int(items[c]["mol_mask"].sum()))))
        t0 = time.time()
        nf = len(done["failures"])
        for R, rs in sample_robust(model, [items[c] for c in ids], R0s, device,
                                   [{"i": c, "refcode": refcodes[c]} for c in ids], done):
            done["R"].append(R)
            done["resid"].append(rs)
        for f in done["failures"][nf:]:
            print(f"  ARM FAILURE {f['refcode']} (i {f['i']}, draw {f.get('draw')}): {f['error']}", flush=True)
        done["sec"] += time.time() - t0
        i = len(done["R"])
        if i % args.save_every < len(ids) or i == n:
            atomic_save(done, part)
            print(f"  {i}/{n} sampled ({done['sec'] / i:.2f} s/crystal)", flush=True)
    t0 = time.time()
    res = pool_map(_lj_work, [(items[c], done["R"][c]) for c in range(n)], args.workers)
    lj_sec = time.time() - t0
    valid = torch.tensor([r[2] for r in res], dtype=torch.bool)
    lj = torch.tensor([r[0] for r in res], dtype=torch.float64)
    resid = torch.stack(done["resid"]).double()
    resid = torch.where(valid & torch.isfinite(resid), resid, torch.full_like(resid, float("inf")))
    meta = {"argv": sys.argv, "git": git_rev(), "torch": torch.__version__, "device": str(device),
            "ckpt": repo_rel(ckpt), "ckpt_sha256": ckpt_sha, "epoch": args.epoch,
            "lr": ck["lr"], "train_seed": seed, "rk4_steps": RK4_STEPS, "draws": args.draws, "resid_t": RESID_T,
            "prior_seed": "700000 + 10000 s + 1000 split + i", "trim": True, "encoder": enc,
            "crystals_per_batch": args.crystals_per_batch, "device_type": device.type, "sampling": samp,
            "arm_failures": done["failures"], "retries": done["retries"], "invalid_draws": int((~valid).sum()),
            "sample_sec": done["sec"], "lj_sec": lj_sec, "lj_evals": int(valid.numel()),
            "n_cosets": ck["n_cosets"], "table_sha256": ck["table_sha256"], "items_sha256": blob["sha256"]}
    sel, cells = {"lj": lj, "resid": resid}, [r[1] for r in res]
    if args.limit:              # smoke: a partial set cannot pass save_bundle's frozen-order check; same schema
        atomic_save({"arm": arm, "seed": seed, "set": args.set, "refcodes": refcodes[:n], "kind": "cells", "R": None,
                     "cells": cells, "valid": valid, "sel": sel, "meta": dict(meta, smoke_limit=n)}, path)
    else:
        C.save_bundle(path, arm=arm, seed=seed, set_name=args.set, refcodes=refcodes, kind="cells", valid=valid,
                      sel=sel, meta=meta, cells=cells)
    atomic_save({"refcodes": refcodes[:n], "R": done["R"], "bundle_sha256": C.sha256(path)}, path[:-3] + ".rot.pt")
    os.remove(part)
    print(f"-> {path}  ({done['sec']:.0f} s sampling, {lj_sec:.0f} s LJ, {time.time() - t_start:.0f} s total; "
          f"{len(done['failures'])} failed draws, {int((~valid).sum())} invalid draws)")


# ------------------------------------------------------------------------------------------ iv-S
def _ivs_work(task):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from g2_asym_baselines import expand, energy
    a, Ra = to_pt(task)
    lj = []
    for j in range(Ra.shape[0]):
        e = energy(*expand(a, Ra[j])) if bool(torch.isfinite(Ra[j]).all()) else float("inf")
        lj.append(e if math.isfinite(e) else float("inf"))
    return lj


def cmd_ivs(args):
    """iv-S (§3.4 diagnostic): R_asym = the returned orientation of the identity-op copy (slot 0), expanded
    exactly with the crystal's spglib ops. sel 'lj' = energy of the expanded cell; sel 'resid' = the
    source iv-P draw's RESID."""
    warnings.filterwarnings("ignore")
    b = C.load_bundle(args.bundle)
    rot = torch.load(args.bundle[:-3] + ".rot.pt", weights_only=False)
    assert rot["refcodes"] == b["refcodes"] and rot["bundle_sha256"] == C.sha256(args.bundle)
    smoke = "smoke_limit" in b["meta"]
    asym = C.load_set(b["set"])[:len(b["refcodes"])]
    assert [a["refcode"] for a in asym] == b["refcodes"]
    Ra = torch.stack([R[:, 0].double() for R in rot["R"]])             # (n, S, 3, 3)
    t0 = time.time()
    lj = torch.tensor(pool_map(_ivs_work, list(zip(asym, Ra)), args.workers), dtype=torch.float64)
    valid = torch.isfinite(Ra).flatten(2).all(-1) & b["valid"]
    arm = b["arm"].replace("ivP", "ivS", 1)
    meta = {"argv": sys.argv, "git": git_rev(), "source_bundle": repo_rel(args.bundle),
            "source_sha256": C.sha256(args.bundle), "lj_sec": time.time() - t0, "lj_evals": int(valid.numel()),
            "resid": "source iv-P draw RESID", "source_arm_failures": b["meta"].get("arm_failures", []),
            "sampling": b["meta"].get("sampling")}
    sel = {"lj": lj, "resid": b["sel"]["resid"].clone()}
    if smoke:
        path = args.bundle.replace(b["arm"], arm, 1)
        atomic_save({"arm": arm, "seed": b["seed"], "set": b["set"], "refcodes": b["refcodes"], "kind": "rasym",
                     "R": Ra, "cells": None, "valid": valid, "sel": sel, "meta": dict(meta, smoke_limit=len(asym))}, path)
    else:
        path = C.bundle_path(arm, b["seed"], b["set"])
        C.save_bundle(path, arm=arm, seed=b["seed"], set_name=b["set"], refcodes=b["refcodes"], kind="rasym",
                      valid=valid, sel=sel, meta=meta, R=Ra)
    print(f"-> {path}")


# ------------------------------------------------------------------------------------------ smoke
def cmd_smoke(args):
    """CPU smoke (no truth matching of any output): two training steps, padded-vs-trimmed equivalence,
    fixed-draw VAL loss timing, 2 VAL crystals x 2 draws -> bundle round trip -> iv-S."""
    import psutil
    import threading
    from torch.utils.data import DataLoader
    from symmc_flow.data import collate
    from symmc_flow.molcrystal import rigid_to_structure
    from g2_asym_baselines import energy, expand
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    tr, va = load_items("train"), load_items("val")
    vpa = corpus_vol_per_atom(tr["items"])
    rep = {"threads": args.threads, "n_cosets": tr["n_cosets"], "vol_per_atom": vpa}
    proc = psutil.Process()
    peak = [0]
    stop = [False]

    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.05)
    threading.Thread(target=watch, daemon=True).start()

    # (1) training steps exactly as `train` runs them (seed 0, published lr, trimmed batches)
    R0v, tv = fixed_val_draws()
    torch.manual_seed(0)
    model, mcfg = build_model(tr["n_cosets"], tr["reserved"], device)
    if args.pack:
        pack_encoder(model)
    rep["params"] = sum(p.numel() for p in model.parameters())
    rep["model_cfg"] = dict(mcfg.__dict__)
    opt = torch.optim.AdamW(model.parameters(), lr=PUBLISHED_LR, weight_decay=0.0)
    dl = DataLoader(tr["items"], batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    steps = []
    for k, batch in enumerate(dl):
        if k == args.steps:
            break
        tb = trim(batch)
        peak[0] = 0
        t0 = time.time()
        lo, ok = train_step(model, opt, batch, device, vpa, True)
        steps.append({"loss": lo, "finite": ok, "sec": round(time.time() - t0, 2), "real_mols": int(tb["mol_mask"].sum()),
                      "trimmed_MxA": list(tb["local"].shape[1:3]), "peak_rss_GB": round(peak[0] / 2**30, 2)})
        print("train step", steps[-1], flush=True)
    rep["train_steps"] = {"batch_size": args.batch_size, "encoder": "packed" if args.pack else "published",
                          "steps": steps}
    rep["train_finite"] = all(s["finite"] and math.isfinite(s["loss"]) for s in steps)
    assert float(model.coset_embed.weight[tr["reserved"]].abs().max()) == 0.0

    # (1b) one fully padded (16 x 64) step at a small batch, to size the padded path
    if args.padded_batch:
        m2, _ = build_model(tr["n_cosets"], tr["reserved"], device)
        o2 = torch.optim.AdamW(m2.parameters(), lr=PUBLISHED_LR, weight_decay=0.0)
        b = collate(tr["items"][:args.padded_batch])
        peak[0] = 0
        t0 = time.time()
        lo, ok = train_step(m2, o2, b, device, vpa, False)
        rep["train_step_padded"] = {"batch": args.padded_batch, "loss": lo, "finite": ok,
                                    "sec": round(time.time() - t0, 2), "peak_rss_GB": round(peak[0] / 2**30, 2)}
        print("padded step", rep["train_step_padded"], flush=True)
        del m2, o2

    # (2) padded (published encoder) vs trimmed (published) vs trimmed + packed: one network function on real slots
    ref, _ = build_model(tr["n_cosets"], tr["reserved"], device)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    pk, _ = build_model(tr["n_cosets"], tr["reserved"], device)
    pk.load_state_dict(model.state_dict())
    pack_encoder(pk).eval()
    b = collate(va["items"][:4])
    with torch.no_grad():
        outs = []
        for net, bb in ((ref, b), (ref, trim(b)), (pk, trim(b))):
            Mb = bb["mol_mask"].shape[1]
            emb = net.encode_molecules(bb["Z"], bb["local"], bb["atom_mask"])
            Rt = M.so3_geodesic(R0v[:4, 0, :Mb], bb["orient"], tv[:4, 0].view(-1, 1))
            outs.append(net(emb, bb["lattice"], bb["centroid"], Rt, tv[:4, 0], bb["sg"], bb["mol_mask"],
                            coset=bb["coset"])[2])
        Mb = outs[1].shape[1]
        mk = b["mol_mask"][:, :Mb]
        rep["max_abs_dvR_padded_vs_trimmed"] = float((outs[0][:, :Mb] - outs[1])[mk].abs().max())
        rep["max_abs_dvR_trimmed_vs_packed"] = float((outs[1] - outs[2])[mk].abs().max())
        rep["max_abs_vR"] = float(outs[0][:, :Mb][mk].abs().max())
        rep["val_loss_4_padded_trimmed_packed"] = [
            fixed_val_loss(ref, va["items"][:4], R0v[:4], tv[:4], device, False),
            fixed_val_loss(ref, va["items"][:4], R0v[:4], tv[:4], device, True),
            fixed_val_loss(pk, va["items"][:4], R0v[:4], tv[:4], device, True)]
    del ref, pk
    t0 = time.time()
    rep["fixed_val_loss_full"] = fixed_val_loss(model, va["items"], R0v, tv, device, True)
    rep["fixed_val_loss_sec"] = round(time.time() - t0, 1)
    print("equivalence", rep["max_abs_dvR_padded_vs_trimmed"], rep["max_abs_dvR_trimmed_vs_packed"],
          rep["val_loss_4_padded_trimmed_packed"], "| val loss", rep["fixed_val_loss_full"],
          rep["fixed_val_loss_sec"], "s", flush=True)

    # (3) 2 VAL crystals x 2 draws with the 2-step model -> smoke bundle -> round trip (no truth matching)
    run = os.path.join(OLD_DIR, "smoke", "run")
    ck = {"model": model.state_dict(), "epoch": 0, "lr": PUBLISHED_LR, "seed": 0, "n_cosets": tr["n_cosets"],
          "reserved": tr["reserved"], "table_sha256": tr["table_sha256"], "val": rep["fixed_val_loss_full"]}
    atomic_save(ck, os.path.join(run, "ep00.pt"))
    t0 = time.time()
    sa = argparse.Namespace(run=run, epoch=0, set="val", limit=2, draws=2, arm="ivP_smoke", device="cpu",
                            threads=args.threads, crystals_per_batch=1, save_every=1, workers=1, pack=False,
                            frozen="")
    cmd_sample(sa)
    rep["sample_2x2_sec"] = round(time.time() - t0, 1)
    path = os.path.join(OLD_DIR, "smoke", "ivP_smoke_s0_val_limit2.pt")
    bl = C.load_bundle(path)
    rot = torch.load(path[:-3] + ".rot.pt", weights_only=False)
    asym = C.load_set("val")[:2]
    rt = {"valid_all": bool(bl["valid"].all()), "lj_shape": list(bl["sel"]["lj"].shape),
          "resid_finite": bool(torch.isfinite(bl["sel"]["resid"]).all())}
    dmax, de, dsp, nat = 0.0, 0.0, 0, []
    for i in range(2):
        it = va["items"][i]
        K = int(it["mol_mask"].sum())
        for j in range(2):
            R = rot["R"][i][j]
            st_b = C.cell_structure(bl["cells"][i][j])
            Rfull = torch.eye(3).expand(MAX_MOLS, 3, 3).clone()
            Rfull[:K] = R
            st_d = rigid_to_structure(it["lattice"], it["Z"], it["local"], it["centroid"], Rfull,
                                      it["atom_mask"], it["mol_mask"])
            df = st_b.frac_coords - st_d.frac_coords
            dmax = max(dmax, float(abs(df - df.round()).max()))
            dsp += int([s.Z for s in st_b.species] != [s.Z for s in st_d.species])
            e = energy(it["lattice"], it["centroid"], Rfull, it["local"], it["Z"], it["atom_mask"], it["mol_mask"])
            de = max(de, abs(e - float(bl["sel"]["lj"][i, j])) / max(1.0, abs(e)))
            nat.append(len(st_b))
    rt.update({"max_frac_diff_bundle_vs_direct": dmax, "species_mismatch": dsp, "lj_rel_diff": de, "atoms": nat})
    ia = argparse.Namespace(bundle=path, workers=1)
    cmd_ivs(ia)
    bs = C.load_bundle(path.replace("ivP_smoke", "ivS_smoke"))
    ok_s, dmax_s = 0, 0.0
    for i in range(2):
        for j in range(2):
            Ra = bs["R"][i, j]
            e = energy(*expand(asym[i], Ra))
            ok_s += int(abs(e - float(bs["sel"]["lj"][i, j])) <= 1e-6 * max(1.0, abs(e)))
            # the identity copy of the iv-P cell and the asym molecule of the iv-S cell coincide
            it = va["items"][i]
            A = int(it["atom_mask"][0].sum())
            X_ivp = bl["cells"][i][j]["cart"][:A]
            L, cent, orient, local, Z, am, mm = expand(asym[i], Ra)
            X_ivs = (cent[0].double() @ L.double()) + local.double()[0] @ orient[0].double().T
            dmax_s = max(dmax_s, float((X_ivp - X_ivs).abs().max()))
    rt.update({"ivs_kind": bs["kind"], "ivs_R_shape": list(bs["R"].shape), "ivs_lj_ok_of_4": ok_s,
               "ivs_identity_copy_max_dx_A": dmax_s})
    rep["bundle_round_trip"] = rt
    stop[0] = True
    rep["peak_rss_GB_since_padded_step"] = round(peak[0] / 2**30, 2)
    path = os.path.join(OLD_DIR, "smoke.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("->", path)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("items")
    p.add_argument("--sets", nargs="+", default=["train", "val", "devtest"])
    p = sub.add_parser("gate")
    p.add_argument("--sets", nargs="+", default=["val", "devtest"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-fit", action="store_true", help="skip the StructureMatcher rebuild (SEL / TEST-B)")
    p = sub.add_parser("smoke")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--padded-batch", type=int, default=2, help="batch of the one fully padded step (0 = skip)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pack", action="store_true", help="packed encoder for the training steps")
    p = sub.add_parser("train")
    p.add_argument("--lr", type=float, required=True, choices=LRS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-trim", action="store_true", help="run the forward on the full 16 x 64 padding")
    p.add_argument("--pack", action="store_true", help="EGNN on real molecules only (pack_encoder)")
    p.add_argument("--steps-per-epoch", type=int, default=0, help="smoke only: cap steps per epoch (0 = full)")
    p.add_argument("--runs-root", default="", help="smoke only: write the run outside results/n3/private/old/runs")
    p = sub.add_parser("select")
    p.add_argument("what", choices=["window", "ivs"])
    p.add_argument("--record", default="", help="ivs: the n3_score.py select record of ivP.json")
    p.add_argument("--runs-root", default="",
                   help="smoke only: read runs from here and write every select output under it")
    p.add_argument("--sample-args", default="",
                   help="window: appended to the printed sample commands (e.g. '--crystals-per-batch 8 --device cuda')")
    p = sub.add_parser("sample")
    p.add_argument("--run", required=True, help="run dir name under results/n3/private/old/runs, or a path")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--set", required=True, choices=["val", "sel", "valsel", "testB", "devtest"])
    p.add_argument("--arm", default="", help="bundle arm name (default ivP_<lr>_ep<epoch>; the frozen pick: ivP)")
    p.add_argument("--draws", type=int, default=S_DRAWS)
    p.add_argument("--limit", type=int, default=0, help="smoke only: first N crystals, written outside draws/")
    p.add_argument("--crystals-per-batch", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--workers", type=int, default=1, help="CPU processes for the LJ energies")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--pack", action="store_true", help="EGNN on real molecules only (pack_encoder)")
    p.add_argument("--frozen", default="", help="testB/devtest: ivP_frozen.json (default results/n3/private/old/)")
    p = sub.add_parser("ivs")
    p.add_argument("--bundle", required=True)
    p.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    # Pool payloads are numpy (to_np); this is the repo's guard should a tensor ever cross (Errno 24 on boxes)
    torch.multiprocessing.set_sharing_strategy("file_system")
    {"items": cmd_items, "gate": cmd_gate, "smoke": cmd_smoke, "train": cmd_train, "select": cmd_select,
     "sample": cmd_sample, "ivs": cmd_ivs}[args.cmd](args)


if __name__ == "__main__":
    main()
