"""G2 (Omega rebuild), part 2: is the asymmetric-unit orientation R_asym LEARNABLE?

Model: an SO(3) flow on R_asym alone. Every other copy is generated exactly by the crystal's own
space-group ops (det +-1, so mirror copies come for free). At flow time t the asym molecule feels a
LEARNED pairwise interaction with all symmetry copies and periodic images around it; the net torque
is the predicted world-frame angular velocity:

    f_i  = sum_j s_theta(d_ij, Z_i, Z_j, t) * (x_i - x_j) / d_ij        (pair "force" on asym atom i)
    w    = sum_i (x_i - c) x f_i                                         (torque -> angular velocity)

Distances/vectors only, so the field is exactly rotation-equivariant. Training = conditional flow
matching on SO(3): R_t = exp(t log(R1 R0^T)) R0 with R0 ~ Haar, target w* = log(R1 R_t^T)/(1-t).
Lattice and asym centroid are the given template (orientation-isolated read, the same setting as
`g2_asym_baselines.py`).

Evaluation on the SAME held-out crystals as the baselines (seeded permutation; test = first --n-test):
for each crystal draw S orientations from the learned flow and S from Haar, then report
  match@1 / match@S    unrelaxed StructureMatcher vs truth
  basin_best           relax all S draws (orientation-only packing press), keep the lowest energy,
                       does it land in the relaxed-truth basin?
Kill criterion (tasks/omega_rebuild_plan.md G2): learned must beat Haar by more than the seed spread.

    python scripts/g2_learned_flow.py --data data/csd_mol/ds_o3.pt --out results/g2_learned_s0.json --seed 0
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from symmc_flow import manifolds as M
from g2_asym_baselines import asym_item, cart_ops, expand, press, energy, to_structure


# ------------------------------------------------------------------ geometry (static per crystal)
def precompute(a, cutoff):
    """Neighbour molecule images of the asym molecule: (centroid_cart (N,3), op index (N,)).
    Only ORIENTATIONS change during the flow, so which images can come within `cutoff` is fixed by
    centroid distance < cutoff + 2 * molecular radius."""
    L = a["L"]
    Rc = cart_ops(a)
    cent = torch.einsum("kij,j->ki", a["W"], a["c0"]) + a["t"]
    cent = cent - torch.floor(cent)
    rmol = float(a["local"].norm(dim=-1).max())
    rng = torch.arange(-2, 3, dtype=torch.float64)
    shifts = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    c0 = a["c0"] @ L
    cc, oo = [], []
    for k in range(a["K"]):
        imgs = (cent[k] + shifts) @ L                        # (125,3)
        d = (imgs - c0).norm(dim=-1)
        keep = d < cutoff + 2 * rmol
        if k == 0:
            keep &= d > 1e-6                                 # not the asym molecule itself
        cc.append(imgs[keep]); oo += [k] * int(keep.sum())
    return {"c0": c0.float(), "img_c": torch.cat(cc).float(), "img_op": torch.tensor(oo),
            "Rc": Rc.float(), "local": a["local"].float(), "Z": a["Z"].long(),
            "R1": a["R0"].float()}


def rbf(d, n=16, cutoff=8.0):
    mu = torch.linspace(0.8, cutoff, n)
    return torch.exp(-((d.unsqueeze(-1) - mu) ** 2) / (2 * (cutoff / n) ** 2))


class TorqueField(nn.Module):
    def __init__(self, h=64, cutoff=8.0):
        super().__init__()
        self.cutoff = cutoff
        self.emb = nn.Embedding(100, 16)
        self.mlp = nn.Sequential(nn.Linear(16 + 32 + 8, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 1))

    def forward(self, g, R, t):
        loc = g["local"]                                     # (A,3)
        Xa = g["c0"] + loc @ R.T                             # (A,3) asym atoms
        Rn = g["Rc"][g["img_op"]] @ R                        # (N,3,3) image orientations
        Xn = g["img_c"][:, None] + torch.einsum("nij,aj->nai", Rn, loc)   # (N,A,3)
        Xn = Xn.reshape(-1, 3)
        Zn = g["Z"].repeat(Rn.shape[0])
        v = Xa[:, None] - Xn[None]                           # (A,NA,3)
        d = v.norm(dim=-1).clamp_min(1e-3)
        w = 0.5 * (torch.cos(math.pi * (d / self.cutoff).clamp(max=1.0)) + 1.0)   # smooth cutoff
        e = self.emb(g["Z"])
        tt = torch.cat([torch.sin(t * torch.arange(1, 5) * math.pi),
                        torch.cos(t * torch.arange(1, 5) * math.pi)])
        feat = torch.cat([rbf(d, cutoff=self.cutoff),
                          e[:, None].expand(-1, d.shape[1], -1),
                          self.emb(Zn)[None].expand(d.shape[0], -1, -1),
                          tt.expand(*d.shape, 8)], -1)
        s = self.mlp(feat).squeeze(-1) * w                   # (A,NA)
        f = (s.unsqueeze(-1) * v / d.unsqueeze(-1)).sum(1)   # (A,3)
        return torch.linalg.cross(Xa - g["c0"], f).sum(0)    # (3,) torque


def target(R0, R1, t):
    Rt = M.so3_exp(t * M.so3_log(R1 @ R0.T)) @ R0
    return Rt, M.so3_log(R1 @ Rt.T) / (1.0 - t)             # world-frame angular velocity


@torch.no_grad()
def sample(model, g, n, steps=40):
    out = []
    for _ in range(n):
        R = M.random_so3(())
        for s in range(steps):
            t = torch.tensor(s / steps)
            R = M.so3_exp(model(g, R, t) / steps) @ R
        out.append(R)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0, help="same as g2_asym_baselines --seed")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--press-steps", type=int, default=80)
    ap.add_argument("--eval-n", type=int, default=0, help="evaluate only the first N test crystals")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--cache", default="data/csd_mol/g2_asym.pt")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    torch.manual_seed(args.seed)

    if os.path.exists(args.cache):
        elig = torch.load(args.cache, weights_only=False)
    else:
        items = torch.load(args.data, weights_only=False)["items"]
        elig = [a for a in (asym_item(it) for it in items) if a is not None]
        torch.save(elig, args.cache)
    g = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(len(elig), generator=g).tolist()   # identical to the baselines' sample
    test = [elig[i] for i in perm[:args.n_test]]
    val = [elig[i] for i in perm[args.n_test:args.n_test + args.n_val]]
    train = [elig[i] for i in perm[args.n_test + args.n_val:]]
    if args.train_limit:
        train = train[:args.train_limit]
    print(f"eligible {len(elig)}  train {len(train)}  val {len(val)}  test {len(test)}", flush=True)

    G_tr = [precompute(a, args.cutoff) for a in train]
    G_va = [precompute(a, args.cutoff) for a in val]
    model = TorqueField(cutoff=args.cutoff)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def loss_on(gr):
        R0 = M.random_so3(())
        t = torch.rand(()) * 0.98
        Rt, u = target(R0, gr["R1"], t)
        return ((model(gr, Rt, t) - u) ** 2).sum()

    hist = []
    best, best_state, start = float("inf"), None, 0
    resume = args.out.replace(".json", ".resume.pt")          # per-epoch checkpoint: a killed run
    if os.path.exists(resume):                                 # restarts from its last epoch
        ck = torch.load(resume, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        hist, best, best_state, start = ck["hist"], ck["best"], ck["best_state"], ck["epoch"] + 1
        print(f"resumed after epoch {ck['epoch']}", flush=True)
    for ep in range(start, args.epochs):
        t0 = time.time()
        model.train()
        tot = 0.0
        for i in torch.randperm(len(G_tr)).tolist():
            opt.zero_grad()
            l = loss_on(G_tr[i])
            if torch.isfinite(l):
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(l)
        model.eval()
        torch.manual_seed(1000 + ep)
        with torch.no_grad():
            vl = sum(float(loss_on(gr)) for gr in G_va for _ in range(4)) / (4 * len(G_va))
        torch.manual_seed(args.seed * 7919 + ep)
        hist.append({"epoch": ep, "train": tot / len(G_tr), "val": vl, "sec": round(time.time() - t0)})
        print(json.dumps(hist[-1]), flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "best": best,
                    "best_state": best_state, "epoch": ep}, resume + ".tmp")
        os.replace(resume + ".tmp", resume)
    model.load_state_dict(best_state)
    ckpt = args.out.replace(".json", ".pt")
    torch.save(best_state, ckpt)

    # ---------------------------------------------------------------- evaluation vs Haar, same S
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher()
    ev = test[:args.eval_n] if args.eval_n else test
    rows = []
    for n_, a in enumerate(ev):
        gr = precompute(a, args.cutoff)
        o = a["orig"]
        truth = to_structure(o["lattice"], o["centroid"], o["orient"], o["local"], o["Z"],
                             o["atom_mask"], o["mol_mask"])
        basin_ref = to_structure(*press(a, a["R0"], args.press_steps))
        row = {"refcode": a["refcode"], "sg": a["sg"]}
        for name, draws in (("learned", sample(model, gr, args.samples)),
                            ("haar", [M.random_so3(()) for _ in range(args.samples)])):
            draws = [R.double() for R in draws]
            ok = [bool(sm.fit(to_structure(*expand(a, R)), truth)) for R in draws]
            geo = [math.degrees(float(M.so3_angle(R.float(), a["R0"].float()))) for R in draws]
            rel = [press(a, R, args.press_steps) for R in draws]
            es = [energy(*c) for c in rel]
            bi = min(range(len(rel)), key=lambda i: es[i])
            row[name] = {"match1": ok[0], "matchS": any(ok), "geo_med": sorted(geo)[len(geo) // 2],
                         "basin_best": bool(sm.fit(to_structure(*rel[bi]), basin_ref))}
        rows.append(row)
        if (n_ + 1) % 10 == 0:
            print(f"eval {n_+1}/{len(ev)}", flush=True)
    summ = {}
    for name in ("learned", "haar"):
        for k in ("match1", "matchS", "basin_best"):
            summ[f"{name}_{k}"] = sum(r[name][k] for r in rows) / max(len(rows), 1)
        summ[f"{name}_geo_med"] = sorted(r[name]["geo_med"] for r in rows)[len(rows) // 2]
    summ["n_eval"] = len(rows)
    json.dump({"args": vars(args), "summary": summ, "history": hist, "rows": rows},
              open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
