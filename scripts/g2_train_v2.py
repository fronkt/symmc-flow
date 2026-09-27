"""N2 (Omega rebuild): train the asym-orientation torque-field flow longer and with less noise.

Same model as G2 (`g2_learned_flow.TorqueField`, so the N1 selector transfers); changed optimisation:
  * B draws per crystal per step (several (R0, t) pairs through the batched forward) instead of one
  * CPU data-parallel over --procs ranks (gloo): each rank takes one crystal per step, gradients are
    averaged, so one step sees procs x B draws
  * cosine learning-rate decay to lr/100 over the run, AdamW, grad clip 1.0
  * FIXED validation draws (seeded (R0, t) per val crystal) so val loss only moves when the model does
  * per-epoch resume checkpoint; best checkpoint by val loss
Split identical to G2 / N1 (split seed 0: test = perm[:200], val = perm[200:300]); test is never read.

    python scripts/g2_train_v2.py --out results/n2_s0.json --seed 0 --procs 8 --epochs 30 --draws 8
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
import torch.distributed as dist

from symmc_flow import manifolds as M
from g2_learned_flow import TorqueField, precompute


def batched_target(R0, R1, t):
    """R0 (B,3,3) Haar, R1 (3,3) truth, t (B,) -> R_t (B,3,3), world-frame velocity (B,3)."""
    L = M.so3_log(R1.unsqueeze(0) @ R0.transpose(-1, -2))            # (B,3)
    Rt = M.so3_exp(t[:, None] * L) @ R0
    return Rt, M.so3_log(R1.unsqueeze(0) @ Rt.transpose(-1, -2)) / (1.0 - t)[:, None]


def run(rank, world, args):
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.port))
        dist.init_process_group("gloo", rank=rank, world_size=world)
    elig = torch.load(args.cache, weights_only=False)
    perm = torch.randperm(len(elig), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    val = [elig[i] for i in perm[200:300]]
    train = [elig[i] for i in perm[300:]]
    if args.train_limit:
        train = train[:args.train_limit]
    G_tr = [precompute(a, 8.0) for a in train]
    G_va = [precompute(a, 8.0) for a in val] if rank == 0 else []
    del elig

    torch.manual_seed(args.seed)                                    # identical init on every rank
    model = TorqueField()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(G_tr) / world)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    # fixed validation draws: seeded, identical for every epoch and every run
    gv = torch.Generator().manual_seed(12345)
    val_draws = [(M.random_so3((args.val_draws,)), torch.rand(args.val_draws, generator=gv) * 0.98)
                 for _ in G_va]

    hist, best, best_state, start = [], float("inf"), None, 0
    resume = args.out.replace(".json", ".resume.pt")
    if os.path.exists(resume):
        ck = torch.load(resume, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        hist, best, best_state, start = ck["hist"], ck["best"], ck["best_state"], ck["epoch"] + 1
        if rank == 0:
            print(f"resumed after epoch {ck['epoch']}", flush=True)

    for ep in range(start, args.epochs):
        t0 = time.time()
        model.train()
        g = torch.Generator().manual_seed(args.seed * 100_003 + ep)   # same shuffle on all ranks
        order = torch.randperm(len(G_tr), generator=g).tolist()
        order += order[:steps_per_epoch * world - len(order)]           # pad so every rank steps equally
        torch.manual_seed(args.seed * 7919 + ep * 131 + rank)           # rank-specific draws
        tot, n = 0.0, 0
        for s in range(steps_per_epoch):
            gr = G_tr[order[s * world + rank]]
            R0 = M.random_so3((args.draws,))
            t = torch.rand(args.draws) * 0.98
            Rt, u = batched_target(R0, gr["R1"], t)
            loss = ((model.forward_batch(gr, Rt, t) - u) ** 2).sum(-1).mean()
            opt.zero_grad()
            ok = torch.tensor([1.0 if torch.isfinite(loss) else 0.0])
            if torch.isfinite(loss):
                loss.backward()
            else:
                for p in model.parameters():
                    p.grad = torch.zeros_like(p)
            if world > 1:
                for p in model.parameters():
                    dist.all_reduce(p.grad)
                dist.all_reduce(ok)
                for p in model.parameters():
                    p.grad /= max(float(ok), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if torch.isfinite(loss):
                tot += float(loss); n += 1
        if rank == 0:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                for gr, (R0, t) in zip(G_va, val_draws):
                    Rt, u = batched_target(R0, gr["R1"], t)
                    vl += float(((model.forward_batch(gr, Rt, t) - u) ** 2).sum(-1).mean())
                vl /= len(G_va)
            hist.append({"epoch": ep, "train": round(tot / max(n, 1), 4), "val": round(vl, 4),
                         "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)})
            print(json.dumps(hist[-1]), flush=True)
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "hist": hist, "best": best, "best_state": best_state, "epoch": ep}, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
        if world > 1:
            dist.barrier()
    if rank == 0:
        torch.save({"best_state": best_state, "hist": hist, "best": best, "args": vars(args)},
                   args.out.replace(".json", ".pt"))
        json.dump({"args": vars(args), "hist": hist, "best_val": best}, open(args.out, "w"), indent=1)
    if world > 1:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--val-draws", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=29511)
    ap.add_argument("--cache", default="data/csd_mol/g2_asym.pt")
    args = ap.parse_args()
    if args.procs > 1:
        torch.multiprocessing.spawn(run, args=(args.procs, args), nprocs=args.procs, join=True)
    else:
        run(0, 1, args)


if __name__ == "__main__":
    main()
