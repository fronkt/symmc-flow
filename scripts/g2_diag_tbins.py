"""Val loss of the saved best G2 checkpoints vs predict-zero, binned by flow time t (fixed draws)."""
import sys, os, json, math
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch
from symmc_flow import manifolds as M
from g2_learned_flow import precompute, TorqueField, target
elig = torch.load("data/csd_mol/g2_asym.pt", weights_only=False)
perm = torch.randperm(len(elig), generator=torch.Generator().manual_seed(0)).tolist()
val = [precompute(elig[i], 8.0) for i in perm[200:300]]
bins = [0.0, 0.25, 0.5, 0.75, 0.9, 0.98]
torch.manual_seed(123)
draws = [(gi, M.random_so3(()), torch.tensor(t)) for gi in range(len(val)) for t in
         [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]]
out = {}
for s in (0, 1, 2):
    ck = torch.load(f"results/g2_learned_s{s}.resume.pt", weights_only=False)
    m = TorqueField(); m.load_state_dict(ck["best_state"]); m.eval()
    agg = {}
    with torch.no_grad():
        for gi, R0, t in draws:
            Rt, u = target(R0, val[gi]["R1"], t)
            lm = float(((m(val[gi], Rt, t) - u) ** 2).sum()); l0 = float((u ** 2).sum())
            k = f"{float(t):.2f}"
            a = agg.setdefault(k, [0.0, 0.0, 0]); a[0] += lm; a[1] += l0; a[2] += 1
    out[s] = {k: round(v[0] / v[1], 3) for k, v in agg.items()}
    out[s]["epoch"] = ck["epoch"]
print(json.dumps(out, indent=1))
print("ratio = model loss / predict-zero loss (1.0 = nothing learned)")
