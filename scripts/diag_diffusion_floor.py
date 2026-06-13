"""Diagnostic: is the diffusion fractional loss (~2.5) at the predict-zero DSM floor?
Compares E||score_target||^2 (loss of a network that outputs 0) against the trained
frac loss. If they're equal, the score net learned ~nothing (objective/setup problem);
if predict-zero is much larger, the net explains real variance (sampler is the issue)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from symmc_flow.carbon24 import Carbon24Dataset
from symmc_flow.data import collate, batch_to_state
from symmc_flow.diffusion import DiffusionProcess, diffusion_targets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
val = Carbon24Dataset(os.path.join(ROOT, "data/raw/carbon_val.csv"), 24,
                      cache_path=os.path.join(ROOT, "data/cache/carbon_val.pt"))
proc = DiffusionProcess(T=1000)
torch.manual_seed(0)
tot, natoms = 0.0, 0
for _ in range(40):
    idx = torch.randint(0, len(val), (128,))
    z1 = batch_to_state(collate([val[int(i)] for i in idx]))
    t_idx = torch.randint(1, proc.T + 1, (128,))
    _, _, _, _, score_tgt, _ = diffusion_targets(proc, z1, t_idx)
    m = z1.mask.unsqueeze(-1).float()
    tot += float(((score_tgt ** 2) * m).sum())
    natoms += float(m.sum())
print(f"predict-zero frac MSE (E||score_tgt||^2 per atom): {tot/natoms:.4f}")
print("trained frac loss was ~2.5 -> if these match, the score net learned nothing")
