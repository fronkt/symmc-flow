"""N3 / G3: RESID pass for an MCF sampling run (tasks/n3_protocol.md §2.3 RESID, §3.2 "RESID pass"; no MCF code
change). Runs right after every MCF-R and MCF-A sampling call, on the box, in the same environment.

  - load the same checkpoint with the run's resolved config (<run>/hydra/inference.yaml, the merged config
    inference.py saved), exactly as EvalRunner does (FlowModule.load_from_checkpoint(cfg=...), eval());
  - iterate MCF's unshuffled test_dataloader (MCDatamodule(...).test_dataloader()) over the same pickle;
  - for each of the S draws, set the saved final state (pred_rotmats -> rotmats_t, pred_trans -> trans_t,
    lattices -> lattice_t) and so3_t = r3_t = l_t = 0.975 (inside MCF's training range [0.01, 0.99]) and run
    FlowModel once under eval() / no_grad;
  - RESID = geodesic angle (rad) between that pass's pred_rotmats and the returned pred_rotmats, mean over the
    crystal's copies (all copies are real in MCF), computed in float64 as atan2(sin, cos) of Ra^T Rb (ML.geodesic:
    within ~4e-8 rad on float32 rotations at every angle; the arccos-of-trace form has a ~4e-4 rad floor near 0,
    where the lowest-RESID pick lives).
  - a draw whose returned state is not finite (an arm failure, §1.3) gets RESID = NaN: its copies enter the
    forward with a finite placeholder (identity rotations, the input lattice_1 / trans_1; the output is
    discarded), because FlowModel's openfold rotation update calls torch.linalg.eigh, which raises on NaN input
    for the whole batch.
Output <run>/out/resid_<S>.pt = {'resid': float64 [S, |set|], 'state_finite': bool [S, |set|],
'pred_rotmats_resid': float32 [S, sum copies, 3, 3] (this pass's rotations, so RESID can be recomputed after the
box is gone), 'branch_cut_edges': long [S, |set|] and 'edges': long [|set|] (edges of MCF's copy graph at the
returned state whose relative-rotation axis lies on gen_edges' azimuth branch cut: where the carried +G RESID is
not guaranteed exact), ...}.

    python scripts/n3_mcf_resid.py --run_dir n3_mcf/infer/mcfR/valsel/s0_last_u3 --device cuda
    python scripts/n3_mcf_resid.py --run_dir <smoke run> --device cpu --shims          # laptop smoke
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import torch

import n3_mcf_lib as ML

ANGLE = ML.GEODESIC


def resid_from_rotmats(R_pass, R_ret, num_bbs):
    """[n] mean-over-copies geodesic between two [sum copies, 3, 3] stacks (NaN where either is non-finite)."""
    from torch_geometric.utils import scatter
    ang = ML.geodesic(R_pass, R_ret)
    idx = torch.repeat_interleave(torch.arange(len(num_bbs), device=ang.device), num_bbs.to(ang.device).long())
    return scatter(ang, idx, dim=0, dim_size=len(num_bbs), reduce="mean")


def resid_pass(run_dir, device="cuda", t=ML.RESID_T):
    from omegaconf import OmegaConf
    from molcrystalflow.data.dataloader import MCDatamodule
    from molcrystalflow.data.dataset import MCDataset
    torch.set_float32_matmul_precision("high")          # as inference.py (module level) sets it
    man, rp = ML.load_manifest(run_dir)
    S = man["num_samples"]
    cfg = OmegaConf.load(rp["config"])
    assert os.path.normcase(os.path.abspath(cfg.inference.ckpt_path)) == os.path.normcase(os.path.abspath(man["ckpt"]))
    assert ML.sha256(man["ckpt"]) == man["ckpt_sha256"], "checkpoint changed since sampling"
    pred = ML.load_predictions(rp["pred"])
    fin = ML.draw_finite(pred)                                       # [S, n]
    t0 = time.time()
    mod = ML.load_flow_module(man["ckpt"], cfg, device)
    ds = MCDataset(cache_path=os.path.join(cfg.data.cache_dir, "test.pt"), dataset_cfg=cfg.data, is_training=False)
    dl = MCDatamodule(data_cfg=cfg.data, train_dataset=None, valid_dataset=None, test_dataset=ds).test_dataloader()
    n = len(ds)
    assert tuple(pred["num_bbs"].shape) == (S, n), (tuple(pred["num_bbs"].shape), S, n)
    Mtot = int(pred["num_bbs"][0].sum())
    resid = torch.full((S, n), float("nan"), dtype=torch.float64)
    cut = torch.zeros(S, n, dtype=torch.long)
    edges = torch.zeros(n, dtype=torch.long)
    R_pass_all = torch.full((S, Mtot, 3, 3), float("nan"), dtype=torch.float32)
    b0 = m0 = 0
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            B, M = int(batch.num_graphs), int(batch.num_bbs.sum())
            assert torch.equal(pred["num_bbs"][0][b0:b0 + B].to(batch.num_bbs.device), batch.num_bbs), "batch order"
            assert torch.equal(pred["gt_data_batch"]["lattice_1"][b0:b0 + B].to(device), batch.lattice_1), "batch order"
            tt = torch.full((B, 1), t, device=batch.lattice_1.device, dtype=batch.lattice_1.dtype)
            t_rep = tt.repeat_interleave(batch.num_bbs, dim=0)
            bb_c = torch.repeat_interleave(torch.arange(B), batch.num_bbs.cpu())
            for s in range(S):
                ok = fin[s, b0:b0 + B]                                  # [B] cpu
                R_ret = pred["pred_rotmats"][s][m0:m0 + M].clone()
                tr = pred["pred_trans"][s][m0:m0 + M].clone()
                lat = pred["lattices"][s][b0:b0 + B].clone()
                if not bool(ok.all()):                                 # finite placeholder, output discarded
                    bad_bb = ~ok[bb_c]
                    R_ret[bad_bb] = torch.eye(3, dtype=R_ret.dtype)
                    tr[bad_bb] = batch.trans_1.cpu().to(tr.dtype)[bad_bb]
                    lat[~ok] = batch.lattice_1.cpu().to(lat.dtype)[~ok]
                e, c = ML.branch_cut_counts(R_ret, batch.num_bbs.cpu())
                edges[b0:b0 + B] = e
                cut[s, b0:b0 + B] = torch.where(ok, c, torch.zeros_like(c))
                R_ret = R_ret.to(device)
                batch["rotmats_t"] = R_ret
                batch["trans_t"] = tr.to(device)
                batch["lattice_t"] = lat.to(device)
                batch["so3_t"] = t_rep
                batch["r3_t"] = t_rep
                batch["l_t"] = tt
                out = mod.model(batch)
                Rp = out["pred_rotmats"].detach()
                r = resid_from_rotmats(Rp, R_ret, batch.num_bbs).cpu()
                r[~ok] = float("nan")
                resid[s, b0:b0 + B] = r
                Rp = Rp.float().cpu()
                Rp[~ok[bb_c]] = float("nan")
                R_pass_all[s, m0:m0 + M] = Rp
            b0, m0 = b0 + B, m0 + M
    assert b0 == n and m0 == Mtot
    rec = {"resid": resid, "state_finite": fin, "pred_rotmats_resid": R_pass_all, "branch_cut_edges": cut,
           "edges": edges, "t": t, "units": "rad", "angle": ANGLE, "reduce": "mean over copies",
           "nonfinite_state": "RESID NaN (arm failure); finite placeholder in the forward, output discarded",
           "ckpt": man["ckpt"], "ckpt_sha256": man["ckpt_sha256"], "config": "hydra/inference.yaml",
           "predictions_sha256": ML.sha256(rp["pred"]), "device": str(device), "shims": ML._STATE["shims"],
           "sec": round(time.time() - t0, 1), "protocol": "tasks/n3_protocol.md §2.3 RESID (MCF)"}
    ML.atomic_torch(rec, rp["resid"])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shims", action="store_true", help="laptop smoke only")
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    ML.setup_mcf(args.shims)
    rec = resid_pass(args.run_dir, args.device)
    r, f = rec["resid"], rec["state_finite"]
    print(json.dumps({"resid": os.path.relpath(ML.run_paths(args.run_dir, r.shape[0])["resid"], REPO).replace("\\", "/"),
                      "shape": list(r.shape), "finite_draws": int(f.sum()), "nonfinite_draws": int((~f).sum()),
                      "resid_finite_on_finite_draws": bool(torch.isfinite(r[f]).all()),
                      "branch_cut_draws": int((rec["branch_cut_edges"] > 0).sum()),
                      "median_deg": round(float(torch.rad2deg(r[torch.isfinite(r)]).median()), 4) if torch.isfinite(r).any() else None,
                      "sec": rec["sec"]}))


if __name__ == "__main__":
    main()
