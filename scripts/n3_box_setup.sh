#!/usr/bin/env bash
# N3 box environments (tasks/n3_protocol.md §7; runbook tasks/n3_runbook.md). Linux, ONE GPU, NVIDIA driver with
# CUDA >= 12.8 (the cu128 wheels below). Run from the unpacked root (n3_box_manifest.py unpack).
#
#   bash scripts/n3_box_setup.sh check        # OS, GPU count, driver CUDA version, disk, RAM, cores
#   bash scripts/n3_box_setup.sh all          # check + conda + the three envs + env file + smoke
#   bash scripts/n3_box_setup.sh mcf|ff|symmc # one env (re-running recreates it)
#   bash scripts/n3_box_setup.sh smoke        # version prints + CUDA import smoke per env
#   bash scripts/n3_box_setup.sh data         # after an upload: manifest verify, set caches, MCF pickle load, UMA
#   N3_DRY_RUN=1 bash scripts/n3_box_setup.sh all   # print every command, run nothing
#
# Environments (pins: scripts/n3_box_constraints_<env>.txt = the versions the N3 units ran locally; torch comes
# from the CUDA index and is pinned here):
#   mcf    python 3.12, torch 2.7.1+cu128, torch-scatter / torch-cluster wheels for torch 2.7.0+cu128, PyG,
#          pytorch-lightning, hydra-core, rdkit, pymatgen, spglib, openbabel (conda-forge), wandb, GPUtil, p_tqdm,
#          setuptools; the MolCrystalFlow fork external/mcf installed editable (no deps).
#          Runs n3_mcf_*.py (training, inference + RESID, Gate C, diagnostics, +G / ii-S post-processing).
#   ff     python 3.12, torch 2.13.0+cu128, fairchem-core 2.23.0 (UMA s-1p1 server of n3_ffstar.py; N3_FF_PYTHON).
#   symmc  python 3.12, torch 2.11.0+cu128 (N3_SYMMC_TORCH / N3_SYMMC_INDEX override), pymatgen, ase, rdkit,
#          spglib, gemmi, networkx, scipy: n3_ours, n3_old_symmc (iv-P), n3_classical, n3_score, set caches.
# Writes $ROOT/.n3_box/env.sh (source it before every command) and $ROOT/.n3_box/versions_<env>.json.
# Nothing here reads CSD data except the `data` mode, which only checks hashes and loads four VAL crystals.
set -euo pipefail

ROOT=${N3_BOX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STATE=$ROOT/.n3_box
DRY=${N3_DRY_RUN:-0}
CU_INDEX=https://download.pytorch.org/whl/cu128
PYG_LINKS=https://data.pyg.org/whl/torch-2.7.0+cu128.html
MCF_TORCH="torch==2.7.1 torchvision==0.22.1"
FF_TORCH="torch==2.13.0"
FF_CORE="fairchem-core==2.23.0"
SYMMC_TORCH=${N3_SYMMC_TORCH:-torch==2.11.0}
SYMMC_INDEX=${N3_SYMMC_INDEX:-$CU_INDEX}
MIN_DRIVER_CUDA=12.8
MIN_DISK_GB=${N3_MIN_DISK_GB:-60}

run() { echo "+ $*"; if [ "$DRY" != 1 ]; then "$@"; fi; }
die() { echo "FAIL: $*" >&2; exit 1; }
mkdir -p "$STATE"

find_conda() {
  for c in "${N3_CONDA:-}" "$(command -v conda 2>/dev/null || true)" /opt/conda/bin/conda \
           "$HOME/miniforge3/bin/conda" /opt/miniforge3/bin/conda; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

cmd_check() {
  if [ "$DRY" = 1 ]; then
    echo "+ check: Linux; nvidia-smi; GPU count (1 expected); driver CUDA >= $MIN_DRIVER_CUDA; >= ${MIN_DISK_GB} GB free; python3"
    return 0
  fi
  [ "$(uname -s)" = Linux ] || die "not Linux"
  command -v nvidia-smi >/dev/null || die "no nvidia-smi"
  local n cu disk mem
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  cu=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
  echo "GPUs: $n   driver CUDA: $cu   cores: $(nproc)   RAM: $(free -g | awk '/Mem:/{print $2}') GB"
  [ "$n" -ge 1 ] || die "no GPU"
  [ "$n" -eq 1 ] || echo "WARNING: $n GPUs; every job is pinned to CUDA_VISIBLE_DEVICES=0 (inference.py shards on >1)"
  awk -v a="$cu" -v b="$MIN_DRIVER_CUDA" 'BEGIN{split(a,x,".");split(b,y,".");exit !((x[1]>y[1])||(x[1]==y[1]&&x[2]>=y[2]))}' \
    || die "driver CUDA $cu < $MIN_DRIVER_CUDA: the cu128 wheels need a newer driver (rent with cuda_max_good>=12.8)"
  disk=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  echo "free disk at $ROOT: ${disk} GB"
  [ "$disk" -ge "$MIN_DISK_GB" ] || die "less than ${MIN_DISK_GB} GB free"
  command -v python3 >/dev/null || die "no python3 (needed by n3_box_manifest.py unpack/verify)"
  echo "check: OK"
}

ensure_conda() {
  if CONDA=$(find_conda); then echo "conda: $CONDA"; else
    local inst=/tmp/miniforge.sh
    run curl -fsSL -o "$inst" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    run bash "$inst" -b -p /opt/miniforge3
    CONDA=/opt/miniforge3/bin/conda
  fi
  CONDA_BASE=$(dirname "$(dirname "$CONDA")")
}

envpy() { echo "$CONDA_BASE/envs/$1/bin/python"; }

make_env() {  # name, python spec, extra conda packages...
  local name=$1; shift
  if [ -d "$CONDA_BASE/envs/$name" ]; then run "$CONDA" env remove -y -n "$name"; fi
  run "$CONDA" create -y -n "$name" -c conda-forge --override-channels "$@"
}

pipi() {  # env, pip args...
  local e=$1; shift
  run "$(envpy "$e")" -m pip install --no-input --progress-bar off "$@"
}

cmd_mcf() {
  make_env mcf python=3.12 openbabel=3.1.1
  pipi mcf --upgrade pip
  pipi mcf $MCF_TORCH --index-url "$CU_INDEX"
  pipi mcf --only-binary=:all: torch-scatter torch-cluster -f "$PYG_LINKS"
  pipi mcf -c "$ROOT/scripts/n3_box_constraints_mcf.txt" torch-geometric "pytorch-lightning>=2.5.0" hydra-core \
    omegaconf wandb rdkit ase pymatgen spglib numpy scipy pandas networkx h5py p_tqdm tqdm rich GPUtil matplotlib \
    plotly seaborn psutil pyyaml setuptools packaging
  pipi mcf --no-deps -e "$ROOT/external/mcf"
}

cmd_ff() {
  make_env ff python=3.12
  pipi ff --upgrade pip
  pipi ff $FF_TORCH --index-url "$CU_INDEX"
  pipi ff -c "$ROOT/scripts/n3_box_constraints_ff.txt" "$FF_CORE"
}

cmd_symmc() {
  make_env symmc python=3.12
  pipi symmc --upgrade pip
  pipi symmc "$SYMMC_TORCH" --index-url "$SYMMC_INDEX"
  pipi symmc -c "$ROOT/scripts/n3_box_constraints_symmc.txt" numpy scipy pymatgen spglib monty ase rdkit gemmi \
    networkx pandas psutil pyyaml pytest packaging
}

write_envfile() {
  local f=$STATE/env.sh
  echo "+ write $f"
  [ "$DRY" = 1 ] && return 0
  cat > "$f" <<EOF
# N3 box environment: source before every command (tasks/n3_runbook.md)
export N3_BOX_ROOT=$ROOT
export N3_ROOT=$ROOT
export N3_SET_CACHE_ONLY=1              # load_set reads per-set caches only; full dev / pool caches never loaded
export N3_MANIFEST_DIR=\$HOME/n3_manifests
export CUDA_VISIBLE_DEVICES=0           # one GPU for every job (inference.py's Trainer would shard on more)
export WANDB_MODE=offline WANDB_SILENT=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no Hugging Face download or token on the box (§3.3)
export PY_MCF=$(envpy mcf)
export PY_FF=$(envpy ff)
export PY_SYMMC=$(envpy symmc)
export N3_FF_PYTHON=\$PY_FF              # n3_classical spawns the UMA server with it
cd $ROOT
EOF
  echo "wrote $f"
}

versions() {  # env, modules...
  local e=$1; shift
  local py; py=$(envpy "$e")
  run "$py" - "$e" "$STATE/versions_$e.json" "$@" <<'PY'
import importlib, importlib.metadata as md, json, platform, sys
env, out, mods = sys.argv[1], sys.argv[2], sys.argv[3:]
v = {"env": env, "python": platform.python_version()}
for m in mods:
    importlib.import_module(m)
    for dist in (m.replace("_", "-"), m, {"hydra": "hydra-core", "fairchem.core": "fairchem-core", "GPUtil": "GPUtil",
                                          "rdkit": "rdkit", "torch_geometric": "torch-geometric"}.get(m, m)):
        try:
            v[m] = md.version(dist); break
        except md.PackageNotFoundError:
            continue
    else:
        v[m] = getattr(sys.modules[m], "__version__", "?")
import torch
v["torch_cuda"] = torch.version.cuda
v["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    v["gpu"] = torch.cuda.get_device_name(0)
    x = torch.randn(512, 512, device="cuda")
    v["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all())
json.dump(v, open(out, "w"), indent=1)
print(json.dumps(v))
assert v["cuda_available"], f"{env}: torch sees no GPU"
PY
}

cmd_smoke() {
  ensure_conda
  versions mcf torch torch_geometric torch_scatter torch_cluster pytorch_lightning hydra omegaconf wandb rdkit ase \
    pymatgen spglib numpy scipy GPUtil p_tqdm
  run "$(envpy mcf)" - "$ROOT" <<'PY'
import os, sys, torch
root = sys.argv[1]
import molcrystalflow, torch_cluster, torch_scatter
assert os.path.abspath(molcrystalflow.__file__).startswith(os.path.join(root, "external", "mcf")), molcrystalflow.__file__
x = torch.randn(64, 3, device="cuda")
e = torch_cluster.radius_graph(x, r=1.5)
s = torch_scatter.scatter_sum(torch.ones(e.shape[1], device="cuda"), e[0], dim=0, dim_size=64)
import GPUtil, p_tqdm  # noqa: F401  (GPUtil needs setuptools' distutils shim on 3.12)
print("mcf smoke: fork", os.path.dirname(molcrystalflow.__file__), "| radius_graph edges", e.shape[1], "| scatter ok",
      bool(torch.isfinite(s).all()))
PY
  run "$CONDA_BASE/envs/mcf/bin/obabel" -V
  versions ff torch fairchem.core ase numpy omegaconf e3nn
  run "$(envpy ff)" -c "from fairchem.core.units.mlip_unit import load_predict_unit; print('ff smoke: load_predict_unit importable')"
  versions symmc torch numpy scipy pymatgen spglib ase rdkit gemmi networkx pandas psutil
  run "$(envpy symmc)" - "$ROOT" <<'PY'
import os, sys
root = sys.argv[1]
sys.path[:0] = [root, os.path.join(root, "scripts")]
import symmc_flow, g2_asym_baselines, g2_learned_flow, n3_common  # noqa: F401
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: F401
print("symmc smoke: repo modules import; n3_common set-cache loader:", hasattr(n3_common, "set_cache_only"))
PY
  echo "smoke: OK (versions in $STATE/versions_*.json)"
}

cmd_data() {  # after an upload + unpack; reads hashes, the set caches' headers and four VAL crystals only
  # shellcheck disable=SC1091
  [ "$DRY" = 1 ] || source "$STATE/env.sh"
  local m
  for m in "$ROOT"/.n3_box/manifests/*.manifest.json; do
    [ -e "$m" ] || continue
    run python3 "$ROOT/scripts/n3_box_manifest.py" verify --root "$ROOT" --label "$(basename "$m" .manifest.json)"
  done
  run "$(envpy symmc)" "$ROOT/scripts/n3_make_set_caches.py" check
  run "$(envpy mcf)" "$ROOT/scripts/n3_mcf_smoke.py" --set val --n 4
  run "$(envpy ff)" "$ROOT/scripts/n3_ffstar.py" info
  echo "data: OK"
}

mode=${1:-all}
case "$mode" in
  check) cmd_check ;;
  mcf|ff|symmc) ensure_conda; "cmd_$mode"; write_envfile ;;
  smoke) cmd_smoke ;;
  data) ensure_conda; cmd_data ;;
  all) cmd_check; ensure_conda; cmd_mcf; cmd_ff; cmd_symmc; write_envfile; cmd_smoke ;;
  *) die "unknown mode $mode (check|all|mcf|ff|symmc|smoke|data)" ;;
esac
