#!/usr/bin/env bash
# N3 box environments (tasks/n3_protocol.md §7; runbook tasks/n3_runbook.md). Linux, ONE GPU, NVIDIA driver with
# CUDA >= 13.0 (FF cu130; MCF/SymMC cu128). Run from the unpacked root (n3_box_manifest.py unpack).
#
#   python scripts/n3_box_manifest.py wheels # laptop BEFORE rental; check pinned CUDA wheel availability
#   bash scripts/n3_box_setup.sh wheels       # same read-only network preflight, no GPU needed
#   bash scripts/n3_box_setup.sh check        # OS, GPU count, driver CUDA version, disk, RAM, cores
#   bash scripts/n3_box_setup.sh all          # check + wheels + conda + the three envs + env file + smoke
#   bash scripts/n3_box_setup.sh mcf|ff|symmc # one env (refuses replacement of an existing env)
#   bash scripts/n3_box_setup.sh smoke        # version prints + CUDA import smoke per env
#   bash scripts/n3_box_setup.sh data         # after an upload: manifest verify, set caches, MCF pickle load, UMA
#   N3_DRY_RUN=1 bash scripts/n3_box_setup.sh all   # print commands; create no directories
#
# Environments (pins: scripts/n3_box_constraints_<env>.txt = the versions the N3 units ran locally; torch comes
# from the CUDA index and is pinned here):
#   mcf    python 3.12, torch 2.7.1+cu128, torch-scatter / torch-cluster wheels for torch 2.7.0+cu128, PyG,
#          torch-scatter 2.1.2+pt27cu128 and torch-cluster 1.6.3+pt27cu128 (protocol A2a binary pins),
#          pytorch-lightning, hydra-core, rdkit, pymatgen, spglib, openbabel (conda-forge), wandb, GPUtil, p_tqdm,
#          setuptools; the MolCrystalFlow fork external/mcf installed editable (no deps).
#          Runs n3_mcf_*.py (training, inference + RESID, Gate C, diagnostics, +G / ii-S post-processing).
#   ff     python 3.12, torch 2.13.0+cu130, fairchem-core 2.23.0 (UMA s-1p1 server of n3_ffstar.py; N3_FF_PYTHON).
#          fairchem-core 2.23.0 requires torch~=2.13.0; the official cu128 index has no such Linux wheel.
#   symmc  python 3.12, torch 2.11.0+cu128 (N3_SYMMC_TORCH / N3_SYMMC_INDEX override), pymatgen, ase, rdkit,
#          spglib, gemmi, networkx, scipy: n3_ours, n3_old_symmc (iv-P), n3_classical, n3_score, set caches.
# Writes $ROOT/.n3_box/env.sh (source it before every command) and $ROOT/.n3_box/versions_<env>.json.
# Nothing here reads CSD data except the `data` mode, which only checks hashes and loads four VAL crystals.
set -euo pipefail

ROOT=${N3_BOX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STATE=$ROOT/.n3_box
DRY=${N3_DRY_RUN:-0}
CU_INDEX=https://download.pytorch.org/whl/cu128
FF_INDEX=https://download.pytorch.org/whl/cu130
PYG_LINKS=https://data.pyg.org/whl/torch-2.7.0+cu128.html
MCF_TORCH="torch==2.7.1 torchvision==0.22.1"
FF_TORCH="torch==2.13.0"
FF_CORE="fairchem-core==2.23.0"
SYMMC_TORCH=${N3_SYMMC_TORCH:-torch==2.11.0}
SYMMC_INDEX=${N3_SYMMC_INDEX:-$CU_INDEX}
MIN_DRIVER_CUDA=13.0
MIN_DISK_GB=${N3_MIN_DISK_GB:-60}

run() { echo "+ $*"; if [ "$DRY" != 1 ]; then "$@"; fi; }
die() { echo "FAIL: $*" >&2; exit 1; }
mode=${1:-all}
[ "$DRY" = 1 ] || [ "$mode" = wheels ] || mkdir -p "$STATE"

cmd_wheels() {
  [ "$SYMMC_TORCH" = "torch==2.11.0" ] && [ "$SYMMC_INDEX" = "$CU_INDEX" ] \
    || die "wheel preflight requires frozen SymMC torch 2.11.0+cu128; review overrides before provisioning"
  run python3 "$ROOT/scripts/n3_box_manifest.py" wheels
}

find_conda() {
  for c in "${N3_CONDA:-}" "$(command -v conda 2>/dev/null || true)" /opt/conda/bin/conda \
           "$HOME/miniforge3/bin/conda" /opt/miniforge3/bin/conda; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

driver_cuda() {  # nvidia-smi banner, else libcuda's cuDriverGetVersion (driver 615 nvidia-smi on Anvil prints no banner)
  local cu
  cu=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
  [ -n "$cu" ] || cu=$(python3 -c 'import ctypes; v = ctypes.c_int()
assert ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(v)) == 0
print("%d.%d" % (v.value // 1000, v.value % 1000 // 10))' 2>/dev/null)
  echo "$cu"
}

cmd_check() {
  if [ "$DRY" = 1 ]; then
    echo "+ check: Linux; nvidia-smi; GPU count (1 expected); driver CUDA >= $MIN_DRIVER_CUDA; >= ${MIN_DISK_GB} GB free; python3"
    return 0
  fi
  [ "$(uname -s)" = Linux ] || die "not Linux"
  local n cu disk mem
  if [ "${N3_NO_GPU_CHECK:-0}" = 1 ]; then  # HPC login node: build the envs here; smoke/data run in the GPU job
    echo "N3_NO_GPU_CHECK=1: GPU / driver checks deferred to smoke (run it on the GPU node)"
  else
  command -v nvidia-smi >/dev/null || die "no nvidia-smi"
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  cu=$(driver_cuda)
  echo "GPUs: $n   driver CUDA: $cu   cores: $(nproc)   RAM: $(free -g | awk '/Mem:/{print $2}') GB"
  [ "$n" -ge 1 ] || die "no GPU"
  [ -n "$cu" ] || die "driver CUDA version unreadable: no nvidia-smi banner and libcuda.so.1 cuDriverGetVersion failed"
  [ "$n" -eq 1 ] || echo "WARNING: $n GPUs; every job is pinned to CUDA_VISIBLE_DEVICES=0 (inference.py shards on >1)"
  awk -v a="$cu" -v b="$MIN_DRIVER_CUDA" 'BEGIN{split(a,x,".");split(b,y,".");exit !((x[1]>y[1])||(x[1]==y[1]&&x[2]>=y[2]))}' \
    || die "driver CUDA $cu < $MIN_DRIVER_CUDA: FF cu130 needs a newer driver (rent with cuda_max_good>=13.0)"
  fi
  disk=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  echo "free disk at $ROOT: ${disk} GB"
  [ "$disk" -ge "$MIN_DISK_GB" ] || die "less than ${MIN_DISK_GB} GB free"
  command -v python3 >/dev/null || die "no python3 (needed by n3_box_manifest.py unpack/verify)"
  echo "C compiler: $(command -v gcc || command -v cc || echo 'none (setup installs gcc)')"
  echo "check: OK"
}

ensure_cc() {  # Triton in the FF env builds its CUDA launcher with a C compiler, even for the uncompiled 'batch' path
  if [ "$DRY" = 1 ]; then echo "+ ensure a C compiler (apt-get install gcc if none)"; return 0; fi
  if command -v gcc >/dev/null || command -v cc >/dev/null; then echo "C compiler: $(command -v gcc || command -v cc)"; return 0; fi
  command -v apt-get >/dev/null || die "no C compiler and no apt-get: FF (Triton) needs gcc"
  run apt-get update -qq
  DEBIAN_FRONTEND=noninteractive run apt-get install -y -qq gcc
  [ "$DRY" = 1 ] || command -v gcc >/dev/null || die "gcc install failed"
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
  # Keep existing environments; replacement must be an explicit operator action.
  [ ! -d "$CONDA_BASE/envs/$name" ] || die "environment $name already exists; inspect it before replacement"
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
  pipi mcf --only-binary=:all: --no-index --no-deps "torch-scatter==2.1.2+pt27cu128" \
    "torch-cluster==1.6.3+pt27cu128" -f "$PYG_LINKS"
  pipi mcf -c "$ROOT/scripts/n3_box_constraints_mcf.txt" torch-geometric "pytorch-lightning>=2.5.0" hydra-core \
    omegaconf wandb rdkit ase pymatgen spglib numpy scipy pandas networkx h5py p_tqdm tqdm rich GPUtil matplotlib \
    plotly seaborn psutil pyyaml setuptools packaging
  pipi mcf --no-deps -e "$ROOT/external/mcf"
}

cmd_ff() {
  make_env ff python=3.12
  pipi ff --upgrade pip
  pipi ff $FF_TORCH --index-url "$FF_INDEX"
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
  mkdir -p "$STATE/tmp" "$STATE/cache/huggingface" "$STATE/cache/wandb" "$STATE/config/wandb"
  local qr qm qf qs qc
  printf -v qr '%q' "$ROOT"
  printf -v qm '%q' "$(envpy mcf)"
  printf -v qf '%q' "$(envpy ff)"
  printf -v qs '%q' "$(envpy symmc)"
  printf -v qc '%q' "${N3_SET_CACHES:-$ROOT/data/n3_sets}"
  cat > "$f" <<EOF
# N3 box environment: source before every command (tasks/n3_runbook.md)
export N3_BOX_ROOT=$qr
export N3_ROOT=$qr
export N3_SET_CACHE_ONLY=1              # load_set reads per-set caches only; full dev / pool caches never loaded
export N3_MANIFEST_DIR=\$HOME/n3_manifests
export N3_SET_CACHES=$qc
export TMPDIR=\$N3_BOX_ROOT/.n3_box/tmp
export HF_HOME=\$N3_BOX_ROOT/.n3_box/cache/huggingface
export TORCH_HOME=\$N3_BOX_ROOT/.n3_box/cache/torch
export TORCH_EXTENSIONS_DIR=\$N3_BOX_ROOT/.n3_box/cache/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=\$N3_BOX_ROOT/.n3_box/cache/torchinductor
export WANDB_CACHE_DIR=\$N3_BOX_ROOT/.n3_box/cache/wandb
export WANDB_CONFIG_DIR=\$N3_BOX_ROOT/.n3_box/config/wandb
export WANDB_DIR=\$N3_BOX_ROOT/.n3_box/wandb
export CUDA_VISIBLE_DEVICES=0           # one GPU for every job (inference.py's Trainer would shard on more)
export WANDB_MODE=offline WANDB_SILENT=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no Hugging Face download or token on the box (§3.3)
export PY_MCF=$qm
export PY_FF=$qf
export PY_SYMMC=$qs
export N3_FF_PYTHON=\$PY_FF              # n3_classical spawns the UMA server with it
cd $qr
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
assert v.get("cuda_matmul_ok"), f"{env}: CUDA matmul is non-finite"
want = {"mcf": "2.7.1", "ff": "2.13.0", "symmc": "2.11.0"}[env]
assert v["torch"].split("+")[0] == want, f"{env}: torch version changed from the tested build"
want_cuda = "13.0" if env == "ff" else "12.8"
assert v["torch_cuda"] == want_cuda, f"{env}: CUDA wheel differs from protocol A2a"
if env == "mcf":
    assert v["torch_scatter"] == "2.1.2+pt27cu128", "MCF torch-scatter differs from protocol A2a"
    assert v["torch_cluster"] == "1.6.3+pt27cu128", "MCF torch-cluster differs from protocol A2a"
if env == "ff":
    assert v["fairchem.core"] == "2.23.0", "FF* fairchem-core differs from protocol A2"
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
assert hasattr(n3_common, "set_cache_only") and hasattr(n3_common, "set_cache_dir"), "per-set loader is missing"
PY
  echo "smoke: OK (versions in $STATE/versions_*.json)"
}

cmd_data() {  # after an upload + unpack; reads hashes, the set caches' headers and four VAL crystals only
  # shellcheck disable=SC1091
  [ "$DRY" = 1 ] || source "$STATE/env.sh"
  local m step
  local found=0
  for m in "$ROOT"/.n3_box/manifests/*.manifest.json; do
    [ -e "$m" ] || continue
    found=1
    run python3 "$ROOT/scripts/n3_box_manifest.py" verify --root "$ROOT" --label "$(basename "$m" .manifest.json)"
  done
  [ "$found" = 1 ] || die "data verification requires an upload manifest"
  if [ "$DRY" = 1 ]; then
    echo "+ validate exact caches permitted by the latest verified upload manifest; check every permitted set"
  else
    step=$(python3 - "$ROOT/.n3_box/manifests" <<'PY'
import json, pathlib, sys
print(max(json.load(open(p))["step"] for p in pathlib.Path(sys.argv[1]).glob("step*.manifest.json")))
PY
)
    local sets=()
    case "$step" in
      4) sets=(train val) ;;
      5) sets=(train val sel valsel) ;;
      7) sets=(train val sel valsel testB devtest) ;;
      *) die "unexpected uploaded protocol step $step" ;;
    esac
    run "$(envpy symmc)" - "$N3_SET_CACHES" "${sets[@]}" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
allowed = set(sys.argv[2:])
assert p.is_dir() and {f.stem for f in p.glob("*.pt")} == allowed, "cache directory differs from the permitted exact set inventory"
assert all((p / (s + ".json")).is_file() for s in allowed), "cache integrity sidecar missing"
PY
    run "$(envpy symmc)" "$ROOT/scripts/n3_make_set_caches.py" check --sets "${sets[@]}"
  fi
  run "$(envpy mcf)" "$ROOT/scripts/n3_mcf_smoke.py" --set val --n 4
  run "$(envpy ff)" "$ROOT/scripts/n3_ffstar.py" info
  # toy cell (not CSD data) through the CUDA server: catches a missing Triton compiler before any FF job
  run "$(envpy symmc)" "$ROOT/scripts/n3_ffstar.py" selftest --device cuda --settings batch --ff-python "$(envpy ff)"
  echo "data: OK"
}

case "$mode" in
  check) cmd_check ;;
  wheels) cmd_wheels ;;
  mcf|symmc) cmd_check; cmd_wheels; ensure_conda; "cmd_$mode"; write_envfile ;;
  ff) cmd_check; cmd_wheels; ensure_cc; ensure_conda; cmd_ff; write_envfile ;;
  smoke) cmd_smoke ;;
  data) ensure_conda; cmd_data ;;
  all) cmd_check; cmd_wheels; ensure_cc; ensure_conda; cmd_mcf; cmd_ff; cmd_symmc; write_envfile; cmd_smoke ;;
  *) die "unknown mode $mode (check|wheels|all|mcf|ff|symmc|smoke|data)" ;;
esac
