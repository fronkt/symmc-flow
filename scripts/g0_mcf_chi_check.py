"""Numerical check of MolCrystalFlow's two get_equivariant_axes implementations:
does chi track handedness, and are local coords consistent across proper/improper copies?"""
import sys, importlib.util
import numpy as np
from scipy.spatial.transform import Rotation

base = r"C:\Users\frank\AppData\Local\Temp\claude\C--Users-frank\40207194-28ba-4c06-bcbf-fb5cb9ac6dff\scratchpad\mcf"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    src = open(path, encoding="utf8").read()
    # extract only the needed functions to avoid heavy imports
    ns = {"np": np, "Tuple": tuple, "Dict": dict}
    import re
    for fn in ["get_pca_axes", "get_equiv_vec", "get_equivariant_axes"]:
        m = re.search(r"^def %s\(.*?(?=^def |\Z)" % fn, src, flags=re.S | re.M)
        code = m.group(0)
        code = re.sub(r"\) -> [^:]+:", "):", code, count=1)
        code = re.sub(r":\s*np\.ndarray", "", code)
        exec(code, ns)
    return ns["get_equivariant_axes"]

impl = {
    "common.py (TRAINING data)": load("c", base + r"\data-preprocess\thurlemann23\common.py"),
    "dataset.py (CSP inference)": load("d", base + r"\molcrystalflow\data\dataset.py"),
}

rng = np.random.default_rng(0)
N = 2000
for name, f in impl.items():
    same_chi_rot = 0; flip_chi_inv = 0; local_same_rot = 0; local_mirror_inv = 0
    for trial in range(N):
        n = rng.integers(6, 30)
        X = rng.normal(size=(n, 3)) * np.array([3.0, 1.5, 0.7])
        Z = rng.choice([1, 6, 7, 8], size=n)
        X = X - X.mean(0)
        R0, e0, c0 = f(X, Z); L0 = X @ R0
        Q = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()
        Xr = X @ Q.T
        R1, e1, c1 = f(Xr, Z); L1 = Xr @ R1
        Xi = -X @ Q.T  # improper copy (inversion then rotation)
        R2, e2, c2 = f(Xi, Z); L2 = Xi @ R2
        same_chi_rot += c0[0] == c1[0]
        flip_chi_inv += c0[0] != c2[0]
        local_same_rot += np.allclose(L0, L1, atol=1e-6)
        # mirror image local coords: one axis sign differs
        ok = any(np.allclose(L0 * s, L2, atol=1e-6) for s in
                 [np.array([-1, 1, 1]), np.array([1, -1, 1]), np.array([1, 1, -1])])
        local_mirror_inv += ok
        assert abs(np.linalg.det(R0) - 1) < 1e-6
    print(f"{name}: N={N}")
    print(f"  chi unchanged under proper rotation: {same_chi_rot}/{N}")
    print(f"  chi flipped under improper op:       {flip_chi_inv}/{N}")
    print(f"  local coords identical, proper copy: {local_same_rot}/{N}")
    print(f"  local coords = one-axis mirror, improper copy: {local_mirror_inv}/{N}")
