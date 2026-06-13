"""SUN metric primitives (Stable / Unique / Novel) for generated crystals.

Unique + Novel + structural-validity are pure structure-set operations and live here;
they are the cheap, dependency-light part of SUN. Stability (E_above_hull via an ML
potential + the Materials Project convex hull) is heavier and lives in `stability.py`.

Efficiency: because the generator is composition-conditioned, two structures of DIFFERENT
reduced composition can never StructureMatcher-match. So uniqueness (dedup within the
generated set) and novelty (vs the training set) only compare WITHIN a composition bucket,
turning an O(N*M) StructureMatcher sweep into the sum of small per-formula comparisons.
"""
from collections import defaultdict


def matcher(ltol=0.3, stol=0.5, angle_tol=10.0):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    return StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)


def _formula(structure):
    return structure.composition.reduced_formula


def structural_validity(structure, min_dist=0.5):
    """CDVAE structural validity: no two atoms closer than `min_dist` Angstrom
    (minimum-image). None / empty structures are invalid."""
    if structure is None or len(structure) == 0:
        return False
    if len(structure) == 1:
        return True
    dm = structure.distance_matrix
    n = len(structure)
    for i in range(n):
        for j in range(i + 1, n):
            if dm[i, j] < min_dist:
                return False
    return True


def unique_mask(structures, sm=None):
    """True marks each structure that is the FIRST occurrence of its (composition,
    geometry) class; duplicates and None are False. sum(mask) == number of distinct
    structures. Comparison is within-formula only."""
    sm = sm or matcher()
    reps = defaultdict(list)            # formula -> list of representative structures
    out = []
    for s in structures:
        if s is None:
            out.append(False)
            continue
        f = _formula(s)
        if any(sm.fit(s, r) for r in reps[f]):
            out.append(False)
        else:
            reps[f].append(s)
            out.append(True)
    return out


def novel_mask(gen_structures, train_structures, sm=None):
    """True for each generated structure that matches NO training structure of the same
    reduced composition (novelty). None gen structures are False (invalid != novel)."""
    sm = sm or matcher()
    train_by_formula = defaultdict(list)
    for t in train_structures:
        if t is not None:
            train_by_formula[_formula(t)].append(t)
    out = []
    for s in gen_structures:
        if s is None:
            out.append(False)
            continue
        ts = train_by_formula.get(_formula(s), [])
        out.append(not any(sm.fit(s, t) for t in ts))
    return out


def sun_summary(gen_structures, train_structures, stable_mask=None, sm=None):
    """Combine the components. `stable_mask` is an optional bool list (from stability.py);
    if None, only U / N / validity / valid-unique-novel are reported and SUN is omitted.
    Returns a dict of rates over the total number of generated structures."""
    sm = sm or matcher()
    n = len(gen_structures)
    valid = [structural_validity(s) for s in gen_structures]
    uniq = unique_mask(gen_structures, sm)
    novel = novel_mask(gen_structures, train_structures, sm)
    res = {
        "n": n,
        "valid_rate": sum(valid) / max(n, 1),
        "unique_rate": sum(uniq) / max(n, 1),
        "novel_rate": sum(novel) / max(n, 1),
        "valid_unique_novel_rate": sum(v and u and nv for v, u, nv in zip(valid, uniq, novel)) / max(n, 1),
    }
    if stable_mask is not None:
        res["stable_rate"] = sum(stable_mask) / max(n, 1)
        res["sun_rate"] = sum(
            st and u and nv for st, u, nv in zip(stable_mask, uniq, novel)
        ) / max(n, 1)
    return res
