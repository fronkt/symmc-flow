"""SUN primitives: structural validity, within-formula uniqueness + novelty."""
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Structure, Lattice
from symmc_flow.sun import structural_validity, unique_mask, novel_mask, sun_summary


def _nacl(a=5.6):
    return Structure(Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _diamond_c(a=3.57):
    return Structure(Lattice.cubic(a), ["C", "C"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def test_structural_validity_rejects_overlap():
    good = _nacl()
    bad = Structure(Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.01, 0, 0]])
    assert structural_validity(good)
    assert not structural_validity(bad)
    assert not structural_validity(None)


def test_unique_mask_dedups_within_formula_only():
    a, a_dup, b = _nacl(), _nacl(), _diamond_c()
    # a and a_dup are identical (dup); b is a different composition (never a dup of a)
    mask = unique_mask([a, a_dup, b])
    assert mask == [True, False, True]          # 2 distinct
    # different composition with identical fractional coords must NOT collapse
    assert unique_mask([_nacl(), _diamond_c()]) == [True, True]


def test_novel_mask_only_compares_same_formula():
    train = [_nacl()]
    gen_known = _nacl()                          # same composition + geometry as train
    gen_novel_same_comp = Structure(Lattice.cubic(6.5), ["Na", "Cl"],
                                    [[0, 0, 0], [0.5, 0.5, 0.0]])  # NaCl, different geom
    gen_other_comp = _diamond_c()                # composition absent from train -> novel
    mask = novel_mask([gen_known, gen_novel_same_comp, gen_other_comp], train)
    assert mask == [False, True, True]


def test_sun_summary_without_stability_omits_sun():
    train = [_nacl()]
    res = sun_summary([_nacl(), _diamond_c()], train)
    assert res["n"] == 2
    assert res["unique_rate"] == 1.0
    assert res["novel_rate"] == 0.5             # NaCl known, C novel
    assert "sun_rate" not in res


def test_sun_summary_with_stability():
    train = [_nacl()]
    gen = [_diamond_c(), _nacl()]               # C novel+unique, NaCl known
    res = sun_summary(gen, train, stable_mask=[True, True])
    # only the C structure is stable AND unique AND novel
    assert res["sun_rate"] == 0.5
    assert res["stable_rate"] == 1.0
