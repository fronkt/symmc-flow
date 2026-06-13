"""Stability (the 'S' in SUN) via a CHGNet-consistent convex hull.

We relax each candidate with the pretrained CHGNet universal potential, then place it on a
convex hull built from CHGNet energies of the Materials Project reference structures in the
SAME chemical system. Using one energy model for both the candidate and every hull anchor
keeps the reference consistent and sidesteps the GGA/GGA+U DFT-correction handling that an
ML-energy-vs-DFT-hull comparison would require. A candidate is 'stable' (metastable) if its
CHGNet E_above_hull is below `threshold` (default 0.1 eV/atom, the usual metastability cut).

Heavy deps (chgnet, mp_api) are imported lazily so the rest of the package stays light.
The MP API key is read from the caller (env MP_API_KEY), never hard-coded.
"""


def relax_structures(structures, fmax=0.1, steps=200):
    """CHGNet-relax each structure; return list of relaxed Structures (None on failure)."""
    from chgnet.model import StructOptimizer
    opt = StructOptimizer()
    out = []
    for s in structures:
        if s is None:
            out.append(None)
            continue
        try:
            res = opt.relax(s, fmax=fmax, steps=steps, verbose=False)
            out.append(res["final_structure"])
        except Exception:
            out.append(None)
    return out


def _chemsys(structure):
    return "-".join(sorted({e.symbol for e in structure.composition.elements}))


def e_above_hull(structures, api_key, net=None, energy_cache=None, ref_cache=None):
    """CHGNet E_above_hull (eV/atom) for each structure vs a CHGNet-energy MP hull of its
    chemical system. Returns a list aligned with `structures` (None where it couldn't be
    computed: None input, empty chemsys entries, or PD failure). `ref_cache` (chemsys ->
    list of ComputedEntry) is reused across calls to amortize MP queries + CHGNet evals."""
    from chgnet.model import CHGNet
    from mp_api.client import MPRester
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    net = net or CHGNet.load()
    ref_cache = {} if ref_cache is None else ref_cache

    def chgnet_total(struct):
        return net.predict_structure(struct)["e"] * len(struct)   # eV/atom -> total eV

    out = []
    with MPRester(api_key) as mpr:
        for s in structures:
            if s is None:
                out.append(None)
                continue
            key = _chemsys(s)
            if key not in ref_cache:
                refs = []
                try:
                    entries = mpr.get_entries_in_chemsys(key.split("-"))
                except Exception:
                    entries = []
                for ent in entries:
                    try:
                        refs.append(ComputedEntry(ent.composition, chgnet_total(ent.structure)))
                    except Exception:
                        pass
                ref_cache[key] = refs
            refs = ref_cache[key]
            if not refs:
                out.append(None)
                continue
            try:
                cand = ComputedEntry(s.composition, chgnet_total(s))
                pd = PhaseDiagram(refs + [cand])
                out.append(pd.get_e_above_hull(cand))
            except Exception:
                out.append(None)
    return out


def stable_mask(structures, api_key, threshold=0.1, fmax=0.1, steps=200, ref_cache=None):
    """Relax then score: True where CHGNet E_above_hull < threshold. Also returns the
    relaxed structures and raw e_above_hull list so callers can reuse them."""
    relaxed = relax_structures(structures, fmax=fmax, steps=steps)
    ehull = e_above_hull(relaxed, api_key, ref_cache=ref_cache)
    mask = [e is not None and e < threshold for e in ehull]
    return mask, relaxed, ehull
