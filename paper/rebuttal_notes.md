# Pre-emptive rebuttal notes — Digital Discovery submission

Prepared responses to the two most likely reviewer objections. All claims below
are already supported in the manuscript; section refs point to where the evidence
lives so the response can cite, not re-argue.

## 1. "Corpus is only ~1,127 crystals; the CSD has ~1.4M. Would more data change the conclusion?"

- The corpus is a *seeded random* CSD sample passed through strict rigid-body,
  ordered, non-disordered, non-polymeric, and Z'/size gates (Methods, Data). It is
  scoped to rigid, C/H/N/O-dominated molecular crystals **by design, to keep the
  orientation benchmark honest**, not purely by compute.
- The central result is **structural/representational, not data-limited**: the
  obstruction is the gauge-freedom of the asymmetric-unit pose R_asym, not a
  shortage of examples. We show this directly —
  - the absolute-orientation floor is **identical at 250 and 1,127 crystals**
    (a 4.5× scaling; §2.3), so it is not data sparsity;
  - the supervised conditional probe (§S3) fails to predict R_asym at the
    **no-information level** regardless of scale.
- The **positive** (learnable relative) result is not memorization that more/less
  data would flip: it survives a leakage-controlled **species-grouped split**
  (28.5±2.9% vs 27.5±2.7% random; §S5) and three seeds.
- We already state in the Discussion that absolute end-to-end numbers are a
  **lower bound** that more data/compute would improve. The diagnostic conclusions
  (decomposition, floor, inference-limited ceiling) **do not depend on corpus
  size**.

## 2. "End-to-end generation is 0% — is this a working method?"

- The paper is explicitly a **characterization/diagnostic study** (title; abstract
  "diagnostic study to characterize what the orientation field can and cannot
  learn"), not a SOTA-generator claim. The orientation-isolated metric is the
  primary read; the contribution is the decomposition, not a finished generator.
- The 0% is **not specific to rigid-body factorization**: an all-atom control —
  the representation used by OXtal/PackFlow — trained on the *identical corpus and
  split* also yields 0% at **both match@1 and match@20**, equal to its random-prior
  floor, with packing-component errors as large as ours (§2.8, §S6). The 0%
  reflects **corpus scale and compute budget**, not a defect of the method.
- It is **not a best-of-k artifact**: match@1 is likewise 0% for both the base and
  the higher-capacity model (§2.8).
- We report the honest joint result precisely so the orientation-isolated 13.7%
  is correctly framed as an **upper bound attainable only when packing is
  supplied**. We make no claim of a finished generator and flag end-to-end
  generation at scale as open (Discussion).

## Bonus: "Coset conditioning leaks the answer (+51%)."

- Already pre-empted: the coset experiment is framed as a **representability /
  upper-bound diagnostic** (§2.6), because the codebook is clustered from the
  observed rotations. The same label is, in principle, derivable at sampling time
  from space group + Wyckoff assignment + copy ordering (cited template-based
  generators); realizing that gain in a sampler is named as future work, not
  claimed here. Headline number is the 3-seed 47.9±3.2%, not the single-run 50.6%.
