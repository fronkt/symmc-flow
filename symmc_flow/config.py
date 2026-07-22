"""Dataclass configs for SymMC-Flow."""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # atom / chemistry
    max_z: int = 118
    atom_embed_dim: int = 64
    # EGNN encoder
    egnn_hidden: int = 128
    egnn_layers: int = 4
    # pair-bias attention
    d_model: int = 128
    n_heads: int = 8
    n_attn_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0
    pair_n_freq: int = 2  # Fourier frequencies on fractional pair offsets (DiffCSP-style)
    # conditioning
    n_space_groups: int = 230
    sg_embed_dim: int = 64
    time_embed_dim: int = 64
    n_cosets: int = 0  # >0 enables per-molecule space-group coset embedding (2c diagnostic)
    # lattice representation (Phase F): "shape10" = (log-vol, det-1 shape) default; "logmetric6"
    # = O(3)-invariant log-metric k in R^6 that supports crystal-family masking.
    lattice_repr: str = "shape10"
    lattice_family_mask: bool = False  # freeze crystal-family-constrained lattice DOF (needs
                                       # lattice_repr="logmetric6"); the deployable extension of
                                       # coset orientation conditioning to the unit cell
    self_cond: bool = False            # Phase F3d: self-conditioning refinement -- the network sees
                                       # its own terminal-state estimate and predicts a correction
                                       # (iterated at sampling). Gated; off = old path byte-identical.
    # flow head loss weights
    lambda_lattice: float = 1.0
    lambda_centroid: float = 1.0
    lambda_orient: float = 1.0


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    steps: int = 400
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"  # "auto" -> cuda if available else cpu
    log_every: int = 25
    # synthetic dataset
    n_train: int = 512
    max_mols: int = 4
    max_atoms_per_mol: int = 12
    sampler_steps: int = 50
    use_group_averaging: bool = True
    use_ot_coupling: bool = False     # optimal-transport prior<->data atom pairing
    prior_vol_per_atom: float = 10.0  # mean cell volume per atom (A^3) of the lattice prior
    prior_logvol_std: float = 0.3     # std of ln(V) in the lattice prior (data-informed ~0.11 for
                                      # molecular crystals; shape10 path keeps the 0.3 default)
    prior_dev_std: float = 0.3        # std of the deviatoric log-metric prior dims (logmetric6)
    centroid_prior_std: float | None = None  # None -> uniform torus prior; float -> wrapped-normal
    fixed_prior: bool = False         # cache one prior per structure (sharper OT target)
    sampler_churn: float = 0.0        # Langevin-style noise in the sampler (0 = deterministic)
    cond_clean_packing: bool = False  # diagnostic: condition the field on TRUE lattice+centroid
                                      # (not z_t) to test if noised packing is what floors orient
    so3_avg_k: int = 1                # >1 enables the SO(3)-averaged orientation objective
                                      # (average the orient FM residual over K prior rotations;
                                      # variance-reduced SO(3) target, cf. arXiv:2507.09785)
