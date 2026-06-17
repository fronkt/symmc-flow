"""SymMC-Flow: Symmetric Molecular Crystal Flow Matching.

Rigid-body conformers (MolCrystalFlow) + space-group-conditioned flow (SGFM)
+ pair-bias attention (Clari) + 2D periodic-table embeddings (CrystalDiT).
"""
from .config import ModelConfig, TrainConfig
from .model import SymMCFlow
from .sampler import rk4_sample
from .molcrystal import MolCrystalDataset, rigid_to_frac, rigid_to_structure

__all__ = ["ModelConfig", "TrainConfig", "SymMCFlow", "rk4_sample",
           "MolCrystalDataset", "rigid_to_frac", "rigid_to_structure"]
__version__ = "0.1.0"
