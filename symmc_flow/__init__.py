"""SymMC-Flow: Symmetric Molecular Crystal Flow Matching.

Rigid-body conformers (MolCrystalFlow) + space-group-conditioned flow (SGFM)
+ pair-bias attention (Clari) + 2D periodic-table embeddings (CrystalDiT).
"""
from .config import ModelConfig, TrainConfig
from .model import SymMCFlow
from .sampler import rk4_sample

__all__ = ["ModelConfig", "TrainConfig", "SymMCFlow", "rk4_sample"]
__version__ = "0.1.0"
