"""Demo: train SymMC-Flow on synthetic data, show the loss drops.

    python scripts/train_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symmc_flow.config import ModelConfig, TrainConfig
from symmc_flow.train import train

if __name__ == "__main__":
    mcfg = ModelConfig(d_model=96, egnn_hidden=96, n_attn_layers=3, egnn_layers=3)
    tcfg = TrainConfig(steps=300, batch_size=16, log_every=25)
    model, hist = train(mcfg, tcfg)
    first = sum(hist[:10]) / 10
    last = sum(hist[-10:]) / 10
    print(f"\nmean loss  first-10 {first:.4f}  ->  last-10 {last:.4f}  "
          f"({100 * (1 - last / first):.1f}% drop)")
