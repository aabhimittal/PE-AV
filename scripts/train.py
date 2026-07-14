#!/usr/bin/env python3
"""Train PE-AV on the synthetic audio-visual dataset.

Usage:
    python scripts/train.py --preset small --epochs 5 --save checkpoints/pe_av.pt
"""

from pe_av.cli import train_main

if __name__ == "__main__":
    train_main()
