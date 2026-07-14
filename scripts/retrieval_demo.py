#!/usr/bin/env python3
"""Text -> audiovisual retrieval demo over a synthetic gallery.

Usage:
    python scripts/retrieval_demo.py --checkpoint checkpoints/pe_av.pt
"""

from pe_av.cli import retrieve_main

if __name__ == "__main__":
    retrieve_main()
