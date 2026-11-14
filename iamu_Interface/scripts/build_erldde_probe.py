#!/usr/bin/env python3
"""Build isolated read-only ERL-DDE C++ probe, never overwriting original .so."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not (ROOT / 'iamu_dynamic' / 'legacy_erldde.py').is_file():
    raise SystemExit('Run python setup_interface.py first')
from iamu_dynamic.legacy_erldde import build_probe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=ROOT.parent/'2024LiERLDDE',
                    help='Directory containing original BACAP_env.cpp')
parser.add_argument('--out', type=Path, default=ROOT/'results'/'erldde_probe')
args = parser.parse_args()
print(build_probe(args.source, args.out))
