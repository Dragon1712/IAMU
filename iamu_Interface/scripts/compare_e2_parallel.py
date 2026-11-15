#!/usr/bin/env python3
"""Check serial versus parallel IAMU-E2 numerical equivalence (ignoring elapsed time).

Use the SAME frozen checkpoint, scientific parameters, ETA case/seed/budget,
policy list and source revision on both sides. A mismatch is a hard failure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parallel_e2_core import compare_csv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', type=Path, required=True,
                   help='Path to serial runs.csv or stability_details.csv')
    p.add_argument('--parallel', type=Path, required=True,
                   help='Path to merged parallel runs.csv or stability_details.csv')
    p.add_argument('--atol', type=float, default=1e-7)
    p.add_argument('--rtol', type=float, default=1e-9)
    a = p.parse_args(argv)
    try:
        n, compared = compare_csv(a.serial,a.parallel,atol=a.atol,rtol=a.rtol)
    except (ValueError,FileNotFoundError,AssertionError) as e:
        print(f'[MISMATCH] {e}',file=sys.stderr)
        return 1
    print(f'[PASS] {n} matching independent records; {compared} numeric cells matched. '
          'Wall-clock timing deliberately excluded.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
