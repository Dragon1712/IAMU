#!/usr/bin/env python3
"""Isolate a frozen IAMU V0.9 environment snapshot within iamu_Interface/.

Copies only selected source files from a sibling iamu_d3dqn checkout. Does not
edit the original source, model checkpoint, result folders or Git history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / 'iamu_d3dqn'
V09_FILES = (
    '__init__.py',
    'core.py',
    'rl_env.py',
    'rl_ddqn.py',
    'rl_experiment.py',
    'risk_selection.py',
    'recovery.py',
    'legacy_erldde.py',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_if_needed(src: Path, dst: Path, force: bool, check_only: bool) -> str:
    if not src.is_file():
        raise FileNotFoundError(f'Missing source file: {src}')
    if dst.exists():
        if not dst.is_file():
            raise RuntimeError(f'Destination is not a file: {dst}')
        if sha256(src) == sha256(dst):
            return 'unchanged'
        if not force:
            raise RuntimeError(
                f'Existing file differs: {dst}\n'
                'Refusing to overwrite a frozen snapshot. Use --force intentionally.'
            )
    elif check_only:
        raise FileNotFoundError(f'Missing snapshot file: {dst}')
    if check_only:
        raise RuntimeError(f'Snapshot differs from selected source: {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        raise IOError(f'Copy failed checksum validation: {dst}')
    return 'copied'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                   help='V0.9 iamu_d3dqn directory (default: sibling directory)')
    p.add_argument('--copy-assets', action='store_true',
                   help='Also copy selected case files and Seed checkpoint/manifest')
    p.add_argument('--case-ids', default='8,9', help='Case IDs to copy when --copy-assets')
    p.add_argument('--seed', type=int, default=42, help='Set-DDQN checkpoint training seed')
    p.add_argument('--force', action='store_true', help='Replace mismatched snapshots explicitly')
    p.add_argument('--check-only', action='store_true', help='Verify snapshot without copying')
    args = p.parse_args(argv)

    source = args.source.expanduser().resolve()
    if source.resolve() == ROOT.resolve():
        p.error('Source must be different from iamu_Interface')
    if not (source / 'iamu_dynamic').is_dir():
        p.error(f'Not a valid IAMU V0.9 source: {source}')
    copied = []
    for name in V09_FILES:
        src = source / 'iamu_dynamic' / name
        dst = ROOT / 'iamu_dynamic' / name
        status = copy_if_needed(src, dst, args.force, args.check_only)
        copied.append(dict(file=f'iamu_dynamic/{name}', sha256=sha256(dst), status=status,
                           original=str(src)))
        print(f'[{status}] {dst.relative_to(ROOT)}', flush=True)

    assets = []
    if args.copy_assets:
        cases_parent = source.parent / '2024LiERLDDE' / 'case_code'
        if not cases_parent.is_dir():
            cases_parent = source.parent / '2024LiERLDDE_GPU' / 'case_code'
        case_ids = [x.strip() for x in args.case_ids.split(',') if x.strip()]
        if not case_ids or not all(s.isdigit() for s in case_ids):
            p.error('Use numeric --case-ids such as 8,9')
        candidates = [(cases_parent / f'{s}.txt', ROOT / 'assets' / 'cases' / f'{s}.txt')
                      for s in case_ids]
        model_dir = source / 'results' / f'v09b_set_ddqn_seed{args.seed}_300'
        candidates += [
            (model_dir / 'best.pt', ROOT / 'assets' / 'checkpoints' / f'set_ddqn_seed{args.seed}' / 'best.pt'),
            (model_dir / 'manifest.json', ROOT / 'assets' / 'checkpoints' / f'set_ddqn_seed{args.seed}' / 'manifest.json'),
        ]
        for src, dst in candidates:
            status = copy_if_needed(src, dst, args.force, args.check_only)
            assets.append(dict(file=str(dst.relative_to(ROOT)), sha256=sha256(dst),
                               status=status, original=str(src)))
            print(f'[{status}] {dst.relative_to(ROOT)}', flush=True)

    if not args.check_only:
        manifest = {
            'snapshot_created_utc': datetime.now(timezone.utc).isoformat(),
            'source_root': str(source),
            'engine_files': copied,
            'assets': assets,
            'v10_scope': 'frozen-policy evaluation and single-vessel ERL-DDE hints',
        }
        path = ROOT / 'dependency_snapshot.json'
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Snapshot manifest: {path}', flush=True)
    print('IAMU V1.0 experiment workspace is ready.')


if __name__ == '__main__':
    main()
