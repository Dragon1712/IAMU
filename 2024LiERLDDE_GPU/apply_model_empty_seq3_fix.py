#!/usr/bin/env python3
"""Safely apply the wait-on-empty-candidate fix to upstream ERL-DDE model.py.

Usage:
    python apply_model_empty_seq3_fix.py model.py

This program does not change neural-network layers or weights. It backs up the
original model.py and fails if the expected original code cannot be recognized.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import shutil

MARKER = 'ERLDDE_EMPTY_SEQ3_WAIT'


def patch_source(src: str) -> str:
    if MARKER in src:
        raise ValueError('model.py is already patched (found marker ' + MARKER + ')')

    # Insert the guard at the exact start of the relevant part of ES.forward.
    guard = re.compile(
        r'(?m)^(?P<ind>[ \t]*)seq1,\s*seq2,\s*seq3,\s*id\s*=\s*self\.cope_state\(state\)[ \t]*$'
    )
    matches = list(guard.finditer(src))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one cope_state assignment; got {len(matches)}')

    def insert_guard(m: re.Match[str]) -> str:
        ind = m.group('ind')
        return (
            m.group(0) + '\n'
            + ind + '# ' + MARKER + ': no schedulable vessel; advance port event time.\n'
            + ind + 'if seq3.shape[0] == 0:\n'
            + ind + '    return np.array([-1, -1, -1, -1, -1])'
        )

    updated = guard.sub(insert_guard, src, count=1)

    # A failed argmax must raise its actual error, not print and then fall
    # through to UnboundLocalError on max_index.
    unsafe = re.compile(
        r'(?m)^(?P<ind>[ \t]*)try:[ \t]*\n'
        r'(?P=ind)[ \t]{4}max_index\s*=\s*torch\.argmax\(attention_weights\)[ \t]*\n'
        r'(?P=ind)except:[ \t]*\n'
        r'(?P=ind)[ \t]{4}print\(seq3\)[ \t]*\n'
        r'(?P=ind)[ \t]{4}print\(state\)[ \t]*'
    )
    unsafe_matches = list(unsafe.finditer(updated))
    if len(unsafe_matches) != 1:
        raise ValueError(
            f'Expected one original try/except around argmax, got {len(unsafe_matches)}. '
            'No files were modified. Apply the instructions in README manually.'
        )

    def replace_unsafe(m: re.Match[str]) -> str:
        ind = m.group('ind')
        return (
            ind + 'if attention_weights.numel() == 0:\n'
            + ind + '    raise RuntimeError("Empty attention weights with nonempty seq3")\n'
            + ind + 'max_index = torch.argmax(attention_weights)'
        )

    updated = unsafe.sub(replace_unsafe, updated, count=1)
    ast.parse(updated, filename='model.py')
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', default='model.py')
    args = parser.parse_args()
    target = Path(args.path)
    src = target.read_text(encoding='utf-8')
    updated = patch_source(src)
    backup = target.with_name(target.name + '.before_empty_seq3_fix')
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}. Preserve it and rename manually.')
    shutil.copy2(target, backup)
    target.write_text(updated, encoding='utf-8')
    print('[PATCHED]', target)
    print('[BACKUP ]', backup)
    print('Changed: empty seq3 -> wait action; removed swallowing argmax exceptions.')


if __name__ == '__main__':
    main()
