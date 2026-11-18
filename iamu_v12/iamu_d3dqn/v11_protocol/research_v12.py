"""Research V1.2: transition semantics, diagnostics and exhaustion-tail compression.

The optional collapse changes the Bellman training target, not the physical
simulation or the realized episode score. It is disabled by default.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import isclose

VERSION = 2
CATEGORIES = ('actionable', 'budget_exhausted', 'forced_with_budget')


def classify_transition(tr):
    if bool(tr.obs_mask[1:].any()):
        return 'actionable'
    flags = tr.reason_flags or {}
    if flags.get('budget_exhausted', False):
        return 'budget_exhausted'
    return 'forced_with_budget'


def class_counts(transitions):
    counts = Counter({k: 0 for k in CATEGORIES})
    choices = Counter({'actionable_keep': 0, 'actionable_edit': 0})
    for tr in transitions:
        kind = classify_transition(tr)
        counts[kind] += 1
        if kind == 'actionable':
            choices['actionable_edit' if tr.action else 'actionable_keep'] += 1
    return {**dict(counts), **dict(choices)}


def _budget(tr):
    before = getattr(tr, 'budget_before', None)
    after = getattr(tr, 'budget_after', None)
    if before is None or after is None:
        raise ValueError('Exhaustion collapse requires explicit V1.2 budget_before / budget_after')
    if before < 0 or after < 0 or after > before or (before - after) != int(tr.action != 0):
        raise ValueError('Transition violates hard edit budget')
    return before, after


def collapse_exhaustion_tail(transitions, *, gamma=1.0):
    """Collapse only the final permanently actionless suffix of one *complete* episode.

    If B starts at zero, collapse the whole episode to one transition.
    If B becomes zero after an Edit, collapse that connecting Edit transition
    and all subsequent forced Keep transitions into one terminal transition.
    The returned prefix is otherwise unchanged (new list, no in-place edits).
    """
    if not (0.0 < gamma <= 1.0):
        raise ValueError('gamma must be in (0, 1]')
    if not transitions:
        raise ValueError('Empty episode')
    if not bool(transitions[-1].done):
        raise ValueError('Require completed episode for tail collapse')
    for i, tr in enumerate(transitions):
        before, after = _budget(tr)
        if i and before != transitions[i - 1].budget_after:
            raise ValueError('Budget discontinuity in episode')
        if i != len(transitions)-1 and tr.done:
            raise ValueError('Internal terminal transition')
    if transitions[0].budget_before == 0:
        boundary = 0
    else:
        boundary = next((i for i, t in enumerate(transitions)
                         if t.budget_before > 0 and t.budget_after == 0), None)
    if boundary is None:
        return list(transitions)
    for tr in transitions[boundary + 1:]:
        if tr.action != 0 or bool(tr.obs_mask[1:].any()) or tr.budget_before != 0 or tr.budget_after != 0:
            raise ValueError('Non-meaningless action exists in alleged exhausted tail')
    total = 0.0
    for tr in reversed(transitions[boundary:]):
        total = float(tr.reward) + gamma * total
    first, last = transitions[boundary], transitions[-1]
    joined = replace(first, reward=total, done=True,
                     next_global=last.next_global.copy(),
                     next_actions=last.next_actions.copy(),
                     next_mask=last.next_mask.copy(),
                     raw_steps=sum(int(getattr(t, 'raw_steps', 1)) for t in transitions[boundary:]),
                     transition_version=VERSION)
    if not bool(joined.next_mask[0]) or bool(joined.next_mask[1:].any()):
        raise ValueError('Collapsed terminal observation must have only Keep available')
    return list(transitions[:boundary]) + [joined]


def discounted_return(transitions, gamma=1.0):
    total = 0.0
    for t in reversed(transitions):
        total = float(t.reward) + gamma * total
    return total


def verify_collapse(raw, collapsed, gamma=1.0, atol=1e-6):
    if not isclose(discounted_return(raw, gamma), discounted_return(collapsed, gamma),
                   rel_tol=1e-6, abs_tol=atol):
        raise AssertionError('Exhaustion collapse changed full discounted episode return')
    if sum(getattr(t, 'raw_steps', 1) for t in collapsed) != sum(getattr(t, 'raw_steps', 1) for t in raw):
        raise AssertionError('Exhaustion collapse lost event accounting')
    if not collapsed[-1].done:
        raise AssertionError('Episode lost terminal transition')
