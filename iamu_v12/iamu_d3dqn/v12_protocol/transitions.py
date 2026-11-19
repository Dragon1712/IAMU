"""V1.2 episode-preserving transition types; do NOT change V1.1 Transition pickle layout."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass(frozen=True)
class RawStep:
    transition: Any  # Original iamu_dynamic.rl_ddqn_v11.Transition
    source_budget: int
    next_budget: int
    source_event_index: int
    next_event_index: int
    source_time: float
    next_time: float
    episode_id: int
    policy_version: int

@dataclass(frozen=True)
class MacroTransition:
    obs_global: np.ndarray
    obs_actions: np.ndarray
    obs_mask: np.ndarray
    action: int
    reward: float
    next_global: np.ndarray
    next_actions: np.ndarray
    next_mask: np.ndarray
    done: bool
    bootstrap_discount: float
    n_raw_steps: int
    elapsed_hours: float
    source_budget: int
    next_budget: int
    episode_id: int
    policy_version: int
    raw_start_index: int
    raw_end_index: int
    reason_flags: dict = field(default_factory=dict)

@dataclass(frozen=True)
class PrefixLedger:
    raw_steps: int
    reward: float  # discounted from episode start
    reward_undiscounted: float
    first_decision_raw_index: int | None

@dataclass(frozen=True)
class CompressionResult:
    transitions: tuple[MacroTransition, ...]
    prefix: PrefixLedger
    audit: dict


def copied_array(a):
    return np.array(a, copy=True)
