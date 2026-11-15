"""Deterministic E2 shard planning, aggregation and comparison; stdlib only.

This module is deliberately independent of PyTorch and the IAMU engine so that
its ordering, accounting and duplicate checks can be unit tested on any host.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

WORKFLOWS = ("multistage", "risk_stability", "risk_stability_joint", "mpc_samples", "h_depth")
SHARD_PRIMARY = ("case", "scenario", "eta_seed", "budget", "policy")
STABILITY_PRIMARY = ("case", "scenario", "eta_seed", "budget", "event_index", "sample_count", "replicate", "decision_policy")
FLOAT_FIELDS = {"weighted_wait", "spent", "elapsed_sec", "decisions", "edits_eta", "edits_clock",
                "gain_vs_periodic", "gain_vs_keep", "gain_vs_set_ddqn",
                "action_agreement", "best_tail_agreement", "best_mean_agreement", "tail_mae",
                "tail_max_error", "gain_mae", "harm_prob_mae", "risk_eval_seconds"}
TIME_FIELDS = {"elapsed_sec", "risk_eval_seconds", "mpc_seconds", "mean_runtime_seconds"}


def parse_ints(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if ":" in piece:
            a, b = map(int, piece.split(":"))
            if b < a:
                raise ValueError("Backwards integer range")
            values.extend(range(a, b + 1))
        else:
            values.append(int(piece))
    if len(values) != len(set(values)):
        raise ValueError("Duplicate integer values")
    return values


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_json(value: object) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    # Individual shard output may have different optional diagnostic columns.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def upper_cvar(values: Iterable[float], alpha: float = .9) -> float:
    arr = sorted((float(x) for x in values), reverse=True)
    if not arr or not (0 < alpha < 1):
        raise ValueError("Invalid empirical CVaR input")
    mass = len(arr) * (1 - alpha)
    left, total = mass, 0.
    for value in arr:
        weight = min(left, 1.)
        total += value * weight
        left -= weight
        if left <= 1e-12:
            break
    return total / mass


def summarize_runs(records: list[dict]) -> list[dict]:
    """Matches original eval_risk_budget_v10.summarize, except wall time metadata."""
    groups = defaultdict(list)
    for r in records:
        for scenario, budget in (("ALL", "ALL"), (r["scenario"], "ALL"),
                                 ("ALL", str(r["budget"])), (r["scenario"], str(r["budget"]))):
            groups[(r["policy"], scenario, budget)].append(r)
    result = []
    for (policy, scenario, budget), group in sorted(groups.items()):
        regret = [max(0., -float(r["gain_vs_periodic"])) for r in group]
        result.append(dict(policy=policy, scenario=scenario, budget=budget,
                           n=len(group),
                           mean_wait=mean(float(r["weighted_wait"]) for r in group),
                           mean_spent=mean(float(r["spent"]) for r in group),
                           mean_gain_vs_periodic=mean(float(r["gain_vs_periodic"]) for r in group),
                           mean_gain_vs_set_ddqn=mean(float(r["gain_vs_set_ddqn"]) for r in group),
                           cvar90_regret_vs_periodic=upper_cvar(regret, .9),
                           mean_runtime_seconds=mean(float(r["elapsed_sec"]) for r in group)))
    return result


def summarize_stability(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["decision_policy"], r["sample_count"], "ALL")].append(r)
        groups[(r["decision_policy"], r["sample_count"], r["scenario"])].append(r)
    result = []
    for (policy, n, scenario), group in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][2])):
        state_replicates = defaultdict(list)
        for r in group:
            key = (r["case"], r["scenario"], r["eta_seed"], r["budget"], r["event_index"])
            state_replicates[key].append(r["selected_action"])
        flips = []
        for choices in state_replicates.values():
            counts = {v: choices.count(v) for v in set(choices)}
            flips.append(1 - max(counts.values()) / len(choices))
        info = dict(policy=policy, sample_count=int(n), scenario=scenario,
                    n_state_replicates=len(group), n_unique_states=len(state_replicates),
                    mean_within_state_action_flip_rate=mean(flips),
                    n_states_with_action_disagreement=sum(x > 0 for x in flips))
        for k in ("action_agreement", "best_tail_agreement", "best_mean_agreement", "tail_mae",
                  "tail_max_error", "gain_mae", "harm_prob_mae", "risk_eval_seconds"):
            info["mean_" + k] = mean(float(r[k]) for r in group)
        result.append(info)
    return result


def validate_unique(rows: list[dict], columns: tuple[str, ...], label: str) -> None:
    seen = set()
    for r in rows:
        key = tuple(str(r[x]) for x in columns)
        if key in seen:
            raise ValueError(f"Duplicate {label}: {key}")
        seen.add(key)


def compare_csv(left: Path, right: Path, *, atol: float = 1e-7, rtol: float = 1e-9):
    a, b = read_rows(left), read_rows(right)
    if not a or not b:
        raise ValueError("Empty comparison input")
    key_cols = SHARD_PRIMARY if "weighted_wait" in a[0] else STABILITY_PRIMARY
    validate_unique(a, key_cols, "left row")
    validate_unique(b, key_cols, "right row")
    d1 = {tuple(r[k] for k in key_cols): r for r in a}
    d2 = {tuple(r[k] for k in key_cols): r for r in b}
    if set(d1) != set(d2):
        raise AssertionError(f"Different key sets: left_only={len(set(d1)-set(d2))}, right_only={len(set(d2)-set(d1))}")
    compared = 0
    for key in sorted(d1):
        x, y = d1[key], d2[key]
        cols = set(x) | set(y)
        for col in cols:
            if col in TIME_FIELDS or "time_seconds" in col or col.startswith("mean_runtime"):
                continue
            if col not in x or col not in y:
                raise AssertionError(f"Missing field {col!r} for {key}")
            sx, sy = x[col], y[col]
            try:
                fx, fy = float(sx), float(sy)
            except (ValueError, TypeError):
                if sx != sy:
                    raise AssertionError(f"Mismatch {key}/{col}: {sx!r} != {sy!r}")
                continue
            if not math.isfinite(fx) or not math.isfinite(fy):
                raise AssertionError(f"Nonfinite value at {key}/{col}")
            if not math.isclose(fx, fy, abs_tol=atol, rel_tol=rtol):
                raise AssertionError(f"Mismatch {key}/{col}: {fx} != {fy}")
            compared += 1
    return len(d1), compared
