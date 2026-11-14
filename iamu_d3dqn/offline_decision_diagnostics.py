#!/usr/bin/env python3
"""Frozen D3QN inference margin sweep and same-state counterfactual Q-gap diagnostics.

Run from the iamu_d3dqn project root. Requires a trusted local best.pt.
This program never trains or updates the checkpoint.

margin: frozen epsilon=0 policy under alternative Keep selection margins, with
        paired Keep and Periodic references over the same IAMU validation grid.
counterfactual: replay reference-policy prefixes to identical reachable states,
        force Keep vs each selected legal Edit, and run the SAME fixed suffix
        policy in both branches. Compares one realized counterfactual return
        difference with the frozen model's raw Q(Edit)-Q(Keep).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from iamu_dynamic import generate_scenario, read_case
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_experiment import choose_action

DEFAULT_MARGINS = "0,0.025,0.05,0.075,0.1,0.125,0.15"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ints(spec: str) -> list[int]:
    result = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            lo, hi = map(int, token.split(":"))
            if hi < lo or hi - lo > 10_000:
                raise ValueError("Bad integer range: " + token)
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(token))
    if not result or len(set(result)) != len(result):
        raise ValueError("Specify nonempty, distinct integers")
    return result


def parse_names(spec: str) -> list[str]:
    result = [s.strip() for s in spec.split(",") if s.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError("Specify nonempty, distinct names")
    return result


def parse_margins(spec: str) -> list[float]:
    result = [float(token) for token in spec.split(",") if token.strip()]
    if not result or any(not math.isfinite(m) or m < 0 for m in result):
        raise ValueError("Keep margins must be finite and nonnegative")
    if len(set(result)) != len(result):
        raise ValueError("Keep margins must be distinct")
    return result


def cvar90_positive_regret(values: list[float]) -> float:
    """Mean of the worst 10% of nonnegative regrets, with fractional tail mass."""
    if not values:
        return float("nan")
    ordered = sorted((max(0.0, float(x)) for x in values), reverse=True)
    mass = 0.1 * len(ordered)
    remaining, total = mass, 0.0
    for x in ordered:
        fraction = min(1.0, remaining)
        if fraction <= 0:
            break
        total += fraction * x
        remaining -= fraction
    return total / mass


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to save in {path}")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


@dataclass
class Protocol:
    cases: list[Path]
    scenarios: list[str]
    seeds: list[int]
    budgets: list[int]
    env: dict[str, Any]
    checkpoint_sha: str
    manifest_sha: str
    frozen_margin: float
    model: DoubleDQNAgent


def setup(args) -> Protocol:
    cp = Path(args.checkpoint).expanduser().resolve()
    mf = Path(args.manifest).expanduser().resolve()
    if not cp.is_file() or not mf.is_file():
        raise FileNotFoundError("Local best.pt and manifest.json must both exist")
    data = json.loads(mf.read_text(encoding="utf-8"))
    cfg = data["training_args"]
    default_cases = cfg.get("val_cases", data.get("val_cases"))
    if not default_cases:
        raise ValueError("Manifest contains no val_cases")
    cases = [Path(s) for s in (args.cases or default_cases)]
    for path in cases:
        if not path.is_file():
            raise FileNotFoundError(f"Case path not found: {path}; run from iamu_d3dqn root")
    scenarios = parse_names(args.scenarios or cfg["scenarios"])
    if not set(scenarios).issubset({"small", "repeated", "shock"}):
        raise ValueError("Unknown scenario name")
    seeds = parse_ints(args.seeds or cfg["val_seeds"])
    budgets = parse_ints(args.budgets or cfg["budgets"])
    if any(b < 0 for b in budgets):
        raise ValueError("Budget must be nonnegative")
    clock = cfg["clock"]
    if clock != "union":
        raise ValueError("This study requires the original union clock for Periodic")
    env_cfg = {
        "candidate_limit": int(cfg["candidate_limit"]),
        "decision_clock": clock,
        "period": float(cfg["period"]),
        "reward_scale": float(cfg["reward_scale"]),
        "candidate_screen": cfg["candidate_screen"],
        "min_gain": float(cfg["min_gain"]),
        "budget_features": cfg["budget_features"],
        "value_features": cfg["value_features"],
        "updates": int(cfg["updates"]),
        "uncertainty": float(cfg.get("validation_uncertainty", 8.0)),
    }
    torch.set_num_threads(1)
    model, meta = DoubleDQNAgent.from_checkpoint(str(cp), device=args.device)
    if args.command == "margin" and model.q_network != "d3qn":
        raise ValueError(f"Margin experiment expects the original CC-D3QN; got {model.q_network}")
    if args.command == "counterfactual" and model.q_network not in ("ddqn", "d3qn", "set_ddqn", "set_d3qn"):
        raise ValueError(f"Unsupported Q network type for counterfactual test: {model.q_network}")
    if "selected_episode" not in meta:
        raise ValueError("Checkpoint has no selected_episode metadata")
    model.online.eval()
    return Protocol(cases, scenarios, seeds, budgets, env_cfg,
                    sha256_file(cp), sha256_file(mf), float(model.keep_margin), model)


def make_instance(case_data, scenario_label: str, seed: int, p: Protocol):
    vessels, port = case_data
    return generate_scenario(vessels, port, label=scenario_label, seed=seed,
                             updates_per_vessel=p.env["updates"],
                             uncertainty=p.env["uncertainty"])


def make_env(instance, budget: int, p: Protocol):
    keys = ("candidate_limit", "decision_clock", "period", "reward_scale",
            "candidate_screen", "min_gain", "budget_features", "value_features")
    return IAMUEventEnv(instance, budget=int(budget),
                        **{k: p.env[k] for k in keys})


def pick_action(env, policy: str, model: DoubleDQNAgent) -> int:
    if policy == "model":
        return model.act(env._observation(), epsilon=0.0)
    if policy not in ("keep", "greedy", "periodic"):
        raise ValueError(f"Unknown policy: {policy}")
    return int(choose_action(policy, env))


def check_action(ob, action: int):
    if not (0 <= action < len(ob.mask) and bool(ob.mask[action])):
        raise AssertionError(f"Illegal action {action} with mask {ob.mask}")


def finish_from(env, policy: str, model: DoubleDQNAgent,
                first_action: int | None = None) -> dict[str, Any]:
    """Cumulative *scaled* reward from THIS state onward and whole-episode wait."""
    reward_sum = 0.0
    decisions = 0
    edits_eta = edits_clock = 0
    while True:
        ob = env._observation()
        action = first_action if first_action is not None else pick_action(env, policy, model)
        first_action = None
        check_action(ob, action)
        if action != 0:
            edits_eta += int(ob.event_type == "eta")
            edits_clock += int(ob.event_type == "clock")
        _, reward, done, info = env.step(action)
        reward_sum += float(reward)
        decisions += 1
        if done:
            return {"G_scaled": reward_sum, "wait": float(info["weighted_wait"]),
                    "spent": int(info["spent"]), "steps": decisions,
                    "edits_eta": edits_eta, "edits_clock": edits_clock}


def grid(p: Protocol):
    for case in p.cases:
        case_data = read_case(str(case))
        for scenario in p.scenarios:
            for seed in p.seeds:
                # This creates identical exogenous ETA streams for every budget/policy.
                instance = make_instance(case_data, scenario, seed, p)
                for budget in p.budgets:
                    yield case, scenario, seed, budget, instance


def key(case, scenario, seed, budget):
    return (str(case), scenario, int(seed), int(budget))


def summary_rows_margin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        for s, b in (("ALL", "ALL"), (row["scenario"], "ALL"),
                     ("ALL", row["budget"]), (row["scenario"], row["budget"])):
            groups[(row["margin"], s, b)].append(row)
    output = []
    for (margin, scenario, budget), values in sorted(groups.items(), key=str):
        waits = [float(r["wait"]) for r in values]
        regrets = [float(r["regret_vs_periodic"]) for r in values]
        output.append({
            "margin": margin, "scenario": scenario, "budget": budget,
            "n": len(values), "mean_wait": statistics.mean(waits),
            "median_wait": statistics.median(waits),
            "mean_spent": statistics.mean(float(r["spent"]) for r in values),
            "zero_edit_rate": statistics.mean(int(r["spent"] == 0) for r in values),
            "mean_gain_vs_keep": statistics.mean(float(r["gain_vs_keep"]) for r in values),
            "mean_gain_vs_periodic": statistics.mean(float(r["gain_vs_periodic"]) for r in values),
            "cvar90_regret_vs_periodic": cvar90_positive_regret(regrets),
            "worst_regret_vs_periodic": max(regrets),
            "worse_than_periodic": sum(float(x) > 1e-6 for x in regrets),
        })
    return output


def run_margin(args, p: Protocol):
    margins = parse_margins(args.margins)
    rows = []
    for idx, (case, scenario, seed, budget, instance) in enumerate(grid(p), start=1):
        references = {}
        for policy in ("keep", "periodic"):
            env = make_env(instance, budget, p)
            env.reset()
            references[policy] = finish_from(env, policy, p.model)
        for margin in margins:
            p.model.keep_margin = float(margin)
            env = make_env(instance, budget, p)
            env.reset()
            result = finish_from(env, "model", p.model)
            # All policies share identical physical scenarios, not necessarily states.
            rows.append({
                "case": str(case), "scenario": scenario, "seed": seed, "budget": budget,
                "margin": margin, "wait": result["wait"], "spent": result["spent"],
                "steps": result["steps"], "edits_eta": result["edits_eta"],
                "edits_clock": result["edits_clock"],
                "keep_wait": references["keep"]["wait"],
                "periodic_wait": references["periodic"]["wait"],
                "gain_vs_keep": references["keep"]["wait"] - result["wait"],
                "gain_vs_periodic": references["periodic"]["wait"] - result["wait"],
                "regret_vs_periodic": max(0.0, result["wait"]-references["periodic"]["wait"]),
            })
        print(f"[margin] {idx} groups complete / {len(p.cases)*len(p.scenarios)*len(p.seeds)*len(p.budgets)}", flush=True)
    p.model.keep_margin = p.frozen_margin
    write_csv(args.out / "margin_runs.csv", rows)
    output = summary_rows_margin(rows)
    write_csv(args.out / "margin_summary.csv", output)
    print("[margin] Aggregate ALL/ALL:")
    for item in output:
        if item["scenario"] == "ALL" and item["budget"] == "ALL":
            print(f"  margin={item['margin']:g} wait={item['mean_wait']:.6f} "
                  f"spent={item['mean_spent']:.3f} CVaR90={item['cvar90_regret_vs_periodic']:.3f}")
    return len(rows)


def score_q(model: DoubleDQNAgent, ob):
    with torch.inference_mode():
        device = model.device
        g = torch.as_tensor(ob.global_features, device=device, dtype=torch.float32)[None, :]
        x = torch.as_tensor(ob.action_features, device=device, dtype=torch.float32)[None, :, :]
        m = torch.as_tensor(ob.mask, device=device, dtype=torch.bool)[None, :]
        q = model.online(g, x, m).squeeze(0).cpu().numpy().astype(float)
    legal = np.asarray(ob.mask, dtype=bool)
    if not bool(legal[0]) or not np.isfinite(q[legal]).all():
        raise AssertionError("Expected legal Keep and finite legal Q values")
    edits = np.flatnonzero(legal[1:]) + 1
    best = int(edits[np.argmax(q[edits])]) if len(edits) else None
    scores = np.where(legal, q, -np.inf).copy()
    scores[0] += model.keep_margin
    selected = int(np.argmax(scores))
    return q, best, selected


def trace_reference(instance, budget, p: Protocol, policy: str):
    """Record public, reproducible prefix actions; never store hidden-arrival features."""
    env = make_env(instance, budget, p)
    env.reset()
    trace = []
    candidates = []
    while True:
        ob = env._observation()
        action = pick_action(env, policy, p.model)
        check_action(ob, action)
        step_idx = len(trace)
        if int(np.count_nonzero(ob.mask)) > 1:
            candidates.append(step_idx)
        trace.append(action)
        _, _, done, _ = env.step(action)
        if done:
            break
    return trace, candidates


def choose_steps(candidates: list[int], limit: int) -> list[int]:
    """Stratify over the eligible-event order instead of always taking first events."""
    if limit == 0 or len(candidates) <= limit:
        return list(candidates)
    indices = sorted(set(min(len(candidates)-1, int((j + 0.5) * len(candidates) / limit))
                         for j in range(limit)))
    return [candidates[i] for i in indices]


def replay_prefix(instance, budget: int, p: Protocol, actions: list[int]):
    env = make_env(instance, budget, p)
    env.reset()
    for action in actions:
        check_action(env._observation(), action)
        _, _, done, _ = env.step(action)
        if done:
            raise AssertionError("Selected decision state lies after the terminal event")
    return env


def sign(value: float, tolerance: float = 1e-8) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def counterfactual_rows_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        for s, b in (("ALL", "ALL"), (r["scenario"], "ALL"),
                     ("ALL", r["budget"]), (r["scenario"], r["budget"])):
            groups[(r["reference"], s, b)].append(r)
    out = []
    for (reference, scenario, budget), items in sorted(groups.items(), key=str):
        meaningful = [r for r in items if abs(r["actual_delta_G"]) > 1e-8]
        chosen = [r for r in items if r["is_best_predicted_edit"]]
        decided = [r for r in chosen if abs(r["actual_delta_G"]) > 1e-8]
        q = np.asarray([r["q_gap"] for r in items], dtype=float)
        target = np.asarray([r["actual_delta_G"] for r in items], dtype=float)
        correlation = (float(np.corrcoef(q, target)[0, 1])
                       if len(q) > 2 and np.std(q) > 1e-9 and np.std(target) > 1e-9 else "")
        out.append({
            "reference": reference, "scenario": scenario, "budget": budget,
            "n_state_action_pairs": len(items),
            "n_unique_states": len(set(r["state_id"] for r in items)),
            "sign_accuracy_raw_nonzero": (statistics.mean(
                sign(r["q_gap"]) == sign(r["actual_delta_G"]) for r in meaningful)
                if meaningful else ""),
            "qgap_vs_realized_delta_mae": float(np.mean(np.abs(q-target))),
            "qgap_vs_realized_delta_bias": float(np.mean(q-target)),
            "qgap_realized_delta_pearson": correlation,
            "n_best_predicted_edit_pairs": len(chosen),
            "chosen_edit_decision_accuracy_nonzero": (
                statistics.mean((r["would_choose_edit"] == (r["actual_delta_G"] > 0))
                                for r in decided) if decided else ""),
            "missed_positive_best_edit": sum(
                not r["would_choose_edit"] and r["actual_delta_G"] > 1e-8 for r in chosen),
            "selected_harmful_best_edit": sum(
                r["would_choose_edit"] and r["actual_delta_G"] < -1e-8 for r in chosen),
        })
    return out


def run_counterfactual(args, p: Protocol):
    references = parse_names(args.references)
    if not set(references).issubset({"keep", "greedy", "periodic", "model"}):
        raise ValueError("References must be keep,greedy,periodic,model")
    if args.states_per_group < 0:
        raise ValueError("states-per-group must be >= 0 (0 means all)")
    if args.action_mode not in ("all", "top_q", "top_gain"):
        raise ValueError("Bad action mode")
    if args.continuation not in ("model", "keep", "greedy", "periodic"):
        raise ValueError("Bad continuation policy")
    rows = []
    state_rows = []
    for idx, (case, scenario, seed, budget, instance) in enumerate(grid(p), start=1):
        for reference in references:
            prefix_actions, eligible = trace_reference(instance, budget, p, reference)
            selected_steps = choose_steps(eligible, args.states_per_group)
            for step in selected_steps:
                # Reconstruct one identical reachable state from the SAME public history.
                env = replay_prefix(instance, budget, p, prefix_actions[:step])
                ob = env._observation()
                q, best_q, model_choice = score_q(p.model, ob)
                legal_edits = (np.flatnonzero(ob.mask[1:]) + 1).tolist()
                assert legal_edits and best_q is not None
                state_id = f"{case.stem}|{scenario}|{seed}|B{budget}|{reference}|t{step}"
                if args.action_mode == "all":
                    action_ids = legal_edits
                elif args.action_mode == "top_q":
                    action_ids = [best_q]
                else:
                    action_ids = [max(legal_edits,
                        key=lambda i: (float(ob.action_features[i, 17]), -i))]
                # The two branches use the SAME selected suffix policy rule, but
                # after the initial intervention their subsequent STATES may differ.
                keep_env = replay_prefix(instance, budget, p, prefix_actions[:step])
                keep_result = finish_from(keep_env, args.continuation, p.model, first_action=0)
                state_results = []
                for action in action_ids:
                    edit_env = replay_prefix(instance, budget, p, prefix_actions[:step])
                    new_ob = edit_env._observation()
                    if (not np.array_equal(new_ob.mask, ob.mask)
                            or not np.allclose(new_ob.global_features, ob.global_features, atol=1e-7)
                            or not np.allclose(new_ob.action_features, ob.action_features, atol=1e-7)):
                        raise AssertionError("Reference prefix replay failed to reproduce original observation")
                    edit_result = finish_from(edit_env, args.continuation, p.model,
                                              first_action=int(action))
                    delta = edit_result["G_scaled"] - keep_result["G_scaled"]
                    wait_gain = keep_result["wait"] - edit_result["wait"]
                    if abs(delta * p.env["reward_scale"] - wait_gain) > 1e-5 * max(1., abs(wait_gain)):
                        raise AssertionError("Reward difference and actual wait difference disagree")
                    gap = float(q[action] - q[0])
                    estimated_gain = float(ob.action_features[action, 17])*p.env["reward_scale"]
                    row = dict(
                        state_id=state_id, case=str(case), scenario=scenario, seed=seed, budget=budget,
                        reference=reference, step=step, event_type=ob.event_type,
                        event_time=float(ob.time), remaining_budget=int(ob.remaining_budget),
                        n_legal_edits=len(legal_edits), selected_action_index=int(action),
                        selected_vessel=int(ob.candidate_ids[action-1]),
                        action_mode=args.action_mode, continuation=args.continuation,
                        keep_margin=float(p.model.keep_margin),
                        q_keep=float(q[0]), q_edit=float(q[action]), q_gap=gap,
                        is_best_predicted_edit=int(action == best_q),
                        model_selected_action=model_choice,
                        would_choose_edit=int(action == best_q and gap > p.model.keep_margin),
                        point_gain_unscaled=estimated_gain,
                        realized_keep_G=keep_result["G_scaled"],
                        realized_edit_G=edit_result["G_scaled"],
                        actual_delta_G=delta,
                        actual_gain_wait_units=wait_gain,
                        qgap_minus_realized_delta=gap-delta,
                        keep_wait=keep_result["wait"], edit_wait=edit_result["wait"],
                        keep_final_spent=keep_result["spent"],
                        edit_final_spent=edit_result["spent"],
                        predicted_sign=sign(gap), realized_sign=sign(delta),
                        missed_beneficial_edit=int(action == best_q and gap <= p.model.keep_margin and delta > 1e-8),
                        harmful_edit_selected=int(action == model_choice and delta < -1e-8),
                    )
                    rows.append(row)
                    state_results.append(row)
                # The actual best evaluated alternative includes Keep as zero gain.
                best_observed = max(state_results, key=lambda r: r["actual_delta_G"])
                state_rows.append(dict(
                    state_id=state_id,case=str(case),scenario=scenario,seed=seed,budget=budget,
                    reference=reference,step=step,n_legal_edits=len(legal_edits),
                    evaluated_edits=len(action_ids),action_mode=args.action_mode,
                    predicted_best_edit=best_q,model_selected_action=model_choice,
                    best_observed_evaluated_edit=int(best_observed["selected_action_index"]),
                    best_evaluated_actual_delta_G=best_observed["actual_delta_G"],
                    any_evaluated_edit_beneficial=int(best_observed["actual_delta_G"] > 1e-8),
                    all_edits_exhaustive=int(len(action_ids) == len(legal_edits)),
                ))
        print(f"[counterfactual] {idx} grid groups complete / "
              f"{len(p.cases)*len(p.scenarios)*len(p.seeds)*len(p.budgets)}", flush=True)
    if not rows:
        raise ValueError("No actionable states found for counterfactual diagnostics")
    write_csv(args.out / "counterfactual_pairs.csv", rows)
    write_csv(args.out / "counterfactual_states.csv", state_rows)
    write_csv(args.out / "counterfactual_summary.csv", counterfactual_rows_summary(rows))
    print(f"[counterfactual] {len(rows)} action pairs across {len(state_rows)} states")
    return len(rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("margin", "counterfactual"):
        p = sub.add_parser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("--checkpoint", required=True, help="Frozen CC-D3QN best.pt")
        p.add_argument("--manifest", required=True, help="Matching training manifest.json")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--cases", nargs="+", default=None, help="Override validation case files (smoke test)")
        p.add_argument("--scenarios", default=None)
        p.add_argument("--seeds", default=None)
        p.add_argument("--budgets", default=None)
        p.add_argument("--overwrite", action="store_true")
        if name == "margin":
            p.add_argument("--margins", default=DEFAULT_MARGINS)
        else:
            p.add_argument("--references", default="keep,periodic")
            p.add_argument("--continuation", choices=("model", "keep", "greedy", "periodic"), default="model")
            p.add_argument("--states-per-group", type=int, default=2,
                           help="Chosen evenly among eligible reference-trajectory states; 0=all")
            p.add_argument("--action-mode", choices=("top_q", "top_gain", "all"), default="top_q")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory not empty: {args.out} (use --overwrite)")
    args.out.mkdir(parents=True, exist_ok=True)
    p = setup(args)
    if args.command == "margin":
        n = run_margin(args, p)
    else:
        n = run_counterfactual(args, p)
    manifest = {
        "experiment": args.command,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": p.checkpoint_sha,
        "training_manifest": str(Path(args.manifest).resolve()),
        "training_manifest_sha256": p.manifest_sha,
        "training_margin": p.frozen_margin,
        "evaluation_args": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()},
        "environment": p.env,
        "cases": [str(x) for x in p.cases],
        "scenarios": p.scenarios, "seeds": p.seeds, "budgets": p.budgets,
        "n_records": n,
        "interpretation": (
            "Margin sweep modifies INFERENCE only; not a newly trained policy."
            if args.command == "margin" else
            "Counterfactual returns are for one realized ETA stream and the stated fixed suffix "
            "policy. Q is trained for its own greedy policy with keep margin; outcomes under "
            "other suffix policies are diagnostic, NOT an unbiased estimate of Q^star."
        ),
    }
    (args.out / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[complete] {args.command}: {n} records -> {args.out}")


if __name__ == "__main__":
    main()
