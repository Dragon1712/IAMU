#!/usr/bin/env python3
"""IAMU independent, paired, multi-training-seed test. No training, no checkpoint tuning.

Run from iamu_d3dqn root AFTER all model checkpoints are complete.
Cases 10..14 and ETA seeds 500001..500005 are held out from 3..7/8..9.
The output is a single frozen test evaluation; NEVER use it for checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

MODELS = {
    "ddqn96": ("ddqn", 96),
    "ddqn130": ("ddqn", 130),
    "d3qn96": ("d3qn", 96),
    "set_ddqn96": ("set_ddqn", 96),
    "set_d3qn96": ("set_d3qn", 96),
}
MODEL_SETS = {
    "core": ("ddqn96", "ddqn130", "set_ddqn96"),
    "all": tuple(MODELS),
}
REFERENCE = ("keep", "greedy", "periodic")
# Existing seed-42 folders are preserved. Every new folder is architecture + seed specific.
SEED42_FOLDERS = {
    "ddqn96": "v082_8workers_sync1_300",
    "ddqn130": "v09a_ddqn130_seed42_300",
    "d3qn96": "v09a_d3qn_seed42_300",
    "set_ddqn96": "v09b_set_ddqn_seed42_300",
    "set_d3qn96": "v09c_set_d3qn_seed42_300",
}


def run_folder(label: str, seed: int) -> str:
    if label not in MODELS:
        raise ValueError(f"Unsupported model: {label}")
    if seed == 42:
        return SEED42_FOLDERS[label]
    prefixes = {
        "ddqn96": "v082_8workers_sync1_seed",
        "ddqn130": "v09a_ddqn130_seed",
        "d3qn96": "v09a_d3qn_seed",
        "set_ddqn96": "v09b_set_ddqn_seed",
        "set_d3qn96": "v09c_set_d3qn_seed",
    }
    return f"{prefixes[label]}{seed}_300"


def parse_ints(x: str) -> list[int]:
    result = []
    for item in x.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, end = map(int, item.split(":"))
            if end < start or end - start > 10000:
                raise ValueError(f"Invalid seed range: {item}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise ValueError("Expected distinct nonempty seeds")
    return result


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No data for {path}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def numeric_mean(vals) -> float:
    v = list(vals)
    return float(statistics.mean(v)) if v else float("nan")


def cvar90_positive_regret(values) -> float:
    """Upper 10% expected shortfall, fraction-weighted (not ceiling based)."""
    arr = sorted((max(0.0, float(x)) for x in values), reverse=True)
    if not arr:
        return float("nan")
    mass = len(arr) * 0.10
    remaining = mass
    total = 0.0
    for x in arr:
        w = min(remaining, 1.0)
        total += w * x
        remaining -= w
        if remaining <= 1e-12:
            break
    return total / mass


def case_hashes(train: list[str], val: list[str], test: list[str]):
    groups = []
    for name, cases in (("train", train), ("val", val), ("test", test)):
        if not cases:
            raise ValueError(f"Missing {name} cases")
        group = {}
        for c in cases:
            p = Path(c)
            if not p.is_file():
                raise FileNotFoundError(p)
            digest = hash_file(p)
            if digest in group.values():
                raise ValueError(f"Duplicate {name} case content: {p}")
            group[str(p.resolve())] = digest
        groups.append(group)
    for x in range(3):
        for y in range(x + 1, 3):
            if set(groups[x].values()) & set(groups[y].values()):
                raise ValueError("Train, validation, and independent test contain identical case bytes")
    return dict(zip(("train", "val", "test"), groups))


def protocol_from_args(cfg: dict) -> dict:
    """Exact environment settings used by 42; also the accepted cross-run comparison signature."""
    def val(key, default=None):
        if key not in cfg and default is None:
            raise KeyError(key)
        return cfg.get(key, default)
    return {
        "train_cases": [hash_file(Path(x)) for x in val("train_cases")],
        "val_cases": [hash_file(Path(x)) for x in val("val_cases")],
        "episodes": int(val("episodes")),
        "workers": int(val("workers")),
        "validation_workers": int(val("validation_workers")),
        "prefetch_per_worker": int(val("prefetch_per_worker")),
        "sync_every": int(val("sync_every")),
        "scenarios": str(val("scenarios")),
        "budgets": str(val("budgets")),
        "train_seed_start": int(val("train_seed_start")),
        "val_seeds": str(val("val_seeds")),
        "uncertainties": str(val("uncertainties")),
        "validation_uncertainty": float(val("validation_uncertainty")),
        "updates": int(val("updates")),
        "clock": str(val("clock")),
        "period": float(val("period")),
        "candidate_limit": int(val("candidate_limit")),
        "reward_scale": float(val("reward_scale")),
        "candidate_screen": str(val("candidate_screen")),
        "min_gain": float(val("min_gain")),
        "budget_features": str(val("budget_features")),
        "value_features": str(val("value_features")),
        "keep_margin": float(val("keep_margin")),
        "greedy_exploration_prob": float(val("greedy_exploration_prob")),
        "lr": float(val("lr")),
        "gamma": float(val("gamma")),
        "replay_capacity": int(val("replay_capacity")),
        "batch_size": int(val("batch_size")),
        "warmup": int(val("warmup")),
        "updates_per_step": float(val("updates_per_step")),
        "max_updates_per_episode": int(val("max_updates_per_episode")),
        "target_update": int(val("target_update")),
        "epsilon_start": float(val("epsilon_start")),
        "epsilon_end": float(val("epsilon_end")),
        "epsilon_decay_episodes": int(val("epsilon_decay_episodes")),
        "eval_every": int(val("eval_every")),
    }


def validation_best_score(curve_path: Path) -> tuple[int, float]:
    curve = load_csv(curve_path)
    if len(curve) != 12 or {int(r["completed"]) for r in curve} != set(range(25, 301, 25)):
        raise ValueError(f"Expected 12 completed validations in {curve_path}")
    if any(int(r["evaluated_runs"]) != 216 for r in curve):
        raise ValueError(f"Not all validation checkpoints used 216 runs: {curve_path}")
    winner = min(curve, key=lambda r: (float(r["mean_wait"]), int(r["completed"])))
    return int(winner["completed"]), float(winner["mean_wait"])


def preflight(results: Path, labels: tuple[str, ...], train_seeds: list[int],
              test_cases: list[str], eta_seeds: list[int], full: bool = True):
    """Validate filenames/manifests without consulting independent test outcomes."""
    expected = set()
    selected = []
    ref_sig = None
    ref_train = ref_val = None
    for seed in train_seeds:
        for label in labels:
            arch, width = MODELS[label]
            folder = results / run_folder(label, seed)
            manifest_path = folder / "manifest.json"
            best = folder / "best.pt"
            curve = folder / "validation_curve.csv"
            perf = folder / "performance.json"
            for f in (manifest_path, best, curve, perf):
                if not f.is_file():
                    raise FileNotFoundError(f"Missing local checkpoint artifact: {f}")
            conf = json.loads(manifest_path.read_text(encoding="utf-8"))
            args = conf["training_args"]
            observed_arch = args.get("q_network", "ddqn")
            if args.get("seed") != seed or observed_arch != arch or int(args["hidden_dim"]) != width:
                raise ValueError(f"Wrong architecture/seed/width in {folder}/manifest.json")
            status = json.loads(perf.read_text(encoding="utf-8"))
            if (status.get("completed_episodes") != 300 or status.get("completed_validations") != 12 or
                    status.get("total_validated_runs") != 2592):
                raise ValueError(f"Incomplete training/validation: {folder}")
            signature = protocol_from_args(args)
            if ref_sig is None:
                ref_sig = signature
                ref_train = list(args["train_cases"])
                ref_val = list(args["val_cases"])
            elif signature != ref_sig:
                raise ValueError(f"Inconsistent training protocol: {folder}")
            selected_episode, selected_score = validation_best_score(curve)
            item = {
                "label": label, "training_seed": seed,
                "architecture": arch, "hidden_dim": width,
                "checkpoint": str(best.resolve()),
                "checkpoint_sha256": hash_file(best) if full else "",
                "training_manifest": str(manifest_path.resolve()),
                "training_manifest_sha256": hash_file(manifest_path),
                "selected_episode": selected_episode,
                "selected_validation_mean_wait": selected_score,
            }
            selected.append(item)
            expected.add((label, seed))
    if len(expected) != len(labels) * len(train_seeds):
        raise ValueError("Repeated model/seed mapping")
    hashes = case_hashes(ref_train, ref_val, test_cases)
    tr_seeds = set(range(ref_sig["train_seed_start"],
                         ref_sig["train_seed_start"] + ref_sig["episodes"]))
    from iamu_dynamic.rl_experiment import parse_ints as _parse_seeds
    val_seeds = set(_parse_seeds(ref_sig["val_seeds"]))
    if (set(eta_seeds) & (val_seeds | tr_seeds)) or (val_seeds & tr_seeds):
        raise ValueError("Independent test ETA seeds overlap training or validation streams")
    # Validate full vs experimental protocol before ever evaluating on test data.
    if (ref_sig["scenarios"] != "shock,repeated,small" or
            ref_sig["budgets"] != "2,5,10" or
            ref_sig["clock"] != "union" or
            abs(ref_sig["gamma"] - 1.0) > 1e-12):
        raise ValueError("Unexpected experimental protocol; inspect before conducting test")
    return selected, ref_sig, hashes


# These globals are initialized inside each CPU-only worker (spawn context).
_WORKER = {}


def init_worker(selected: list[dict], signature: dict):
    import torch
    from iamu_dynamic.rl_ddqn import DoubleDQNAgent
    torch.set_num_threads(1)
    agents = {}
    for record in selected:
        agent, meta = DoubleDQNAgent.from_checkpoint(record["checkpoint"], device="cpu")
        if agent.q_network != record["architecture"] or agent.hidden_dim != record["hidden_dim"]:
            raise ValueError("Checkpoint architecture differs from manifest")
        if agent.keep_margin != signature["keep_margin"]:
            raise ValueError("Checkpoint Keep margin differs from training protocol")
        if int(meta.get("selected_episode", -1)) != record["selected_episode"]:
            raise ValueError("Checkpoint selected episode differs from validation winner")
        if abs(float(meta.get("selection_score", float("nan"))) -
               record["selected_validation_mean_wait"]) > 1e-6:
            raise ValueError("Checkpoint score does not match the best validation curve")
        agent.online.eval()
        agents[(record["label"], record["training_seed"])] = agent
    _WORKER.clear()
    _WORKER.update({"agents": agents, "sig": signature, "case_cache": {}})


def eval_group(task: tuple[str, str, int, int]) -> list[dict]:
    """One exact paired test problem; one scenario realization shared by ALL models."""
    from iamu_dynamic import generate_scenario, read_case
    from iamu_dynamic.rl_experiment import run_episode
    case, scenario, eta_seed, budget = task
    cache = _WORKER["case_cache"]
    if case not in cache:
        cache[case] = read_case(case)
    vessels, port = cache[case]
    sig = _WORKER["sig"]
    instance = generate_scenario(vessels, port, label=scenario, seed=eta_seed,
                                 updates_per_vessel=sig["updates"],
                                 uncertainty=sig["validation_uncertainty"])
    kwargs = dict(candidate_limit=sig["candidate_limit"], decision_clock=sig["clock"],
                  period=sig["period"], reward_scale=sig["reward_scale"],
                  candidate_screen=sig["candidate_screen"], min_gain=sig["min_gain"],
                  budget_features=sig["budget_features"], value_features=sig["value_features"])
    rows = []
    for name in REFERENCE:
        result = run_episode(instance, budget, name, **kwargs)
        rows.append(dict(case=Path(case).stem, scenario=scenario,
                         eta_seed=eta_seed, budget=budget, model=name,
                         training_seed="baseline", architecture="baseline",
                         wait=float(result["wait"]), spent=int(result["spent"]),
                         steps=int(result["steps"]),
                         edits_eta=int(result["edits_by_eta"]),
                         edits_clock=int(result["edits_by_clock"])))
    for (name, seed), agent in sorted(_WORKER["agents"].items()):
        result = run_episode(instance, budget, "ddqn", agent=agent, **kwargs)
        rows.append(dict(case=Path(case).stem, scenario=scenario,
                         eta_seed=eta_seed, budget=budget, model=name,
                         training_seed=seed, architecture=agent.q_network,
                         wait=float(result["wait"]), spent=int(result["spent"]),
                         steps=int(result["steps"]),
                         edits_eta=int(result["edits_by_eta"]),
                         edits_clock=int(result["edits_by_clock"])))
    return rows


def group_key(r: dict) -> tuple:
    return (str(r["case"]), str(r["scenario"]), int(r["eta_seed"]), int(r["budget"]))


def enrich_rows(rows: list[dict], labels: tuple[str, ...], train_seeds: list[int]):
    expected = len(REFERENCE) + len(labels) * len(train_seeds)
    grouped = defaultdict(list)
    for r in rows:
        grouped[group_key(r)].append(r)
    model_keys = {("keep", "baseline"), ("greedy", "baseline"),
                  ("periodic", "baseline")} | {(m, s) for m in labels for s in train_seeds}
    for key, group in grouped.items():
        if len(group) != expected or {(r["model"], r["training_seed"]) for r in group} != model_keys:
            raise ValueError(f"Missing or duplicate policy evaluations for {key}")
        base = {r["model"]: r for r in group if r["model"] in REFERENCE}
        for r in group:
            wait = float(r["wait"])
            r["gain_vs_keep"] = float(base["keep"]["wait"]) - wait
            r["gain_vs_periodic"] = float(base["periodic"]["wait"]) - wait
            r["regret_vs_keep"] = max(0.0, wait - float(base["keep"]["wait"]))
            r["regret_vs_periodic"] = max(0.0, wait - float(base["periodic"]["wait"]))
    return grouped


def sum_rows(rows: list[dict], labels: tuple[str, ...], train_seeds: list[int]):
    groups = defaultdict(list)
    for r in rows:
        for scenario, budget in (("ALL", "ALL"), (r["scenario"], "ALL"),
                                 ("ALL", r["budget"]), (r["scenario"], r["budget"])):
            groups[(r["model"], str(r["training_seed"]), scenario, str(budget))].append(r)
    result = []
    for (model, seed, scenario, budget), a in sorted(groups.items()):
        if model not in labels and model not in REFERENCE:
            continue
        result.append(dict(model=model, training_seed=seed, scenario=scenario,
            budget=budget, n=len(a), mean_wait=numeric_mean(float(x["wait"]) for x in a),
            mean_spent=numeric_mean(int(x["spent"]) for x in a),
            no_edit_fraction=numeric_mean(int(x["spent"]) == 0 for x in a),
            mean_gain_vs_keep=numeric_mean(float(x["gain_vs_keep"]) for x in a),
            mean_gain_vs_periodic=numeric_mean(float(x["gain_vs_periodic"]) for x in a),
            cvar90_regret_vs_keep=cvar90_positive_regret(x["regret_vs_keep"] for x in a),
            cvar90_regret_vs_periodic=cvar90_positive_regret(x["regret_vs_periodic"] for x in a),
            worse_periodic=sum(float(x["gain_vs_periodic"]) < -1e-7 for x in a)))
    return result


def get_contrasts(labels: tuple[str, ...]) -> list[tuple[str, str]]:
    options = (("set_ddqn96", "ddqn96"), ("set_ddqn96", "ddqn130"),
               ("set_d3qn96", "d3qn96"), ("set_d3qn96", "set_ddqn96"))
    return [(a,b) for a,b in options if a in labels and b in labels]


def contrast_rows(rows: list[dict], labels: tuple[str, ...], seeds: list[int]):
    table = {(group_key(r), r["model"], str(r["training_seed"])):r for r in rows}
    group_keys = sorted(set(group_key(r) for r in rows))
    diffs = []
    contrast_summary = []
    for model, baseline in get_contrasts(labels):
        for seed in seeds:
            for key in group_keys:
                a = table[(key, model, str(seed))]
                b = table[(key, baseline, str(seed))]
                diffs.append(dict(case=key[0],scenario=key[1],eta_seed=key[2],budget=key[3],
                                  training_seed=seed, contrast=f"{model} - {baseline}",
                                  wait_diff=float(a["wait"]) - float(b["wait"]),
                                  spent_diff=int(a["spent"]) - int(b["spent"])))
    for (comparison, seed), a in sorted(_group_by(diffs, lambda r:(r["contrast"],r["training_seed"])).items()):
        for scenario, budget in (("ALL", "ALL"), ("shock", "ALL"), ("repeated", "ALL"),
                                 ("small", "ALL"), ("ALL", 2), ("ALL", 5), ("ALL", 10)):
            z = [r for r in a if (scenario == "ALL" or r["scenario"] == scenario) and
                 (budget == "ALL" or r["budget"] == budget)]
            d = [float(r["wait_diff"]) for r in z]
            contrast_summary.append(dict(contrast=comparison,training_seed=seed,
                scenario=scenario,budget=budget,n=len(z),mean_wait_diff=numeric_mean(d),
                mean_spent_diff=numeric_mean(float(r["spent_diff"]) for r in z),
                n_set_better=sum(v < -1e-7 for v in d),n_set_worse=sum(v > 1e-7 for v in d),
                n_tie=sum(abs(v) <= 1e-7 for v in d)))
    return diffs, contrast_summary


def _group_by(rows, key):
    z = defaultdict(list)
    for r in rows:
        z[key(r)].append(r)
    return z


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    h = (len(xs) - 1) * q
    k = int(h)
    w = h - k
    return (1 - w) * xs[k] + w * xs[min(len(xs) - 1, k + 1)]


def bootstrap_contrasts(diffs: list[dict], seed: int = 20261114, reps: int = 4000):
    """Exploratory hierarchical bootstrap: resample training seeds and test case×ETA-seed blocks.

    Exactly 3 training seeds: CIs reflect model uncertainty imperfectly. No p-value.
    Scenario and budget observations from the same case and ETA seed stay grouped.
    """
    rng = random.Random(seed)
    result = []
    for name, ar in sorted(_group_by(diffs, lambda r: r["contrast"]).items()):
        seeds = sorted(set(int(r["training_seed"]) for r in ar))
        blocks = sorted(set((r["case"], int(r["eta_seed"])) for r in ar))
        matrix = {}
        for s in seeds:
            for block in blocks:
                z = [float(r["wait_diff"]) for r in ar if r["training_seed"] == s and
                     (r["case"], int(r["eta_seed"])) == block]
                if not z:
                    raise ValueError("Missing paired test block")
                matrix[(s, block)] = statistics.mean(z)
        per_seed = {s:statistics.mean(matrix[(s,b)] for b in blocks) for s in seeds}
        overall = statistics.mean(per_seed.values())
        simulations = []
        for _ in range(reps):
            sampled_seeds = [rng.choice(seeds) for _ in seeds]
            sampled_blocks = [rng.choice(blocks) for _ in blocks]
            simulations.append(statistics.mean(matrix[(s,b)] for s in sampled_seeds
                                               for b in sampled_blocks))
        result.append(dict(contrast=name, n_training_seeds=len(seeds),
                           n_test_blocks=len(blocks),mean_wait_diff=overall,
                           sd_of_training_seed_means=statistics.stdev(per_seed.values()) if len(seeds)>1 else 0.,
                           seed_mean_min=min(per_seed.values()),seed_mean_max=max(per_seed.values()),
                           seed_wins=sum(x < 0 for x in per_seed.values()),
                           exploratory_bootstrap_ci95_low=percentile(simulations,.025),
                           exploratory_bootstrap_ci95_high=percentile(simulations,.975),
                           bootstrap_replicates=reps))
    return result


def test_main(args):
    labels = MODEL_SETS[args.mode]
    train_seeds = parse_ints(args.train_seeds)
    eta_seeds = parse_ints(args.eta_seeds)
    test_cases = [str(Path(x)) for x in args.test_cases]
    selected, sig, hashes = preflight(Path(args.results_root),labels,train_seeds,test_cases,eta_seeds)
    if args.dry_run:
        print(json.dumps({"models":list(labels),"training_seeds":train_seeds,
              "test_cases":test_cases,"eta_seeds":eta_seeds,
              "expected_groups":len(test_cases)*len(sig["scenarios"].split(","))*len(eta_seeds)*3,
              "checkpoints":selected},indent=2,ensure_ascii=False))
        return
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}; use a NEW output directory")
    out.mkdir(parents=True,exist_ok=True)
    expected_groups = len(test_cases) * 3 * len(eta_seeds) * 3
    tasks = [(x,sc,eta_seed,b) for x in test_cases for sc in ("shock","repeated","small")
             for eta_seed in eta_seeds for b in (2,5,10)]
    results=[]
    if args.workers == 1:
        init_worker(selected,sig)
        for i,task in enumerate(tasks,1):
            results.extend(eval_group(task))
            if i % 5 == 0 or i == expected_groups:
                print(f"[test] {i}/{expected_groups} paired groups",flush=True)
    else:
        # Process isolation: CPU model copies are never trained or mutated.
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),
             initializer=init_worker,initargs=(selected,sig)) as pool:
            for i,rows in enumerate(pool.map(eval_group,tasks,chunksize=1),1):
                results.extend(rows)
                if i % 5 == 0 or i == expected_groups:
                    print(f"[test] {i}/{expected_groups} paired groups",flush=True)
    grouped=enrich_rows(results,labels,train_seeds)
    if len(grouped)!=expected_groups:
        raise AssertionError("Unexpected number of test instances")
    summaries=sum_rows(results,labels,train_seeds)
    diffs,contrast_summary=contrast_rows(results,labels,train_seeds)
    bootstrap=bootstrap_contrasts(diffs,reps=args.bootstrap_reps)
    save_csv(out/"independent_test_runs.csv",results)
    save_csv(out/"model_seed_summary.csv",summaries)
    save_csv(out/"paired_differences.csv",diffs)
    save_csv(out/"paired_seed_summary.csv",contrast_summary)
    save_csv(out/"architecture_bootstrap.csv",bootstrap)
    source_files = ["train_budgeted_ddqn_parallel.py", "parallel_validation_v081.py",
                    "iamu_dynamic/rl_ddqn.py", "iamu_dynamic/rl_env.py",
                    "iamu_dynamic/rl_experiment.py", "independent_multiseed_eval.py"]
    code_sha256 = {name: hash_file(Path(name)) if Path(name).is_file() else None
                   for name in source_files}
    git_head = subprocess.run(["git", "rev-parse", "HEAD"],
                              text=True, capture_output=True, check=False)
    repo_commit = git_head.stdout.strip() if git_head.returncode == 0 else "unknown"
    import torch
    (out/"audit_manifest.json").write_text(json.dumps(dict(
        repo_commit=repo_commit,code_sha256=code_sha256,
        python_version=sys.version,torch_version=torch.__version__,
        test_is_independent=True, no_test_selection=True,
        warning="Do not use this test to select checkpoints, thresholds or architecture variants.",
        models=list(labels),training_seeds=train_seeds,test_eta_seeds=eta_seeds,
        test_cases=test_cases,test_case_sha256=hashes["test"],
        train_case_sha256=hashes["train"],validation_case_sha256=hashes["val"],
        training_protocol=sig,checkpoints=selected,
        test_groups=expected_groups,total_test_runs=len(results),
        procedure="Frozen best.pt on independent case contents and ETA seeds; only CPU inference.",
    ),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"[DONE] {expected_groups} paired groups, {len(results)} runs -> {out}")
    for row in bootstrap:
        print(f"{row['contrast']}: mean Δwait={row['mean_wait_diff']:.4f}; "
              f"seed wins={row['seed_wins']}/{row['n_training_seeds']}; "
              f"exploratory bootstrap95=[{row['exploratory_bootstrap_ci95_low']:.4f}, "
              f"{row['exploratory_bootstrap_ci95_high']:.4f}]")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=tuple(MODEL_SETS),default="core")
    p.add_argument("--train-seeds",default="42,43,44")
    p.add_argument("--eta-seeds",default="500001:500005")
    p.add_argument("--test-cases",nargs="+",default=[f"../2024LiERLDDE/case_code/{i}.txt" for i in range(10,15)])
    p.add_argument("--results-root",default="results")
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--bootstrap-reps",type=int,default=4000)
    p.add_argument("--out",default="results/v10_independent_multiseed_test_core")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args(argv)
    if not 1<=args.workers<=32 or args.bootstrap_reps<100:
        p.error("Invalid workers or bootstrap-reps")
    test_main(args)


if __name__ == "__main__":
    main()
