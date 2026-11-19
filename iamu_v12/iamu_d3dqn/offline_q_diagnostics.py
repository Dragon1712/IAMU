#!/usr/bin/env python3
"""IAMU three-model frozen-checkpoint offline Q diagnostics (NO gradient updates).

Run from the iamu_d3dqn project root. Requires local, trusted best.pt weights.
Experiments: original DDQN-96; width-matched DDQN-130; masked-mean CC-D3QN-96.

Two *different* diagnostics are intentionally separated:
  (1) common reference-policy observations: evaluate all three Q networks on
      EXACTLY THE SAME observations, without allowing their decisions to change
      those reference trajectories;
  (2) on-policy deterministic rollouts: each model actually acts; compare
      realized waiting and single-realization Monte-Carlo return-to-go against
      the Q value of the chosen action.

This script never calls .update(), never modifies model weights, and does not
peek at hidden arrivals to construct observations or choose actions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from iamu_dynamic import generate_scenario, read_case
from iamu_dynamic.rl_ddqn import DoubleDQNAgent
from iamu_dynamic.rl_env import IAMUEventEnv
from iamu_dynamic.rl_experiment import parse_ints, parse_names

MODEL_SPECS = {"ddqn96": ("ddqn", 96), "ddqn130": ("ddqn", 130), "d3qn96": ("d3qn", 96)}
REF_POLICIES = ("keep", "greedy", "periodic")
ENV_OPTIONS = (
    "candidate_limit", "period", "reward_scale", "candidate_screen",
    "min_gain", "budget_features", "value_features", "updates",
    "validation_uncertainty", "clock",
)
STATE_FIELDS = (
    "state_id", "source", "reference", "model", "case", "scenario", "seed", "budget", "step",
    "time", "event_type", "remaining_budget", "num_legal", "num_edits", "keep_only",
    "q_keep", "q_best_edit", "q_gap", "q_chosen", "value", "adv_keep", "adv_best_edit",
    "network_action", "network_vessel", "executed_action", "executed_vessel", "estimated_gain",
)
ACTION_FIELDS = (
    "state_id", "source", "reference", "model", "case", "scenario", "seed", "budget", "step",
    "time", "event_type", "remaining_budget", "action", "vessel_id", "q_raw", "q_minus_keep",
    "selection_score", "adv_raw", "adv_centered", "value", "point_gain", "network_selected", "executed",
)
EPISODE_FIELDS = (
    "case", "scenario", "seed", "budget", "policy", "source", "wait", "spent", "steps",
    "edits_eta", "edits_clock", "gain_vs_keep", "gain_vs_periodic", "regret_vs_keep",
    "regret_vs_periodic",
)
COMMON_FIELDS = (
    "state_id", "reference", "case", "scenario", "seed", "budget", "step", "event_type",
    "num_edits", "ddqn96_action", "ddqn130_action", "d3qn96_action", "ddqn96_q_gap",
    "ddqn130_q_gap", "d3qn96_q_gap", "all_agree", "ddqn96_ddqn130_agree",
    "ddqn96_d3qn96_agree", "ddqn130_d3qn96_agree",
)
MC_FIELDS = (
    "state_id", "model", "case", "scenario", "seed", "budget", "step", "event_type",
    "chosen_action", "q_chosen", "mc_return_scaled", "error_q_minus_mc", "abs_error",
)


class CsvSink:
    def __init__(self, path: Path, fields: tuple[str, ...]):
        self.file = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields, extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def describe(values):
    z = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))], dtype=float)
    if not len(z):
        return {"n": 0, "mean": "", "sd": "", "p10": "", "p50": "", "p90": ""}
    return {
        "n": len(z), "mean": float(z.mean()),
        "sd": float(z.std(ddof=1)) if len(z) > 1 else 0.,
        "p10": float(np.percentile(z, 10)), "p50": float(np.percentile(z, 50)),
        "p90": float(np.percentile(z, 90)),
    }


def cvar90_positive_regret(values):
    """Fractional top-10% average of max(0, regret); descriptive, not a CI."""
    z = sorted((max(0., float(v)) for v in values), reverse=True)
    if not z:
        return ""
    mass, remain, weighted_sum = .1 * len(z), .1 * len(z), 0.
    for v in z:
        w = min(1., remain)
        weighted_sum += v * w
        remain -= w
        if remain <= 1e-12:
            break
    return weighted_sum / mass


def checksum(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(manifests, args):
    """Require three training runs to share all environment / evaluation settings."""
    present = {n: m for n, m in manifests.items() if m is not None}
    if len(present) == 3:
        baseline = present["ddqn96"]["training_args"]
        for model, manifest in present.items():
            cfg = manifest["training_args"]
            differences = [key for key in ENV_OPTIONS if cfg.get(key) != baseline.get(key)]
            if differences:
                raise ValueError(f"{model} manifest differs in evaluation fields: {differences}")
            if cfg.get("val_seeds") != baseline.get("val_seeds"):
                raise ValueError(f"{model} manifest has different validation seeds")
        print("[protocol] all three model manifests match on evaluation settings", flush=True)
    else:
        print("[protocol] some manifest.json files are missing; verify checkpoint provenance manually", flush=True)
    if args.reference_policies and "periodic" in args.reference_policies and args.clock != "union":
        raise ValueError("Periodic reference policy requires --clock union")


def score_observation(model: DoubleDQNAgent, ob):
    """Return only legal-action values and a deterministic epsilon=0 decision."""
    with torch.inference_mode():
        g = torch.as_tensor(ob.global_features, dtype=torch.float32, device=model.device)[None, :]
        a = torch.as_tensor(ob.action_features, dtype=torch.float32, device=model.device)[None, :, :]
        mask_t = torch.as_tensor(ob.mask, dtype=torch.bool, device=model.device)[None, :]
        if model.q_network == "d3qn":
            value_t, advantage_t, q_t = model.online.forward_components(g, a, mask_t)
            value = float(value_t[0, 0].item())
            advantage = advantage_t[0].cpu().numpy().astype(float)
        else:
            q_t = model.online(g, a, mask_t)
            value, advantage = None, None
        q = q_t[0].cpu().numpy().astype(float)
    legal = np.asarray(ob.mask, dtype=bool)
    if not legal[0]:
        raise RuntimeError("Keep should be legal")
    if not np.isfinite(q[legal]).all():
        raise RuntimeError("Non-finite Q prediction on a legal candidate")
    scores = np.where(legal, q, -np.inf).copy()
    scores[0] += model.keep_margin
    action = int(np.argmax(scores))
    edits = np.flatnonzero(legal[1:]) + 1
    best_edit = int(edits[np.argmax(q[edits])]) if len(edits) else None
    gap = float(q[best_edit] - q[0]) if best_edit is not None else None
    return {"q": q, "scores": scores, "a": advantage, "v": value, "action": action,
            "best_edit": best_edit, "gap": gap}


def reference_action(name, env):
    if name == "keep":
        return 0
    if name == "greedy":
        return int(env.point_greedy_action())
    if name == "periodic":
        return int(env.point_greedy_action()) if env.event_type == "clock" else 0
    raise ValueError(f"Unknown reference policy: {name}")


def create_env(instance, budget, cfg):
    return IAMUEventEnv(
        instance, budget=int(budget), candidate_limit=int(cfg["candidate_limit"]),
        decision_clock=cfg["clock"], period=float(cfg["period"]),
        reward_scale=float(cfg["reward_scale"]),
        candidate_screen=cfg["candidate_screen"], min_gain=float(cfg["min_gain"]),
        budget_features=cfg["budget_features"], value_features=cfg["value_features"],
    )


def diag_state(model_name, result, ob, env, case_name, scenario, seed, budget,
               step, state_id, source, reference, executed, state_sink, action_sink):
    legal = np.asarray(ob.mask, dtype=bool)
    num_legal = int(legal.sum())
    vals, adv = result["q"], result["a"]
    chosen = result["action"]
    best_edit = result["best_edit"]
    def vessel(idx):
        return "" if idx == 0 else int(ob.candidate_ids[idx - 1])
    est = 0. if executed == 0 else float(env._candidate_data[executed - 1]["gain"])
    row = dict(
        state_id=state_id, source=source, reference=reference or "", model=model_name,
        case=case_name, scenario=scenario, seed=seed, budget=budget, step=step,
        time=float(ob.time), event_type=ob.event_type, remaining_budget=ob.remaining_budget,
        num_legal=num_legal, num_edits=num_legal-1, keep_only=int(num_legal == 1),
        q_keep=float(vals[0]), q_best_edit=float(vals[best_edit]) if best_edit is not None else "",
        q_gap=result["gap"] if result["gap"] is not None else "",
        q_chosen=float(vals[chosen]), value=result["v"] if result["v"] is not None else "",
        adv_keep=float(adv[0]) if adv is not None else "",
        adv_best_edit=float(adv[best_edit]) if adv is not None and best_edit is not None else "",
        network_action=chosen, network_vessel=vessel(chosen),
        executed_action=executed, executed_vessel=vessel(executed), estimated_gain=est,
    )
    state_sink.write(row)
    mean_adv = float(adv[legal].mean()) if adv is not None else None
    for ix in np.flatnonzero(legal):
        ix = int(ix)
        point_gain = 0. if ix == 0 else float(env._candidate_data[ix-1]["gain"])
        action_sink.write(dict(
            state_id=state_id, source=source, reference=reference or "", model=model_name,
            case=case_name, scenario=scenario, seed=seed, budget=budget, step=step,
            time=float(ob.time), event_type=ob.event_type, remaining_budget=ob.remaining_budget,
            action=ix, vessel_id=vessel(ix), q_raw=float(vals[ix]), q_minus_keep=float(vals[ix]-vals[0]),
            selection_score=float(result["scores"][ix]),
            adv_raw=float(adv[ix]) if adv is not None else "",
            adv_centered=float(adv[ix]-mean_adv) if adv is not None else "",
            value=result["v"] if result["v"] is not None else "",
            point_gain=point_gain, network_selected=int(ix == chosen), executed=int(ix == executed),
        ))
    return row


def make_instance(vessels, port, scenario, seed, cfg):
    return generate_scenario(
        vessels, port, label=scenario, seed=seed,
        updates_per_vessel=int(cfg["updates"]), uncertainty=float(cfg["validation_uncertainty"]),
    )


def run_reference(inst, case_name, scenario, seed, budget, policy, models, cfg,
                  state_sink, action_sink, common_sink):
    env = create_env(inst, budget, cfg)
    ob = env.reset()
    step = 0
    wait = None
    eta_edits = clock_edits = 0
    while not env.done:
        state_id = f"{case_name}|{scenario}|{seed}|B{budget}|ref:{policy}|step:{step}"
        evaluations = {name: score_observation(model, ob) for name,model in models.items()}
        executed = reference_action(policy, env)
        if not bool(ob.mask[executed]):
            raise RuntimeError(f"Illegal {policy} action {executed}")
        for name, result in evaluations.items():
            diag_state(name, result, ob, env, case_name, scenario, seed, budget, step,
                       state_id, "reference", policy, executed, state_sink, action_sink)
        choices = {k: v["action"] for k,v in evaluations.items()}
        common_sink.write(dict(
            state_id=state_id, reference=policy, case=case_name, scenario=scenario,
            seed=seed, budget=budget, step=step, event_type=ob.event_type,
            num_edits=int(ob.mask.sum()) - 1,
            ddqn96_action=choices["ddqn96"], ddqn130_action=choices["ddqn130"],
            d3qn96_action=choices["d3qn96"],
            ddqn96_q_gap=evaluations["ddqn96"]["gap"] if evaluations["ddqn96"]["gap"] is not None else "",
            ddqn130_q_gap=evaluations["ddqn130"]["gap"] if evaluations["ddqn130"]["gap"] is not None else "",
            d3qn96_q_gap=evaluations["d3qn96"]["gap"] if evaluations["d3qn96"]["gap"] is not None else "",
            all_agree=int(len(set(choices.values())) == 1),
            ddqn96_ddqn130_agree=int(choices["ddqn96"] == choices["ddqn130"]),
            ddqn96_d3qn96_agree=int(choices["ddqn96"] == choices["d3qn96"]),
            ddqn130_d3qn96_agree=int(choices["ddqn130"] == choices["d3qn96"]),
        ))
        if executed:
            eta_edits += int(ob.event_type == "eta")
            clock_edits += int(ob.event_type == "clock")
        ob, _, done, info = env.step(executed)
        step += 1
        if done:
            wait = float(info["weighted_wait"])
    return dict(case=case_name,scenario=scenario,seed=seed,budget=budget,policy=policy,
                source="reference",wait=wait,spent=int(env.used),steps=step,
                edits_eta=eta_edits,edits_clock=clock_edits)


def run_model(inst, case_name, scenario, seed, budget, name, model, cfg,
              state_sink, action_sink, mc_sink):
    env = create_env(inst, budget, cfg)
    ob = env.reset()
    step = 0
    history = []
    eta_edits = clock_edits = 0
    while not env.done:
        state_id = f"{case_name}|{scenario}|{seed}|B{budget}|on:{name}|step:{step}"
        result = score_observation(model, ob)
        action = result["action"]
        diag_state(name, result, ob, env, case_name, scenario, seed, budget, step,
                   state_id, "onpolicy", "", action, state_sink, action_sink)
        if action:
            eta_edits += int(ob.event_type == "eta")
            clock_edits += int(ob.event_type == "clock")
        event_type = ob.event_type
        ob, reward, done, info = env.step(action)
        history.append(dict(state_id=state_id,model=name,case=case_name,scenario=scenario,
                            seed=seed,budget=budget,step=step,event_type=event_type,
                            chosen_action=action,q_chosen=float(result["q"][action]),
                            reward=float(reward)))
        step += 1
        if done:
            wait = float(info["weighted_wait"])
    reward_to_go = 0.
    for entry in reversed(history):
        reward_to_go += entry["reward"]  # gamma==1.0 in the controlled experiment
        error = entry["q_chosen"] - reward_to_go
        mc_sink.write(dict(state_id=entry["state_id"],model=name,case=case_name,
                           scenario=scenario,seed=seed,budget=budget,step=entry["step"],
                           event_type=entry["event_type"],chosen_action=entry["chosen_action"],
                           q_chosen=entry["q_chosen"],mc_return_scaled=reward_to_go,
                           error_q_minus_mc=error,abs_error=abs(error)))
    if abs(-sum(e["reward"] for e in history)*float(cfg["reward_scale"])-wait) > 1e-4*max(1.,wait):
        raise RuntimeError("Reward sum and realized wait differ")
    return dict(case=case_name,scenario=scenario,seed=seed,budget=budget,policy=name,
                source="onpolicy",wait=wait,spent=int(env.used),steps=step,
                edits_eta=eta_edits,edits_clock=clock_edits)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def postprocess(out: Path, episodes, do_onpolicy):
    """Produce audit-friendly summaries; never mix common and on-policy states."""
    waits = {(x["case"],x["scenario"],x["seed"],x["budget"],x["policy"]):x["wait"]
             for x in episodes}
    for e in episodes:
        base = (e["case"],e["scenario"],e["seed"],e["budget"])
        keep = waits.get((*base,"keep"))
        periodic = waits.get((*base,"periodic"))
        e["gain_vs_keep"] = keep-e["wait"] if keep is not None else ""
        e["gain_vs_periodic"] = periodic-e["wait"] if periodic is not None else ""
        e["regret_vs_keep"] = max(0.,e["wait"]-keep) if keep is not None else ""
        e["regret_vs_periodic"] = max(0.,e["wait"]-periodic) if periodic is not None else ""
    write_csv(out/"episode_outcomes.csv", EPISODE_FIELDS, episodes)

    performance = []
    grouped = defaultdict(list)
    for e in episodes:
        for scenario,budget in ((e["scenario"],e["budget"]), (e["scenario"],"ALL"),
                                ("ALL",e["budget"]), ("ALL","ALL")):
            grouped[(e["policy"],scenario,budget)].append(e)
    for (policy,scenario,budget),items in sorted(grouped.items(),key=lambda x:str(x[0])):
        regret = [x["regret_vs_periodic"] for x in items if x["regret_vs_periodic"] != ""]
        gain_keep = [x["gain_vs_keep"] for x in items if x["gain_vs_keep"] != ""]
        gain_per = [x["gain_vs_periodic"] for x in items if x["gain_vs_periodic"] != ""]
        performance.append(dict(policy=policy,scenario=scenario,budget=budget,n=len(items),
            mean_wait=mean(x["wait"] for x in items),mean_spent=mean(x["spent"] for x in items),
            mean_gain_vs_keep=mean(gain_keep) if gain_keep else "",
            mean_gain_vs_periodic=mean(gain_per) if gain_per else "",
            cvar90_regret_vs_periodic=cvar90_positive_regret(regret) if regret else "",
            worse_than_periodic=sum(float(x)>1e-6 for x in regret) if regret else ""))
    write_csv(out/"performance_summary.csv",
              ("policy","scenario","budget","n","mean_wait","mean_spent",
               "mean_gain_vs_keep","mean_gain_vs_periodic","cvar90_regret_vs_periodic",
               "worse_than_periodic"),performance)

    qstats = defaultdict(lambda:dict(count=0,keep_only=0,act=0,edit_choice=0,gaps=[],
                                       q_keep=[],values=[],adv_keep=[]))
    with (out/"q_states.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys = [(row["model"],row["source"],row["reference"],"ALL","ALL"),
                    (row["model"],row["source"],row["reference"],row["scenario"],row["budget"]) ]
            for k in keys:
                st=qstats[k];st["count"]+=1
                st["keep_only"] += int(row["keep_only"])
                if int(row["num_edits"])>0:
                    st["act"]+=1
                    st["edit_choice"]+=int(row["network_action"])!=0
                    st["gaps"].append(float(row["q_gap"]))
                st["q_keep"].append(float(row["q_keep"]))
                if row["value"]!="":st["values"].append(float(row["value"]))
                if row["adv_keep"]!="":st["adv_keep"].append(float(row["adv_keep"]))
    summaries=[]
    for (model,source,reference,scenario,budget),st in sorted(qstats.items()):
        g=describe(st["gaps"])
        summaries.append(dict(model=model,source=source,reference=reference,scenario=scenario,
                              budget=budget,states=st["count"],actionable=st["act"],
                              keep_only_fraction=st["keep_only"]/st["count"],
                              edit_choice_rate_when_actionable=st["edit_choice"]/st["act"] if st["act"] else "",
                              mean_qgap=g["mean"],median_qgap=g["p50"],
                              qgap_p10=g["p10"],qgap_p90=g["p90"],
                              mean_qkeep=mean(st["q_keep"]),
                              mean_value=mean(st["values"]) if st["values"] else "",
                              mean_raw_adv_keep=mean(st["adv_keep"]) if st["adv_keep"] else ""))
    write_csv(out/"q_summary.csv",
              ("model","source","reference","scenario","budget","states","actionable",
               "keep_only_fraction","edit_choice_rate_when_actionable","mean_qgap",
               "median_qgap","qgap_p10","qgap_p90","mean_qkeep","mean_value",
               "mean_raw_adv_keep"),summaries)

    comparison=read_csv(out/"common_state_comparison.csv")
    pairs=defaultdict(list)
    for e in comparison:
        for group in ((e["reference"],"ALL","ALL"), (e["reference"],e["scenario"],e["budget"])):
            pairs[group].append(e)
    agree=[]
    for (reference,scenario,budget),data in sorted(pairs.items()):
        actionable = [x for x in data if int(x["num_edits"]) > 0]
        def rate(rows, key):
            return mean(int(x[key]) for x in rows) if rows else ""
        agree.append(dict(reference=reference,scenario=scenario,budget=budget,n=len(data),
                          actionable=len(actionable),
                          all_agree_rate=rate(data,"all_agree"),
                          all_agree_actionable_rate=rate(actionable,"all_agree"),
                          ddqn96_ddqn130_agree_rate=rate(data,"ddqn96_ddqn130_agree"),
                          ddqn96_ddqn130_actionable_rate=rate(actionable,"ddqn96_ddqn130_agree"),
                          ddqn96_d3qn96_agree_rate=rate(data,"ddqn96_d3qn96_agree"),
                          ddqn96_d3qn96_actionable_rate=rate(actionable,"ddqn96_d3qn96_agree"),
                          ddqn130_d3qn96_agree_rate=rate(data,"ddqn130_d3qn96_agree"),
                          ddqn130_d3qn96_actionable_rate=rate(actionable,"ddqn130_d3qn96_agree")))
    write_csv(out/"choice_agreement_summary.csv",
              ("reference","scenario","budget","n","actionable","all_agree_rate",
               "all_agree_actionable_rate",
               "ddqn96_ddqn130_agree_rate","ddqn96_ddqn130_actionable_rate",
               "ddqn96_d3qn96_agree_rate","ddqn96_d3qn96_actionable_rate",
               "ddqn130_d3qn96_agree_rate","ddqn130_d3qn96_actionable_rate"),agree)

    if do_onpolicy:
        calib=defaultdict(list)
        with (out/"onpolicy_mc_returns.csv").open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k in ((row["model"],"ALL","ALL"),(row["model"],row["scenario"],row["budget"])):
                    calib[k].append(float(row["error_q_minus_mc"]))
        rows=[]
        for (model,scenario,budget),z in sorted(calib.items()):
            rows.append(dict(model=model,scenario=scenario,budget=budget,n=len(z),
                             mean_bias_q_minus_mc=float(np.mean(z)),
                             mae=float(np.mean(np.abs(z))),
                             rmse=float(np.sqrt(np.mean(np.square(z)))),
                             p90_abs_error=float(np.percentile(np.abs(z),90))))
        write_csv(out/"q_calibration_summary.csv",
                  ("model","scenario","budget","n","mean_bias_q_minus_mc","mae",
                   "rmse","p90_abs_error"),rows)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ddqn96",type=Path,default=Path("results/v082_8workers_sync1_300/best.pt"))
    p.add_argument("--ddqn130",type=Path,default=Path("results/v09a_ddqn130_seed42_300/best.pt"))
    p.add_argument("--d3qn96",type=Path,default=Path("results/v09a_d3qn_seed42_300/best.pt"))
    p.add_argument("--manifest",type=Path,default=Path("results/v09a_d3qn_seed42_300/manifest.json"))
    p.add_argument("--cases",nargs="+",default=None,help="Case TSV paths; defaults to manifest.val_cases")
    p.add_argument("--scenarios",default=None,help="Default from manifest")
    p.add_argument("--seeds",default=None,help="Example: 400001:400003")
    p.add_argument("--budgets",default=None,help="Example: 2,5,10")
    p.add_argument("--reference-policies",default="keep,greedy,periodic")
    p.add_argument("--no-onpolicy",action="store_true",help="Only shared-reference-state diagnostics")
    p.add_argument("--max-groups",type=int,default=0,help="First N validation groups (smoke testing)")
    p.add_argument("--device",default="cpu",help="Frozen network inference device")
    p.add_argument("--out",type=Path,default=Path("results/offline_q_three_seed42"))
    p.add_argument("--overwrite",action="store_true",help="Allow overwriting existing output CSVs")
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    torch.set_num_threads(1)
    if args.max_groups < 0:
        raise ValueError("--max-groups must be >= 0")
    paths={name:getattr(args,name) for name in MODEL_SPECS}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing local PyTorch weights: "+", ".join(missing)+
            ". GitHub CSV/JSON results do not contain best.pt. Pass --ddqn96/--ddqn130/--d3qn96 "+
            "pointing to the trusted SERVER files (or matching frozen validation snapshots).")
    manifest=read_manifest(args.manifest)
    if manifest is None:
        raise FileNotFoundError(f"Required evaluation protocol manifest missing: {args.manifest}")
    cfg=manifest["training_args"]
    cases=args.cases or manifest["val_cases"]
    scenarios=parse_names(args.scenarios or cfg["scenarios"])
    seeds=parse_ints(args.seeds or cfg["val_seeds"])
    budgets=parse_ints(args.budgets or cfg["budgets"])
    reference_policies=parse_names(args.reference_policies) if args.reference_policies else []
    if not set(reference_policies).issubset(REF_POLICIES):
        raise ValueError("Reference policies must be from keep,greedy,periodic")
    if "keep" not in reference_policies:
        raise ValueError("Keep reference is required for paired performance diagnostics")
    manifests={name:read_manifest(path.parent/"manifest.json") for name,path in paths.items()}
    args.reference_policies=reference_policies
    args.clock=cfg["clock"]
    validate_protocol(manifests,args)
    if not math.isclose(float(cfg["gamma"]),1.0,abs_tol=1e-12):
        raise ValueError("Current on-policy MC return-to-go implementation requires gamma=1")
    for case in cases:
        if not Path(case).is_file():
            raise FileNotFoundError(f"Case file missing: {case} (run from iamu_d3dqn root)")
    if args.out.exists() and not args.overwrite and any(args.out.glob("*.csv")):
        raise FileExistsError(f"Output already exists: {args.out}; choose a fresh --out or --overwrite")
    args.out.mkdir(parents=True,exist_ok=True)
    models={}
    provenance={}
    for name,path in paths.items():
        model,meta=DoubleDQNAgent.from_checkpoint(str(path),device=args.device)
        arch,width=MODEL_SPECS[name]
        if model.q_network!=arch or model.hidden_dim!=width:
            raise ValueError(f"{name}: expected {arch}-{width}, got {model.q_network}-{model.hidden_dim}")
        if not math.isclose(model.keep_margin,float(cfg["keep_margin"]),rel_tol=0,abs_tol=1e-7):
            raise ValueError(f"{name}: keep_margin differs from evaluation protocol")
        model.online.eval()
        models[name]=model
        provenance[name]=dict(path=str(path.resolve()),sha256=checksum(path),architecture=arch,
                              width=width,parameters=sum(p.numel() for p in model.online.parameters()),
                              selected_episode=meta.get("selected_episode"),
                              gradient_updates=meta.get("n_gradient_updates"))
        print(f"[checkpoint] {name} {arch}-{width} epoch={meta.get('selected_episode')} {path}",flush=True)
    print(f"[grid] {len(cases)} cases x {len(scenarios)} scenarios x {len(seeds)} seeds x {len(budgets)} budgets",flush=True)
    sinks={
        "state":CsvSink(args.out/"q_states.csv",STATE_FIELDS),
        "action":CsvSink(args.out/"action_q_values.csv",ACTION_FIELDS),
        "common":CsvSink(args.out/"common_state_comparison.csv",COMMON_FIELDS),
        "mc":CsvSink(args.out/"onpolicy_mc_returns.csv",MC_FIELDS),
    }
    episodes=[]
    groups=0
    try:
        for case in cases:
            vessels,port=read_case(case)
            case_name=Path(case).stem
            for scenario in scenarios:
                for seed in seeds:
                    inst=make_instance(vessels,port,scenario,seed,cfg)
                    for budget in budgets:
                        groups+=1
                        for reference in reference_policies:
                            episodes.append(run_reference(inst,case_name,scenario,seed,budget,reference,
                                                          models,cfg,sinks["state"],sinks["action"],sinks["common"]))
                        if not args.no_onpolicy:
                            for name,model in models.items():
                                episodes.append(run_model(inst,case_name,scenario,seed,budget,name,model,cfg,
                                                          sinks["state"],sinks["action"],sinks["mc"]))
                        if groups%3==0 or (args.max_groups and groups>=args.max_groups):
                            print(f"[progress] {groups} validation groups done",flush=True)
                        if args.max_groups and groups>=args.max_groups:
                            break
                    if args.max_groups and groups>=args.max_groups:break
                if args.max_groups and groups>=args.max_groups:break
            if args.max_groups and groups>=args.max_groups:break
    finally:
        for s in sinks.values():s.close()
    postprocess(args.out,episodes,not args.no_onpolicy)
    config=dict(source="IAMU frozen V0.9-A experiments",grid_groups=groups,
                cases=[str(x) for x in cases],case_sha256={str(x):checksum(Path(x)) for x in cases},
                scenarios=scenarios,seeds=seeds,budgets=budgets,
                reference_policies=reference_policies,onpolicy=not args.no_onpolicy,
                evaluation_config={k:cfg[k] for k in ENV_OPTIONS},models=provenance,
                note=("Common-state Q-gap and action agreement are comparable because observations are shared; "
                      "raw Q levels across different visited states are not directly comparable. "
                      "On-policy MC error is a single realized return-to-go under latent ETA uncertainty, "
                      "not an oracle expected-Q target. Models were not trained or updated."))
    (args.out/"diagnostic_manifest.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[complete] {groups} groups, {len(episodes)} episode rollouts -> {args.out}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
