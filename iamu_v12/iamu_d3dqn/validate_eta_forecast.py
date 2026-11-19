#!/usr/bin/env python3
"""Independent, strictly out-of-sample validation of IAMU V0.4 ETA beliefs.

Does NOT invoke simulate(), RealizedRuntime, SlotPlanner or RL. All candidate
forecasts depend only on the current public ETA, event time and fixed model
parameters. Hidden actual arrivals are used EXCLUSIVELY as evaluation labels.

The 'censored' family matches V0.4's max(now, N(eta, sigma)) sampler exactly
at the distributional level. The 'truncated' family is a diagnostic comparator
for the condition that an ETA announcement implies the ship has not arrived.

All data here are synthetic: conclusions do NOT validate real-world port ETAs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist, mean

from iamu_dynamic import generate_scenario, read_case

_NORMAL = NormalDist()
_SQRT2 = math.sqrt(2.0)
_LOG_SQRT2PI = .5 * math.log(2 * math.pi)
_COVERAGES = (50, 80, 90, 95)
_DEFAULT_GRID = 32


def parse_seeds(value: str) -> list[int]:
    """Comma-separated ints or inclusive ranges: '1001:1020'."""
    result = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            parts = item.split(':')
            if len(parts) != 2:
                raise ValueError('Seed range syntax: START:END')
            start, stop = (int(p) for p in parts)
            if stop < start or stop - start > 10000:
                raise ValueError('Invalid seed range')
            result.extend(range(start, stop + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise ValueError('Seed list must be nonempty and contain no duplicates')
    return result


def parse_csv(value: str) -> list[str]:
    items = [x.strip() for x in value.split(',') if x.strip()]
    if not items or len(set(items)) != len(items):
        raise ValueError('Expected nonempty list without duplicates')
    return items


@dataclass(frozen=True)
class ForecastObservation:
    split: str
    case: str
    scenario: str
    seed: int
    vessel_id: int
    stage: str
    event_id: int
    time: float
    eta: float
    true_arrival: float   # OFFLINE LABEL ONLY; not accepted by ArrivalBelief

    @property
    def episode(self) -> tuple[str, str, int]:
        return self.case, self.scenario, self.seed


def build_observations(case: str, scenarios: list[str], seeds: list[int],
                       split: str, updates: int, uncertainty: float,
                       include_initial: bool = True) -> list[ForecastObservation]:
    """One observational target per vessel's public ETA announcement.

    Announcement existence is public. Consequently true_arrival > time for
    all update rows. Do NOT evaluate 'all unstarted vessels' by privately
    filtering on hidden true arrivals; that would create an oracle cohort.
    """
    vessels, port = read_case(case)
    records = []
    for label in scenarios:
        for seed in seeds:
            sc = generate_scenario(vessels, port, label, seed, updates, uncertainty)
            if include_initial:
                for v in vessels:
                    records.append(ForecastObservation(
                        split, str(Path(case).resolve()), label, seed, v.id,
                        'initial', -1, 0., float(v.eta0), float(sc.true_etas[v.id])))
            for event in sc.updates:
                # This check is for offline data validity, not a feature of
                # the distribution or a condition on predicted action quality.
                truth = float(sc.true_etas[event.vessel_id])
                if not (event.time < truth):
                    raise AssertionError('ETA event generated after actual arrival')
                if event.new_eta < event.time - 1e-9:
                    raise AssertionError('Observed ETA is earlier than its announcement')
                records.append(ForecastObservation(
                    split, str(Path(case).resolve()), label, seed,
                    event.vessel_id, 'update', event.update_id,
                    float(event.time), float(event.new_eta), truth))
    return records


def std_normal_cdf(x: float) -> float:
    # erfc remains reliable for tails where 1-erf suffers cancellation.
    return .5 * math.erfc(-x / _SQRT2)


def std_normal_survival(x: float) -> float:
    return .5 * math.erfc(x / _SQRT2)


def std_normal_pdf(x: float) -> float:
    return math.exp(-.5 * x * x) / math.sqrt(2 * math.pi)


@dataclass(frozen=True)
class ArrivalBelief:
    family: str
    sigma: float

    def __post_init__(self):
        if self.family not in ('censored', 'truncated'):
            raise ValueError('family must be censored or truncated')
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError('sigma must be finite and > 0 for probabilistic scoring')

    @property
    def name(self) -> str:
        return f'{self.family}_sigma{self.sigma:g}'

    def quantile(self, p: float, mu: float, now: float) -> float:
        if not 0 < p < 1:
            raise ValueError('probability must be in (0,1)')
        if self.family == 'censored':
            return max(now, mu + self.sigma * _NORMAL.inv_cdf(p))
        cutoff = std_normal_cdf((now - mu) / self.sigma)
        transformed = cutoff + p * (1 - cutoff)
        return max(now, mu + self.sigma * _NORMAL.inv_cdf(min(transformed, 1 - 1e-15)))

    def cdf(self, y: float, mu: float, now: float) -> float:
        if y < now:
            return 0.
        base = std_normal_cdf((y - mu) / self.sigma)
        if self.family == 'censored':
            return base
        cutoff = std_normal_cdf((now - mu) / self.sigma)
        return max(0., min(1., (base - cutoff) / max(1e-300, 1 - cutoff)))

    def predictive_mean(self, mu: float, now: float) -> float:
        a = (now - mu) / self.sigma
        surv = std_normal_survival(a)
        density = std_normal_pdf(a)
        if self.family == 'censored':
            return now * (1-surv) + mu * surv + self.sigma * density
        return mu + self.sigma * density / max(surv, 1e-300)

    def nll(self, actual: float, mu: float, now: float) -> float:
        if actual < now - 1e-9:
            raise ValueError('Actual arrival < now: inadmissible for prearrival model')
        a = (now - mu) / self.sigma
        if actual <= now + 1e-12:
            if self.family == 'truncated':
                return math.inf
            return -math.log(max(std_normal_cdf(a), 1e-300))
        z = (actual - mu) / self.sigma
        base_nll = .5*z*z + math.log(self.sigma) + _LOG_SQRT2PI
        if self.family == 'truncated':
            base_nll += math.log(max(std_normal_survival(a), 1e-300))
        return base_nll

    def crps(self, actual: float, mu: float, now: float,
             grid: int = _DEFAULT_GRID) -> float:
        """Deterministic quantile quadrature of proper CRPS; no MC RNG.

        CRPS = 2 * integral_0^1 pinball_p(actual, quantile_p) dp.
        Midpoint grid incurs an approximation error, controlled by --grid.
        """
        if grid < 8:
            raise ValueError('grid must have >=8 midpoint quantiles')
        total = 0.
        for j in range(grid):
            p = (j+.5) / grid
            q = self.quantile(p, mu, now)
            diff = actual - q
            total += (p if diff >= 0 else p - 1) * diff
        return 2 * total / grid


def score_one(obs: ForecastObservation, model: ArrivalBelief,
              grid: int = _DEFAULT_GRID) -> dict:
    # By design, the predictive object is constructed only from public mu/now.
    mu, now, actual = obs.eta, obs.time, obs.true_arrival
    pred_mean = model.predictive_mean(mu, now)
    q90 = model.quantile(.9,mu,now)
    q95 = model.quantile(.95,mu,now)
    out = dict(split=obs.split, case=obs.case, scenario=obs.scenario,
               seed=obs.seed, vessel_id=obs.vessel_id, stage=obs.stage,
               event_id=obs.event_id, time=now, public_eta=mu,
               true_arrival=actual, method=model.name,
               crps=model.crps(actual,mu,now,grid),
               nll=model.nll(actual,mu,now),
               abs_error=abs(pred_mean-actual),
               sq_error=(pred_mean-actual)**2,
               signed_error=pred_mean-actual,
               pit=model.cdf(actual,mu,now),
               high90=float(actual > q90),
               high95=float(actual > q95),
               excess90=max(0.,actual-q90),
               excess95=max(0.,actual-q95),
               pinball90=(.9 if actual>=q90 else -.1)*(actual-q90),
               pinball95=(.95 if actual>=q95 else -.05)*(actual-q95),
               interval_width90=(model.quantile(.95,mu,now)-model.quantile(.05,mu,now)))
    for level in _COVERAGES:
        lo, hi = (1-level/100)/2, (1+level/100)/2
        out[f'coverage{level}']=float(model.quantile(lo,mu,now)
                                         <= actual <= model.quantile(hi,mu,now))
    return out


def episode_means(scored: list[dict], field: str) -> list[float]:
    by_episode: dict[tuple[str,str,int],list[float]] = defaultdict(list)
    for r in scored:
        by_episode[(r['case'],r['scenario'],r['seed'])].append(float(r[field]))
    return [mean(values) for _,values in sorted(by_episode.items())]


def summarize(scored: list[dict], *, split: str,
              scenario: str, stage: str, method: str) -> dict:
    if not scored:
        return {}
    e = lambda field: mean(episode_means(scored,field))
    # Episode-macro statistics prevent variable numbers of ETA updates in
    # separate synthetic trajectories from silently reweighting the results.
    return dict(split=split, scenario=scenario, stage=stage, method=method,
                n_forecasts=len(scored), n_episodes=len(episode_means(scored,'crps')),
                crps=e('crps'), nll=e('nll'), mae=e('abs_error'),
                rmse=math.sqrt(e('sq_error')), bias=e('signed_error'),
                coverage50=e('coverage50'),coverage80=e('coverage80'),
                coverage90=e('coverage90'),coverage95=e('coverage95'),
                high90=e('high90'),high95=e('high95'),
                excess90=e('excess90'),excess95=e('excess95'),
                pinball90=e('pinball90'),pinball95=e('pinball95'),
                interval_width90=e('interval_width90'),pit_mean=e('pit'))


def percentile(values: list[float], frac: float) -> float:
    items = sorted(values)
    if not items: raise ValueError('No values')
    at = (len(items)-1) * frac
    i = int(at)
    return items[i] * (1-(at-i)) + items[min(i+1,len(items)-1)]*(at-i)


def bootstrap_test(all_test: list[dict], selected: str,
                   reference: str, iterations: int, seed: int) -> list[dict]:
    """Cluster bootstrap over *complete scenario-seed episodes*.

    The paired CRPS difference uses identical forecast targets for both
    methods; comparisons are 'reference - selected' (positive is beneficial).
    """
    methods = sorted(set(r['method'] for r in all_test))
    by: dict[str,dict[tuple[str,str,int],list[dict]]] = defaultdict(lambda:defaultdict(list))
    for r in all_test:
        by[r['method']][(r['case'],r['scenario'],r['seed'])].append(r)
    epis = sorted(by[selected]);rng=random.Random(seed)
    out=[]
    for method in methods:
        eps = sorted(by[method])
        if eps != epis: raise ValueError('Non-paired test episode IDs')
        per = [(mean(z['crps'] for z in by[method][ep]),
                mean(z['coverage90'] for z in by[method][ep]),
                mean(z['high95'] for z in by[method][ep])) for ep in epis]
        observed = [mean(row[col] for row in per) for col in range(3)]
        distributions=[[],[],[]]
        for _ in range(iterations):
            sample=[per[rng.randrange(len(per))] for _ in epis]
            for j in range(3):distributions[j].append(mean(v[j] for v in sample))
        out.append(dict(method=method,n_episodes=len(epis),
            crps=observed[0],crps_ci_low=percentile(distributions[0],.025),
            crps_ci_high=percentile(distributions[0],.975),
            coverage90=observed[1],cov90_ci_low=percentile(distributions[1],.025),
            cov90_ci_high=percentile(distributions[1],.975),
            high95=observed[2],high95_ci_low=percentile(distributions[2],.025),
            high95_ci_high=percentile(distributions[2],.975)))
    # Shared resampling for paired differences; no independent bootstrap
    # from the two methods, which would unnecessarily inflate uncertainty.
    if reference in by and selected in by:
        diffs = [mean(x['crps'] for x in by[reference][ep]) -
                 mean(x['crps'] for x in by[selected][ep]) for ep in epis]
        draw = [mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
                for _ in range(iterations)]
        out.append(dict(method=f'PAIRED_{reference}_minus_{selected}',
            n_episodes=len(epis),crps=mean(diffs),
            crps_ci_low=percentile(draw,.025),crps_ci_high=percentile(draw,.975)))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def hash_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_disjoint(args, train_seeds, val_seeds, test_seeds) -> None:
    groups=[set(str(Path(x).resolve()) for x in getattr(args, f'{split}_cases'))
            for split in ('train','val','test')]
    if any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Train/validation/test case files MUST be disjoint')
    # Same instance copied under a different filename must NOT evade the split.
    digests=[{hash_file(p) for p in cases} for cases in groups]
    if any(digests[i]&digests[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Train/validation/test case CONTENTS MUST be disjoint')
    if (set(train_seeds)&set(val_seeds) or set(train_seeds)&set(test_seeds)
            or set(val_seeds)&set(test_seeds)):
        raise ValueError('Train/validation/test seed values MUST be disjoint')
    for cases in groups:
        for name in cases:
            if not Path(name).is_file(): raise FileNotFoundError(name)


def run(args) -> dict:
    train_s=parse_seeds(args.train_seeds)
    val_s=parse_seeds(args.val_seeds)
    test_s=parse_seeds(args.test_seeds)
    verify_disjoint(args,train_s,val_s,test_s)
    scenarios=parse_csv(args.scenarios)
    if not set(scenarios).issubset({'shock','repeated','small'}):
        raise ValueError('Supported scenarios: shock,repeated,small')
    if args.updates < 1 or not math.isfinite(args.uncertainty) or args.uncertainty<0:
        raise ValueError('Invalid scenario generator settings')
    if args.grid<8 or args.bootstrap<20:
        raise ValueError('--grid must be >=8; --bootstrap must be >=20')
    if args.ood_uncertainty is not None and (not math.isfinite(args.ood_uncertainty)
                                            or args.ood_uncertainty<0):
        raise ValueError('Invalid OOD uncertainty')
    sigmas=[float(v) for v in parse_csv(args.sigmas)]
    if any(not math.isfinite(v) or v<=0 for v in sigmas):
        raise ValueError('All sigma values must be positive and finite')
    families=parse_csv(args.families)
    if not set(families).issubset({'censored','truncated'}):
        raise ValueError('Invalid probability family')

    split_args=[('train',args.train_cases,train_s,args.uncertainty),
                ('validation',args.val_cases,val_s,args.uncertainty),
                ('test',args.test_cases,test_s,args.uncertainty)]
    if args.ood_uncertainty is not None:
        split_args.append(('test_ood',args.test_cases,test_s,args.ood_uncertainty))
    observations=[]
    for split,cases,seeds,uncertainty in split_args:
        for case in cases:
            observations.extend(build_observations(case,scenarios,seeds,split,
                args.updates,uncertainty,include_initial=not args.no_initial))
    training=[r for r in observations if r.split=='train' and r.stage=='update']
    if not training: raise ValueError('No training update examples available')
    # Training-only calibration baseline. Does not use validation/test labels.
    fitted_sigma=max(.1,math.sqrt(mean((r.true_arrival-r.eta)**2 for r in training)))
    models = [ArrivalBelief(kind,s) for kind in families
              for s in sigmas+[fitted_sigma]]
    # Distinct fitted and fixed sigma may coincide; do not duplicate a model.
    models=list({m.name:m for m in models}.values())
    if 'censored_sigma8' not in {m.name for m in models}:
        print('NOTE: censored_sigma8 not included; paired CI will omit V0.4 baseline.')

    scored=[]
    for obs in observations:
        for m in models:
            scored.append(score_one(obs,m,args.grid))
    group: dict[tuple[str,str,str,str],list[dict]] = defaultdict(list)
    for r in scored:
        group[(r['split'],r['scenario'],r['stage'],r['method'])].append(r)
        group[(r['split'],'ALL',r['stage'],r['method'])].append(r)
    summaries=[summarize(rows,split=s,scenario=sc,stage=st,method=method)
               for (s,sc,st,method),rows in sorted(group.items())]

    validation=[r for r in summaries if r['split']=='validation' and
                r['scenario']=='ALL' and r['stage']=='update']
    if len(validation)!=len(models):
        raise RuntimeError('Validation missing forecasts for a candidate')
    # LOCK model identity here, strictly before inspecting test scores.
    best=min(validation,key=lambda r:(r['crps'],r['nll'],r['method']))
    selected=best['method']
    deployed_best=min((v for v in validation if v['method'].startswith('censored_')),
                      key=lambda v:(v['crps'],v['nll'],v['method']))['method']
    test=[r for r in scored if r['split']=='test' and r['stage']=='update']
    ci=bootstrap_test(test,selected,'censored_sigma8',args.bootstrap,args.bootstrap_seed)

    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'model_metrics.csv',summaries)
    write_csv(output/'test_bootstrap_ci.csv',ci)
    selected_rows=[r for r in summaries if r['method'] in (selected,deployed_best) and
                   r['stage']=='update' and r['split'] in ('validation','test','test_ood')]
    write_csv(output/'selected_model_metrics.csv',selected_rows)
    reliability=[]
    for split in ('test','test_ood'):
        testrows=[r for r in scored if r['split']==split and r['stage']=='update']
        for method in sorted({r['method'] for r in testrows}):
            these=[r for r in testrows if r['method']==method]
            for bin_id in range(10):
                n=sum(1 for r in these if min(9,int(r['pit']*10))==bin_id)
                reliability.append(dict(split=split,method=method,
                  pit_bin_low=bin_id/10,pit_bin_high=(bin_id+1)/10,
                  count=n,total=len(these),fraction=n/len(these)))
    write_csv(output/'test_pit_bins.csv',reliability)
    if args.save_predictions:
        write_csv(output/'predictions.csv',scored)
    selected_obj={'selected_on':'validation/update/episode_macro_CRPS',
                  'selected_model':selected,'validation_crps':best['crps'],
                  'best_v04_deployable_censored_model':deployed_best,
                  'training_fitted_sigma':fitted_sigma,
                  'test_not_used_for_model_selection':True,
                  'test_label_usage':'offline scoring only',
                  'note':'truncated is a diagnostic comparator, not deployed by V0.4'}
    (output/'selected_model.json').write_text(json.dumps(selected_obj,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'arguments':vars(args),'train_seeds':train_s,
              'validation_seeds':val_s,'test_seeds':test_s,
              'case_sha256':{str(Path(p).resolve()):hash_file(p)
                for ps in (args.train_cases,args.val_cases,args.test_cases) for p in ps},
              'n_observations_by_split':{s:sum(r.split==s for r in observations)
                    for s,_,_,_ in split_args},
              'train_only_fitted_sigma':fitted_sigma,
              'generator':'iamu_dynamic.generate_scenario',
              'belief_current':'max(now, Normal(public_eta, sigma))',
              'metrics_weighting':'macro mean by (case,scenario,seed) episode',
              'synthetic_data_not_real_world_validation':True}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

    for split in ('validation','test','test_ood'):
        best_r=next((r for r in summaries if r['split']==split and r['scenario']=='ALL'
                      and r['stage']=='update' and r['method']==selected),None)
        if best_r:
            print(f'{split:>10} selected={selected} n={best_r["n_forecasts"]} '
                  f'episodes={best_r["n_episodes"]} CRPS={best_r["crps"]:.4f} '
                  f'coverage90={best_r["coverage90"]:.3f} '
                  f'P(actual>q90)={best_r["high90"]:.3f}')
    print(f'Training-only sigma={fitted_sigma:.4f}; locked by validation CRPS; '
          f'outputs: {output.resolve()}')
    return selected_obj


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--train-cases',nargs='+',default=['../2024LiERLDDE/case_code/0.txt'])
    p.add_argument('--val-cases',nargs='+',default=['../2024LiERLDDE/case_code/1.txt'])
    p.add_argument('--test-cases',nargs='+',default=['../2024LiERLDDE/case_code/2.txt'])
    p.add_argument('--train-seeds',default='101:105')
    p.add_argument('--val-seeds',default='201:205')
    p.add_argument('--test-seeds',default='1001:1020')
    p.add_argument('--scenarios',default='shock,repeated,small')
    p.add_argument('--updates',type=int,default=2)
    p.add_argument('--uncertainty',type=float,default=8.)
    p.add_argument('--ood-uncertainty',type=float,default=12.,
                   help='Held-out intensity stress test, NEVER used to tune the model')
    p.add_argument('--sigmas',default='8,14.4,20')
    p.add_argument('--families',default='censored,truncated')
    p.add_argument('--grid',type=int,default=32,help='Midpoint quantile grid for approximate CRPS')
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--bootstrap-seed',type=int,default=913)
    p.add_argument('--no-initial',action='store_true')
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--out',default='results/v05_forecast_validation')
    args=p.parse_args()
    run(args)


if __name__=='__main__':
    main()
