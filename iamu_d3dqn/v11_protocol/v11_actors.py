"""Generated fork of IAMU CPU actor with job-keyed RNG."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import inspect,os,queue,time,math,traceback
import torch
from iamu_dynamic import read_case,generate_scenario
from iamu_dynamic.rl_ddqn_v11 import DoubleDQNAgent,Transition
from iamu_dynamic.rl_env_v11 import IAMUEventEnv

@dataclass(frozen=True)
class RolloutJob:
    episode: int
    case: str
    scenario: str
    seed: int
    uncertainty: float
    budget: int
    epsilon: float


def _supports(function, parameter: str) -> bool:
    return parameter in inspect.signature(function).parameters


def _actor_env_kwargs(settings: dict) -> dict:
    kwargs = dict(budget=settings['budget'], candidate_limit=settings['candidate_limit'],
                  decision_clock=settings['clock'], period=settings['period'],
                  reward_scale=settings['reward_scale'])
    optional = {
        'candidate_screen': settings['candidate_screen'],
        'min_gain': settings['min_gain'],
        'budget_features': settings['budget_features'],
        'value_features': settings['value_features'],
    }
    for name, value in optional.items():
        if _supports(IAMUEventEnv.__init__, name):
            kwargs[name] = value
        else:
            defaults = {'candidate_screen': 'none', 'min_gain': .1,
                        'budget_features': 'legacy', 'value_features': 'legacy'}
            if value != defaults[name]:
                raise RuntimeError(f'Installed environment does not support --{name.replace("_", "-")}')
    return kwargs


def _actor_loop(worker_id: int, inbox, outbox, args: dict):
    """Top-level spawn entry: intentionally never imports/initializes CUDA."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        # Distinct seeds; no global actor random generator used for test cases.
        agent_kwargs = dict(hidden_dim=args['hidden_dim'], seed=args['seed'] + worker_id + 10000,
                            device='cpu', q_network=args.get('q_network','ddqn'))
        if _supports(DoubleDQNAgent.__init__, 'keep_margin'):
            agent_kwargs['keep_margin'] = args['keep_margin']
        elif args['keep_margin'] != 0.:
            raise RuntimeError('Installed DDQN does not support --keep-margin')
        agent_kwargs.update(execution_margin=args['execution_margin'],
                            bootstrap_margin=args['bootstrap_margin'])
        agent = DoubleDQNAgent(**agent_kwargs)
        agent.online.eval()
        outbox.put(('actor-online', worker_id, os.getpid()))
        case_cache: dict[str, Any] = {}
        model_version = -1
        while True:
            message = inbox.get()
            if message[0] == 'stop':
                break
            if message[0] == 'sync':
                _, weights, model_version = message
                agent.online.load_state_dict(weights, strict=True)
                agent.online.eval()
                continue
            if message[0] != 'rollout':
                raise ValueError(f'Unknown worker message: {message[0]}')
            _, job = message
            # Worker assignment does not determine per-job exploration RNG.
            agent.rng.seed(args['seed']*1000003+job.episode*10007+job.seed)
            if model_version < 0:
                raise RuntimeError('Actor attempted rollout before weight synchronization')
            started = time.perf_counter()
            outbox.put(('actor-start', worker_id, os.getpid(), job.episode, model_version))
            if job.case not in case_cache:
                case_cache[job.case] = read_case(job.case)
            vessels, port = case_cache[job.case]
            scenario = generate_scenario(vessels, port, label=job.scenario,
                seed=job.seed, updates_per_vessel=args['updates'],
                uncertainty=job.uncertainty)
            env_settings = dict(args, budget=job.budget)
            env = IAMUEventEnv(scenario, **_actor_env_kwargs(env_settings))
            obs = env.reset()
            transitions = []
            total_reward = 0.
            teacher_steps = 0
            last_heartbeat = started
            while True:
                teacher = None
                if args['greedy_exploration_prob'] > 0:
                    if not _supports(agent.act, 'teacher_action'):
                        raise RuntimeError('Installed DDQN does not support Greedy exploration')
                    teacher = env.point_greedy_action()
                act_kw = dict(epsilon=job.epsilon)
                if teacher is not None:
                    act_kw.update(teacher_action=teacher,
                                  teacher_prob=args['greedy_exploration_prob'])
                action = agent.act(obs, **act_kw)
                teacher_steps += int(teacher is not None and action == teacher)
                nxt, reward, done, info = env.step(action)
                transitions.append(Transition(
                    obs.global_features.copy(), obs.action_features.copy(),
                    obs.mask.copy(), int(action), float(reward),
                    nxt.global_features.copy(), nxt.action_features.copy(),
                    nxt.mask.copy(), bool(done),
                    dict(obs.decision_flags or {}),int(job.episode)))
                total_reward += reward
                obs = nxt
                # Nonblocking heartbeat: slow candidate projections can take
                # substantial CPU time and must be observable in live logs.
                if time.perf_counter()-last_heartbeat >= 15.:
                    try:
                        outbox.put_nowait(('actor-heartbeat', worker_id,
                                           os.getpid(),job.episode,len(transitions),
                                           round(time.perf_counter()-started,2)))
                    except queue.Full:
                        pass
                    last_heartbeat=time.perf_counter()
                if done:
                    break
            raw_wait = float(info['weighted_wait'])
            if not math.isclose(-total_reward*args['reward_scale'], raw_wait,
                                abs_tol=1e-5*max(1.,raw_wait), rel_tol=1e-6):
                raise AssertionError('Episode rewards do not sum to -weighted_wait')
            if info['spent'] > job.budget:
                raise AssertionError('Actor violated edit budget')
            audit = getattr(env, 'action_audit', [])
            edits = [a for a in audit if a.get('action', 0) != 0]
            flag_counts={}
            for tr in transitions:
                for key,value in (tr.reason_flags or {}).items():
                    if isinstance(value,bool):flag_counts[key]=flag_counts.get(key,0)+int(value)
            report = {
                'worker_id': worker_id, 'worker_pid': os.getpid(),
                'policy_version': model_version, 'episode': job.episode,
                'case': job.case, 'scenario': job.scenario, 'seed': job.seed,
                'uncertainty': job.uncertainty, 'budget': job.budget,
                'epsilon': job.epsilon, 'spent': int(info['spent']),
                'weighted_wait': raw_wait, 'reward_sum_scaled': total_reward,
                'steps': len(transitions), 'forced_flags': flag_counts,
                'actor_seconds': time.perf_counter()-started,
                'teacher_action_matches': teacher_steps,
                'screened_proposals': getattr(env, 'screened_proposals', 0),
                'last_revision_time': float(getattr(env, 'last_revision_time', 0.) or 0.),
                'mean_estimated_edit_gain': (
                    sum(x['estimated_gain'] for x in edits)/len(edits) if edits else 0.),
            }
            outbox.put(('ok', report, transitions))
    except BaseException:
        # Child failures cannot be allowed to leave the learner hanging forever.
        try:
            outbox.put(('error', worker_id, traceback.format_exc()), timeout=2)
        except Exception:
            pass
        raise


