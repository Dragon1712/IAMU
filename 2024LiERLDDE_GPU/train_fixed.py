"""ERL-DDE-KDE-Fixed training loop.

Adapted from Dragon1712/IAMU train.py (GitHub SHA
221a99c0a82970830cc66e65fe2dfa8a7c6691ff, accessed 2026-11-12).

DELIBERATE algorithm differences:
  * Reference-count normalized KDE and newly calibrated thresholds.
  * Bounded rejection sampling with minimum-density fallback.
  * Bounded population reinitialization with minimum-density fallback.
  * No unused append to the preallocated candidate-model list.
  * Optional deterministic candidate seed stream and full-state checkpoints.
Original C++ scheduling environment, ES model architecture, rollout and
rank-based evolution/update are preserved.
"""
from __future__ import absolute_import, division, print_function

import os
import time
import queue
import math
import tempfile
from copy import deepcopy
from math import log, sqrt, exp

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ES
from env import Bay_Env
from kde_fixed import ReferenceBank, bounded_candidate, bounded_reinitialize

averageReward = []
maxReward = []
minReward = []
episodeCounter = []
really_reward_list = []


def update_progress_bar(progress):
    import sys
    bar_length = 40
    block = int(round(bar_length * progress))
    sys.stdout.write('\r[' + '=' * block + '-' * (bar_length - block) + '] ' +
                     '{0:.1%}'.format(progress))
    sys.stdout.flush()


def do_rollouts(args, model, random_seeds, return_queue, test=False, case=None):
    env = Bay_Env(args.bay_case_file if case is None else case)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = torch.from_numpy(state)
    this_model_return = 0
    this_model_num_frames = 0
    with torch.no_grad():
        for step in range(args.max_episode_length):
            state = state.float().unsqueeze(0)
            action = model(Variable(state).to(args.device))
            state, reward, done, _, _ = env.step(action)
            this_model_return += reward
            this_model_num_frames += 1
            if done:
                break
            state = torch.from_numpy(state)
    # If an adapted env offers close(), release native resources.
    close = getattr(env, 'close', None)
    if callable(close):
        close()
    if test:
        return this_model_return
    return_queue.put((random_seeds, this_model_return, this_model_num_frames))


def perturb_model(args, model, new_model, random_seed, i):
    # Keep original ES perturbation exactly to retain the update structure.
    new_model.load_state_dict(model.state_dict())
    torch.manual_seed(int(random_seed))
    index = 0
    for k, v in new_model.es_params():
        z = torch.randn(v.size(), device=args.device)
        r = torch.randn(v.size(), device=args.device)
        perturb = args.sigma[i] * (sqrt(1 - args.c_cov) * z) + \
                  sqrt(args.c_cov) * torch.mul(args.p[i][index], r)
        v += perturb.float().to(args.device)
        index += 1
    return new_model


def generate_seeds_and_models(args, synced_model, new_model, i):
    if args.seed_rng is None:
        # Original behavior (non-reproducible entropy reseeding each attempt).
        np.random.seed()
        random_seed = int(np.random.randint(2 ** 31 - 1))
    else:
        random_seed = int(args.seed_rng.integers(2 ** 31 - 1))
    model = perturb_model(args, synced_model, new_model, random_seed, i)
    return random_seed, model.to(args.device)


def gradient_update(args, synced_model, returns, random_seeds,
                    num_eps, num_frames, unperturbed_results):
    # Original rank-based evolution and parameter update.
    nth_place = [sum(r > unperturbed_results[i] for r in returns[i])
                 for i in range(args.cluster_n)]
    args.rank_d = nth_place
    sort_returns = [[] for _ in range(args.cluster_n)]
    rankings = [[0 for _ in range(args.lambda_)] for _ in range(args.cluster_n)]
    for i in range(args.cluster_n):
        sort_returns[i] = sorted(((r, num) for num, r in enumerate(returns[i])),
                                 key=lambda x: x[0], reverse=True)
        for rank, (_, j) in enumerate(sort_returns[i]):
            rankings[i][j] = rank

    really_reward = [do_rollouts(args, synced_model[i], i, None, test=True,
                                 case=args.test_case)
                     for i in range(args.cluster_n)]
    if not args.silent:
        print(f'Iteration num: {args.episode_num}\n'
              f'Episode num: {num_eps}\n'
              f'Average reward: {np.mean(really_reward)}\n'
              f'Variance in rewards: {np.var(really_reward)}\n'
              f'Max reward on test case: {np.max(really_reward)}\n'
              f'Min reward on test case: {np.min(really_reward)}\n'
              f'Total population number: {args.cluster_n*args.lambda_}\n'
              f'Sigma: {args.sigma}\n'
              f's: {args.s}\n'
              f'Learning rate: {args.lr}\n'
              f'Total num frames seen: {num_frames}\n'
              f'Test reward: {really_reward}\n'
              f'Unperturbed reward: {unperturbed_results}\n'
              f'Unperturbed rank: {nth_place} out of theta\n', flush=True)

    averageReward.append(np.mean(really_reward))
    episodeCounter.append(num_eps)
    maxReward.append(np.max(really_reward))
    minReward.append(np.min(really_reward))
    really_reward_list.append(really_reward)

    np.save(os.path.join(args.folder_path, 'really_reward.npy'),
            np.array(really_reward_list))
    np.save(os.path.join(args.folder_path, 'max_reward.npy'),
            np.array(maxReward))
    plt.clf()
    plt.plot(range(len(episodeCounter)), maxReward, label='max')
    plt.ylabel('rewards')
    plt.xlabel('Iteration num')
    plt.legend()
    plt.xticks(np.arange(0, len(episodeCounter),
                         max(int(len(episodeCounter) // 10), 1)))
    plt.gcf().savefig(os.path.join(args.folder_path, 'graph.png'), dpi=100)

    for i in range(args.cluster_n):
        for j in range(args.lambda_):
            if rankings[i][j] >= args.miu:
                continue
            torch.manual_seed(int(random_seeds[i][j]))
            index = 0
            for k, v in synced_model[i].es_params():
                z = torch.randn(v.size(), device=args.device)
                r = torch.randn(v.size(), device=args.device)
                perturb = args.w[rankings[i][j]] * args.sigma[i] * (
                    sqrt(1 - args.c_cov) * z +
                    sqrt(args.c_cov) * torch.mul(args.p[i][index], r))
                v += (args.lr * perturb).float()
                index += 1
    return synced_model


def render_env(args, model, case=None):
    value = do_rollouts(args, model, None, None, test=True, case=case)
    print('Reward: %f' % value)


def upgrade_args(args, model_parameters, last_model_parameters, results,
                 last_results):
    # Original CMA-like evolution state updates.
    p = [[] for _ in range(args.cluster_n)]
    for i in range(args.cluster_n):
        index = 0
        for v1, v2 in zip(last_model_parameters[i], model_parameters[i]):
            p[i].append((1 - args.c) * args.p[i][index] +
                        sqrt(args.c * (2 - args.c) * args.miu_eff) *
                        (v2 - v1) / args.sigma[i])
            index += 1
    args.p = p
    q = [2 * min(args.rank_d[i], args.miu) / args.miu - 1
         for i in range(args.cluster_n)]
    for i in range(args.cluster_n):
        args.s[i] = (1 - args.c_s) * args.s[i] + args.c_s * (q[i] - args.q_)
        args.sigma[i] = args.sigma[i] * exp(args.s[i] / args.d_sig)
    for i in range(args.cluster_n):
        if args.sigma[i] < 1e-6:
            args.need_restart[i] = True
            print(f'Restart {i} population for convergence')
    for i in range(args.cluster_n):
        if args.sigma[i] > 1:
            args.sigma[i] = 0.05
            print(f'Restart {i} population for sigma too large')


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _checkpoint_save(args, models, next_iter, num_eps, total_frames,
                     last_results, forced_count):
    """Save full state at completed-iteration boundary; atomic file replace."""
    output = os.path.join(args.folder_path, 'checkpoint_full_latest.pt')
    state = {
        'format': 'ERL-DDE-KDE-Fixed-v1',
        'next_iter': next_iter,
        'models': [_cpu_state(m) for m in models],
        'history_theta': [[_cpu_state(m) for m in hist]
                          for hist in args.history_theta],
        'sigma': list(args.sigma),
        's': list(args.s),
        'p': [[v.detach().cpu().clone() for v in bank] for bank in args.p],
        'need_restart': list(args.need_restart),
        'rank_d': list(getattr(args, 'rank_d', [0] * args.cluster_n)),
        'last_results': deepcopy(last_results),
        'num_eps': num_eps,
        'total_frames': total_frames,
        'forced_count': forced_count,
        'stats': {
            'averageReward': averageReward[:], 'maxReward': maxReward[:],
            'minReward': minReward[:], 'episodeCounter': episodeCounter[:],
            'really_reward_list': really_reward_list[:],
        },
        'torch_rng': torch.get_rng_state(),
        'numpy_rng': np.random.get_state(),
        'seed_rng': None if args.seed_rng is None else args.seed_rng.bit_generator.state,
        'config': {
            'cluster_n': args.cluster_n, 'lambda_': args.lambda_,
            'train_case_num': args.train_case_num,
            'train_loop': args.train_loop, 'n': args.n,
            'kde_r1': args.kde_r1, 'kde_r2': args.kde_r2,
            'kde_max_attempts': args.kde_max_attempts,
            'kde_restart_attempts': args.kde_restart_attempts,
        },
    }
    if args.device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    fd, temp_name = tempfile.mkstemp(
        prefix='.checkpoint_full_', suffix='.tmp', dir=args.folder_path)
    os.close(fd)
    try:
        torch.save(state, temp_name)
        os.replace(temp_name, output)
        print(f'[CHECKPOINT] saved next_iter={next_iter} path={output}', flush=True)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def restore_full_checkpoint(args, synced_model, path):
    """Restore a *fixed-version* checkpoint; not original author's .pth."""
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only. Only load
        # checkpoints created locally by this program (pickle is trusted).
        checkpoint = torch.load(path, map_location='cpu')
    if checkpoint.get('format') != 'ERL-DDE-KDE-Fixed-v1':
        raise ValueError('not an ERL-DDE-KDE-Fixed full checkpoint')
    cfg = checkpoint['config']
    for key in ('cluster_n', 'lambda_', 'train_case_num', 'train_loop', 'n',
                'kde_r1', 'kde_r2', 'kde_max_attempts', 'kde_restart_attempts'):
        if cfg[key] != getattr(args, key):
            raise ValueError(f'checkpoint mismatch {key}: '
                             f'{cfg[key]} != {getattr(args, key)}')
    for model, state in zip(synced_model, checkpoint['models']):
        model.load_state_dict(state)
    args.history_theta = []
    for bank in checkpoint['history_theta']:
        history = []
        for state in bank:
            m = ES().to(args.device)
            m.load_state_dict(state)
            for param in m.parameters():
                param.requires_grad = False
            history.append(m)
        args.history_theta.append(history)
    args.sigma = checkpoint['sigma']
    args.s = checkpoint['s']
    args.p = [[v.to(args.device) for v in bank] for bank in checkpoint['p']]
    args.need_restart = checkpoint['need_restart']
    args.rank_d = checkpoint['rank_d']
    args.start_iteration = checkpoint['next_iter']
    args.restored_last_results = checkpoint['last_results']
    args.restored_num_eps = checkpoint['num_eps']
    args.restored_total_frames = checkpoint['total_frames']
    args.restored_forced_count = checkpoint['forced_count']
    for name, target in (
        ('averageReward', averageReward), ('maxReward', maxReward),
        ('minReward', minReward), ('episodeCounter', episodeCounter),
        ('really_reward_list', really_reward_list),
    ):
        target[:] = checkpoint['stats'][name]
    torch.set_rng_state(checkpoint['torch_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    if args.seed_rng is not None and checkpoint['seed_rng'] is not None:
        args.seed_rng.bit_generator.state = checkpoint['seed_rng']
    if args.device.type == 'cuda' and 'cuda_rng' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    print(f'[CHECKPOINT] restored next_iter={args.start_iteration}', flush=True)
    return synced_model


def train_loop(args, synced_model):
    print('Num params in network %d' % synced_model[0].count_parameters())
    print('[KDE-FIXED] NORMALIZED mean-similarity KDE: '
          f'candidate threshold={args.kde_r1}, '
          f'population threshold={args.kde_r2}; '
          f'max attempts/candidate={args.kde_max_attempts}; '
          f'restart attempts={args.kde_restart_attempts}', flush=True)
    all_models = [[ES().to(args.device) for _ in range(args.lambda_)]
                  for _ in range(args.cluster_n)]
    num_eps = getattr(args, 'restored_num_eps', 0)
    total_num_frames = getattr(args, 'restored_total_frames', 0)
    last_results = getattr(args, 'restored_last_results',
                           [[] for _ in range(args.cluster_n)])
    forced_count = getattr(args, 'restored_forced_count', 0)
    start = getattr(args, 'start_iteration', 0)
    end = args.train_loop * args.train_case_num
    if start >= end:
        print('[KDE-FIXED] Training already complete.')
        return
    for iteration in range(start, end):
        it_start = time.perf_counter()
        case = iteration % args.train_case_num
        args.bay_case_file = args.train_folder + f'/{case}.txt'
        args.episode_num = iteration
        return_queue = [queue.Queue() for _ in range(args.cluster_n)]
        all_seeds = [[] for _ in range(args.cluster_n)]
        last_model_parameters = [[deepcopy(v) for _, v in synced_model[i].es_params()]
                                 for i in range(args.cluster_n)]
        # Candidate generation. Fixed population references remain unchanged
        # during this phase; build one ReferenceBank per population.
        candidate_start = time.perf_counter()
        for i in range(args.cluster_n):
            refs = (synced_model[:i] + synced_model[i+1:] +
                    [m for bank in args.history_theta[:i] for m in bank] +
                    [m for bank in args.history_theta[i+1:] for m in bank])
            bank = ReferenceBank(refs, args.h, args.kde_device)
            total_attempts = 0
            num_forced = 0
            min_score, max_score = float('inf'), float('-inf')
            for j in range(args.lambda_):
                seed, model, score, tries, forced = bounded_candidate(
                    args, i, all_models[i][j], synced_model[i], bank,
                    generate_seeds_and_models, perturb_model,
                    args.kde_max_attempts, args.kde_r1,
                )
                all_seeds[i].append(seed)
                total_attempts += tries
                num_forced += int(forced)
                forced_count += int(forced)
                min_score = min(min_score, score)
                max_score = max(max_score, score)
                # NOTE: preallocated all_models[i][j] already holds the model.
                # Upstream appends this same object unnecessarily each episode;
                # NOT appending prevents memory growth without changing rollout.
            print(f'[KDE CANDIDATES] iter={iteration} pop={i} '
                  f'references={bank.count} accepted={args.lambda_} '
                  f'accepted_by_threshold={args.lambda_-num_forced} '
                  f'forced_min_density={num_forced} '
                  f'attempts={total_attempts} '
                  f'score_min={min_score:.7f} score_max={max_score:.7f} '
                  f'elapsed_sec={time.perf_counter()-candidate_start:.1f}',
                  flush=True)
        print(f'[TIMING] iter={iteration} candidate_generation_sec='
              f'{time.perf_counter()-candidate_start:.2f}', flush=True)

        # Rollouts as in the original algorithm.
        rollout_start = time.perf_counter()
        test_num = 0
        for i in range(args.cluster_n):
            for j in range(args.lambda_):
                do_rollouts(args, all_models[i][j], all_seeds[i][j],
                            return_queue[i], test=False)
                test_num += 1
                update_progress_bar(test_num / (args.cluster_n * args.lambda_))
            do_rollouts(args, synced_model[i], 'dummy_seed',
                        return_queue[i], test=False)
        print(f'\nThe {iteration}th iteration is finished: '
              f'{time.perf_counter()-rollout_start:.2f} s', flush=True)

        unperturbed_results = []
        results = [[] for _ in range(args.cluster_n)]
        seeds = [[] for _ in range(args.cluster_n)]
        num_frames = [[] for _ in range(args.cluster_n)]
        for i in range(args.cluster_n):
            for _ in range(args.lambda_ + 1):
                seed, reward, frame = return_queue[i].get()
                if seed == 'dummy_seed':
                    unperturbed_results.append(reward)
                    continue
                results[i].append(reward)
                seeds[i].append(seed)
                num_frames[i].append(frame)
        assert len(unperturbed_results) == args.cluster_n
        total_num_frames += sum(sum(k) for k in num_frames)
        num_eps += sum(len(k) for k in results)
        args.miu = int((iteration / end) *
                       (args.lambda_ - args.miu_0) + args.miu_0)
        args.sum_miu = sum(log(j) for j in range(1, args.miu + 1))
        args.w = [log((args.miu+1)/i) /
                  (args.miu*log(args.miu+1)-args.sum_miu)
                  for i in range(1, args.miu + 1)]
        args.miu_eff = 1 / sum(w*w for w in args.w)
        assert args.miu <= args.lambda_
        synced_model = gradient_update(args, synced_model, results, seeds,
                                       num_eps, total_num_frames,
                                       unperturbed_results)
        model_parameters = [[deepcopy(v) for _, v in synced_model[i].es_params()]
                            for i in range(args.cluster_n)]
        upgrade_args(args, model_parameters, last_model_parameters,
                     results, last_results)
        last_results = deepcopy(results)
        rank = []
        main_return_queue = queue.Queue()
        for i in range(args.cluster_n):
            do_rollouts(args, synced_model[i], i, main_return_queue,
                        test=False)
        for _ in range(args.cluster_n):
            seed, reward, frame = main_return_queue.get()
            rank.append((reward, seed))
        rank.sort(key=lambda x: x[0], reverse=True)
        mean_reward = sum(reward for reward, _ in rank) / args.cluster_n
        # Population exclusion: normalized density + bounded reinitialization.
        if args.cluster_n > 1:
            for reward, i in rank:
                refs = synced_model[:i] + synced_model[i+1:] + args.history_theta[i]
                bank = ReferenceBank(refs, args.h, args.kde_device)
                density = bank.score(synced_model[i], args.kde_r2)
                if density > args.kde_r2 or args.need_restart[i]:
                    if reward >= mean_reward or i <= (args.cluster_n // 2):
                        print(f'[KDE RESTART] iter={iteration} pop={i} '
                              f'score={density:.7f}', flush=True)
                    else:
                        args.history_theta[i] = []
                        print(f'[KDE DELETE] iter={iteration} pop={i} '
                              f'score={density:.7f}', flush=True)
                        # Original reset randomizes mean repeatedly until
                        # density <= threshold, which can also hang.
                        # Here we bound it and retain the least dense sample.
                        reset_bank = ReferenceBank(
                            synced_model[:i] + synced_model[i+1:],
                            args.h, args.kde_device,
                        )
                        reset_score, reset_tries, reset_forced = bounded_reinitialize(
                            synced_model[i], reset_bank,
                            args.kde_restart_attempts, args.kde_r2,
                        )
                        print(f'[KDE REINIT] iter={iteration} pop={i} '
                              f'tries={reset_tries} score={reset_score:.7f} '
                              f'forced={int(reset_forced)}', flush=True)
                    args.need_restart[i] = False
                    args.sigma[i] = 0.05
                    args.p[i] = [torch.zeros_like(v) for v in model_parameters[i]]
                    args.s[i] = 0
                    last_results[i] = []

        # Keep original population means as standalone weights for evaluation.
        for i in range(args.cluster_n):
            torch.save(synced_model[i].state_dict(),
                       os.path.join(args.folder_path,
                                    f'synced_model_{i}_{args.para_number}.pth'))
            args.history_theta[i].append(deepcopy(synced_model[i]))
        print(f'[ITER DONE] iter={iteration} '
              f'wall_sec={time.perf_counter()-it_start:.2f} '
              f'cumulative_forced={forced_count} '
              f'history={[len(h) for h in args.history_theta]}', flush=True)
        if args.full_checkpoint_every and (
            ((iteration + 1) % args.full_checkpoint_every == 0) or
            (iteration + 1 == end)
        ):
            _checkpoint_save(args, synced_model, iteration+1,
                             num_eps, total_num_frames,
                             last_results, forced_count)
