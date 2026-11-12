"""Entry point for modified ERL-DDE-KDE-Fixed (NOT original paper reproduction)."""
import argparse
import os
from math import sqrt, log

import numpy as np
import torch

from model import ES
from train_fixed import train_loop, render_env, restore_full_checkpoint


def build_parser():
    p = argparse.ArgumentParser(description='ERL-DDE-KDE-Fixed (modified baseline)')
    p.add_argument('--device', default='cpu', help='CPU recommended for serial rollouts')
    p.add_argument('--kde-device', default='cpu', help='KDE device: cpu or cuda:0')
    p.add_argument('--cases', type=int, default=200)
    p.add_argument('--loops', type=int, default=1)
    p.add_argument('--populations', type=int, default=5)
    p.add_argument('--lambda-size', type=int, default=80)
    p.add_argument('--max-episode-length', type=int, default=1000)
    p.add_argument('--train-folder', default='./case_code')
    p.add_argument('--test-case', default='./case_code/test.txt')
    p.add_argument('--kde-r1', type=float, default=0.9975,
                   help='Normalized candidate mean-similarity threshold; experimental')
    p.add_argument('--kde-r2', type=float, default=0.9990,
                   help='Normalized population-exclusion threshold; experimental')
    p.add_argument('--kde-max-attempts', type=int, default=24,
                   help='Max proposals per candidate, then pick least-dense proposal')
    p.add_argument('--kde-restart-attempts', type=int, default=64,
                   help='Max random reinitializations, then pick least-dense')
    p.add_argument('--seed', type=int, default=42,
                   help='Fixed run seed (not a bit-for-bit reproduction of upstream)')
    p.add_argument('--legacy-seeds', action='store_true',
                   help='Use upstream np.random.seed() entropy reseeding each proposal')
    p.add_argument('--full-checkpoint-every', type=int, default=10,
                   help='Save complete fixed-version state every N iterations; 0 disables')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume fixed-version checkpoint_full_latest.pt')
    p.add_argument('--warmstart-directory', type=str, default=None,
                   help='Load five original .pth weights only; restarts evolution from iter 0')
    p.add_argument('--silent', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--output-prefix', default='BACAP_KDE_FIXED')
    return p


def make_args(cli):
    if cli.cases < 1 or cli.loops < 1 or cli.populations < 1 or cli.lambda_size < 2:
        raise ValueError('cases/loops/populations >=1 and lambda-size >=2 required')
    if not (0 < cli.kde_r1 <= 1 and 0 < cli.kde_r2 <= 1):
        raise ValueError('KDE mean-similarity thresholds must be in (0, 1]')
    if cli.kde_max_attempts < 1 or cli.kde_restart_attempts < 1:
        raise ValueError('KDE attempt limits must be positive')
    if cli.full_checkpoint_every < 0:
        raise ValueError('full-checkpoint-every cannot be negative')
    if cli.resume and cli.warmstart_directory:
        raise ValueError('--resume and --warmstart-directory are mutually exclusive')
    args = cli
    args.device = torch.device(cli.device)
    args.kde_device = torch.device(cli.kde_device)
    if (args.device.type == 'cuda' or args.kde_device.type == 'cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this PyTorch environment')
    args.train_loop = cli.loops
    args.train_case_num = cli.cases
    args.train_folder = cli.train_folder
    args.test_case = cli.test_case
    args.max_episode_length = cli.max_episode_length
    args.cluster_n = cli.populations
    args.lambda_ = cli.lambda_size
    args.kde_r1 = cli.kde_r1
    args.kde_r2 = cli.kde_r2
    args.kde_max_attempts = cli.kde_max_attempts
    args.kde_restart_attempts = cli.kde_restart_attempts
    args.full_checkpoint_every = cli.full_checkpoint_every
    args.seed_rng = None if cli.legacy_seeds else np.random.default_rng(cli.seed)
    if cli.seed is not None:
        torch.manual_seed(cli.seed)
        np.random.seed(cli.seed)
    args.n = ES().count_parameters()
    args.para_number = args.n
    args.h = sqrt(args.n)
    args.need_restart = [False] * args.cluster_n
    args.miu_0 = args.lambda_ // 2
    args.miu = args.miu_0
    args.sum_miu = sum(log(j) for j in range(1, args.miu + 1))
    args.w = [log((args.miu + 1) / i) /
              (args.miu * log(args.miu + 1) - args.sum_miu)
              for i in range(1, args.miu + 1)]
    args.miu_eff = 1 / sum(w*w for w in args.w)
    args.c_cov = 1 / (3 * sqrt(args.n) + 5)
    args.c = 2 / (args.n + 7)
    args.q_ = 0.3
    args.c_s = 0.3
    args.d_sig = 1
    args.sigma = [0.05] * args.cluster_n
    args.s = [0] * args.cluster_n
    args.lr = 1
    args.p = []
    args.history_theta = [[] for _ in range(args.cluster_n)]
    args.output_prefix = cli.output_prefix
    return args


def make_output_dir(args):
    if args.resume:
        directory = os.path.abspath(os.path.dirname(args.resume))
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(args.resume)
        print('Resume in existing output folder:', directory)
        return directory
    parent = os.path.join(os.getcwd(), args.output_prefix + '_checkpoints')
    os.makedirs(parent, exist_ok=True)
    for i in range(10000):
        directory = os.path.join(parent, str(i))
        try:
            os.mkdir(directory)
            print('Create folder', directory)
            return directory
        except FileExistsError:
            pass
    raise RuntimeError('No free checkpoint directory')


def main():
    cli = build_parser().parse_args()
    args = make_args(cli)
    args.folder_path = make_output_dir(args)
    models = [ES().to(args.device) for _ in range(args.cluster_n)]
    for m in models:
        for param in m.parameters():
            param.requires_grad = False
    args.p = [[torch.zeros_like(v) for _, v in models[i].es_params()]
              for i in range(args.cluster_n)]
    if args.resume:
        restore_full_checkpoint(args, models, args.resume)
    elif args.warmstart_directory:
        for i in range(args.cluster_n):
            path = os.path.join(args.warmstart_directory,
                                f'synced_model_{i}_{args.n}.pth')
            try:
                state = torch.load(path, map_location=args.device, weights_only=True)
            except TypeError:
                state = torch.load(path, map_location=args.device)
            models[i].load_state_dict(state)
        print('[WARMSTART] Weights loaded; history, sigma and iteration reset! '
              'This is not continuation of the original run.', flush=True)
    else:
        print('[KDE-FIXED] New training session (modified algorithm).')
    if args.test:
        render_env(args, models[0], args.test_case)
    else:
        train_loop(args, models)


if __name__ == '__main__':
    main()
