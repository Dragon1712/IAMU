"""Minimal C++/Python ABI test, independent of the original PyTorch model.

Usage:
    python smoke_test.py case_code/0.txt
    python smoke_test.py --synthetic
"""
import argparse
import tempfile
from pathlib import Path
import numpy as np
from env import Bay_Env, Array2D, StepResult
import ctypes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', nargs='?', default='case_code/0.txt')
    p.add_argument('--synthetic', action='store_true')
    args = p.parse_args()

    print('Array2D size:', ctypes.sizeof(Array2D))
    print('StepResult size:', ctypes.sizeof(StepResult))
    print('reward offset:', StepResult.reward.offset)
    print('done offset:', StepResult.done.offset)

    tmp = None
    if args.synthetic:
        tmp = tempfile.TemporaryDirectory()
        case = Path(tmp.name) / 'test.txt'
        case.write_text('0\t4\t150\t2\t4\n3\t4\t150\t2\t4\n6\t4\t150\t2\t4\n')
    else:
        case = Path(args.case)
    print('case:', case)
    try:
        with Bay_Env(str(case)) as env:
            obs = env.reset()
            assert obs.shape == (30, 8), obs.shape
            assert np.isfinite(obs).all()
            print('reset: PASS', obs.shape, 'first row:', obs[0])
            for i in range(3):
                # A small, finite synthetic action; on author data this tests ABI,
                # not whether the action is optimal or legal in every port state.
                action = [0.0, 0.1, 0.0, 0.0, 0.5]
                obs, reward, done, _, _ = env.step(action)
                assert obs.shape == (30, 8)
                assert np.isfinite(obs).all()
                assert np.isfinite(reward)
                print(f'step {i + 1}: PASS reward={reward:.4f}, done={done}')
                if done:
                    break
        print('SMOKE TEST PASSED')
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == '__main__':
    main()
