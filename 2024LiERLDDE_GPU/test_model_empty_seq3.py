"""Smoke-test empty seq3 path using your real ERL-DDE model.py (no C++ env needed)."""
import types
import numpy as np
import torch
from model import ES

model = ES().eval()

def empty_candidates(self, state):
    return (None, None, torch.empty(0, 3, device=state.device),
            torch.empty(0, dtype=torch.long, device=state.device))

model.cope_state = types.MethodType(empty_candidates, model)
with torch.no_grad():
    action = model(torch.zeros(1, 30, 8))
assert isinstance(action, np.ndarray), type(action)
assert np.array_equal(action, np.array([-1, -1, -1, -1, -1])), action
print('[PASS] Empty seq3 returns the native C++ environment wait/advance action')
