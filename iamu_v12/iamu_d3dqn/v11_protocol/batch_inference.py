"""Stateless batched CPU/GPU greedy inference, OPTIONAL optimization.

Does not alter RL training or exploration; GPU GEMM reordering may produce
small floating differences, especially with actions near a margin threshold.
Use only for fixed-model inference after numerical parity testing.
"""
from __future__ import annotations

def action_batch(agent, observations):
    if not observations:return []
    import numpy as np,torch
    if len({tuple(o.action_features.shape) for o in observations})!=1:
        raise ValueError('Cannot batch variable candidate slot dimensions without padding')
    agent.online.eval()
    with torch.no_grad():
        g=torch.as_tensor(np.stack([o.global_features for o in observations]),device=agent.device,dtype=torch.float32)
        a=torch.as_tensor(np.stack([o.action_features for o in observations]),device=agent.device,dtype=torch.float32)
        mask=torch.as_tensor(np.stack([o.mask for o in observations]),device=agent.device,dtype=torch.bool)
        if not bool(mask[:,0].all()):raise AssertionError('Keep must always be legal')
        q=agent.online(g,a,mask).masked_fill(~mask,-torch.inf).clone()
        q[:,0]+=agent.execution_margin
        return [int(i) for i in q.argmax(dim=1).cpu().tolist()]
