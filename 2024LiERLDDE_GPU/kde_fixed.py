"""KDE-Fixed density metric and bounded sampling.

This is a MODIFIED algorithm, not a numerically equivalent implementation of
ERL-DDE's sum-of-kernels / bandwidth threshold (r1=0.5, r2=0.2).

Metric: mean(exp(-0.5 * squared_euclidean_distance / bandwidth**2)).
Range: [0, 1] for nonempty reference banks. Lower means more diverse.
"""
from __future__ import annotations

import math
import torch
from torch.nn.functional import pairwise_distance


def model_vector(model, device=None):
    """Flatten model parameters in the original ES.es_params() ordering."""
    with torch.no_grad():
        v = torch.cat([param.detach().reshape(-1) for _, param in model.es_params()])
        return v.to(device=device) if device is not None else v


class ReferenceBank:
    def __init__(self, models, bandwidth: float, device=None):
        if bandwidth <= 0:
            raise ValueError('bandwidth must be positive')
        self.bandwidth = float(bandwidth)
        self.device = torch.device(device) if device is not None else None
        self.count = len(models)
        self.matrix = None
        if self.count:
            with torch.no_grad():
                self.matrix = torch.stack([model_vector(m, self.device) for m in models])
                # Cache both reference matrix and squared norms for faster mv.
                self.norm2 = self.matrix.square().sum(1)

    @torch.no_grad()
    def score(self, model, threshold=None):
        """Return normalized mean similarity in [0,1]. Empty bank -> 0.

        Fast path uses the squared-distance norm identity with a cached
        reference matrix and norms. For near-threshold values, recompute
        with the original pairwise_distance to reduce numerical flips.
        """
        if not self.count:
            return 0.0
        x = model_vector(model, self.matrix.device)
        if x.numel() != self.matrix.shape[1]:
            raise ValueError('candidate/reference parameter dimension mismatch')
        d2 = (self.norm2 + x.square().sum() -
              2 * torch.mv(self.matrix, x)).clamp_min_(0.)
        score = float(torch.exp(-0.5 * d2 /
                                (self.bandwidth * self.bandwidth)).mean().item())
        if threshold is not None and abs(score - threshold) < 2e-5:
            d = pairwise_distance(x.unsqueeze(0), self.matrix)
            score = float(torch.exp(-0.5 * (d / self.bandwidth).square()).mean().item())
        if not math.isfinite(score):
            raise FloatingPointError('KDE score is not finite')
        return score


def bounded_candidate(args, population, slot, parent, bank, generate,
                      perturb, max_attempts: int, threshold: float):
    """Generate one candidate; take first passing sample or lowest density.

    Returns (seed, model, selected_score, tries, forced). The selected model is
    ALWAYS the preallocated `slot`; rejected attempts overwrite the same slot.
    """
    if max_attempts < 1:
        raise ValueError('max_attempts must be >= 1')
    best_seed, best_score = None, float('inf')
    for tries in range(1, max_attempts + 1):
        seed, model = generate(args, parent, slot, population)
        score = bank.score(model, threshold)
        if score < best_score:
            best_seed, best_score = seed, score
        if score <= threshold:
            return seed, model, score, tries, False
    # Deliberate deviation from upstream: guarantee progress by choosing the
    # least dense candidate among the generated attempts. Regenerate it using
    # the original perturbation and its saved seed, not an aliased model.
    model = perturb(args, parent, slot, best_seed, population)
    return best_seed, model, best_score, max_attempts, True


def bounded_reinitialize(model, bank, max_attempts: int, threshold: float):
    """Bound the upstream while True around ES.init_weight().

    If no sample passes, restore the lowest-density sampled initialization.
    Returns (score, tries, forced). Note: this deliberately changes upstream
    semantics whenever the bounded fallback is used.
    """
    if max_attempts < 1:
        raise ValueError('max_attempts must be >= 1')
    best_score, best_state = float('inf'), None
    for tries in range(1, max_attempts + 1):
        model.init_weight()
        score = bank.score(model, threshold)
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
        if score <= threshold:
            return score, tries, False
    model.load_state_dict(best_state)
    return best_score, max_attempts, True
