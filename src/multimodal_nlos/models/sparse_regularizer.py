from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSparseRegularizer(nn.Module):
    """Figure block 3: learnable piecewise sparse penalty for fused feature z."""

    def __init__(self) -> None:
        super().__init__()
        self.low_threshold = nn.Parameter(torch.tensor(0.001))
        self.threshold_gap = nn.Parameter(torch.tensor(0.03))
        self.low_slope = nn.Parameter(torch.tensor(1.0))
        self.high_slope = nn.Parameter(torch.tensor(1.0))

    def forward(self, fused_feature: torch.Tensor) -> torch.Tensor:
        magnitude = fused_feature.abs()
        b1 = F.softplus(self.low_threshold).clamp(min=1e-4)
        b2 = b1 + F.softplus(self.threshold_gap).clamp(min=1e-4)
        w1 = F.softplus(self.low_slope).clamp(min=1e-3)
        w2 = F.softplus(self.high_slope).clamp(min=1e-3)

        penalty = torch.zeros_like(magnitude)
        mid = (magnitude >= b1) & (magnitude < b2)
        high = magnitude >= b2
        penalty[mid] = w1 * (magnitude[mid] - b1)
        penalty[high] = w2 * (magnitude[high] - b2) + w1 * (b2 - b1)
        return penalty.mean()

