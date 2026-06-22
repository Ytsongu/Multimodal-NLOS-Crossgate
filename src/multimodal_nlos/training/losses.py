from __future__ import annotations

import torch
import torch.nn as nn


def hybrid_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sparse_loss: torch.Tensor,
    *,
    lambda_sparse: float,
    criterion: nn.Module | None = None,
) -> torch.Tensor:
    criterion = criterion or nn.CrossEntropyLoss()
    return criterion(logits, labels) + lambda_sparse * sparse_loss
