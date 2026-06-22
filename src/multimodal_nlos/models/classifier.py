from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .crossgate import BidirectionalCrossGateFusion
from .encoders import DualBranchEncoder
from .sparse_regularizer import LearnableSparseRegularizer


class CrossGateLOSNLOSClassifier(nn.Module):
    """End-to-end figure-level model skeleton."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DualBranchEncoder(config)
        self.fusion = BidirectionalCrossGateFusion(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.sparse_regularizer = LearnableSparseRegularizer()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self,
        gnss_current: torch.Tensor,
        imu_current: torch.Tensor,
        lidar_context: torch.Tensor,
        satellite_context: torch.Tensor,
        temporal_sequence: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        spatial_feature, temporal_feature = self.encoder(
            gnss_current=gnss_current,
            imu_current=imu_current,
            lidar_context=lidar_context,
            satellite_context=satellite_context,
            temporal_sequence=temporal_sequence,
        )
        fused_feature, fusion_aux = self.fusion(spatial_feature, temporal_feature)
        sparse_loss = (
            self.sparse_regularizer(fused_feature)
            if self.config.use_sparse_regularizer
            else fused_feature.new_tensor(0.0)
        )
        logits = self.classifier(fused_feature)

        if return_aux:
            return logits, sparse_loss, {
                "spatial_feature": spatial_feature,
                "temporal_feature": temporal_feature,
                "fused_feature": fused_feature,
                "fusion": fusion_aux,
            }
        return logits, sparse_loss

