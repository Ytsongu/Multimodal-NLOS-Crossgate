from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class TemporalFeatureEncoder(nn.Module):
    """Encodes GNSS/IMU temporal sequence features."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, temporal_sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(temporal_sequence)
        return self.norm(hidden[-1])


class SpatialFeatureEncoder(nn.Module):
    """Encodes current spatial context from GNSS, IMU, LiDAR, and satellite features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        h = config.hidden_dim
        self.gnss_proj = nn.Linear(config.gnss_dim, h)
        self.imu_proj = nn.Linear(config.imu_dim, h)
        self.lidar_proj = nn.Linear(config.lidar_dim, h)
        self.satellite_proj = nn.Linear(config.satellite_dim, h)
        self.modality_attention = nn.MultiheadAttention(h, config.num_heads, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        gnss_current: torch.Tensor,
        imu_current: torch.Tensor,
        lidar_context: torch.Tensor,
        satellite_context: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.gnss_proj(gnss_current),
                self.imu_proj(imu_current),
                self.lidar_proj(lidar_context),
                self.satellite_proj(satellite_context),
            ],
            dim=1,
        )
        attended, _ = self.modality_attention(tokens, tokens, tokens)
        spatial_feature = attended.mean(dim=1)
        return self.out(spatial_feature)


class DualBranchEncoder(nn.Module):
    """Figure block 1: spatial and temporal branches."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.spatial = SpatialFeatureEncoder(config)
        self.temporal = TemporalFeatureEncoder(config.temporal_dim, config.hidden_dim, config.dropout)

    def forward(
        self,
        gnss_current: torch.Tensor,
        imu_current: torch.Tensor,
        lidar_context: torch.Tensor,
        satellite_context: torch.Tensor,
        temporal_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_feature = self.spatial(
            gnss_current=gnss_current,
            imu_current=imu_current,
            lidar_context=lidar_context,
            satellite_context=satellite_context,
        )
        temporal_feature = self.temporal(temporal_sequence)
        return spatial_feature, temporal_feature

