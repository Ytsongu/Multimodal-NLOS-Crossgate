from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    gnss_dim: int = 5
    imu_dim: int = 6
    lidar_dim: int = 64
    satellite_dim: int = 22
    temporal_dim: int = 48
    hidden_dim: int = 64
    num_heads: int = 2
    num_classes: int = 2
    dropout: float = 0.3
    use_sparse_regularizer: bool = True

