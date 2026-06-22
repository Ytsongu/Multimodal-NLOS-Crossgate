from __future__ import annotations

import torch
import torch.nn as nn


class CrossGateRefinement(nn.Module):
    """Cross-attention -> aligned feature -> concat -> gate -> refined feature."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.candidate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query = query_feature.unsqueeze(1)
        context = context_feature.unsqueeze(1)
        aligned, attention = self.cross_attention(query, context, context)
        aligned = aligned.squeeze(1)

        concat = torch.cat([context_feature, aligned], dim=-1)
        candidate = self.candidate(concat)
        gate = self.gate(concat)
        refined = gate * candidate + (1.0 - gate) * context_feature

        return refined, {
            "aligned": aligned,
            "gate": gate,
            "attention": attention,
        }


class BidirectionalCrossGateFusion(nn.Module):
    """Figure block 2: temporal-guided spatial and spatial-guided temporal fusion."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.temporal_guided_spatial = CrossGateRefinement(hidden_dim, num_heads, dropout)
        self.spatial_guided_temporal = CrossGateRefinement(hidden_dim, num_heads, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        spatial_feature: torch.Tensor,
        temporal_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        refined_spatial, spatial_aux = self.temporal_guided_spatial(
            query_feature=temporal_feature,
            context_feature=spatial_feature,
        )
        refined_temporal, temporal_aux = self.spatial_guided_temporal(
            query_feature=spatial_feature,
            context_feature=temporal_feature,
        )
        fused_feature = self.fuse(torch.cat([refined_spatial, refined_temporal], dim=-1))

        return fused_feature, {
            "refined_spatial": refined_spatial,
            "refined_temporal": refined_temporal,
            "temporal_guided_spatial": spatial_aux,
            "spatial_guided_temporal": temporal_aux,
        }

