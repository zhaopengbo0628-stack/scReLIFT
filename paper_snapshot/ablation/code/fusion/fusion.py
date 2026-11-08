"""Isolated 64-D fusion operators for the fusion ablation."""
from __future__ import annotations

import torch
from torch import nn


class MeanFusion(nn.Module):
    def forward(self, z_r: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        return (z_r + z_f) * 0.5


class ConcatFusion(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(2 * dim, 128), nn.GELU(),
                                        nn.Linear(128, dim), nn.LayerNorm(dim))

    def forward(self, z_r, z_f):
        return self.projection(torch.cat((z_r, z_f), dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, z_r, z_f):
        gate = torch.sigmoid(self.gate(torch.cat((z_r, z_f), dim=-1)))
        return gate * z_r + (1 - gate) * z_f


class InteractionFusion(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(4 * dim, 128), nn.GELU(),
                                        nn.Linear(128, dim), nn.LayerNorm(dim))

    def forward(self, z_r, z_f):
        h = torch.cat((z_r, z_f, z_r * z_f, torch.abs(z_r - z_f)), dim=-1)
        return self.projection(h)


def build_fusion(name: str, dim: int = 64) -> nn.Module:
    constructors = {"mean": MeanFusion, "concat": ConcatFusion,
                    "gate": GatedFusion, "interaction": InteractionFusion}
    try:
        return constructors[name](dim) if name != "mean" else MeanFusion()
    except KeyError as exc:
        raise ValueError(f"unknown fusion: {name}") from exc
