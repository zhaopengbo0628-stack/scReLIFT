from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RNAStudentEncoder(nn.Module):
    def __init__(self, genes: int, hidden_dim_1: int = 512, hidden_dim_2: int = 256, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(genes, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, latent_dim),
        )

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(rna), dim=1)


class RNAAnchoredStudentEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        latent_dim: int = 64,
        adapter_hidden_dim: int = 128,
        max_correction: float = 0.25,
        initial_correction: float = 0.05,
        fixed_correction: float | None = None,
    ):
        super().__init__()
        if not 0.0 < max_correction < 1.0:
            raise ValueError("max_correction must satisfy 0 < max < 1")
        if fixed_correction is None and not 0.0 < initial_correction < max_correction:
            raise ValueError("initial_correction must satisfy 0 < initial < max")
        if fixed_correction is not None and not 0.0 < fixed_correction <= max_correction:
            raise ValueError("fixed_correction must satisfy 0 < fixed <= max")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
        )
        self.max_correction = float(max_correction)
        if fixed_correction is None:
            initial_ratio = torch.tensor(initial_correction / max_correction)
            self.correction_logit = nn.Parameter(torch.atanh(initial_ratio))
        else:
            self.register_buffer("_fixed_correction", torch.tensor(float(fixed_correction)))

    def correction_scale(self) -> torch.Tensor:
        if hasattr(self, "_fixed_correction"):
            return self._fixed_correction
        return self.max_correction * torch.tanh(self.correction_logit)

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(self.backbone(rna), dim=1)
        correction = F.normalize(self.adapter(anchor), dim=1)
        return F.normalize(anchor + self.correction_scale() * correction, dim=1)


class RNAScaffoldBoundedResidualEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, latent_dim: int = 64, hidden_dim: int = 128, bound: float = 0.10):
        super().__init__()
        if bound <= 0:
            raise ValueError("bound must be positive")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.bound = float(bound)

    def forward_raw(self, rna: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scaffold = self.backbone(rna)
        delta = self.bound * torch.tanh(self.adapter(scaffold))
        return scaffold + delta

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_raw(rna), dim=1)


class RNAScaffoldVectorNormResidualEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, latent_dim: int = 64, hidden_dim: int = 128, rho: float = 0.10, eps: float = 1e-8):
        super().__init__()
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.rho = float(rho)
        self.eps = float(eps)

    def decompose_raw(self, rna: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            scaffold = self.backbone(rna)
        raw = self.adapter(scaffold)
        scaffold_norm = torch.linalg.vector_norm(scaffold, dim=1, keepdim=True).clamp_min(self.eps)
        budget = self.rho * scaffold_norm
        raw_norm_sq = torch.sum(raw * raw, dim=1, keepdim=True)
        denominator = torch.sqrt(raw_norm_sq + budget * budget + self.eps)
        delta = budget * raw / denominator
        return scaffold, delta, scaffold + delta

    def forward_raw(self, rna: torch.Tensor) -> torch.Tensor:
        return self.decompose_raw(rna)[2]

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_raw(rna), dim=1)
