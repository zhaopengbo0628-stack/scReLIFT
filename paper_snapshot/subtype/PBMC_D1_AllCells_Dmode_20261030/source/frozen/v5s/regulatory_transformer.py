from __future__ import annotations

import torch
from torch import nn


class LocalCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, peaks: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(query, peaks, peaks, key_padding_mask=padding)
        query = self.norm_attention(query + attended)
        return self.norm_ffn(query + self.ffn(query))


class RegulatoryTransformer(nn.Module):
    def __init__(
        self,
        candidate_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        distance_bp: torch.Tensor,
        b_reg: torch.Tensor,
        num_peaks: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        window_bp: int = 250_000,
    ):
        super().__init__()
        self.num_genes = int(candidate_index.shape[0])
        self.window_bp = float(window_bp)
        self.register_buffer("candidate_index", candidate_index.long())
        self.register_buffer("candidate_mask", candidate_mask.bool())
        self.register_buffer("abs_weight", b_reg.abs().T.float())
        self.register_buffer("sign_weight", b_reg.sign().T.float())
        self.register_buffer("distance", (distance_bp.float() / self.window_bp).clamp(0.0, 1.0))
        self.peak_embedding = nn.Embedding(num_peaks, d_model)
        self.gene_embedding = nn.Parameter(torch.randn(self.num_genes, d_model) * 0.02)
        self.peak_features = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(
            [LocalCrossAttentionBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.scalar_head = nn.Linear(d_model, 1)
        self.peak_reconstruction_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        atac: torch.Tensor,
        gene_start: int = 0,
        gene_stop: int | None = None,
        masked: bool = False,
        mask_rate: float = 0.15,
        return_reconstruction: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gene_stop = self.num_genes if gene_stop is None else gene_stop
        candidate_index = self.candidate_index[gene_start:gene_stop]
        candidate_mask = self.candidate_mask[gene_start:gene_stop]
        original_values = atac[:, candidate_index]
        mask = torch.zeros_like(original_values, dtype=torch.bool)
        values = original_values
        if masked:
            mask = torch.rand_like(values) < mask_rate
            values = values.masked_fill(mask, 0.0)
        peak_tokens = self.peak_embedding(candidate_index).unsqueeze(0).expand(atac.shape[0], -1, -1, -1)
        features = torch.stack(
            (
                values,
                self.abs_weight[gene_start:gene_stop].gather(1, candidate_index).unsqueeze(0).expand(atac.shape[0], -1, -1),
                self.sign_weight[gene_start:gene_stop].gather(1, candidate_index).unsqueeze(0).expand(atac.shape[0], -1, -1),
                self.distance[gene_start:gene_stop].unsqueeze(0).expand(atac.shape[0], -1, -1),
            ),
            dim=-1,
        )
        peak_tokens = peak_tokens + self.peak_features(features)
        batch, genes, candidates, dim = peak_tokens.shape
        query = self.gene_embedding[gene_start:gene_stop].unsqueeze(0).expand(batch, -1, -1)
        query = query.reshape(batch * genes, 1, dim)
        peak_tokens = peak_tokens.reshape(batch * genes, candidates, dim)
        safe_candidate_mask = candidate_mask.clone()
        safe_candidate_mask[~safe_candidate_mask.any(dim=1), 0] = True
        padding = (~safe_candidate_mask).unsqueeze(0).expand(batch, -1, -1).reshape(batch * genes, candidates)
        for block in self.blocks:
            query = block(query, peak_tokens, padding)
        regulatory = self.scalar_head(query[:, 0]).reshape(batch, genes)
        regulatory = regulatory.masked_fill(~candidate_mask.any(dim=1).unsqueeze(0), 0.0)
        if not return_reconstruction:
            return regulatory
        query_for_peaks = query.expand(-1, candidates, -1)
        reconstruction = self.peak_reconstruction_head(torch.cat((peak_tokens, query_for_peaks), dim=-1)).squeeze(-1)
        reconstruction = reconstruction.reshape(batch, genes, candidates).masked_fill(~candidate_mask.unsqueeze(0), 0.0)
        return regulatory, reconstruction, mask
