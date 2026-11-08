from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V5SConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 256
    dropout: float = 0.1
    window_bp: int = 250_000
    max_candidates: int = 50
    consistency_folds: int = 5
    predictability_epsilon: float = 0.1
    gene_confidence_epsilon: float = 0.1
    selection_seed: int = 20020130
    selection_shuffle_seed: int = 20020230
    mask_rate: float = 0.15
    consistency_weight: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 80
    patience: int = 10
    gene_chunk_size: int = 128
    grad_clip: float = 1.0
    seed: int = 20020130
    evaluation_seed: int = 20020130

    def to_dict(self) -> dict:
        return asdict(self)
