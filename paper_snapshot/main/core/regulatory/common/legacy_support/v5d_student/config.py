from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V5DStudentConfig:
    hidden_dim_1: int = 512
    hidden_dim_2: int = 256
    latent_dim: int = 128
    dropout: float = 0.1
    neighbor_k: int = 15
    random_k: int = 15
    sampled_local_per_anchor: int = 8
    sampled_random_per_anchor: int = 8
    batch_anchor: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    grad_clip: float = 5.0
    pair_bank_seed: int = 20020130
    distillation_loss: str = "pair_mse"
    sampler: str = "fixed_pair_bank"
    freeze_mode: str = "none"
    teacher_target: str = "teacher_cosine_geometry"
    checkpoint_metric: str = "teacher_recall_at_10"
    contrastive_temperature: float = 0.1
    local_teacher_temperature: float = 0.07
    local_student_temperature: float = 0.07
    local_geometry_weight: float = 0.8
    boundary_geometry_weight: float = 0.2
    anchor_geometry_weight: float = 0.1
    correspondence_floor: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)
