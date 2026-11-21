from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .poe import product_of_experts


@dataclass(frozen=True)
class StageConfig:
    window_size: int
    stride: int
    local_layers: int = 1
    global_layers: int = 3


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 768
    latent_dim: int = 32
    encoder_hidden_dim: int = 256
    num_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.2
    num_classes: int = 2
    fusion_type: str = "poe"
    include_prior_expert: bool = True
    hierarchy_type: str = "full"
    pooling_type: str = "mean"
    informer_layers: int = 3
    informer_factor: int = 5
    informer_distil: bool = True
    # Disabled by default so historical checkpoints and all external comparisons
    # retain the exact original architecture.  When enabled, the head produces a
    # score for every *input behaviour* (before the hierarchy coarsens time) and
    # its normalized MIL evidence is added to the spammer-vs-normal margin.
    temporal_mil: bool = False
    temporal_mil_scale: float = 1.0
    temporal_mil_temperature: float = 1.0
    # Optional independent user-level views.  A zero dimension preserves the
    # historical behaviour-only architecture and checkpoint interface exactly.
    profile_dim: int = 0
    graph_dim: int = 0
    branch_aux_weight: float = 0.0
    gate_temperature: float = 1.0
    behavior_priority: bool = False
    stages: tuple[StageConfig, ...] = field(
        default_factory=lambda: (
            StageConfig(window_size=64, stride=32, local_layers=1, global_layers=3),
            StageConfig(window_size=64, stride=4, local_layers=1, global_layers=3),
        )
    )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        values = dict(values)
        values["stages"] = tuple(StageConfig(**item) for item in values["stages"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        return self.mean(hidden), self.log_var(hidden).clamp(-12.0, 12.0)


class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class PoEMVAE(nn.Module):
    """Two-modality VAE with an explicit Gaussian Product-of-Experts fusion."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.include_prior_expert = (
            config.include_prior_expert and config.fusion_type != "poe_no_prior"
        )
        self.text_encoder = ModalityEncoder(
            config.input_dim, config.encoder_hidden_dim, config.latent_dim
        )
        self.image_encoder = ModalityEncoder(
            config.input_dim, config.encoder_hidden_dim, config.latent_dim
        )
        self.text_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )
        self.image_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        sequence_mask: torch.Tensor,
        image_present: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        text_mean, text_log_var = self.text_encoder(text)
        image_mean, image_log_var = self.image_encoder(image)
        means = torch.stack((text_mean, image_mean), dim=-2)
        log_vars = torch.stack((text_log_var, image_log_var), dim=-2)
        expert_mask = torch.stack((sequence_mask, sequence_mask & image_present), dim=-1)
        joint_mean, joint_log_var = product_of_experts(
            means, log_vars, expert_mask, include_prior=self.include_prior_expert
        )

        if sample is None:
            sample = self.training
        if sample:
            noise = torch.randn_like(joint_mean)
            latent = joint_mean + noise * torch.exp(0.5 * joint_log_var)
        else:
            latent = joint_mean
        latent = latent * sequence_mask.unsqueeze(-1)

        return {
            "latent": latent,
            "joint_mean": joint_mean,
            "joint_log_var": joint_log_var,
            "text_mean": text_mean,
            "text_log_var": text_log_var,
            "image_mean": image_mean,
            "image_log_var": image_log_var,
            "text_reconstruction": self.text_decoder(latent),
            "image_reconstruction": self.image_decoder(latent),
        }


def sample_gaussian(
    mean: torch.Tensor, log_var: torch.Tensor, sample: bool | None, training: bool
) -> torch.Tensor:
    if sample is None:
        sample = training
    if not sample:
        return mean
    return mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)


class DirectFusionTokenizer(nn.Module):
    """Controlled deterministic fusion baselines with no reconstruction objective."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fusion_type = config.fusion_type
        if self.fusion_type in {"text_only", "image_only"}:
            self.projection = nn.Linear(config.input_dim, config.latent_dim)
        elif self.fusion_type == "concat":
            self.projection = nn.Linear(config.input_dim * 2, config.latent_dim)
        elif self.fusion_type == "sum":
            self.text_projection = nn.Linear(config.input_dim, config.latent_dim)
            self.image_projection = nn.Linear(config.input_dim, config.latent_dim)
        else:
            raise ValueError(f"Unsupported direct fusion type: {self.fusion_type}")

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        sequence_mask: torch.Tensor,
        image_present: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        del sample
        image_mask = (sequence_mask & image_present).unsqueeze(-1)
        if self.fusion_type == "text_only":
            latent = self.projection(text)
        elif self.fusion_type == "image_only":
            latent = self.projection(image) * image_mask
        elif self.fusion_type == "concat":
            latent = self.projection(torch.cat((text, image * image_mask), dim=-1))
        else:
            text_latent = self.text_projection(text)
            image_latent = self.image_projection(image) * image_mask
            count = 1.0 + image_mask.to(text_latent.dtype)
            latent = (text_latent + image_latent) / count
        latent = latent * sequence_mask.unsqueeze(-1)
        return {
            "latent": latent,
            "joint_mean": latent,
            "joint_log_var": torch.zeros_like(latent),
        }


class DeterministicMultimodalAE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.encoder_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.encoder_hidden_dim),
            nn.Linear(config.encoder_hidden_dim, config.latent_dim),
        )
        self.text_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )
        self.image_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        sequence_mask: torch.Tensor,
        image_present: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        del sample
        image_mask = (sequence_mask & image_present).unsqueeze(-1)
        latent = self.encoder(torch.cat((text, image * image_mask), dim=-1))
        latent = latent * sequence_mask.unsqueeze(-1)
        return {
            "latent": latent,
            "joint_mean": latent,
            "joint_log_var": torch.zeros_like(latent),
            "text_reconstruction": self.text_decoder(latent),
            "image_reconstruction": self.image_decoder(latent),
        }


class IndependentVAEFusion(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.text_encoder = ModalityEncoder(
            config.input_dim, config.encoder_hidden_dim, config.latent_dim
        )
        self.image_encoder = ModalityEncoder(
            config.input_dim, config.encoder_hidden_dim, config.latent_dim
        )
        self.fusion = nn.Linear(config.latent_dim * 2, config.latent_dim)
        self.text_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )
        self.image_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        sequence_mask: torch.Tensor,
        image_present: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        text_mean, text_log_var = self.text_encoder(text)
        image_mean, image_log_var = self.image_encoder(image)
        image_mask = (sequence_mask & image_present).unsqueeze(-1)
        text_latent = sample_gaussian(text_mean, text_log_var, sample, self.training)
        image_latent = sample_gaussian(image_mean, image_log_var, sample, self.training)
        latent = self.fusion(torch.cat((text_latent, image_latent * image_mask), dim=-1))
        latent = latent * sequence_mask.unsqueeze(-1)
        count = 1.0 + image_mask.to(text_log_var.dtype)
        joint_log_var = (text_log_var + image_log_var * image_mask) / count
        joint_mean = self.fusion(torch.cat((text_mean, image_mean * image_mask), dim=-1))
        return {
            "latent": latent,
            "joint_mean": joint_mean,
            "joint_log_var": joint_log_var,
            "text_reconstruction": self.text_decoder(latent),
            "image_reconstruction": self.image_decoder(latent),
        }


class AverageMVAE(nn.Module):
    """Two variational encoders fused by a masked arithmetic mean instead of PoE."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.text_encoder = ModalityEncoder(
            config.input_dim, config.encoder_hidden_dim, config.latent_dim
        )
        self.image_encoder = ModalityEncoder(
            config.input_dim, config.encoder_hidden_dim, config.latent_dim
        )
        self.text_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )
        self.image_decoder = ModalityDecoder(
            config.latent_dim, config.encoder_hidden_dim, config.input_dim
        )

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        sequence_mask: torch.Tensor,
        image_present: torch.Tensor,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        text_mean, text_log_var = self.text_encoder(text)
        image_mean, image_log_var = self.image_encoder(image)
        image_mask = (sequence_mask & image_present).unsqueeze(-1)
        count = 1.0 + image_mask.to(text_mean.dtype)
        joint_mean = (text_mean + image_mean * image_mask) / count
        joint_log_var = (text_log_var + image_log_var * image_mask) / count
        latent = sample_gaussian(joint_mean, joint_log_var, sample, self.training)
        latent = latent * sequence_mask.unsqueeze(-1)
        return {
            "latent": latent,
            "joint_mean": joint_mean,
            "joint_log_var": joint_log_var,
            "text_reconstruction": self.text_decoder(latent),
            "image_reconstruction": self.image_decoder(latent),
        }


class IndependentGaussianMean(AverageMVAE):
    """Independent Gaussian average: Var(mean)=sum(variance)/M**2."""
    def forward(self,text,image,sequence_mask,image_present,sample=None):
        tm,tl=self.text_encoder(text);im,il=self.image_encoder(image)
        mask=(sequence_mask & image_present).unsqueeze(-1).to(tm.dtype)
        count=1.0+mask
        mean=(tm+im*mask)/count
        variance=(tl.exp()+il.exp()*mask)/count.square()
        log_var=variance.clamp_min(torch.finfo(variance.dtype).tiny).log()
        z=sample_gaussian(mean,log_var,sample,self.training)*sequence_mask.unsqueeze(-1)
        return dict(latent=z,joint_mean=mean,joint_log_var=log_var,
                    text_reconstruction=self.text_decoder(z),image_reconstruction=self.image_decoder(z))


class SplitWindowStage(nn.Module):
    """Overlapping local attention followed by attention over window summaries."""

    def __init__(
        self,
        dim: int,
        config: ModelConfig,
        stage: StageConfig,
        *,
        local_enabled: bool = True,
        global_enabled: bool = True,
    ) -> None:
        super().__init__()
        if stage.window_size <= 0 or stage.stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if dim % config.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.window_size = stage.window_size
        self.stride = stage.stride
        self.local_enabled = local_enabled
        self.global_enabled = global_enabled

        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=config.num_heads,
                dim_feedforward=dim * config.ff_multiplier,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.local = (
            nn.TransformerEncoder(layer(), num_layers=stage.local_layers)
            if local_enabled
            else None
        )
        self.global_encoder = (
            nn.TransformerEncoder(layer(), num_layers=stage.global_layers)
            if global_enabled
            else None
        )
        self.local_norm = nn.LayerNorm(dim) if local_enabled else None
        self.global_norm = nn.LayerNorm(dim) if global_enabled else None

    def _window(self, tensor: torch.Tensor, pad_value: float | bool) -> torch.Tensor:
        batch, length = tensor.shape[:2]
        starts = list(range(0, max(length, 1), self.stride))
        windows = []
        for start in starts:
            part = tensor[:, start : start + self.window_size]
            missing = self.window_size - part.shape[1]
            if missing:
                pad_shape = (batch, missing, *tensor.shape[2:])
                pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                part = torch.cat((part, pad), dim=1)
            windows.append(part)
        return torch.stack(windows, dim=1)

    def forward(
        self, inputs: torch.Tensor, sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = inputs.shape
        windows = self._window(inputs, 0.0)
        masks = self._window(sequence_mask, False)
        num_windows = windows.shape[1]
        flat_windows = windows.reshape(batch * num_windows, self.window_size, dim)
        flat_masks = masks.reshape(batch * num_windows, self.window_size)

        window_valid = flat_masks.any(dim=1)
        safe_masks = flat_masks.clone()
        safe_masks[~window_valid, 0] = True
        if self.local is not None and self.local_norm is not None:
            local = self.local(flat_windows, src_key_padding_mask=~safe_masks)
            local = self.local_norm(local)
        else:
            local = flat_windows
        weights = flat_masks.to(local.dtype).unsqueeze(-1)
        summaries = (local * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        summaries = summaries.reshape(batch, num_windows, dim)
        summary_mask = window_valid.reshape(batch, num_windows)
        summaries = summaries * summary_mask.unsqueeze(-1)

        if self.global_encoder is not None and self.global_norm is not None:
            global_features = self.global_encoder(
                summaries, src_key_padding_mask=~summary_mask
            )
            global_features = self.global_norm(global_features)
        else:
            global_features = summaries
        global_features = global_features * summary_mask.unsqueeze(-1)
        return global_features, summary_mask


class ProbSparseSelfAttention(nn.Module):
    """Informer-style ProbSparse self-attention for non-causal classification."""

    def __init__(self, dim: int, num_heads: int, factor: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        if factor <= 0:
            raise ValueError("informer_factor must be positive")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.factor = factor
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, query_length, dim = inputs.shape
        key_length = query_length
        queries = self._heads(self.query(inputs))
        keys = self._heads(self.key(inputs))
        values = self._heads(self.value(inputs))
        logarithm = max(1, math.ceil(math.log(key_length + 1)))
        sample_k = min(key_length, self.factor * logarithm)
        top_queries = min(query_length, self.factor * logarithm)

        # The original Informer query-sparsity estimate samples keys per query.
        # Always include key zero so a short, right-padded sequence has a valid sample.
        sampled_indices = torch.randint(
            key_length, (query_length, sample_k), device=inputs.device
        )
        sampled_indices[:, 0] = 0
        sampled_keys = keys[:, :, sampled_indices, :]
        sampled_scores = (
            queries.unsqueeze(-2) * sampled_keys
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        sampled_mask = mask[:, None, sampled_indices]
        sampled_scores = sampled_scores.masked_fill(~sampled_mask, float("-inf"))
        valid_count = sampled_mask.sum(dim=-1).clamp_min(1)
        finite_scores = sampled_scores.masked_fill(~sampled_mask, 0.0)
        sparsity = sampled_scores.amax(dim=-1) - finite_scores.sum(dim=-1) / valid_count
        sparsity = sparsity.masked_fill(~mask[:, None, :], float("-inf"))
        top_indices = sparsity.topk(top_queries, dim=-1, sorted=False).indices

        selected_queries = torch.gather(
            queries,
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )
        full_scores = torch.matmul(selected_queries, keys.transpose(-2, -1))
        full_scores = full_scores / math.sqrt(self.head_dim)
        full_scores = full_scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attention = self.dropout(full_scores.softmax(dim=-1))
        selected_context = torch.matmul(attention, values)

        weights = mask[:, None, :, None].to(values.dtype)
        mean_context = (values * weights).sum(dim=2, keepdim=True)
        mean_context = mean_context / weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        context = mean_context.expand(-1, -1, query_length, -1).clone()
        context.scatter_(
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
            selected_context.to(context.dtype),
        )
        context = context.transpose(1, 2).reshape(batch, query_length, dim)
        return self.output(context) * mask.unsqueeze(-1)


class InformerEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.latent_dim
        self.attention = ProbSparseSelfAttention(
            dim, config.num_heads, config.informer_factor, config.dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * config.ff_multiplier),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * config.ff_multiplier, dim),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = self.norm1(inputs + self.dropout(self.attention(inputs, mask)))
        inputs = self.norm2(inputs + self.dropout(self.feed_forward(inputs)))
        return inputs * mask.unsqueeze(-1)


class InformerDistilLayer(nn.Module):
    """Informer encoder distillation: Conv1d, normalization, ELU and max-pooling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(dim)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = inputs.transpose(1, 2)
        features = self.pool(self.activation(self.norm(self.conv(features))))
        next_mask = F.max_pool1d(
            mask.to(features.dtype).unsqueeze(1), kernel_size=3, stride=2, padding=1
        ).squeeze(1).bool()
        features = features.transpose(1, 2) * next_mask.unsqueeze(-1)
        return features, next_mask


class InformerEncoder(nn.Module):
    """ProbSparse Informer encoder with optional convolutional distillation."""

    def __init__(self, config: ModelConfig, max_positions: int = 32768) -> None:
        super().__init__()
        if config.informer_layers <= 0:
            raise ValueError("informer_layers must be positive")
        self.layers = nn.ModuleList(
            [InformerEncoderLayer(config) for _ in range(config.informer_layers)]
        )
        self.distil_layers = nn.ModuleList(
            [InformerDistilLayer(config.latent_dim) for _ in range(config.informer_layers - 1)]
            if config.informer_distil
            else []
        )
        self.norm = nn.LayerNorm(config.latent_dim)
        positions = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, config.latent_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.latent_dim)
        )
        encoding = torch.zeros(max_positions, config.latent_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position_encoding", encoding, persistent=False)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape[1] > self.position_encoding.shape[0]:
            raise ValueError(
                f"Informer length {inputs.shape[1]} exceeds positional capacity "
                f"{self.position_encoding.shape[0]}"
            )
        features = inputs + self.position_encoding[: inputs.shape[1]].to(inputs.dtype)
        features = features * mask.unsqueeze(-1)
        for index, layer in enumerate(self.layers):
            features = layer(features, mask)
            if index < len(self.distil_layers):
                features, mask = self.distil_layers[index](features, mask)
        return self.norm(features) * mask.unsqueeze(-1), mask


class AuxiliaryViewEncoder(nn.Module):
    """Project one independent user view into the shared decision space."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MUDformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.fusion_type in {"text_only", "image_only", "concat", "sum"}:
            self.mvae = DirectFusionTokenizer(config)
        elif config.fusion_type == "deterministic_ae":
            self.mvae = DeterministicMultimodalAE(config)
        elif config.fusion_type == "independent_vae":
            self.mvae = IndependentVAEFusion(config)
        elif config.fusion_type == "mean_gaussian":
            self.mvae = IndependentGaussianMean(config)
        elif config.fusion_type == "mvae_no_poe":
            self.mvae = AverageMVAE(config)
        elif config.fusion_type in {"poe", "poe_no_prior", "corruption_aware"}:
            self.mvae = PoEMVAE(config)
        else:
            raise ValueError(f"Unsupported fusion_type: {config.fusion_type}")
        self.full_encoder: nn.TransformerEncoder | None = None
        self.informer_encoder: InformerEncoder | None = None
        if config.hierarchy_type == "none":
            stages: list[SplitWindowStage] = []
        elif config.hierarchy_type == "full_mha":
            full_layer = nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.num_heads,
                dim_feedforward=config.latent_dim * config.ff_multiplier,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            full_depth = max(
                1,
                sum(stage.local_layers + stage.global_layers for stage in config.stages),
            )
            self.full_encoder = nn.TransformerEncoder(full_layer, num_layers=full_depth)
            stages = []
        elif config.hierarchy_type == "informer":
            self.informer_encoder = InformerEncoder(config)
            stages = []
        elif config.hierarchy_type == "single_stage":
            stages = [SplitWindowStage(config.latent_dim, config, config.stages[0])]
        elif config.hierarchy_type == "sw_only":
            stages = [
                SplitWindowStage(
                    config.latent_dim, config, stage, local_enabled=True, global_enabled=False
                )
                for stage in config.stages
            ]
        elif config.hierarchy_type == "w_only":
            stages = [
                SplitWindowStage(
                    config.latent_dim, config, stage, local_enabled=False, global_enabled=True
                )
                for stage in config.stages
            ]
        elif config.hierarchy_type == "full":
            stages = [SplitWindowStage(config.latent_dim, config, stage) for stage in config.stages]
        else:
            raise ValueError(f"Unsupported hierarchy_type: {config.hierarchy_type}")
        self.stages = nn.ModuleList(stages)
        if config.pooling_type == "attention":
            self.pooling_score = nn.Linear(config.latent_dim, 1)
        elif config.pooling_type == "cls":
            self.cls_query = nn.Parameter(torch.zeros(1, 1, config.latent_dim))
            nn.init.normal_(self.cls_query, std=0.02)
            self.cls_attention = nn.MultiheadAttention(
                config.latent_dim, config.num_heads, dropout=config.dropout, batch_first=True
            )
        elif config.pooling_type not in {"mean", "max"}:
            raise ValueError(f"Unsupported pooling_type: {config.pooling_type}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.num_classes),
        )
        if config.profile_dim < 0 or config.graph_dim < 0:
            raise ValueError("auxiliary feature dimensions cannot be negative")
        if config.branch_aux_weight < 0.0 or config.gate_temperature <= 0.0:
            raise ValueError("branch auxiliary weight must be non-negative and gate temperature positive")
        self.profile_encoder = (
            AuxiliaryViewEncoder(config.profile_dim, config.latent_dim, config.dropout)
            if config.profile_dim > 0 else None
        )
        self.graph_encoder = (
            AuxiliaryViewEncoder(config.graph_dim, config.latent_dim, config.dropout)
            if config.graph_dim > 0 else None
        )
        self.robust_multibranch = self.profile_encoder is not None or self.graph_encoder is not None
        self.profile_classifier = (
            nn.Sequential(
                nn.LayerNorm(config.latent_dim), nn.Dropout(config.dropout),
                nn.Linear(config.latent_dim, config.num_classes),
            )
            if self.profile_encoder is not None else None
        )
        self.graph_classifier = (
            nn.Sequential(
                nn.LayerNorm(config.latent_dim), nn.Dropout(config.dropout),
                nn.Linear(config.latent_dim, config.num_classes),
            )
            if self.graph_encoder is not None else None
        )
        if self.robust_multibranch:
            branch_count = 1 + int(self.profile_encoder is not None) + int(self.graph_encoder is not None)
            self.branch_gate = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(config.latent_dim), nn.Linear(config.latent_dim, 1))
                for _ in range(branch_count)
            )
            self.branch_gate_bias = nn.Parameter(torch.zeros(branch_count))
        if config.temporal_mil:
            if config.temporal_mil_scale <= 0.0 or config.temporal_mil_temperature <= 0.0:
                raise ValueError("temporal MIL scale and temperature must be positive")
            self.temporal_risk_head = nn.Sequential(
                nn.LayerNorm(config.latent_dim),
                nn.Linear(config.latent_dim, 1),
            )
            # The temporal extension starts as an identity for the historical
            # user classifier.  Synthetic supervision, rather than random
            # initialization, is what introduces a temporal evidence margin.
            nn.init.zeros_(self.temporal_risk_head[-1].weight)
            nn.init.zeros_(self.temporal_risk_head[-1].bias)

    def _temporal_mil_evidence(
        self, risk_logits: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Length-normalized LogSumExp evidence from per-behaviour risk scores."""
        temperature = self.config.temporal_mil_temperature
        scaled = risk_logits / temperature
        scaled = scaled.masked_fill(~sequence_mask, float("-inf"))
        valid = sequence_mask.sum(dim=1).clamp_min(1).to(risk_logits.dtype)
        return temperature * (torch.logsumexp(scaled, dim=1) - valid.log())

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        sequence_mask: torch.Tensor,
        image_present: torch.Tensor,
        sample_latent: bool | None = None,
        *,
        profile: torch.Tensor | None = None,
        graph: torch.Tensor | None = None,
        profile_present: torch.Tensor | None = None,
        graph_present: torch.Tensor | None = None,
        behavior_present: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.mvae(
            text, image, sequence_mask, image_present, sample=sample_latent
        )
        features, mask = outputs["latent"], sequence_mask
        risk_logits: torch.Tensor | None = None
        if self.config.temporal_mil:
            # These are deliberately computed on the uncompressed sequence, so
            # one score remains aligned with one retained user behaviour.
            risk_logits = self.temporal_risk_head(features).squeeze(-1)
            risk_logits = risk_logits.masked_fill(~sequence_mask, float("-inf"))
            outputs["temporal_risk_logits"] = risk_logits
            outputs["temporal_mil_evidence"] = self._temporal_mil_evidence(
                risk_logits, sequence_mask
            )
        if self.full_encoder is not None:
            features = self.full_encoder(features, src_key_padding_mask=~mask)
            features = features * mask.unsqueeze(-1)
        if self.informer_encoder is not None:
            features, mask = self.informer_encoder(features, mask)
        for stage in self.stages:
            features, mask = stage(features, mask)
        if self.config.pooling_type == "mean":
            weights = mask.to(features.dtype).unsqueeze(-1)
            pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        elif self.config.pooling_type == "max":
            pooled = features.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        elif self.config.pooling_type == "attention":
            scores = self.pooling_score(features).squeeze(-1).masked_fill(~mask, float("-inf"))
            weights = scores.softmax(dim=1).unsqueeze(-1)
            pooled = (features * weights).sum(dim=1)
        else:
            query = self.cls_query.expand(features.shape[0], -1, -1)
            pooled, _ = self.cls_attention(
                query, features, features, key_padding_mask=~mask, need_weights=False
            )
            pooled = pooled.squeeze(1)
        if self.robust_multibranch:
            if behavior_present is None:
                observed = (text.abs().amax(dim=-1) > 1e-8) | (image.abs().amax(dim=-1) > 1e-8)
                behavior_present = (observed & sequence_mask).any(dim=1)
            else:
                behavior_present = behavior_present.to(dtype=torch.bool, device=pooled.device)
            embeddings = [pooled]
            branch_logits = [self.classifier(pooled)]
            availability = [behavior_present]
            branch_names = ["behavior"]
            if self.profile_encoder is not None:
                if profile is None:
                    profile = pooled.new_zeros((pooled.shape[0], self.config.profile_dim))
                    profile_present = torch.zeros(pooled.shape[0], dtype=torch.bool, device=pooled.device)
                if profile.shape != (pooled.shape[0], self.config.profile_dim):
                    raise ValueError("profile feature shape does not match model config")
                if profile_present is None:
                    profile_present = torch.ones(pooled.shape[0], dtype=torch.bool, device=pooled.device)
                profile_embedding = self.profile_encoder(profile)
                embeddings.append(profile_embedding)
                assert self.profile_classifier is not None
                branch_logits.append(self.profile_classifier(profile_embedding))
                availability.append(profile_present.to(dtype=torch.bool, device=pooled.device))
                branch_names.append("profile")
            if self.graph_encoder is not None:
                if graph is None:
                    graph = pooled.new_zeros((pooled.shape[0], self.config.graph_dim))
                    graph_present = torch.zeros(pooled.shape[0], dtype=torch.bool, device=pooled.device)
                if graph.shape != (pooled.shape[0], self.config.graph_dim):
                    raise ValueError("graph feature shape does not match model config")
                if graph_present is None:
                    graph_present = torch.ones(pooled.shape[0], dtype=torch.bool, device=pooled.device)
                graph_embedding = self.graph_encoder(graph)
                embeddings.append(graph_embedding)
                assert self.graph_classifier is not None
                branch_logits.append(self.graph_classifier(graph_embedding))
                availability.append(graph_present.to(dtype=torch.bool, device=pooled.device))
                branch_names.append("graph")
            branch_embeddings = torch.stack(embeddings, dim=1)
            branch_present = torch.stack(availability, dim=1)
            # A jointly erased input contains no evidence.  Keeping the zeroed
            # behaviour branch numerically active avoids NaNs without claiming
            # robustness to removal of every input view.
            no_evidence = ~branch_present.any(dim=1)
            if no_evidence.any():
                branch_present = branch_present.clone()
                branch_present[no_evidence, 0] = True
            gate_scores = torch.cat(
                [gate(value).squeeze(-1).unsqueeze(1) for gate, value in zip(self.branch_gate, embeddings, strict=True)],
                dim=1,
            ) + self.branch_gate_bias
            gate_scores = gate_scores.masked_fill(~branch_present, float("-inf"))
            branch_weights = (gate_scores / self.config.gate_temperature).softmax(dim=1)
            if self.config.behavior_priority:
                behavior_only = torch.zeros_like(branch_weights)
                behavior_only[:, 0] = 1.0
                branch_weights = torch.where(
                    branch_present[:, :1], behavior_only, branch_weights
                )
            stacked_logits = torch.stack(branch_logits, dim=1)
            user_logits = (stacked_logits * branch_weights.unsqueeze(-1)).sum(dim=1)
            outputs["branch_embeddings"] = branch_embeddings
            outputs["branch_present"] = branch_present
            outputs["branch_weights"] = branch_weights
            outputs["branch_logits"] = stacked_logits
            outputs["branch_names"] = branch_names
        else:
            user_logits = self.classifier(pooled)
        outputs["user_logits"] = user_logits
        logits = user_logits
        if risk_logits is not None:
            # Add equal and opposite terms to make the added evidence an exact
            # increment of the spammer-minus-normal decision margin.
            adjustment = self.config.temporal_mil_scale * outputs["temporal_mil_evidence"]
            outputs["audit_logits"] = user_logits + torch.stack(
                (-0.5 * adjustment, 0.5 * adjustment), dim=-1
            )
        # ``logits`` remains the original user-identification head.  This
        # preserves all historical user-level training/evaluation interfaces;
        # ``audit_logits`` is explicitly consumed only by the new temporal
        # behaviour-audit objective and frozen localization evaluator.
        outputs["logits"] = logits
        outputs["final_sequence_mask"] = mask
        return outputs

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        text: torch.Tensor,
        image: torch.Tensor,
        sequence_mask: torch.Tensor,
        image_present: torch.Tensor,
        labels: torch.Tensor,
        mvae_weight: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        class_loss = F.cross_entropy(outputs["logits"], labels)

        # Empty histories use the existing profile/graph fallback. The numerical
        # attention slot must not become a reconstruction or prior-training target.
        if "branch_present" in outputs:
            sequence_mask = sequence_mask & outputs["branch_present"][:, :1]

        zero = class_loss.new_zeros(())
        branch_loss = zero
        if "branch_logits" in outputs and self.config.branch_aux_weight > 0.0:
            branch_logits = outputs["branch_logits"]
            branch_present = outputs["branch_present"]
            expanded_labels = labels.unsqueeze(1).expand_as(branch_present)
            branch_loss = F.cross_entropy(
                branch_logits[branch_present], expanded_labels[branch_present]
            )
        reconstruction_enabled = self.config.fusion_type in {
            "deterministic_ae", "independent_vae", "mvae_no_poe", "mean_gaussian",
            "poe", "poe_no_prior", "corruption_aware",
        }
        if reconstruction_enabled:
            text_error = F.mse_loss(
                outputs["text_reconstruction"], text, reduction="none"
            ).mean(-1)
            text_loss = (text_error * sequence_mask).sum() / sequence_mask.sum().clamp_min(1)
            valid_images = sequence_mask & image_present
            image_error = F.mse_loss(
                outputs["image_reconstruction"], image, reduction="none"
            ).mean(-1)
            image_loss = (image_error * valid_images).sum() / valid_images.sum().clamp_min(1)
        else:
            text_loss = zero
            image_loss = zero

        mean = outputs["joint_mean"]
        log_var = outputs["joint_log_var"]
        kl_per_token = -0.5 * (1.0 + log_var - mean.square() - log_var.exp()).sum(-1)
        if self.config.fusion_type == "deterministic_ae" or not reconstruction_enabled:
            kl_loss = zero
        else:
            kl_loss = (kl_per_token * sequence_mask).sum() / sequence_mask.sum().clamp_min(1)
        mvae_loss = text_loss + image_loss + kl_loss
        total = class_loss + mvae_weight * mvae_loss + self.config.branch_aux_weight * branch_loss
        return {
            "total": total,
            "classification": class_loss,
            "mvae": mvae_loss,
            "text_reconstruction": text_loss,
            "image_reconstruction": image_loss,
            "kl": kl_loss,
            "branch_classification": branch_loss,
        }
