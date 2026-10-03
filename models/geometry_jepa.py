"""Geometry JEPA: action-conditioned prediction of frozen VGGT latents.

This is a training-time auxiliary branch. Given leakage-free student tokens from
the current observation, an action chunk covering ``t -> t + k`` and the horizon
``k``, it predicts the pre-computed VGGT teacher latent of the future frame.

The protocol (query shape ``[M, D]``, pooled-cosine + token-MSE loss, adaptive
pooling for the feature/token compression) follows ``LDJEPA_transfer.md``.

Nothing in this module performs I/O: teacher targets are supplied by the caller.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_CONDITION_SOURCES = ("full", "zero")
TARGET_MODES = ("absolute", "residual", "joint_future_slot")


@dataclass
class GeometryJEPAConfig:
    """Resolved configuration of the Geometry JEPA branch."""

    enable: bool = False
    lambda_geo: float = 0.03
    teacher_protocol_id: str = "vggt_tlayout_3view_percam_q32_d768_centered_v3"

    # Teacher query shape. Must match the cache manifest.
    # M = queries_per_camera * num_cameras (32 * 3 for the RoboTwin T-mosaic).
    num_queries: int = 96
    hidden_dim: int = 768

    # Horizon sampling. ``horizon`` is 1-based and indexes the predicted video
    # frames, i.e. ``k`` selects ``video_frames[:, k - 1]``.
    horizon_choices: List[int] = field(default_factory=lambda: [1])
    max_horizon: int = 8

    # Perceiver resampler adapter.
    adapter_num_heads: int = 8
    adapter_depth: int = 1
    adapter_dropout: float = 0.0

    # Action-conditioned predictor.
    predictor_depth: int = 2
    predictor_num_heads: int = 8
    predictor_dropout: float = 0.0
    action_hidden_dim: int = 768
    action_condition_source: str = "full"

    # Loss.
    # "residual" predicts ``u_{t+k} - u_t``. "joint_future_slot" directly
    # predicts a jointly encoded pair target and uses J(t,t) as its baseline.
    target_mode: str = "absolute"
    alpha_cos: float = 1.0
    beta_mse: float = 0.05
    gamma_token_cos: float = 0.0
    use_pool_cosine: bool = True
    use_token_mse: bool = True
    detach_target: bool = True

    # Gradient / optimizer policy.
    freeze_wam_for_geo: bool = True
    lr_scale: float = 1.0
    log_grad_norm: bool = True

    def validate(self) -> None:
        if self.hidden_dim % self.adapter_num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"adapter_num_heads ({self.adapter_num_heads})"
            )
        if self.hidden_dim % self.predictor_num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"predictor_num_heads ({self.predictor_num_heads})"
            )
        if self.adapter_depth < 1:
            raise ValueError(
                "adapter_depth must be >= 1, otherwise the learned queries never "
                "read the student context"
            )
        if not self.horizon_choices:
            raise ValueError("horizon_choices must not be empty")
        if min(self.horizon_choices) < 1:
            raise ValueError(f"horizon_choices must be >= 1, got {self.horizon_choices}")
        if max(self.horizon_choices) > self.max_horizon:
            raise ValueError(
                f"max(horizon_choices)={max(self.horizon_choices)} exceeds "
                f"max_horizon={self.max_horizon}"
            )
        if self.num_queries < 1 or self.hidden_dim < 1:
            raise ValueError("num_queries and hidden_dim must be positive")
        if self.target_mode not in TARGET_MODES:
            raise ValueError(
                f"target_mode must be one of {TARGET_MODES}, got {self.target_mode!r}"
            )
        if self.action_condition_source not in ACTION_CONDITION_SOURCES:
            raise ValueError(
                "action_condition_source must be one of "
                f"{ACTION_CONDITION_SOURCES}, got {self.action_condition_source!r}"
            )
        if self.lr_scale <= 0:
            raise ValueError(
                f"lr_scale must be > 0, got {self.lr_scale}; use lambda_geo=0 or "
                "enable=false to switch the branch off"
            )

    def fingerprint(self) -> Dict[str, Any]:
        """Fields that must match when resuming (LDJEPA_transfer.md section 12.2).

        Everything that changes a parameter shape or the meaning of the teacher
        target. Purely optimisation-side knobs (`lambda_geo`, `lr_scale`,
        `freeze_wam_for_geo`, loss weights) are deliberately excluded so they can
        be retuned across a resume. `action_condition_source` is checked
        separately by `assert_checkpoint_compatible`: it changes training
        semantics without changing the state-dict structure.
        """
        return {field: getattr(self, field) for field in CHECKPOINT_FINGERPRINT_FIELDS}


CHECKPOINT_FINGERPRINT_FIELDS = (
    "teacher_protocol_id",
    "num_queries",
    "hidden_dim",
    "adapter_depth",
    "adapter_num_heads",
    "predictor_depth",
    "predictor_num_heads",
    "action_hidden_dim",
    "max_horizon",  # sizes the horizon embedding table
    "target_mode",  # redefines what the predictor head regresses to
)


def assert_checkpoint_compatible(
    saved: Dict[str, Any], current: GeometryJEPAConfig
) -> None:
    """Fail fast when a checkpoint was written with a different JEPA geometry."""
    expected = current.fingerprint()
    mismatches = [
        f"{field}: checkpoint={saved.get(field)!r} config={value!r}"
        for field, value in expected.items()
        if field in saved and saved[field] != value
    ]
    saved_action_source = saved.get("action_condition_source")
    if saved_action_source is None:
        warnings.warn(
            "Geometry JEPA checkpoint has no action_condition_source; treating it "
            "as legacy 'full' conditioning",
            UserWarning,
            stacklevel=2,
        )
        saved_action_source = "full"
    if saved_action_source != current.action_condition_source:
        mismatches.append(
            "action_condition_source: "
            f"checkpoint={saved_action_source!r} "
            f"config={current.action_condition_source!r}"
        )
    if mismatches:
        raise ValueError(
            "Geometry JEPA checkpoint is incompatible with the current config:\n  "
            + "\n  ".join(mismatches)
        )


def resize_feature_dim(tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Resample the last dimension of ``[B, N, C]`` to ``[B, N, target_dim]``.

    The feature vector of each token is treated as a 1-D signal and resampled
    with adaptive average pooling. This has no learnable parameters, so the
    teacher cache stays independent of the student architecture.
    """
    batch, num_tokens, channels = tokens.shape
    if channels == target_dim:
        return tokens
    flat = tokens.reshape(batch * num_tokens, 1, channels).float()
    resized = F.adaptive_avg_pool1d(flat, target_dim)
    return resized.reshape(batch, num_tokens, target_dim).to(tokens.dtype)


def _flatten_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """Normalise a student representation to ``[B, N, C]``."""
    if x.ndim == 2:  # [B, C]
        return x.unsqueeze(1)
    if x.ndim == 3:  # [B, N, C]
        return x
    if x.ndim == 4:  # [B, C, H, W]
        return x.flatten(2).transpose(1, 2)
    if x.ndim == 5:  # [B, C, T, H, W]
        return x.flatten(2).transpose(1, 2)
    raise ValueError(f"Unsupported student token shape: {tuple(x.shape)}")


class _CrossAttentionBlock(nn.Module):
    """One Perceiver block: cross-attention from queries to context, then MLP."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normed_kv = self.norm_kv(context)
        attn_out, _ = self.attn(
            self.norm_q(queries), normed_kv, normed_kv, need_weights=False
        )
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm_mlp(queries))
        return queries


class PerceiverResamplerAdapter(nn.Module):
    """Compress an arbitrary student representation into ``[B, M, D]`` queries.

    Args:
        num_queries: Number of learned latent queries ``M``.
        hidden_dim: Query/context width ``D``.
        num_heads: Cross-attention heads.
        depth: Number of Perceiver blocks.
        dropout: Attention dropout.
        num_pos_tokens: When > 0, a learnable positional embedding of that many
            tokens is added to the context. The student token grid carries no
            positional information at the extraction point, so this restores the
            spatial identity of each patch.
    """

    def __init__(
        self,
        num_queries: int = 16,
        hidden_dim: int = 768,
        num_heads: int = 8,
        depth: int = 1,
        dropout: float = 0.0,
        num_pos_tokens: int = 0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_pos_tokens = num_pos_tokens

        self.learned_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        if num_pos_tokens > 0:
            self.pos_embed = nn.Parameter(torch.randn(1, num_pos_tokens, hidden_dim) * 0.02)
        else:
            self.register_parameter("pos_embed", None)

        self.blocks = nn.ModuleList(
            [_CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, student_tokens: torch.Tensor) -> torch.Tensor:
        """``student_tokens`` -> ``[B, M, D]``."""
        tokens = _flatten_to_tokens(student_tokens)
        tokens = tokens.to(self.input_proj.weight.dtype)
        tokens = resize_feature_dim(tokens, self.hidden_dim)
        context = self.input_norm(self.input_proj(tokens))

        if self.pos_embed is not None:
            if context.shape[1] != self.num_pos_tokens:
                raise ValueError(
                    f"Adapter was built for {self.num_pos_tokens} student tokens but "
                    f"received {context.shape[1]}"
                )
            context = context + self.pos_embed

        queries = self.learned_queries.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.out_norm(queries)


class ActionConditionedLDPredictor(nn.Module):
    """Predict the future teacher query from ``(q_t, action chunk, horizon)``."""

    def __init__(
        self,
        num_queries: int = 16,
        hidden_dim: int = 768,
        action_hidden_dim: int = 768,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_horizon: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.action_hidden_dim = action_hidden_dim
        self.max_horizon = max_horizon

        self.action_encoder = nn.Sequential(
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.horizon_embedding = nn.Embedding(max_horizon + 1, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        q_t: torch.Tensor,
        actions: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q_t: ``[B, M, D]`` current JEPA queries.
            actions: ``[B, T_k, A]`` actions covering ``t -> t + k``, no padding.
            horizon: ``[B]`` integer horizons in ``[1, max_horizon]``.

        Returns:
            ``[B, M, D]`` predicted teacher query.
        """
        batch = q_t.shape[0]
        dtype = self.horizon_embedding.weight.dtype

        if actions.ndim != 3:
            raise ValueError(f"actions must be [B, T_k, A], got {tuple(actions.shape)}")
        if actions.shape[0] != batch:
            raise ValueError(
                f"batch mismatch: q_t={q_t.shape[0]} actions={actions.shape[0]}"
            )

        # Flatten the action chunk and resample it to a fixed width. Ordering is
        # preserved implicitly by the flatten layout, so the caller must keep the
        # [time, action_dim] axis order stable and strip any padding.
        action_flat = actions.reshape(batch, 1, -1).float()
        action_pooled = F.adaptive_avg_pool1d(action_flat, self.action_hidden_dim)
        action_token = self.action_encoder(action_pooled.to(dtype))  # [B, 1, D]

        horizon = horizon.to(device=q_t.device, dtype=torch.long)
        if horizon.ndim != 1 or horizon.shape[0] != batch:
            raise ValueError(f"horizon must be [B], got {tuple(horizon.shape)}")
        if int(horizon.min()) < 1 or int(horizon.max()) > self.max_horizon:
            raise ValueError(
                f"horizon out of range [1, {self.max_horizon}]: "
                f"min={int(horizon.min())} max={int(horizon.max())}"
            )
        horizon_token = self.horizon_embedding(horizon).unsqueeze(1)  # [B, 1, D]

        sequence = torch.cat([action_token, horizon_token, q_t.to(dtype)], dim=1)
        out = self.transformer(sequence)
        return self.out_norm(out[:, 2:])


class LDJEPALoss(nn.Module):
    """Pooled cosine + token MSE + per-token cosine between prediction and target."""

    def __init__(
        self,
        alpha_cos: float = 1.0,
        beta_mse: float = 0.05,
        gamma_token_cos: float = 0.0,
        use_pool_cosine: bool = True,
        use_token_mse: bool = True,
        detach_target: bool = True,
    ):
        super().__init__()
        self.alpha_cos = alpha_cos
        self.beta_mse = beta_mse
        self.gamma_token_cos = gamma_token_cos
        self.use_pool_cosine = use_pool_cosine
        self.use_token_mse = use_token_mse
        self.detach_target = detach_target

    def forward(
        self, q_pred: torch.Tensor, u_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Always compute in float32: the branch runs under bf16 autocast/DeepSpeed.
        pred = q_pred.float()
        target = u_target.float()
        if self.detach_target:
            target = target.detach()

        zero = pred.new_zeros(())

        if self.use_pool_cosine:
            cos_loss = (
                1.0
                - F.cosine_similarity(pred.mean(dim=1), target.mean(dim=1), dim=-1)
            ).mean()
        else:
            cos_loss = zero

        mse_loss = F.mse_loss(pred, target) if self.use_token_mse else zero

        # Pooling over M hides the token structure; this term scores each query.
        if self.gamma_token_cos:
            token_cos_loss = (
                1.0 - F.cosine_similarity(pred, target, dim=-1)
            ).mean()
        else:
            token_cos_loss = zero

        total = (
            self.alpha_cos * cos_loss
            + self.beta_mse * mse_loss
            + self.gamma_token_cos * token_cos_loss
        )
        return {
            "ld_jepa_loss": total,
            "ld_cos_loss": cos_loss,
            "ld_mse_loss": mse_loss,
            "ld_token_cos_loss": token_cos_loss,
        }


class GeometryJEPAModule(nn.Module):
    """Adapter + action-conditioned predictor + loss.

    Teacher targets are loaded by the data pipeline and passed in, so this module
    stays free of storage-backend assumptions.
    """

    def __init__(self, config: GeometryJEPAConfig, student_num_tokens: Optional[int] = None):
        super().__init__()
        config.validate()
        self.config = config

        self.adapter = PerceiverResamplerAdapter(
            num_queries=config.num_queries,
            hidden_dim=config.hidden_dim,
            num_heads=config.adapter_num_heads,
            depth=config.adapter_depth,
            dropout=config.adapter_dropout,
            num_pos_tokens=int(student_num_tokens or 0),
        )
        self.predictor = ActionConditionedLDPredictor(
            num_queries=config.num_queries,
            hidden_dim=config.hidden_dim,
            action_hidden_dim=config.action_hidden_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_num_heads,
            dropout=config.predictor_dropout,
            max_horizon=config.max_horizon,
        )
        self.loss_fn = LDJEPALoss(
            alpha_cos=config.alpha_cos,
            beta_mse=config.beta_mse,
            gamma_token_cos=config.gamma_token_cos,
            use_pool_cosine=config.use_pool_cosine,
            use_token_mse=config.use_token_mse,
            detach_target=config.detach_target,
        )

    def forward(
        self,
        student_tokens: torch.Tensor,
        actions: torch.Tensor,
        horizon: torch.Tensor,
        teacher_target: torch.Tensor,
        teacher_anchor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_tokens: ``[B, N, C]`` leakage-free current-observation tokens.
            actions: ``[B, T_k, A]`` actions covering ``t -> t + k``.
            horizon: ``[B]`` horizons.
            teacher_target: ``[B, M, D]`` frozen VGGT latents at ``t + k``.
            teacher_anchor: ``[B, M, D]`` endpoint latent at ``t`` for residual
                mode, or the static pair target ``J(t,t)`` for joint mode.
        """
        q_t = self.adapter(student_tokens)
        q_pred = self.predictor(q_t, actions, horizon)

        expected = (q_pred.shape[0], self.config.num_queries, self.config.hidden_dim)
        target = self._check_teacher(teacher_target, expected, "teacher_target")
        anchor = (
            None
            if teacher_anchor is None
            else self._check_teacher(teacher_anchor, expected, "teacher_anchor")
        )

        if self.config.target_mode == "residual":
            if anchor is None:
                raise ValueError(
                    "target_mode='residual' requires `teacher_anchor` (the teacher "
                    "latent at t, i.e. horizon 0)"
                )
            target = target - anchor
        elif self.config.target_mode == "joint_future_slot" and anchor is None:
            raise ValueError(
                "target_mode='joint_future_slot' requires `teacher_anchor` "
                "containing the static pair target J(t,t)"
            )

        losses = self.loss_fn(q_pred, target)
        outputs = {
            "geometry_jepa_loss": losses["ld_jepa_loss"],
            "geo_cos_loss": losses["ld_cos_loss"],
            "geo_mse_loss": losses["ld_mse_loss"],
            "geo_token_cos_loss": losses["ld_token_cos_loss"],
            "q_t": q_t,
            "q_pred": q_pred,
        }
        if anchor is not None:
            outputs.update(self._copy_baseline(anchor, target))
        return outputs

    def _check_teacher(
        self, tensor: torch.Tensor, expected, name: str
    ) -> torch.Tensor:
        tensor = tensor.to(device=self.adapter.learned_queries.device)
        if tuple(tensor.shape) != tuple(expected):
            raise ValueError(
                f"{name} shape mismatch: expected {tuple(expected)}, got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN/Inf")
        return tensor

    @torch.no_grad()
    def _copy_baseline(
        self, anchor: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Score the trivial predictor that repeats the present teacher latent.

        Section 17.3: a low absolute JEPA loss is meaningless on its own, because
        neighbouring VGGT latents are already highly aligned. ``geo_skill > 0``
        means the branch actually beats "predict no change".
        """
        trivial = torch.zeros_like(anchor) if self.config.target_mode == "residual" else anchor
        baseline = self.loss_fn(trivial, target)
        return {
            "geo_copy_loss": baseline["ld_jepa_loss"],
            "geo_copy_cos_loss": baseline["ld_cos_loss"],
        }
