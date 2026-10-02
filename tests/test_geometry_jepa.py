"""Validation tests for the Geometry JEPA branch (LDJEPA_transfer.md section 15).

These tests exercise the JEPA modules and the transition/teacher-cache plumbing.
They do not require VGGT, the WAN checkpoints or a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.utils.teacher_store import (
    TemporalPairTeacherStore,
    TeacherCacheError,
    TeacherStore,
    create_teacher_store,
    make_sample_key,
    save_index,
    split_sample_key,
)
from data.utils.vggt_teacher import (
    DEFAULT_PROTOCOL,
    JOINT_FUTURE_SLOT_PROTOCOL_ID,
    TEMPORAL_PAIR_TARGET,
    WINDOW_CACHE_LAYOUT,
    TeacherProtocol,
    assert_manifest_matches,
    assert_protocol_id_covers_overrides,
    build_camera_tags,
    build_temporal_pair_target,
    build_teacher_target,
    letterbox,
    split_t_layout,
    write_manifest,
)
from models.geometry_jepa import (
    CHECKPOINT_FINGERPRINT_FIELDS,
    GeometryJEPAConfig,
    GeometryJEPAModule,
    LDJEPALoss,
    PerceiverResamplerAdapter,
    assert_checkpoint_compatible,
)

BATCH = 2
NUM_TOKENS = 120  # (384 // 32) * (320 // 32)
STUDENT_DIM = 3072
QUERIES_PER_CAMERA, NUM_VIEWS = 32, 3
M, D = QUERIES_PER_CAMERA * NUM_VIEWS, 768  # 96 x 768
ACTION_DIM = 14
FREQ_RATIO = 2
DOWNSAMPLE = 3
WINDOW_STRIDE = FREQ_RATIO * DOWNSAMPLE  # 6


@pytest.fixture
def config() -> GeometryJEPAConfig:
    return GeometryJEPAConfig(
        enable=True,
        num_queries=M,
        hidden_dim=D,
        horizon_choices=[1, 2],
        max_horizon=8,
        action_hidden_dim=D,
    )


@pytest.fixture
def module(config: GeometryJEPAConfig) -> GeometryJEPAModule:
    torch.manual_seed(0)
    return GeometryJEPAModule(config, student_num_tokens=NUM_TOKENS)


def _student() -> torch.Tensor:
    return torch.randn(BATCH, NUM_TOKENS, STUDENT_DIM)


def _actions(horizon: int) -> torch.Tensor:
    return torch.randn(BATCH, horizon * FREQ_RATIO, ACTION_DIM)


def _horizon(k: int) -> torch.Tensor:
    return torch.full((BATCH,), k, dtype=torch.long)


# --- 15.1 shapes ---------------------------------------------------------


def test_shapes(module: GeometryJEPAModule):
    out = module(
        student_tokens=_student(),
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=torch.randn(BATCH, M, D),
    )
    assert out["q_t"].shape == (BATCH, M, D)
    assert out["q_pred"].shape == (BATCH, M, D)
    assert out["geometry_jepa_loss"].ndim == 0
    assert torch.isfinite(out["geometry_jepa_loss"])


@pytest.mark.parametrize(
    "shape",
    [(BATCH, STUDENT_DIM), (BATCH, 8, STUDENT_DIM), (BATCH, STUDENT_DIM, 4, 5)],
)
def test_adapter_accepts_arbitrary_student_layouts(shape):
    adapter = PerceiverResamplerAdapter(num_queries=M, hidden_dim=D)
    assert adapter(torch.randn(*shape)).shape == (shape[0], M, D)


def test_adapter_rejects_wrong_token_count_when_pos_embed_is_used():
    adapter = PerceiverResamplerAdapter(
        num_queries=M, hidden_dim=D, num_pos_tokens=NUM_TOKENS
    )
    with pytest.raises(ValueError, match="student tokens"):
        adapter(torch.randn(BATCH, NUM_TOKENS + 1, STUDENT_DIM))


# --- 15.2 future leakage -------------------------------------------------


def test_student_tokens_are_free_of_future_leakage():
    """Only the first `tokens_per_frame` tokens may feed the JEPA branch.

    Motus replaces latent frame 0 with the clean condition-frame latent and the
    WAN patch embedding uses a temporal kernel of 1, so mutating the future
    latent frames must leave the student slice untouched.
    """
    torch.manual_seed(0)
    channels, lat_t, lat_h, lat_w = 48, 3, 12, 10
    patch = torch.nn.Conv3d(channels, STUDENT_DIM, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    tokens_per_frame = (lat_h // 2) * (lat_w // 2)

    latent_a = torch.randn(BATCH, channels, lat_t, lat_h, lat_w)
    latent_b = latent_a.clone()
    latent_b[:, :, 1:] = torch.randn_like(latent_b[:, :, 1:])  # only the future changes

    with torch.no_grad():
        tokens_a = patch(latent_a).flatten(2).transpose(1, 2)
        tokens_b = patch(latent_b).flatten(2).transpose(1, 2)

    torch.testing.assert_close(
        tokens_a[:, :tokens_per_frame], tokens_b[:, :tokens_per_frame]
    )
    assert not torch.allclose(tokens_a, tokens_b), "future tokens should differ"


# --- 15.3 / 15.4 action and horizon sensitivity --------------------------


def test_predictor_is_sensitive_to_actions(module: GeometryJEPAModule):
    module.eval()
    q_t = torch.randn(BATCH, M, D)
    with torch.no_grad():
        pred_a = module.predictor(q_t, _actions(1), _horizon(1))
        pred_b = module.predictor(q_t, _actions(1) + 5.0, _horizon(1))
    assert not torch.allclose(pred_a, pred_b, atol=1e-4)


def test_predictor_is_sensitive_to_horizon(module: GeometryJEPAModule):
    module.eval()
    q_t = torch.randn(BATCH, M, D)
    actions = _actions(2)
    with torch.no_grad():
        pred_1 = module.predictor(q_t, actions[:, :FREQ_RATIO], _horizon(1))
        pred_2 = module.predictor(q_t, actions, _horizon(2))
    assert not torch.allclose(pred_1, pred_2, atol=1e-4)


def test_predictor_rejects_out_of_range_horizon(module: GeometryJEPAModule):
    with pytest.raises(ValueError, match="horizon out of range"):
        module.predictor(torch.randn(BATCH, M, D), _actions(1), _horizon(99))


# --- loss ----------------------------------------------------------------


def test_loss_is_zero_for_a_perfect_prediction():
    loss_fn = LDJEPALoss()
    target = torch.randn(BATCH, M, D)
    losses = loss_fn(target.clone(), target)
    assert losses["ld_jepa_loss"].abs() < 1e-5
    assert losses["ld_cos_loss"].abs() < 1e-5
    assert losses["ld_mse_loss"].abs() < 1e-5


# --- 17.3 residual target and the copy baseline --------------------------


def _collinear_teacher(drift: float = 0.3):
    """Two teacher rows shaped like the cache: neighbours are nearly collinear."""
    anchor = torch.randn(BATCH, M, D)
    return anchor, anchor + drift * torch.randn(BATCH, M, D)


def _residual_module() -> GeometryJEPAModule:
    torch.manual_seed(0)
    return GeometryJEPAModule(
        GeometryJEPAConfig(
            enable=True,
            num_queries=M,
            hidden_dim=D,
            action_hidden_dim=D,
            target_mode="residual",
        ),
        student_num_tokens=NUM_TOKENS,
    )


def test_absolute_mode_copy_baseline_is_nearly_free():
    """Why `target_mode='absolute'` is degenerate: repeating `u_t` almost wins."""
    anchor, target = _collinear_teacher()
    loss_fn = LDJEPALoss()
    copy_cos = loss_fn(anchor, target)["ld_cos_loss"]
    random_cos = loss_fn(torch.randn_like(target), target)["ld_cos_loss"]
    assert copy_cos < 0.2 * random_cos


def test_copy_baseline_scores_the_trivial_predictor(module: GeometryJEPAModule):
    anchor, target = _collinear_teacher()
    out = module(
        student_tokens=_student(),
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=target,
        teacher_anchor=anchor,
    )
    expected = module.loss_fn(anchor, target)
    torch.testing.assert_close(out["geo_copy_loss"], expected["ld_jepa_loss"])
    torch.testing.assert_close(out["geo_copy_cos_loss"], expected["ld_cos_loss"])


def test_residual_mode_targets_the_difference():
    module = _residual_module()
    anchor, target = _collinear_teacher()
    out = module(
        student_tokens=_student(),
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=target,
        teacher_anchor=anchor,
    )
    expected = module.loss_fn(out["q_pred"], target - anchor)["ld_jepa_loss"]
    torch.testing.assert_close(out["geometry_jepa_loss"], expected)


def test_residual_mode_makes_the_copy_baseline_expensive():
    """The whole point: "predict no change" must stop being a cheap solution."""
    module = _residual_module()
    anchor, target = _collinear_teacher()
    out = module(
        student_tokens=_student(),
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=target,
        teacher_anchor=anchor,
    )
    assert out["geo_copy_cos_loss"] > 0.99


def test_residual_mode_requires_an_anchor():
    module = _residual_module()
    with pytest.raises(ValueError, match="requires `teacher_anchor`"):
        module(
            student_tokens=_student(),
            actions=_actions(1),
            horizon=_horizon(1),
            teacher_target=torch.randn(BATCH, M, D),
        )


def test_residual_target_carries_no_gradient():
    """`detach_target` must apply to the residual, not just to `u_{t+k}`."""
    module = _residual_module()
    anchor = torch.randn(BATCH, M, D, requires_grad=True)
    target = torch.randn(BATCH, M, D, requires_grad=True)
    out = module(
        student_tokens=_student(),
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=target,
        teacher_anchor=anchor,
    )
    out["geometry_jepa_loss"].backward()
    assert anchor.grad is None
    assert target.grad is None


def test_joint_mode_uses_static_pair_as_the_baseline():
    torch.manual_seed(0)
    module = GeometryJEPAModule(
        GeometryJEPAConfig(
            enable=True,
            num_queries=M,
            hidden_dim=D,
            action_hidden_dim=D,
            target_mode="joint_future_slot",
        ),
        student_num_tokens=NUM_TOKENS,
    )
    static = torch.randn(BATCH, M, D)
    target = torch.randn(BATCH, M, D)
    out = module(
        student_tokens=_student(),
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=target,
        teacher_anchor=static,
    )

    expected_target = module.loss_fn(out["q_pred"], target)["ld_jepa_loss"]
    expected_static = module.loss_fn(static, target)["ld_jepa_loss"]
    torch.testing.assert_close(out["geometry_jepa_loss"], expected_target)
    torch.testing.assert_close(out["geo_copy_loss"], expected_static)


def test_joint_mode_requires_the_static_pair():
    module = GeometryJEPAModule(
        GeometryJEPAConfig(
            enable=True,
            num_queries=M,
            hidden_dim=D,
            action_hidden_dim=D,
            target_mode="joint_future_slot",
        ),
        student_num_tokens=NUM_TOKENS,
    )
    with pytest.raises(ValueError, match="static pair"):
        module(
            student_tokens=_student(),
            actions=_actions(1),
            horizon=_horizon(1),
            teacher_target=torch.randn(BATCH, M, D),
        )


def test_zero_residual_stays_finite():
    """A static window gives `u_{t+k} == u_t`; cosine of a zero target must not NaN."""
    module = _residual_module()
    anchor = torch.randn(BATCH, M, D)
    out = module(
        student_tokens=_student(),
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=anchor.clone(),
        teacher_anchor=anchor,
    )
    for key in ("geometry_jepa_loss", "geo_cos_loss", "geo_token_cos_loss", "geo_copy_loss"):
        assert torch.isfinite(out[key]), key
    # An undefined direction scores a full miss rather than a free win.
    torch.testing.assert_close(out["geo_cos_loss"], torch.ones(()))


def test_pooling_is_permutation_invariant_but_token_cosine_is_not():
    torch.manual_seed(0)
    target = torch.randn(BATCH, M, D)
    pred = torch.randn(BATCH, M, D)
    perm = torch.randperm(M)
    pooled = LDJEPALoss(alpha_cos=1.0, beta_mse=0.0, gamma_token_cos=0.0)
    tokens = LDJEPALoss(alpha_cos=0.0, beta_mse=0.0, gamma_token_cos=1.0)

    # Shuffling the queries cannot move a loss that averages over them ...
    torch.testing.assert_close(
        pooled(pred, target)["ld_jepa_loss"],
        pooled(pred[:, perm], target)["ld_jepa_loss"],
    )
    # ... so a permuted copy of the target is a perfect pooled prediction ...
    assert pooled(target[:, perm], target)["ld_jepa_loss"] < 1e-5
    # ... and only the per-token term rejects it.
    assert tokens(target[:, perm], target)["ld_jepa_loss"] > 0.5


def test_module_rejects_mismatched_target(module: GeometryJEPAModule):
    with pytest.raises(ValueError, match="teacher_target shape mismatch"):
        module(
            student_tokens=_student(),
            actions=_actions(1),
            horizon=_horizon(1),
            teacher_target=torch.randn(BATCH, M, D + 1),
        )


def test_module_rejects_non_finite_target(module: GeometryJEPAModule):
    target = torch.randn(BATCH, M, D)
    target[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        module(
            student_tokens=_student(),
            actions=_actions(1),
            horizon=_horizon(1),
            teacher_target=target,
        )


# --- 15.6 gradients ------------------------------------------------------


def test_frozen_mode_keeps_gradients_out_of_the_wam(module: GeometryJEPAModule):
    upstream = torch.nn.Linear(STUDENT_DIM, STUDENT_DIM)
    student = upstream(_student())

    out = module(
        student_tokens=student.detach(),  # freeze_wam_for_geo=True
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=torch.randn(BATCH, M, D),
    )
    out["geometry_jepa_loss"].backward()

    assert module.adapter.learned_queries.grad is not None
    assert module.predictor.horizon_embedding.weight.grad is not None
    assert upstream.weight.grad is None


def test_open_mode_propagates_into_the_wam(module: GeometryJEPAModule):
    upstream = torch.nn.Linear(STUDENT_DIM, STUDENT_DIM)
    student = upstream(_student())

    out = module(
        student_tokens=student,  # freeze_wam_for_geo=False
        actions=_actions(1),
        horizon=_horizon(1),
        teacher_target=torch.randn(BATCH, M, D),
    )
    out["geometry_jepa_loss"].backward()

    assert upstream.weight.grad is not None
    assert torch.isfinite(upstream.weight.grad).all()
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


# --- config validation ---------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"adapter_num_heads": 7}, "divisible"),
        ({"adapter_depth": 0}, "adapter_depth"),
        ({"horizon_choices": []}, "must not be empty"),
        ({"horizon_choices": [0]}, ">= 1"),
        ({"horizon_choices": [9], "max_horizon": 8}, "exceeds"),
    ],
)
def test_config_validation(overrides, match):
    cfg = GeometryJEPAConfig(num_queries=M, hidden_dim=D, **overrides)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


# --- teacher protocol ----------------------------------------------------


def test_split_t_layout_recovers_three_views():
    # RoboTwin mosaic: head 480x640 on top, two 240x320 wrists below.
    head = torch.zeros(1, 3, 480, 640)
    left = torch.ones(1, 3, 240, 320)
    right = torch.full((1, 3, 240, 320), 0.5)
    mosaic = torch.cat([head, torch.cat([left, right], dim=3)], dim=2)
    assert mosaic.shape == (1, 3, 720, 640)

    views = split_t_layout(mosaic)
    assert views.shape[:2] == (1, 3)
    assert views[0, 0].mean() == pytest.approx(0.0)
    assert views[0, 1].mean() == pytest.approx(1.0, abs=1e-3)
    assert views[0, 2].mean() == pytest.approx(0.5, abs=1e-3)


def test_letterbox_preserves_aspect_ratio_with_black_padding():
    out = letterbox(torch.ones(1, 3, 480, 640), (518, 518))
    assert out.shape == (1, 3, 518, 518)
    # 640 -> 518, 480 -> 388, so 65 black rows top and bottom.
    assert out[0, :, :64, :].abs().max() == 0.0
    assert out[0, :, 260, 260] == pytest.approx(1.0)


def test_camera_tags_are_deterministic_and_order_dependent():
    a = build_camera_tags(3, D)
    b = build_camera_tags(3, D)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a[0], a[1])


def test_protocol_rejects_inconsistent_query_counts():
    with pytest.raises(ValueError, match="num_queries"):
        TeacherProtocol(queries_per_camera=32, num_queries=16)


def _joint_protocol(**overrides):
    values = dict(
        protocol_id=JOINT_FUTURE_SLOT_PROTOCOL_ID,
        target_construction=TEMPORAL_PAIR_TARGET,
        cache_layout=WINDOW_CACHE_LAYOUT,
        sequence_length=2,
        pair_order="current,future",
        selected_temporal_slot="future",
        static_baseline="repeat_current_pair",
        cached_horizons=(1, 2, 4, 8),
        slot_schema=("static", "h1", "h2", "h4", "h8"),
        num_video_frames=8,
    )
    values.update(overrides)
    return TeacherProtocol(**values)


def test_joint_protocol_freezes_the_window_slot_contract():
    protocol = _joint_protocol()
    assert protocol.to_dict()["cached_horizons"] == [1, 2, 4, 8]
    with pytest.raises(ValueError, match="slot_schema"):
        _joint_protocol(slot_schema=("static", "h1"))
    with pytest.raises(ValueError, match="current,future"):
        _joint_protocol(pair_order="future,current")


class _FakeVGGT:
    """Deterministic stand-in whose tokens depend only on the given view."""

    N_TOKENS, C_V = 20, 2048

    def __init__(self):
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def aggregator(self, images):  # images: [B*V, 1, 3, H, W]
        signature = images.squeeze(1).mean(dim=(2, 3))  # [B*V, 3]
        basis = torch.linspace(0.1, 1.0, self.N_TOKENS * self.C_V).reshape(
            1, self.N_TOKENS, self.C_V
        )
        tokens = signature[:, :1, None] * basis + signature[:, 1:2, None]
        return [None, tokens.unsqueeze(1)], 5


class _FakeTemporalVGGT(_FakeVGGT):
    """Fake temporal aggregator whose selected slot depends on the ordered pair."""

    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def aggregator(self, images):
        self.last_input_shape = tuple(images.shape)
        signature = images.mean(dim=(3, 4))  # [B*V, S, 3]
        basis = torch.linspace(0.1, 1.0, self.N_TOKENS * self.C_V).reshape(
            1, 1, self.N_TOKENS, self.C_V
        )
        position_weight = torch.arange(
            1, signature.shape[1] + 1, dtype=signature.dtype
        ).view(1, -1, 1)
        ordered_context = (signature * position_weight).sum(dim=1, keepdim=True)
        scale = signature[..., :1] + 0.25 * ordered_context[..., 1:2]
        curve = signature[..., 2:3] + 0.1 * ordered_context[..., :1]
        tokens = scale.unsqueeze(-2) * basis + curve.unsqueeze(-2) * basis.square()
        return [None, tokens], 5


def test_temporal_pair_target_uses_two_slots_per_camera():
    protocol = TeacherProtocol(
        queries_per_camera=QUERIES_PER_CAMERA, num_queries=M, hidden_dim=D
    )
    teacher = _FakeTemporalVGGT()
    current = torch.rand(2, NUM_VIEWS, 3, 60, 80)
    future = torch.rand_like(current)

    target = build_temporal_pair_target(teacher, current, future, protocol)

    assert target.shape == (2, M, D)
    assert teacher.last_input_shape[:2] == (2 * NUM_VIEWS, 2)
    assert torch.isfinite(target).all()


def test_temporal_pair_target_keeps_camera_blocks_independent():
    protocol = TeacherProtocol(
        queries_per_camera=QUERIES_PER_CAMERA, num_queries=M, hidden_dim=D
    )
    teacher = _FakeTemporalVGGT()
    current_a = torch.rand(1, NUM_VIEWS, 3, 60, 80)
    current_b = current_a.clone()
    current_b[:, 2] = torch.rand(1, 3, 60, 80)
    future = torch.rand_like(current_a)

    target_a = build_temporal_pair_target(teacher, current_a, future, protocol)
    target_b = build_temporal_pair_target(teacher, current_b, future, protocol)

    shared = 2 * QUERIES_PER_CAMERA
    torch.testing.assert_close(target_a[:, :shared], target_b[:, :shared])
    assert not torch.allclose(target_a[:, shared:], target_b[:, shared:])


def test_per_camera_pooling_keeps_each_query_inside_one_camera():
    """Changing one camera must only change that camera's query block."""
    protocol = TeacherProtocol(
        queries_per_camera=QUERIES_PER_CAMERA, num_queries=M, hidden_dim=D
    )
    teacher = _FakeVGGT()

    torch.manual_seed(0)
    views_a = torch.rand(1, NUM_VIEWS, 3, 60, 80)
    views_b = views_a.clone()
    views_b[:, 2] = torch.rand(1, 3, 60, 80)  # only the right wrist changes

    target_a = build_teacher_target(teacher, views_a, protocol)
    target_b = build_teacher_target(teacher, views_b, protocol)
    assert target_a.shape == (1, M, D)

    shared = 2 * QUERIES_PER_CAMERA
    torch.testing.assert_close(target_a[:, :shared], target_b[:, :shared])
    assert not torch.allclose(target_a[:, shared:], target_b[:, shared:])


def test_centring_removes_the_shared_common_mode():
    """Per-view centring makes the target invariant to a constant token offset.

    Raw VGGT tokens put ~96% of their energy on one shared direction; removing it
    is what keeps the pooled queries from collapsing onto each other.
    """
    protocol = TeacherProtocol(
        queries_per_camera=QUERIES_PER_CAMERA, num_queries=M, hidden_dim=D
    )
    assert protocol.token_centering == "per_view_mean_subtraction"
    assert protocol.camera_tag == "none", "a camera tag re-injects the common mode"

    class _OffsetVGGT(_FakeVGGT):
        offset = 0.0

        def aggregator(self, images):
            tokens, psi = super().aggregator(images)
            return [t + self.offset if t is not None else None for t in tokens], psi

    torch.manual_seed(0)
    views = torch.rand(1, NUM_VIEWS, 3, 60, 80)

    plain = _OffsetVGGT()
    shifted = _OffsetVGGT()
    shifted.offset = 7.5

    # Exact in real arithmetic; the tolerance covers float32 cancellation when the
    # offset is subtracted back out.
    torch.testing.assert_close(
        build_teacher_target(plain, views, protocol),
        build_teacher_target(shifted, views, protocol),
        atol=1e-3,
        rtol=1e-2,
    )


def test_uncentred_protocol_collapses_the_queries():
    """Guards the diagnosis: without centring the queries become near-identical."""
    torch.manual_seed(0)
    views = torch.rand(1, NUM_VIEWS, 3, 60, 80)
    teacher = _FakeVGGT()

    def query_cosine(**overrides):
        protocol = TeacherProtocol(
            queries_per_camera=QUERIES_PER_CAMERA,
            num_queries=M,
            hidden_dim=D,
            **overrides,
        )
        target = build_teacher_target(teacher, views, protocol)[0]
        normed = torch.nn.functional.normalize(target, dim=-1)
        sim = normed @ normed.T
        return sim[~torch.eye(M, dtype=bool)].mean()

    assert query_cosine() < query_cosine(token_centering="none")


# --- 15.5 cache alignment ------------------------------------------------


def test_cached_rows_match_a_fresh_recompute_at_the_targeted_frame(tmp_path: Path):
    """Section 15.5: the row served for `(key, k)` must be the teacher of the
    frame the dataset predicts at horizon `k`.

    The frames carry a per-frame signature and `_FakeVGGT` is a pure function of
    the pixels, so an off-by-one between the writer's row grid and the reader's
    `start + k * stride` shows up as a numeric mismatch rather than passing
    silently.
    """
    protocol = TeacherProtocol(
        queries_per_camera=QUERIES_PER_CAMERA, num_queries=M, hidden_dim=D,
        window_stride=WINDOW_STRIDE,
    )
    teacher = _FakeVGGT()
    num_frames, episode = 192, "clean/task/0"

    def views_for(frame: int) -> torch.Tensor:
        # Distinct, deterministic pixels per frame.
        torch.manual_seed(frame)
        return torch.rand(1, NUM_VIEWS, 3, 60, 80)

    # Writer: one row per stride-aligned frame, exactly as precompute does.
    rows = [
        build_teacher_target(teacher, views_for(f), protocol)[0]
        for f in range(0, num_frames, WINDOW_STRIDE)
    ]
    shard = tmp_path / "shards" / f"{episode}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard, torch.stack(rows).numpy().astype(np.float16))
    write_manifest(tmp_path, protocol)
    save_index(
        tmp_path,
        {episode: {"file": f"shards/{episode}.npy", "num_frames": num_frames,
                   "num_rows": len(rows)}},
    )

    store = _store(tmp_path)

    def direction_match(a: torch.Tensor, b: torch.Tensor) -> float:
        # The fake teacher's centred outputs are tiny in absolute terms, so an
        # absolute tolerance would make this vacuous; compare directions.
        return float(
            torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0)
        )

    for start in (0, 6, 60):
        for horizon in (1, 2, 4, 8):
            cached = store.get(make_sample_key(episode, start), horizon)
            # The frame the dataset assigns to this horizon (section 2.3).
            recomputed = build_teacher_target(
                teacher, views_for(start + horizon * WINDOW_STRIDE), protocol
            )[0]
            assert direction_match(cached, recomputed) > 0.99, (start, horizon)

    # A wrong horizon must not match, or the check above is vacuous.
    off_by_one = build_teacher_target(
        teacher, views_for(2 * WINDOW_STRIDE), protocol
    )[0]
    assert direction_match(store.get(make_sample_key(episode, 0), 1), off_by_one) < 0.5


# --- teacher store -------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    protocol = TeacherProtocol(
        queries_per_camera=QUERIES_PER_CAMERA,
        num_queries=M,
        hidden_dim=D,
        window_stride=WINDOW_STRIDE,
    )
    write_manifest(tmp_path, protocol)

    key = "clean/adjust_bottle/0"
    shard = tmp_path / "shards" / f"{key}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # 32 stride-aligned rows == frames 0, 6, ..., 186
    np.save(shard, rng.standard_normal((32, M, D)).astype(np.float16))
    save_index(
        tmp_path,
        {key: {"file": f"shards/{key}.npy", "num_frames": 192, "num_rows": 32}},
    )
    return tmp_path


def _store(cache_dir: Path) -> TeacherStore:
    return TeacherStore(
        str(cache_dir), num_queries=M, hidden_dim=D, window_stride=WINDOW_STRIDE
    )


def test_sample_key_roundtrip():
    key = make_sample_key("clean/adjust_bottle/0", 42)
    assert key == "clean/adjust_bottle/0@42"
    assert split_sample_key(key) == ("clean/adjust_bottle/0", 42)
    with pytest.raises(TeacherCacheError, match="malformed sample_key"):
        split_sample_key("clean/adjust_bottle/0")


def test_teacher_store_resolves_window_and_horizon(cache_dir: Path):
    store = _store(cache_dir)
    key = make_sample_key("clean/adjust_bottle/0", 12)

    single = store.get(key, 1)
    assert single.shape == (M, D)
    assert single.dtype == torch.float32

    # horizon k must resolve to frame start + k * stride, i.e. the same latent a
    # window starting there would see at horizon 0 distance.
    torch.testing.assert_close(single, store.get(make_sample_key("clean/adjust_bottle/0", 18), 0))

    many = store.get_horizons(key, [1, 4, 8])
    assert many.shape == (3, M, D)
    torch.testing.assert_close(many[0], single)
    assert not torch.allclose(many[0], many[1])

    batch = store.get_batch([key, key], [1, 4])
    assert batch.shape == (2, M, D)
    torch.testing.assert_close(batch[1], many[1])


def test_teacher_store_rejects_an_index_from_another_protocol(cache_dir: Path):
    """A half-finished re-run leaves a new manifest beside the old index.json."""
    index = json.loads((cache_dir / "index.json").read_text())
    for entry in index.values():
        entry["protocol_id"] = "superseded_v2"
    save_index(cache_dir, index)

    with pytest.raises(TeacherCacheError, match="different protocol"):
        _store(cache_dir)


def test_teacher_store_accepts_a_legacy_index_without_protocol_stamps(cache_dir: Path):
    """Caches written before the stamp existed must keep loading."""
    entries = json.loads((cache_dir / "index.json").read_text())
    assert all("protocol_id" not in entry for entry in entries.values())
    assert len(_store(cache_dir)) == len(entries)


def test_teacher_store_fails_fast(cache_dir: Path):
    store = _store(cache_dir)
    with pytest.raises(TeacherCacheError, match="not in teacher cache"):
        store.get(make_sample_key("clean/missing/0", 0), 1)
    with pytest.raises(TeacherCacheError, match="out of range"):
        store.get(make_sample_key("clean/adjust_bottle/0", 186), 8)
    with pytest.raises(TeacherCacheError, match="not a multiple of"):
        store.get(make_sample_key("clean/adjust_bottle/0", 7), 1)


@pytest.fixture
def pair_cache_dir(tmp_path: Path) -> Path:
    protocol = _joint_protocol(window_stride=WINDOW_STRIDE)
    write_manifest(tmp_path, protocol, extra={"complete": True})
    episode = "clean/adjust_bottle/0"
    num_frames = 72
    num_windows = (num_frames - (8 * WINDOW_STRIDE + 1)) // WINDOW_STRIDE + 1
    rng = np.random.default_rng(4)
    values = rng.standard_normal((num_windows, 5, M, D)).astype(np.float16)
    shard = tmp_path / "shards" / f"{episode}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard, values)
    save_index(
        tmp_path,
        {
            episode: {
                "file": f"shards/{episode}.npy",
                "protocol_id": protocol.protocol_id,
                "num_frames": num_frames,
                "num_windows": num_windows,
                "num_slots": 5,
            }
        },
    )
    return tmp_path


def _pair_store(cache_dir: Path) -> TemporalPairTeacherStore:
    return TemporalPairTeacherStore(
        str(cache_dir),
        num_queries=M,
        hidden_dim=D,
        window_stride=WINDOW_STRIDE,
        cached_horizons=(1, 2, 4, 8),
        num_video_frames=8,
    )


@pytest.mark.parametrize("requested_horizons", [(1, 2, 4, 8), (1,), (8,), (8, 1)])
def test_temporal_pair_store_maps_window_rows_and_horizon_slots(
    pair_cache_dir: Path, requested_horizons: tuple[int, ...]
):
    store = create_teacher_store(
        str(pair_cache_dir), M, D, WINDOW_STRIDE, requested_horizons, 8
    )
    key = make_sample_key("clean/adjust_bottle/0", 12)
    horizons = [0, *requested_horizons]
    selected = store.get_horizons(key, horizons)
    raw = torch.from_numpy(
        np.load(pair_cache_dir / "shards/clean/adjust_bottle/0.npy")[2]
    ).float()

    slots = [(0, 1, 2, 4, 8).index(horizon) for horizon in horizons]
    assert selected.shape == (len(horizons), M, D)
    torch.testing.assert_close(selected, raw[slots])
    assert isinstance(store, TemporalPairTeacherStore)


@pytest.mark.parametrize("dataset_type", ["robotwin", "lerobot"])
def test_teacher_store_factory_preserves_legacy_layout(
    cache_dir: Path, dataset_type: str
):
    from data.dataset import apply_geometry_jepa_params

    config = OmegaConf.create(
        {
            "common": {
                "num_video_frames": 8,
                "video_action_freq_ratio": FREQ_RATIO,
                "global_downsample_rate": DOWNSAMPLE,
            },
            "dataset": {"type": dataset_type},
            "geometry_jepa": {
                "enable": True,
                "cache_dir": str(cache_dir),
                "teacher_protocol_id": DEFAULT_PROTOCOL.protocol_id,
                "num_queries": M,
                "hidden_dim": D,
                "target_mode": "residual",
                "horizon_choices": [1, 2, 4, 8],
            },
        }
    )
    params = {}
    apply_geometry_jepa_params(config, params)
    assert "teacher_cached_horizons" not in params
    assert params["teacher_cache_dir"] == str(cache_dir)
    assert params["deterministic_windows"] is True
    assert isinstance(
        create_teacher_store(str(cache_dir), M, D, WINDOW_STRIDE), TeacherStore
    )


@pytest.mark.parametrize("requested_horizons", [(1, 2, 4, 8), (1,), (8,), (8, 1)])
def test_geometry_jepa_params_dispatch_pair_cache_only_for_robotwin(
    pair_cache_dir: Path, requested_horizons: tuple[int, ...]
):
    from data.dataset import apply_geometry_jepa_params

    config = OmegaConf.create(
        {
            "common": {
                "num_video_frames": 8,
                "video_action_freq_ratio": FREQ_RATIO,
                "global_downsample_rate": DOWNSAMPLE,
            },
            "dataset": {"type": "robotwin"},
            "geometry_jepa": {
                "enable": True,
                "cache_dir": str(pair_cache_dir),
                "teacher_protocol_id": JOINT_FUTURE_SLOT_PROTOCOL_ID,
                "num_queries": M,
                "hidden_dim": D,
                "target_mode": "joint_future_slot",
                "horizon_choices": list(requested_horizons),
            },
        }
    )
    params = {}
    apply_geometry_jepa_params(config, params)
    assert params["teacher_cached_horizons"] == list(requested_horizons)
    assert params["deterministic_windows"] is True

    config.geometry_jepa.horizon_choices = [3]
    with pytest.raises(ValueError, match="cached_horizons"):
        apply_geometry_jepa_params(config, {})

    config.dataset.type = "lerobot"
    with pytest.raises(ValueError, match="robotwin only"):
        apply_geometry_jepa_params(config, {})


def test_temporal_pair_store_rejects_invalid_addressing(pair_cache_dir: Path):
    store = _pair_store(pair_cache_dir)
    episode = "clean/adjust_bottle/0"
    with pytest.raises(TeacherCacheError, match="no horizons"):
        store.get(make_sample_key(episode, 0), 3)
    with pytest.raises(TeacherCacheError, match="not a multiple"):
        store.get(make_sample_key(episode, 1), 1)
    with pytest.raises(TeacherCacheError, match="out of range"):
        store.get(make_sample_key(episode, 24), 8)

    with pytest.raises(TeacherCacheError, match="cached_horizons"):
        create_teacher_store(str(pair_cache_dir), M, D, WINDOW_STRIDE, (3,), 8)


def test_temporal_pair_store_rejects_wrong_shard_shape(pair_cache_dir: Path):
    store = _pair_store(pair_cache_dir)
    shard = pair_cache_dir / "shards/clean/adjust_bottle/0.npy"
    np.save(shard, np.zeros((4, 4, M, D), dtype=np.float16))
    with pytest.raises(TeacherCacheError, match="shape"):
        store.get(make_sample_key("clean/adjust_bottle/0", 0), 1)


def test_temporal_pair_store_rejects_mixed_protocols(pair_cache_dir: Path):
    index = json.loads((pair_cache_dir / "index.json").read_text())
    next(iter(index.values()))["protocol_id"] = "other_v9"
    save_index(pair_cache_dir, index)
    with pytest.raises(TeacherCacheError, match="protocol_id"):
        _pair_store(pair_cache_dir)


def test_manifest_mismatch_is_rejected(cache_dir: Path):
    ok = dict(
        protocol_id=DEFAULT_PROTOCOL.protocol_id,
        num_queries=M,
        hidden_dim=D,
        window_stride=WINDOW_STRIDE,
    )
    assert_manifest_matches(cache_dir, **ok)
    for field, bad in (
        ("hidden_dim", 1024),
        ("protocol_id", "other_v9"),
        ("window_stride", 5),
    ):
        with pytest.raises(ValueError, match=field):
            assert_manifest_matches(cache_dir, **{**ok, field: bad})


def test_joint_manifest_requires_the_pair_cache_contract(tmp_path: Path):
    protocol = _joint_protocol(window_stride=WINDOW_STRIDE)
    write_manifest(tmp_path, protocol, extra={"complete": False})
    expected = dict(
        protocol_id=protocol.protocol_id,
        num_queries=protocol.num_queries,
        hidden_dim=protocol.hidden_dim,
        window_stride=WINDOW_STRIDE,
        target_construction=TEMPORAL_PAIR_TARGET,
        cache_layout=WINDOW_CACHE_LAYOUT,
        cached_horizons=(1, 2, 4, 8),
        num_video_frames=8,
    )
    with pytest.raises(ValueError, match="complete"):
        assert_manifest_matches(tmp_path, **expected)

    write_manifest(tmp_path, protocol, extra={"complete": True})
    assert_manifest_matches(tmp_path, **expected)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.pop("cached_horizons")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cached_horizons"):
        assert_manifest_matches(tmp_path, **expected)


def test_legacy_manifest_without_pair_fields_stays_readable(cache_dir: Path):
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for field in (
        "target_construction",
        "cache_layout",
        "sequence_length",
        "pair_order",
        "selected_temporal_slot",
        "static_baseline",
        "cached_horizons",
        "slot_schema",
        "num_video_frames",
    ):
        manifest.pop(field, None)
    manifest_path.write_text(json.dumps(manifest))

    assert_manifest_matches(
        cache_dir,
        protocol_id=DEFAULT_PROTOCOL.protocol_id,
        num_queries=M,
        hidden_dim=D,
        window_stride=WINDOW_STRIDE,
    )


# --- cache writer protocol identity --------------------------------------


def test_a_local_checkpoint_path_is_not_protocol_drift():
    """`teacher_revision` is a path here and a hub id by default; same model."""
    local = TeacherProtocol(teacher_revision="pretrained_models/VGGT-1B")
    assert assert_protocol_id_covers_overrides(local) is local
    with pytest.raises(ValueError, match="teacher_revision"):
        assert_protocol_id_covers_overrides(
            TeacherProtocol(teacher_revision="pretrained_models/VGGT-2B")
        )


def test_custom_protocol_id_is_applied_to_the_returned_protocol():
    """Discarding the return value would write a drifted cache under the old id."""
    drifted = TeacherProtocol(hidden_dim=512)
    updated = assert_protocol_id_covers_overrides(drifted, explicit_id="probe_v1")
    assert updated.protocol_id == "probe_v1"
    assert updated.hidden_dim == 512
    assert drifted.protocol_id == DEFAULT_PROTOCOL.protocol_id


def test_explicit_protocol_id_wins_without_detected_drift():
    """The manual escape hatch for two checkpoints that share a model name."""
    same_name = TeacherProtocol(teacher_revision="some/other/dir/VGGT-1B")
    updated = assert_protocol_id_covers_overrides(same_name, explicit_id="retrained_v1")
    assert updated.protocol_id == "retrained_v1"


def test_merge_index_rejects_shards_from_another_protocol(tmp_path: Path):
    """Mixed protocols are invisible downstream: every shard has the same shape."""
    from tools.precompute_vggt_cache import merge_index

    write_manifest(tmp_path, TeacherProtocol(window_stride=WINDOW_STRIDE))
    entry = {"file": "shards/clean/task/0.npy", "num_frames": 192, "num_rows": 32}
    (tmp_path / "index.part-000.json").write_text(
        json.dumps({"clean/task/0": {**entry, "protocol_id": DEFAULT_PROTOCOL.protocol_id}})
    )
    (tmp_path / "index.part-001.json").write_text(
        json.dumps({"clean/task/1": {**entry, "protocol_id": "other_v9"}})
    )

    with pytest.raises(ValueError, match="not encoded under"):
        merge_index(tmp_path)
    assert not (tmp_path / "index.json").exists()


def test_merge_index_finalizes_temporal_pair_provenance(tmp_path: Path):
    from tools.precompute_vggt_cache import merge_index

    protocol = _joint_protocol(window_stride=WINDOW_STRIDE)
    write_manifest(tmp_path, protocol, extra={"complete": False})
    episode = "clean/task/0"
    shard = tmp_path / "shards" / f"{episode}.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard, np.zeros((1, 5, M, D), dtype=np.float16))
    entry = {
        "file": f"shards/{episode}.npy",
        "protocol_id": protocol.protocol_id,
        "num_frames": 49,
        "num_rows": 1,
        "num_windows": 1,
        "num_slots": 5,
    }
    (tmp_path / "index.part-000.json").write_text(json.dumps({episode: entry}))

    merge_index(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["total_windows"] == 1
    assert manifest["total_pairs"] == 5
    assert manifest["total_bytes"] == shard.stat().st_size
    assert len(manifest["index_sha256"]) == 64
    assert len(manifest["frame_counts_sha256"]) == 64


def test_temporal_pair_window_starts_match_full_robotwin_windows():
    from tools.precompute_vggt_cache import temporal_pair_window_starts

    protocol = _joint_protocol(window_stride=WINDOW_STRIDE)
    assert temporal_pair_window_starts(48, protocol) == []
    assert temporal_pair_window_starts(49, protocol) == [0]
    assert temporal_pair_window_starts(72, protocol) == [0, 6, 12, 18]


def test_temporal_pair_writer_emits_window_slot_shard(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from tools.precompute_vggt_cache import encode_temporal_pair_episode

    class FakeBatch:
        def __init__(self, values):
            self.values = values

        def asnumpy(self):
            return self.values

    class FakeVideoReader:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_batch(self, indices):
            frames = np.empty((len(indices), 6, 4, 3), dtype=np.uint8)
            for position, frame in enumerate(indices):
                frames[position, :4] = 10 + frame
                frames[position, 4:, :2] = 40 + frame
                frames[position, 4:, 2:] = 80 + frame
            return FakeBatch(frames)

    monkeypatch.setitem(
        sys.modules,
        "decord",
        SimpleNamespace(VideoReader=FakeVideoReader, cpu=lambda _index: None),
    )
    protocol = _joint_protocol(
        input_size=(8, 8),
        queries_per_camera=2,
        num_queries=6,
        hidden_dim=8,
        window_stride=2,
        cached_horizons=(1, 2),
        slot_schema=("static", "h1", "h2"),
        num_video_frames=2,
    )
    path = tmp_path / "episode.tmp.npy"
    num_windows = encode_temporal_pair_episode(
        _FakeTemporalVGGT(),
        "ignored.mp4",
        total_frames=9,
        protocol=protocol,
        batch_size=2,
        storage_dtype=np.dtype("float16"),
        device=torch.device("cpu"),
        output_path=path,
    )
    shard = np.load(path)

    assert num_windows == 3
    assert shard.shape == (3, 3, 6, 8)
    assert np.isfinite(shard).all()
    assert not np.array_equal(shard[:, 0], shard[:, 1])


def test_temporal_pair_writer_handles_an_episode_without_full_windows(
    tmp_path: Path, monkeypatch
):
    from types import SimpleNamespace

    from tools.precompute_vggt_cache import encode_temporal_pair_episode

    monkeypatch.setitem(
        sys.modules,
        "decord",
        SimpleNamespace(VideoReader=None, cpu=lambda _index: None),
    )
    protocol = _joint_protocol(
        input_size=(8, 8),
        queries_per_camera=2,
        num_queries=6,
        hidden_dim=8,
        window_stride=2,
        cached_horizons=(1, 2),
        slot_schema=("static", "h1", "h2"),
        num_video_frames=2,
    )
    path = tmp_path / "empty.npy"
    count = encode_temporal_pair_episode(
        _FakeTemporalVGGT(),
        "unused.mp4",
        total_frames=4,
        protocol=protocol,
        batch_size=2,
        storage_dtype=np.dtype("float16"),
        device=torch.device("cpu"),
        output_path=path,
    )

    assert count == 0
    assert np.load(path).shape == (0, 3, 6, 8)


# --- transition alignment ------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 2, 4, 8])
def test_transition_alignment_matches_the_dataset_sampler(horizon):
    """horizon k -> video frame k-1 and the first k*ratio actions.

    Mirrors `RobotWinTaskDataset._sampling_indices_from_start`.
    """
    num_video_frames = 8
    condition_frame_idx = 18  # window starts are multiples of WINDOW_STRIDE
    chunk = num_video_frames * FREQ_RATIO

    action_indices = [condition_frame_idx + (i + 1) * DOWNSAMPLE for i in range(chunk)]
    video_indices = [
        action_indices[(i + 1) * FREQ_RATIO - 1] for i in range(num_video_frames)
    ]

    num_actions = horizon * FREQ_RATIO
    assert action_indices[num_actions - 1] == video_indices[horizon - 1]
    # This identity is what lets the teacher store resolve (sample_key, horizon).
    assert video_indices[horizon - 1] == condition_frame_idx + horizon * WINDOW_STRIDE
    assert video_indices[horizon - 1] % WINDOW_STRIDE == 0


# --- LeRobot window enumeration ------------------------------------------
#
# `_build_window_index` is driven on a stub so the key schema and the
# stride/frame-count contracts can be checked without a real LeRobot dataset.


def _lerobot_window_stub(lengths, frame_counts=None):
    from itertools import accumulate
    from types import SimpleNamespace

    from data.lerobot.lerobot_dataset import LeRobotMotusDataset

    bounds = list(accumulate(lengths))
    stub = SimpleNamespace(
        repo_id="RoboDojo",
        task_mode="single",
        episode_ids=list(range(len(lengths))),
        window_stride=WINDOW_STRIDE,
        min_episode_frames=NUM_VIDEO_FRAMES * FREQ_RATIO * DOWNSAMPLE + 1,
        frame_count_cache=None,
        lerobot_dataset=SimpleNamespace(
            episode_data_index={
                "from": torch.tensor([0] + bounds[:-1]),
                "to": torch.tensor(bounds),
            }
        ),
    )
    stub._load_cached_frame_counts = lambda: dict(frame_counts or {})
    stub._build_window_index = LeRobotMotusDataset._build_window_index.__get__(stub)
    return stub


def test_lerobot_episode_key_survives_the_sample_key_roundtrip():
    from data.lerobot.lerobot_dataset import make_lerobot_episode_key

    key = make_lerobot_episode_key("RoboDojo_lerobot_v21_video", 42)
    assert key == "lerobot/RoboDojo_lerobot_v21_video/000042"
    # Three segments, so the key still matches the schema declared in the manifest.
    assert key.count("/") == DEFAULT_PROTOCOL.sample_key_schema.count("/")
    assert split_sample_key(make_sample_key(key, 18)) == (key, 18)


def test_lerobot_windows_are_stride_aligned_and_stable():
    lengths = [200, 40, 120]  # the middle episode cannot fit a full window
    stub = _lerobot_window_stub(lengths)
    stub._build_window_index()

    starts = [w["start_frame"] for w in stub.windows]
    assert all(start % WINDOW_STRIDE == 0 for start in starts)
    assert len(stub.windows) == len(range(0, 200 - 48, WINDOW_STRIDE)) + len(
        range(0, 120 - 48, WINDOW_STRIDE)
    )
    assert {w["episode_key"] for w in stub.windows} == {
        "lerobot/RoboDojo/000000",
        "lerobot/RoboDojo/000002",
    }
    assert len(set(w["sample_key"] for w in stub.windows)) == len(stub.windows)
    assert len(stub.sample_weights) == len(stub.windows)

    rebuilt = _lerobot_window_stub(lengths)
    rebuilt._build_window_index()
    assert [w["sample_key"] for w in rebuilt.windows] == [
        w["sample_key"] for w in stub.windows
    ]


def test_lerobot_window_index_rejects_a_truncated_concat_video():
    """The cache row count and the parquet index are the only cross-check there is."""
    stub = _lerobot_window_stub(
        [200], frame_counts={"lerobot/RoboDojo/000000": 180}
    )
    with pytest.raises(ValueError, match="disagree"):
        stub._build_window_index()


# --- Motus integration ---------------------------------------------------
#
# `Motus._geometry_jepa_step` is exercised directly so the horizon sampling,
# action slicing, target indexing and gradient policy can be checked without
# instantiating the 5B backbone.


class _FakeMotus:
    """Minimal stand-in exposing what `_geometry_jepa_step` reads."""

    def __init__(self, geo_config: GeometryJEPAConfig):
        from types import SimpleNamespace

        from models.motus import Motus

        self.config = SimpleNamespace(
            geometry_jepa=geo_config,
            video_action_freq_ratio=FREQ_RATIO,
            num_video_frames=NUM_VIDEO_FRAMES,
        )
        self.geometry_jepa = GeometryJEPAModule(
            geo_config, student_num_tokens=NUM_TOKENS
        )
        self.device = torch.device("cpu")
        self._horizon_generator = torch.Generator()
        self._horizon_generator.manual_seed(0)
        self._geometry_jepa_grad_norms = Motus._geometry_jepa_grad_norms.__get__(self)

    def step(self, student_tokens, actions, teacher_latents):
        from models.motus import Motus

        return Motus._geometry_jepa_step(
            self, student_tokens, actions, teacher_latents
        )


NUM_VIDEO_FRAMES = 8
EXPECTED_METRICS = {
    "geometry_jepa_loss",
    "geo_raw_loss",
    "geo_cos_loss",
    "geo_mse_loss",
    "geo_token_cos_loss",
    "geo_copy_loss",
    "geo_skill",
    "geo_horizon",
    "geo_grad_norm",
}
# Only produced once the backbone is open (section 11.2).
OPEN_MODE_METRIC = "geo_student_grad_norm"


def _full_batch():
    return (
        _student(),
        torch.randn(BATCH, NUM_VIDEO_FRAMES * FREQ_RATIO, ACTION_DIM),
        # Row 0 is the anchor at `t`, rows 1..K are the horizons.
        torch.randn(BATCH, NUM_VIDEO_FRAMES + 1, M, D),
    )


def test_motus_step_selects_the_target_and_actions_for_the_sampled_horizon():
    torch.manual_seed(0)
    cfg = GeometryJEPAConfig(
        enable=True, num_queries=M, hidden_dim=D,
        horizon_choices=[1, 4, 8], max_horizon=8, lambda_geo=0.03,
    )
    fake = _FakeMotus(cfg)
    student, actions, teacher = _full_batch()

    seen = set()
    for _ in range(30):
        loss, metrics = fake.step(student, actions, teacher)
        k = int(metrics["geo_horizon"])
        seen.add(k)
        assert torch.isfinite(loss)
        # Per-horizon accounting rides along when the set has more than one entry.
        assert set(metrics) == EXPECTED_METRICS | {f"geo_h{k}_sum", f"geo_h{k}_n"}
        assert metrics[f"geo_h{k}_n"] == 1.0
        torch.testing.assert_close(metrics[f"geo_h{k}_sum"], metrics["geo_raw_loss"])
        assert metrics["geo_grad_norm"] > 0
        assert torch.isfinite(metrics["geo_grad_norm"])
    assert seen == {1, 4, 8}, f"horizon sampling did not cover the config: {seen}"


def test_motus_step_applies_lambda_geo():
    torch.manual_seed(0)
    cfg = GeometryJEPAConfig(
        enable=True, num_queries=M, hidden_dim=D,
        horizon_choices=[1], max_horizon=8, lambda_geo=0.25,
    )
    fake = _FakeMotus(cfg)
    student, actions, teacher = _full_batch()

    loss, metrics = fake.step(student, actions, teacher)
    raw = fake.geometry_jepa(
        student_tokens=student.detach(),
        actions=actions[:, :FREQ_RATIO],
        horizon=_horizon(1),
        teacher_target=teacher[:, 1],
        teacher_anchor=teacher[:, 0],
    )["geometry_jepa_loss"]
    torch.testing.assert_close(loss, 0.25 * raw)
    torch.testing.assert_close(metrics["geometry_jepa_loss"], loss.detach())


@pytest.mark.parametrize("freeze", [True, False])
def test_motus_step_respects_the_gradient_policy(freeze):
    torch.manual_seed(0)
    cfg = GeometryJEPAConfig(
        enable=True, num_queries=M, hidden_dim=D,
        horizon_choices=[1], max_horizon=8, freeze_wam_for_geo=freeze,
    )
    fake = _FakeMotus(cfg)
    upstream = torch.nn.Linear(STUDENT_DIM, STUDENT_DIM)
    _, actions, teacher = _full_batch()

    loss, metrics = fake.step(upstream(_student()), actions, teacher)
    loss.backward()

    assert (upstream.weight.grad is None) is freeze
    assert fake.geometry_jepa.adapter.learned_queries.grad is not None

    # Section 11.2 / 15.6: opening the WAM must expose the gradient the branch
    # pushes into the student path; frozen mode must not report one.
    assert (OPEN_MODE_METRIC in metrics) is not freeze
    if not freeze:
        assert metrics[OPEN_MODE_METRIC] > 0
        assert torch.isfinite(metrics[OPEN_MODE_METRIC])
        assert torch.isfinite(upstream.weight.grad).all()


def test_lambda_geo_scales_the_gradient_pushed_into_the_wam():
    """`lambda_geo` is the knob that bounds the branch's effect on the backbone."""
    _, actions, teacher = _full_batch()
    student = _student()

    def student_grad_norm(lambda_geo):
        torch.manual_seed(0)
        cfg = GeometryJEPAConfig(
            enable=True, num_queries=M, hidden_dim=D,
            horizon_choices=[1], max_horizon=8,
            freeze_wam_for_geo=False, lambda_geo=lambda_geo,
        )
        source = student.clone().requires_grad_()
        _, metrics = _FakeMotus(cfg).step(source, actions, teacher)
        return metrics[OPEN_MODE_METRIC]

    small, large = student_grad_norm(0.01), student_grad_norm(0.04)
    torch.testing.assert_close(large, small * 4.0, rtol=1e-4, atol=1e-6)


def test_motus_step_requires_teacher_latents():
    cfg = GeometryJEPAConfig(enable=True, num_queries=M, hidden_dim=D)
    fake = _FakeMotus(cfg)
    student, actions, _ = _full_batch()
    with pytest.raises(ValueError, match="no `teacher_latents`"):
        fake.step(student, actions, None)


def test_motus_step_requires_the_anchor_row():
    cfg = GeometryJEPAConfig(enable=True, num_queries=M, hidden_dim=D)
    fake = _FakeMotus(cfg)
    student, actions, teacher = _full_batch()
    with pytest.raises(ValueError, match="horizons 0"):
        fake.step(student, actions, teacher[:, 1:])


def test_motus_step_reports_skill_against_the_copy_baseline():
    torch.manual_seed(0)
    cfg = GeometryJEPAConfig(
        enable=True, num_queries=M, hidden_dim=D,
        horizon_choices=[1], max_horizon=8, target_mode="residual",
    )
    fake = _FakeMotus(cfg)
    student, actions, teacher = _full_batch()

    _, metrics = fake.step(student, actions, teacher)
    expected = 1.0 - metrics["geo_raw_loss"] / metrics["geo_copy_loss"]
    torch.testing.assert_close(metrics["geo_skill"], expected)


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_motus_step_maps_joint_horizon_to_sparse_cache_slot(horizon: int):
    torch.manual_seed(0)
    cfg = GeometryJEPAConfig(
        enable=True,
        num_queries=M,
        hidden_dim=D,
        horizon_choices=[horizon],
        max_horizon=8,
        target_mode="joint_future_slot",
    )
    fake = _FakeMotus(cfg)
    student = _student()
    actions = torch.randn(BATCH, NUM_VIDEO_FRAMES * FREQ_RATIO, ACTION_DIM)
    teacher = torch.randn(BATCH, 2, M, D)

    loss, metrics = fake.step(student, actions, teacher)
    expected = fake.geometry_jepa(
        student_tokens=student.detach(),
        actions=actions[:, : horizon * FREQ_RATIO],
        horizon=_horizon(horizon),
        teacher_target=teacher[:, 1],
        teacher_anchor=teacher[:, 0],
    )

    torch.testing.assert_close(loss, cfg.lambda_geo * expected["geometry_jepa_loss"])
    torch.testing.assert_close(metrics["joint_static_loss"], expected["geo_copy_loss"])
    torch.testing.assert_close(
        metrics["joint_skill"],
        1.0 - metrics["geo_raw_loss"] / metrics["joint_static_loss"].clamp_min(1e-8),
    )
    torch.testing.assert_close(
        metrics["joint_target_rms"], teacher[:, 1].float().square().mean().sqrt()
    )
    assert "geo_copy_loss" not in metrics
    assert "geo_skill" not in metrics


def test_motus_step_rejects_a_too_short_action_chunk():
    cfg = GeometryJEPAConfig(
        enable=True, num_queries=M, hidden_dim=D,
        horizon_choices=[8], max_horizon=8,
    )
    fake = _FakeMotus(cfg)
    student, _, teacher = _full_batch()
    short = torch.randn(BATCH, 4, ACTION_DIM)  # horizon 8 needs 16 actions
    with pytest.raises(ValueError, match="needs 16 actions"):
        fake.step(student, short, teacher)


# --- checkpointing (section 12.2) ----------------------------------------


def _jepa(**overrides) -> GeometryJEPAModule:
    cfg = GeometryJEPAConfig(enable=True, num_queries=M, hidden_dim=D, **overrides)
    return GeometryJEPAModule(cfg, student_num_tokens=NUM_TOKENS)


def test_checkpoint_roundtrip_restores_the_branch(tmp_path: Path):
    torch.manual_seed(0)
    saved = _jepa()
    path = tmp_path / "geometry_jepa.pt"
    torch.save(saved.state_dict(), path)

    torch.manual_seed(1)
    restored = _jepa()
    assert not torch.allclose(
        saved.adapter.learned_queries, restored.adapter.learned_queries
    ), "fresh module should differ before loading"

    restored.load_state_dict(torch.load(path))
    for (name, a), (_, b) in zip(
        saved.state_dict().items(), restored.state_dict().items()
    ):
        torch.testing.assert_close(a, b, msg=name)

    saved.eval()
    restored.eval()
    student, actions, horizon = _student(), _actions(1), _horizon(1)
    with torch.no_grad():
        torch.testing.assert_close(
            saved.predictor(saved.adapter(student), actions, horizon),
            restored.predictor(restored.adapter(student), actions, horizon),
        )


def test_fingerprint_covers_shape_defining_fields():
    cfg = GeometryJEPAConfig(enable=True, num_queries=M, hidden_dim=D)
    fingerprint = cfg.fingerprint()
    assert set(fingerprint) == set(CHECKPOINT_FINGERPRINT_FIELDS)
    # Tuning knobs must stay out so they can be changed across a resume.
    for tunable in ("lambda_geo", "lr_scale", "freeze_wam_for_geo", "beta_mse"):
        assert tunable not in fingerprint


@pytest.mark.parametrize(
    "field, bad",
    [
        ("num_queries", 16),
        ("hidden_dim", 1024),
        ("predictor_depth", 4),
        ("max_horizon", 16),
        ("teacher_protocol_id", "other_v9"),
        ("action_hidden_dim", 512),
    ],
)
def test_incompatible_checkpoint_is_rejected(field, bad):
    current = GeometryJEPAConfig(enable=True, num_queries=M, hidden_dim=D)
    saved = current.fingerprint()
    assert_checkpoint_compatible(saved, current)  # identical -> fine

    assert_checkpoint_compatible({**saved, "unrelated": 1}, current)  # extra keys ok
    with pytest.raises(ValueError, match=field):
        assert_checkpoint_compatible({**saved, field: bad}, current)


def test_horizon_embedding_size_follows_max_horizon():
    """max_horizon sizes a parameter, so it must be in the fingerprint."""
    assert "max_horizon" in CHECKPOINT_FINGERPRINT_FIELDS
    small = _jepa(max_horizon=8)
    large = _jepa(max_horizon=16)
    assert small.predictor.horizon_embedding.weight.shape[0] == 9
    assert large.predictor.horizon_embedding.weight.shape[0] == 17
    with pytest.raises(RuntimeError, match="size mismatch"):
        small.load_state_dict(large.state_dict())


def test_deployment_state_dict_isolates_the_jepa_weights():
    """Section 12.3: the branch lives under one prefix, so it can be dropped."""
    jepa = _jepa()
    keys = list(jepa.state_dict())
    assert keys, "module should expose parameters"
    assert all(k.startswith(("adapter.", "predictor.")) for k in keys)
