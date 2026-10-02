"""Frozen VGGT teacher protocol for the Geometry JEPA branch.

This module owns everything that defines the teacher latent space. Any change
here produces a different teacher space and therefore requires regenerating the
cache with a new ``protocol_id``.

RoboTwin videos are stored as a single T-shaped mosaic produced by
``data/robotwin2/robotwin_data_convert/robotwin_converter.py``:

    +-----------------------------+
    |                             |
    |        head  (2/3 H)        |
    |                             |
    +--------------+--------------+
    | left_wrist   | right_wrist  |   (1/3 H, half width each)
    +--------------+--------------+

The mosaic is split back into three views at the *native* video resolution
(before ``resize_with_padding`` adds letterbox bars) and each view is encoded by
VGGT independently. Every view's tokens are then centred, pooled down to
``queries_per_camera`` queries *per camera*, and the per-camera blocks are
concatenated in a fixed order into ``[M, D]`` with
``M = queries_per_camera * num_views``.

The RobotWin Joint ablation applies the same compression after an ordered
``S=2`` same-camera forward. It selects the future temporal slot and stores it
in a separate window-addressed cache, leaving the single-frame cache unchanged.

Two protocol choices matter a lot for how discriminative the targets are:

* **Per-view centring.** Raw VGGT tokens are strongly anisotropic: ~96% of their
  energy sits on a single shared direction, so any two tokens have cosine ~0.96
  before any compression. Subtracting the per-view mean token removes that common
  mode and drops the intra-frame query cosine from ~0.96 to ~0.01, which is what
  makes the JEPA target carry usable signal.
* **Pooling per camera** (rather than over the concatenated ``V*N`` tokens) keeps
  every query inside a single view. Camera identity is therefore encoded
  positionally by the fixed concatenation order, and an explicit camera tag is
  both redundant and harmful -- it re-injects exactly the constant common mode
  that centring removes.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

MANIFEST_FILENAME = "manifest.json"
SINGLE_FRAME_TARGET = "single_frame"
TEMPORAL_PAIR_TARGET = "temporal_pair_future_slot"
FRAME_CACHE_LAYOUT = "frame_rows"
WINDOW_CACHE_LAYOUT = "window_horizon_slots"
JOINT_FUTURE_SLOT_PROTOCOL_ID = (
    "vggt_tlayout_3view_percam_q32_d768_centered_joint_future_slot_v1"
)


@dataclass
class TeacherProtocol:
    """Immutable description of how a teacher latent is produced."""

    protocol_id: str = "vggt_tlayout_3view_percam_q32_d768_centered_v3"
    teacher_name: str = "VGGT"
    teacher_revision: str = "facebook/VGGT-1B"
    teacher_implementation_revision: str = (
        "facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce"
    )
    input_range: Sequence[float] = (0.0, 1.0)
    input_size: Sequence[int] = (518, 518)
    resize_mode: str = "letterbox_pad"
    camera_keys: Sequence[str] = ("head", "left_wrist", "right_wrist")
    layout: str = "robotwin_t_mosaic"
    view_mode: str = "per_camera_pool"
    aggregator_token_selector: str = "last_cached_layer_all_tokens"
    token_centering: str = "per_view_mean_subtraction"
    feature_compression: str = "adaptive_avg_pool_1d"
    camera_tag: str = "none"
    token_compression: str = "per_camera_adaptive_avg_pool_1d"
    queries_per_camera: int = 32
    num_queries: int = 96
    hidden_dim: int = 768
    storage_dtype: str = "float16"
    # Addressing: a sample key names a training window and `horizon` counts video
    # transitions inside it, so the target frame is `start_frame + k * window_stride`.
    # window_stride == video_action_freq_ratio * global_downsample_rate.
    window_stride: int = 6
    horizon_unit: str = "sampled_video_transition"
    sample_key_schema: str = "split/task/episode@start_frame"
    target_construction: str = SINGLE_FRAME_TARGET
    cache_layout: str = FRAME_CACHE_LAYOUT
    sequence_length: int = 1
    pair_order: str = "single"
    selected_temporal_slot: str = "only"
    static_baseline: str = "none"
    cached_horizons: Sequence[int] = ()
    slot_schema: Sequence[str] = ()
    num_video_frames: int = 0

    def __post_init__(self) -> None:
        expected = self.queries_per_camera * self.num_views
        if self.num_queries != expected:
            raise ValueError(
                f"num_queries ({self.num_queries}) must equal queries_per_camera "
                f"({self.queries_per_camera}) * num_views ({self.num_views}) = {expected}"
            )
        if self.target_construction == TEMPORAL_PAIR_TARGET:
            expected_slots = ("static",) + tuple(
                f"h{int(horizon)}" for horizon in self.cached_horizons
            )
            if self.cache_layout != WINDOW_CACHE_LAYOUT:
                raise ValueError(
                    f"{TEMPORAL_PAIR_TARGET} requires cache_layout={WINDOW_CACHE_LAYOUT!r}"
                )
            if self.sequence_length != 2 or self.pair_order != "current,future":
                raise ValueError(
                    f"{TEMPORAL_PAIR_TARGET} requires an ordered current,future pair"
                )
            if self.selected_temporal_slot != "future":
                raise ValueError(
                    f"{TEMPORAL_PAIR_TARGET} requires selected_temporal_slot='future'"
                )
            if self.static_baseline != "repeat_current_pair":
                raise ValueError(
                    f"{TEMPORAL_PAIR_TARGET} requires static_baseline='repeat_current_pair'"
                )
            if not self.cached_horizons or any(
                int(horizon) < 1 for horizon in self.cached_horizons
            ):
                raise ValueError("cached_horizons must contain positive horizons")
            if len(set(int(horizon) for horizon in self.cached_horizons)) != len(
                self.cached_horizons
            ):
                raise ValueError("cached_horizons must not contain duplicates")
            if tuple(self.slot_schema) != expected_slots:
                raise ValueError(
                    f"slot_schema must be {expected_slots}, got {tuple(self.slot_schema)}"
                )
            if self.num_video_frames < max(int(h) for h in self.cached_horizons):
                raise ValueError(
                    "num_video_frames must cover every cached temporal-pair horizon"
                )
        elif self.target_construction != SINGLE_FRAME_TARGET:
            raise ValueError(
                f"unknown target_construction: {self.target_construction!r}"
            )

    @property
    def num_views(self) -> int:
        return len(self.camera_keys)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["input_range"] = list(self.input_range)
        data["input_size"] = list(self.input_size)
        data["camera_keys"] = list(self.camera_keys)
        data["cached_horizons"] = list(self.cached_horizons)
        data["slot_schema"] = list(self.slot_schema)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherProtocol":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


DEFAULT_PROTOCOL = TeacherProtocol()

# Everything that changes the teacher space and therefore requires a new
# `protocol_id` (section 2.1). `window_stride` is excluded: it comes from the
# dataset config, not the teacher.
PROTOCOL_SEMANTIC_FIELDS = (
    "teacher_name",
    "teacher_revision",
    "teacher_implementation_revision",
    "input_range",
    "input_size",
    "resize_mode",
    "camera_keys",
    "layout",
    "view_mode",
    "aggregator_token_selector",
    "token_centering",
    "feature_compression",
    "camera_tag",
    "token_compression",
    "queries_per_camera",
    "num_queries",
    "hidden_dim",
    "storage_dtype",
    "target_construction",
    "cache_layout",
    "sequence_length",
    "pair_order",
    "selected_temporal_slot",
    "static_baseline",
    "cached_horizons",
    "slot_schema",
    "num_video_frames",
)


def split_t_layout(frames: torch.Tensor) -> torch.Tensor:
    """Split the T-shaped mosaic into the three source camera views.

    Args:
        frames: ``[B, 3, H, W]`` RGB in ``[0, 1]`` at the *native* mosaic
            resolution (no letterbox padding).

    Returns:
        ``[B, 3, 3, H_v, W_v]`` -- ``(batch, view, channel, height, width)`` with
        views ordered ``(head, left_wrist, right_wrist)``. Views are resized to a
        common size so they can be stacked; the head view defines that size.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected [B, 3, H, W], got {tuple(frames.shape)}")

    _, _, height, width = frames.shape
    # combined_h = orig_h + orig_h // 2, so orig_h == ceil(2 * combined_h / 3).
    head_h = math.ceil(2 * height / 3)
    half_w = width // 2
    if head_h >= height or half_w < 1:
        raise ValueError(
            f"frame {height}x{width} is not a valid RoboTwin T-mosaic"
        )

    head = frames[:, :, :head_h, :]
    left = frames[:, :, head_h:, :half_w]
    right = frames[:, :, head_h:, half_w:]

    target_hw = (head.shape[-2], head.shape[-1])
    views = [
        view
        if view.shape[-2:] == target_hw
        else F.interpolate(view, size=target_hw, mode="bilinear", align_corners=False)
        for view in (head, left, right)
    ]
    return torch.stack(views, dim=1)


def letterbox(images: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Aspect-preserving resize + centre pad with black, mirroring the dataset."""
    target_h, target_w = int(size[0]), int(size[1])
    _, _, height, width = images.shape
    scale = min(target_h / height, target_w / width)
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))

    resized = F.interpolate(
        images, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True
    )
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    return F.pad(
        resized,
        (pad_left, target_w - new_w - pad_left, pad_top, target_h - new_h - pad_top),
        value=0.0,
    )


def build_camera_tags(
    num_cameras: int,
    hidden_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic sinusoidal tag per camera position -> ``[V, D]``."""
    positions = torch.arange(num_cameras, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(hidden_dim, device=device, dtype=torch.float32).unsqueeze(0)
    tags = torch.sin((positions + 1.0) / (10000.0 ** (freqs / hidden_dim)))
    return tags.to(dtype)


def load_vggt(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load and freeze the VGGT teacher. Imported lazily to keep training light."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The `vggt` package is required to build teacher targets. "
            "Install it with `pip install vggt` (or from "
            "https://github.com/facebookresearch/vggt)."
        ) from exc

    model = VGGT.from_pretrained(checkpoint_path)
    model.eval().to(device=device, dtype=dtype)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _deepest_aggregator_tokens(aggregated) -> torch.Tensor:
    tokens = next((item for item in reversed(aggregated) if item is not None), None)
    if tokens is None:
        raise ValueError("VGGT aggregator returned no cached layer outputs")
    return tokens


def _compress_view_tokens(
    tokens: torch.Tensor,
    protocol: TeacherProtocol,
) -> torch.Tensor:
    """Compress ``[B, V, N, C]`` tokens into the protocol's ``[B, M, D]``."""
    if tokens.ndim != 4:
        raise ValueError(f"expected [B, V, N, C], got {tuple(tokens.shape)}")
    batch, num_views = tokens.shape[:2]
    if num_views != protocol.num_views:
        raise ValueError(
            f"protocol expects {protocol.num_views} views, got {num_views}"
        )

    from models.geometry_jepa import resize_feature_dim

    tokens = tokens.float()
    if protocol.token_centering == "per_view_mean_subtraction":
        tokens = tokens - tokens.mean(dim=-2, keepdim=True)
    elif protocol.token_centering != "none":
        raise ValueError(f"unknown token_centering: {protocol.token_centering!r}")

    tokens = resize_feature_dim(
        tokens.reshape(batch * num_views, -1, tokens.shape[-1]), protocol.hidden_dim
    ).reshape(batch, num_views, -1, protocol.hidden_dim)

    if protocol.camera_tag == "fixed_sinusoidal":
        tokens = tokens + build_camera_tags(
            num_views, protocol.hidden_dim, device=tokens.device
        ).view(1, num_views, 1, protocol.hidden_dim)
    elif protocol.camera_tag != "none":
        raise ValueError(f"unknown camera_tag: {protocol.camera_tag!r}")

    per_camera = tokens.reshape(batch * num_views, -1, protocol.hidden_dim)
    per_camera = F.adaptive_avg_pool1d(
        per_camera.transpose(1, 2), protocol.queries_per_camera
    ).transpose(1, 2)

    pooled = per_camera.reshape(batch, protocol.num_queries, protocol.hidden_dim)
    target = F.layer_norm(pooled, (protocol.hidden_dim,))
    if not torch.isfinite(target).all():
        raise ValueError("VGGT produced non-finite teacher latents")
    return target.float().cpu()


@torch.inference_mode()
def build_teacher_target(
    teacher,
    views: torch.Tensor,
    protocol: TeacherProtocol = DEFAULT_PROTOCOL,
) -> torch.Tensor:
    """Encode multi-view RGB into a single ``[B, M, D]`` teacher latent.

    Args:
        teacher: Frozen VGGT model.
        views: ``[B, V, 3, H, W]`` RGB in ``[0, 1]``, cameras in protocol order.
        protocol: Teacher protocol.

    Returns:
        ``[B, M, D]`` float32 CPU tensor.
    """
    if views.ndim != 5:
        raise ValueError(f"expected [B, V, 3, H, W], got {tuple(views.shape)}")
    batch, num_views = views.shape[0], views.shape[1]
    if num_views != protocol.num_views:
        raise ValueError(
            f"protocol expects {protocol.num_views} views, got {num_views}"
        )

    param = next(teacher.parameters())
    device, dtype = param.device, param.dtype

    images = views.reshape(batch * num_views, *views.shape[2:])
    images = letterbox(images.float(), protocol.input_size)
    images = images.clamp(0.0, 1.0).to(device=device, dtype=dtype)

    # VGGT consumes [B, S, 3, H, W]; each camera is an independent single-view
    # sample, so S == 1.
    aggregated, _patch_start_idx = teacher.aggregator(images.unsqueeze(1))
    tokens = _deepest_aggregator_tokens(aggregated)
    if tokens.ndim == 4:  # [B*V, S, N, C_v] with S == 1
        tokens = tokens.squeeze(1)
    if tokens.ndim != 3:
        raise ValueError(f"unexpected aggregator token shape {tuple(tokens.shape)}")
    tokens = tokens.float().reshape(batch, num_views, -1, tokens.shape[-1])
    return _compress_view_tokens(tokens, protocol)


@torch.inference_mode()
def build_temporal_pair_target(
    teacher,
    current_views: torch.Tensor,
    future_views: torch.Tensor,
    protocol: TeacherProtocol = DEFAULT_PROTOCOL,
    selected_slot: int = 1,
) -> torch.Tensor:
    """Jointly encode same-camera ``(current, future)`` pairs and select one slot."""
    if current_views.ndim != 5 or future_views.ndim != 5:
        raise ValueError(
            "current_views and future_views must both be [B, V, 3, H, W]"
        )
    if current_views.shape != future_views.shape:
        raise ValueError(
            f"pair view shapes differ: {tuple(current_views.shape)} vs "
            f"{tuple(future_views.shape)}"
        )
    if selected_slot not in (0, 1):
        raise ValueError(f"selected_slot must be 0 or 1, got {selected_slot}")

    batch, num_views = current_views.shape[:2]
    if num_views != protocol.num_views:
        raise ValueError(
            f"protocol expects {protocol.num_views} views, got {num_views}"
        )

    param = next(teacher.parameters())
    device, dtype = param.device, param.dtype
    pairs = torch.stack((current_views, future_views), dim=2)
    images = pairs.reshape(batch * num_views * 2, *pairs.shape[3:])
    images = letterbox(images.float(), protocol.input_size)
    images = images.clamp(0.0, 1.0).reshape(
        batch * num_views, 2, *images.shape[1:]
    )
    images = images.to(device=device, dtype=dtype)

    aggregated, _patch_start_idx = teacher.aggregator(images)
    tokens = _deepest_aggregator_tokens(aggregated)
    if tokens.ndim != 4 or tokens.shape[1] != 2:
        raise ValueError(
            "temporal-pair VGGT output must be [B*V, 2, N, C], got "
            f"{tuple(tokens.shape)}"
        )
    selected = tokens[:, selected_slot].reshape(
        batch, num_views, -1, tokens.shape[-1]
    )
    return _compress_view_tokens(selected, protocol)


def write_manifest(cache_dir: Path, protocol: TeacherProtocol, extra: Optional[Dict] = None) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol.to_dict()
    if extra:
        payload.update(extra)
    path = cache_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_manifest(cache_dir: Path) -> Dict[str, Any]:
    path = Path(cache_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher cache manifest not found: {path}. "
            "Run tools/precompute_vggt_cache.py first."
        )
    return json.loads(path.read_text())


def _teacher_identity(protocol: "TeacherProtocol", name: str) -> Any:
    """Comparable value of one protocol field.

    A local checkpoint directory and a hub id name the same teacher, so only the
    trailing model name of `teacher_revision` is compared; pointing at a
    different model still trips the drift check.
    """
    value = getattr(protocol, name)
    return Path(value).name if name == "teacher_revision" else value


def assert_protocol_id_covers_overrides(
    protocol: "TeacherProtocol", explicit_id: Optional[str] = None
) -> "TeacherProtocol":
    """Refuse to write a cache whose semantics drifted from its `protocol_id`.

    Section 2.1: any change to the teacher pipeline must produce a new protocol.
    `window_stride` is exempt -- it is derived from the dataset config, not from
    the teacher, and the reader validates it separately.
    """
    default = TeacherProtocol()
    drifted = {
        name: getattr(protocol, name)
        for name in PROTOCOL_SEMANTIC_FIELDS
        if _teacher_identity(protocol, name) != _teacher_identity(default, name)
    }
    if not drifted:
        # An explicit id still wins: matching fields do not prove a matching
        # teacher, since `teacher_revision` only compares the model name.
        return protocol if explicit_id is None else replace(protocol, protocol_id=explicit_id)
    if explicit_id is None or explicit_id == default.protocol_id:
        raise ValueError(
            "the teacher protocol was changed but protocol_id was not:\n  "
            + "\n  ".join(f"{k}: {default.__dict__.get(k)!r} -> {v!r}" for k, v in drifted.items())
            + "\nPass --protocol-id with a new identifier; mixing protocols in one "
            "cache silently corrupts the teacher space."
        )
    return replace(protocol, protocol_id=explicit_id)


def assert_manifest_matches(
    cache_dir: Path,
    protocol_id: str,
    num_queries: int,
    hidden_dim: int,
    window_stride: int,
    target_construction: Optional[str] = None,
    cache_layout: Optional[str] = None,
    cached_horizons: Optional[Sequence[int]] = None,
    num_video_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Fail fast when the cache on disk does not match the training config."""
    manifest = load_manifest(cache_dir)
    default = TeacherProtocol()
    expected = {
        "protocol_id": protocol_id,
        "num_queries": int(num_queries),
        "hidden_dim": int(hidden_dim),
        "window_stride": int(window_stride),
        "horizon_unit": default.horizon_unit,
        "sample_key_schema": default.sample_key_schema,
    }
    optional_expected = {
        "target_construction": target_construction,
        "cache_layout": cache_layout,
        "cached_horizons": (
            None if cached_horizons is None else [int(k) for k in cached_horizons]
        ),
        "num_video_frames": (
            None if num_video_frames is None else int(num_video_frames)
        ),
    }
    expected.update(
        {key: value for key, value in optional_expected.items() if value is not None}
    )
    if target_construction == TEMPORAL_PAIR_TARGET:
        if cached_horizons is None:
            raise ValueError(
                "cached_horizons is required for a temporal-pair teacher cache"
            )
        expected.update(
            {
                "sequence_length": 2,
                "pair_order": "current,future",
                "selected_temporal_slot": "future",
                "static_baseline": "repeat_current_pair",
                "slot_schema": ["static"]
                + [f"h{int(horizon)}" for horizon in cached_horizons],
                "complete": True,
            }
        )
    # `teacher_revision` is a local path and so environment-specific; the writer
    # side (`assert_protocol_id_covers_overrides`) is what stops it drifting
    # without a new protocol_id.
    # Compared through `to_dict()` so sequences match the JSON round-trip, and
    # never overriding what the caller passed in explicitly.
    as_json = default.to_dict()
    expected.update(
        {
            name: as_json[name]
            for name in PROTOCOL_SEMANTIC_FIELDS
            if name != "teacher_revision" and name in manifest and name not in expected
        }
    )
    mismatches = [
        f"{key}: cache={manifest.get(key)!r} config={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Teacher cache manifest does not match the training config:\n  "
            + "\n  ".join(mismatches)
        )
    return manifest
