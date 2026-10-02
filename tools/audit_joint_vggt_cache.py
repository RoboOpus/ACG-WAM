#!/usr/bin/env python3
"""Audit a RobotWin joint-future-slot VGGT cache before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.utils.teacher_store import TemporalPairTeacherStore  # noqa: E402
from data.utils.vggt_teacher import (  # noqa: E402
    TEMPORAL_PAIR_TARGET,
    TeacherProtocol,
    build_teacher_target,
    build_temporal_pair_target,
    load_manifest,
    load_vggt,
    split_t_layout,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    delta = (left.float() - right.float()).flatten(1).norm(dim=1)
    scale = right.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    return delta / scale


def per_sample_loss(config, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    loss = torch.zeros(prediction.shape[0])
    if bool(config.use_pool_cosine):
        loss += float(config.alpha_cos) * (
            1.0
            - F.cosine_similarity(
                prediction.mean(dim=1), target.mean(dim=1), dim=-1
            )
        )
    if bool(config.use_token_mse):
        loss += float(config.beta_mse) * (prediction - target).square().mean((1, 2))
    if float(config.gamma_token_cos):
        loss += float(config.gamma_token_cos) * (
            1.0 - F.cosine_similarity(prediction, target, dim=-1)
        ).mean(dim=1)
    return loss


def load_views(reader, frame_indices: Sequence[int], device: torch.device) -> torch.Tensor:
    frames = reader.get_batch(list(frame_indices)).asnumpy()
    frames = (
        torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    )
    return split_t_layout(frames)


def encode_pairs(
    teacher,
    reader,
    pairs: Sequence[Tuple[int, int]],
    protocol: TeacherProtocol,
    selected_slot: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for begin in range(0, len(pairs), batch_size):
        chunk = pairs[begin : begin + batch_size]
        indices = [frame for pair in chunk for frame in pair]
        views = load_views(reader, indices, device)
        outputs.append(
            build_temporal_pair_target(
                teacher,
                views[0::2],
                views[1::2],
                protocol,
                selected_slot=selected_slot,
            )
        )
    return torch.cat(outputs) if outputs else torch.empty(0)


def encode_frames(
    teacher,
    reader,
    frames: Sequence[int],
    protocol: TeacherProtocol,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for begin in range(0, len(frames), batch_size):
        chunk = frames[begin : begin + batch_size]
        views = load_views(reader, chunk, device)
        outputs.append(build_teacher_target(teacher, views, protocol))
    return torch.cat(outputs) if outputs else torch.empty(0)


def choose_episode(index: dict, requested: str | None) -> str:
    if requested is not None:
        if requested not in index:
            raise KeyError(f"episode {requested!r} is not present in the cache")
        return requested
    matches = sorted(key for key in index if key.startswith("clean/adjust_bottle/"))
    if not matches:
        raise KeyError("cache has no clean/adjust_bottle episode")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robotwin_joint.yaml")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--vggt-checkpoint", default="pretrained_models/VGGT-1B")
    parser.add_argument("--episode-key", default=None)
    parser.add_argument("--num-windows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    cache_dir = Path(args.cache_dir or config.geometry_jepa.cache_dir)
    manifest = load_manifest(cache_dir)
    protocol = TeacherProtocol.from_dict(manifest)
    if protocol.target_construction != TEMPORAL_PAIR_TARGET:
        raise ValueError(f"not a temporal-pair cache: {cache_dir}")

    store = TemporalPairTeacherStore(
        str(cache_dir),
        num_queries=int(config.geometry_jepa.num_queries),
        hidden_dim=int(config.geometry_jepa.hidden_dim),
        window_stride=int(protocol.window_stride),
        cached_horizons=config.geometry_jepa.horizon_choices,
        num_video_frames=int(config.common.num_video_frames),
    )
    episode_key = choose_episode(store.index, args.episode_key)
    entry = store.index[episode_key]
    num_windows = int(entry["num_windows"])
    if num_windows < 2:
        raise ValueError("audit requires at least two valid windows")
    rng = random.Random(args.seed)
    rows = sorted(rng.sample(range(num_windows), min(args.num_windows, num_windows)))
    starts = [row * protocol.window_stride for row in rows]
    horizons = [int(k) for k in protocol.cached_horizons]

    from decord import VideoReader, cpu

    reader = VideoReader(entry["video_path"], ctx=cpu(0), num_threads=4)
    device = torch.device(args.device)
    teacher = load_vggt(args.vggt_checkpoint, device=device)

    slot_horizons = [0] + horizons
    cache_targets = torch.stack(
        [store.get_horizons(f"{episode_key}@{start}", slot_horizons) for start in starts]
    )
    cache_pairs = [
        (start, start + horizon * protocol.window_stride)
        for start in starts
        for horizon in slot_horizons
    ]
    recomputed = encode_pairs(
        teacher, reader, cache_pairs, protocol, 1, args.batch_size, device
    ).reshape(len(starts), len(slot_horizons), protocol.num_queries, protocol.hidden_dim)

    repeat = encode_pairs(
        teacher,
        reader,
        cache_pairs[: min(4, len(cache_pairs))],
        protocol,
        1,
        args.batch_size,
        device,
    )
    determinism_max_abs = float(
        (repeat - recomputed.reshape(-1, protocol.num_queries, protocol.hidden_dim)[: len(repeat)])
        .abs()
        .max()
    )
    quantized = recomputed.to(torch.float16).float()
    cache_quantized_exact = bool(torch.equal(cache_targets, quantized))

    transition_pairs = [
        (start, start + horizon * protocol.window_stride)
        for start in starts
        for horizon in horizons
    ]
    future_frames = [future for _, future in transition_pairs]
    independent = encode_frames(
        teacher, reader, future_frames, protocol, args.batch_size, device
    )
    joint = recomputed[:, 1:].reshape(-1, protocol.num_queries, protocol.hidden_dim)

    alternate_starts = starts[1:] + starts[:1]
    context_pairs = [
        (alternate, start + horizon * protocol.window_stride)
        for start, alternate in zip(starts, alternate_starts)
        for horizon in horizons
    ]
    changed_context = encode_pairs(
        teacher, reader, context_pairs, protocol, 1, args.batch_size, device
    )
    reversed_pairs = [(future, current) for current, future in transition_pairs]
    reversed_same_future = encode_pairs(
        teacher, reader, reversed_pairs, protocol, 0, args.batch_size, device
    )

    pair_signal = relative_l2(joint, independent)
    context_signal = relative_l2(changed_context, joint)
    order_signal = relative_l2(reversed_same_future, joint)
    static = recomputed[:, :1].expand(-1, len(horizons), -1, -1).reshape_as(joint)
    baseline = per_sample_loss(config.geometry_jepa, static, joint)
    finite = bool(torch.isfinite(recomputed).all() and torch.isfinite(cache_targets).all())
    shape_ok = tuple(cache_targets.shape[1:]) == (
        len(slot_horizons),
        protocol.num_queries,
        protocol.hidden_dim,
    )

    threshold = 1e-4
    signal_fraction = {
        "pair_dependence": float((pair_signal > threshold).float().mean()),
        "current_context_dependence": float((context_signal > threshold).float().mean()),
        "order_sensitivity": float((order_signal > threshold).float().mean()),
    }
    denominator_fraction = float(
        (torch.isfinite(baseline) & (baseline > 1e-8)).float().mean()
    )
    gates = {
        "shape": shape_ok,
        "finite": finite,
        "fp32_determinism": determinism_max_abs <= 1e-5,
        "cache_matches_fp16_quantization": cache_quantized_exact,
        "pair_dependence": signal_fraction["pair_dependence"] >= 0.95,
        "current_context_dependence": signal_fraction["current_context_dependence"] >= 0.95,
        "order_sensitivity": signal_fraction["order_sensitivity"] >= 0.95,
        "dynamic_static_baseline": denominator_fraction >= 0.95,
    }
    report = {
        "passed": all(gates.values()),
        "gates": gates,
        "seed": args.seed,
        "episode_key": episode_key,
        "num_windows": len(starts),
        "starts": starts,
        "horizons": horizons,
        "camera_order": list(protocol.camera_keys),
        "teacher_implementation_revision": protocol.teacher_implementation_revision,
        "shape": list(cache_targets.shape),
        "determinism_max_abs": determinism_max_abs,
        "signal_threshold": threshold,
        "signal_fraction": signal_fraction,
        "signal_relative_l2_median": {
            "pair_dependence": float(pair_signal.median()),
            "current_context_dependence": float(context_signal.median()),
            "order_sensitivity": float(order_signal.median()),
        },
        "target_rms": float(joint.square().mean().sqrt()),
        "static_denominator_valid_fraction": denominator_fraction,
        "static_denominator_mean_by_horizon": {
            str(horizon): float(baseline.reshape(len(starts), -1)[:, index].mean())
            for index, horizon in enumerate(horizons)
        },
        "cache_hit_rate": 1.0,
        "cache_bytes": (cache_dir / entry["file"]).stat().st_size,
        "shard_sha256": sha256(cache_dir / entry["file"]),
        "index_sha256": sha256(cache_dir / "index.json"),
        "manifest_sha256": sha256(cache_dir / "manifest.json"),
    }
    output = Path(args.output or cache_dir / f"audit_seed{args.seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()