#!/usr/bin/env python3
"""Pre-compute the frozen VGGT teacher latent cache for Geometry JEPA.

The default cache is keyed by ``(episode_key, absolute_frame_index)``. A config
with ``target_mode: joint_future_slot`` instead writes a separate window-level
cache with slots ``(static, *horizon_choices)``.

Single GPU::

    python tools/precompute_vggt_cache.py \
        --config configs/robotwin.yaml \
        --cache-dir /mnt/workspace/Tao/cache/vggt_teacher_robotwin \
        --vggt-checkpoint pretrained_models/VGGT-1B

Multi GPU (one process per shard, then merge)::

    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python tools/precompute_vggt_cache.py \
            --config configs/robotwin.yaml --cache-dir <dir> \
            --num-shards 4 --shard-id $i &
    done; wait
    python tools/precompute_vggt_cache.py --cache-dir <dir> --merge-index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))

from data.utils.image_utils import get_video_frame_count  # noqa: E402
from data.utils.teacher_store import (  # noqa: E402
    FRAME_COUNTS_FILENAME,
    SHARD_DIRNAME,
    save_index,
    save_json,
)
from data.utils.vggt_teacher import (  # noqa: E402
    FRAME_CACHE_LAYOUT,
    JOINT_FUTURE_SLOT_PROTOCOL_ID,
    TEMPORAL_PAIR_TARGET,
    WINDOW_CACHE_LAYOUT,
    TeacherProtocol,
    assert_protocol_id_covers_overrides,
    build_temporal_pair_target,
    build_teacher_target,
    load_manifest,
    load_vggt,
    split_t_layout,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_episode_list(config) -> List[Dict]:
    """Enumerate every episode of the configured dataset as `(episode_key, video_path)`."""
    dataset_type = str(config.dataset.get("type", "robotwin"))
    if dataset_type == "lerobot":
        from data.lerobot.lerobot_dataset import iter_lerobot_episodes

        params = config.dataset.params
        return iter_lerobot_episodes(
            repo_id=params.repo_id, root=params.get("root", None)
        )
    if dataset_type != "robotwin":
        raise ValueError(
            f"teacher cache generation is not implemented for dataset.type={dataset_type!r}"
        )

    from data.robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset

    dataset_cfg = config.dataset
    dataset = RobotWinTaskDataset(
        dataset_dir=dataset_cfg.dataset_dir,
        data_mode=dataset_cfg.get("data_mode", "clean"),
        task_mode=dataset_cfg.get("task_mode", "multi"),
        task_name=dataset_cfg.get("task_name", None),
        randomized_limit_per_task=dataset_cfg.get("randomized_limit_per_task", None),
        max_episodes=dataset_cfg.get("max_episodes", None),
        num_video_frames=config.common.num_video_frames,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        global_downsample_rate=config.common.global_downsample_rate,
        video_size=(config.common.video_height, config.common.video_width),
        vlm_checkpoint_path=None,
        teacher_cache_dir=None,
    )
    return dataset.iter_episodes()


def manifest_dataset_fields(config) -> Dict[str, str]:
    """Provenance of the encoded episodes, recorded alongside the protocol."""
    dataset_cfg = config.dataset
    dataset_type = str(dataset_cfg.get("type", "robotwin"))
    if dataset_type == "lerobot":
        params = dataset_cfg.params
        return {
            "dataset_type": dataset_type,
            "repo_id": str(params.repo_id),
            "root": str(params.get("root", "")),
            "video_key": "observation.images.cam_concatenated",
            "episode_key_schema": "lerobot/<repo_id>/<episode_index:06d>",
        }
    return {
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_cfg.dataset_dir),
        "data_mode": str(dataset_cfg.get("data_mode", "clean")),
    }


def encode_episode(
    teacher,
    video_path: str,
    total_frames: int,
    protocol: TeacherProtocol,
    batch_size: int,
    storage_dtype: np.dtype,
    device: torch.device,
) -> np.ndarray:
    """Encode the stride-aligned frames of one episode -> ``[num_rows, M, D]``.

    Window starts and horizons are both multiples of ``window_stride``, so only
    those frames can ever be a teacher target. Row ``i`` holds frame
    ``i * window_stride``.
    """
    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    frames = list(range(0, total_frames, protocol.window_stride))
    out = np.empty(
        (len(frames), protocol.num_queries, protocol.hidden_dim), dtype=storage_dtype
    )

    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        batch = reader.get_batch(chunk).asnumpy()  # [b, H, W, 3] uint8 RGB
        batch = (
            torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0).to(device)
        )
        views = split_t_layout(batch)  # [b, V, 3, Hv, Wv]
        target = build_teacher_target(teacher, views, protocol)  # [b, M, D] cpu fp32
        out[start : start + len(chunk)] = target.numpy().astype(storage_dtype)

    return out


def temporal_pair_window_starts(
    total_frames: int, protocol: TeacherProtocol
) -> List[int]:
    """Enumerate the same full-window starts as ``RobotWinTaskDataset``."""
    max_start = int(total_frames) - (
        int(protocol.num_video_frames) * int(protocol.window_stride) + 1
    )
    if max_start < 0:
        return []
    return list(range(0, max_start + 1, int(protocol.window_stride)))


def encode_temporal_pair_episode(
    teacher,
    video_path: str,
    total_frames: int,
    protocol: TeacherProtocol,
    batch_size: int,
    storage_dtype: np.dtype,
    device: torch.device,
    output_path: Path,
) -> int:
    """Write one ``[window, static+horizons, M, D]`` pair shard."""
    from decord import VideoReader, cpu

    if protocol.target_construction != TEMPORAL_PAIR_TARGET:
        raise ValueError("encode_temporal_pair_episode requires a pair protocol")

    starts = temporal_pair_window_starts(total_frames, protocol)
    slot_horizons = (0,) + tuple(int(k) for k in protocol.cached_horizons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (
        len(starts),
        len(slot_horizons),
        protocol.num_queries,
        protocol.hidden_dim,
    )
    if not starts:
        np.save(output_path, np.empty(shape, dtype=storage_dtype))
        return 0

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=storage_dtype,
        shape=shape,
    )

    pairs = [
        (window_idx, slot_idx, start, start + horizon * protocol.window_stride)
        for window_idx, start in enumerate(starts)
        for slot_idx, horizon in enumerate(slot_horizons)
    ]
    for begin in range(0, len(pairs), batch_size):
        chunk = pairs[begin : begin + batch_size]
        frame_indices = [
            frame
            for _, _, current, future in chunk
            for frame in (current, future)
        ]
        frames = reader.get_batch(frame_indices).asnumpy()
        frames = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
            .to(device)
        )
        views = split_t_layout(frames)
        targets = build_temporal_pair_target(
            teacher, views[0::2], views[1::2], protocol
        )
        for target, (window_idx, slot_idx, _, _) in zip(targets, chunk):
            out[window_idx, slot_idx] = target.numpy().astype(storage_dtype)

    out.flush()
    del out
    return len(starts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_manifest(cache_dir: Path, index: Dict[str, Dict]) -> None:
    """Record measured cache size and counts after all shards are published."""
    manifest = load_manifest(cache_dir)
    index_path = Path(cache_dir) / "index.json"
    frame_counts_path = Path(cache_dir) / FRAME_COUNTS_FILENAME
    total_bytes = sum(
        (Path(cache_dir) / entry["file"]).stat().st_size for entry in index.values()
    )
    manifest.update(
        {
            "complete": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "num_episodes": len(index),
            "total_bytes": total_bytes,
            "index_sha256": _sha256(index_path),
            "frame_counts_sha256": _sha256(frame_counts_path),
        }
    )
    if manifest.get("cache_layout", FRAME_CACHE_LAYOUT) == WINDOW_CACHE_LAYOUT:
        total_windows = sum(int(entry["num_windows"]) for entry in index.values())
        manifest.update(
            {
                "total_windows": total_windows,
                "total_pairs": total_windows * len(manifest["slot_schema"]),
            }
        )
    save_json(Path(cache_dir) / "manifest.json", manifest)


def merge_index(cache_dir: Path) -> None:
    """Combine per-shard index parts into the final ``index.json``."""
    merged: Dict[str, Dict] = {}
    parts = sorted(cache_dir.glob("index.part-*.json"))
    if not parts:
        raise FileNotFoundError(f"no index.part-*.json found in {cache_dir}")
    for part in parts:
        merged.update(json.loads(part.read_text()))

    # Nothing downstream can detect a cache that mixes protocols: the shards all
    # have the same shape and only the manifest records which one produced them.
    # Caches written before the stamp existed carry None and stay loadable.
    manifest = load_manifest(cache_dir)
    expected = manifest["protocol_id"]
    stale = sorted(
        key
        for key, entry in merged.items()
        if entry.get("protocol_id") not in (None, expected)
    )
    if stale:
        raise ValueError(
            f"{len(stale)}/{len(merged)} episodes were not encoded under "
            f"{expected!r} (e.g. {stale[:3]}); re-run those shards with --overwrite"
        )
    unstamped = sum(1 for entry in merged.values() if entry.get("protocol_id") is None)
    if (
        manifest.get("cache_layout", FRAME_CACHE_LAYOUT) == WINDOW_CACHE_LAYOUT
        and unstamped
    ):
        raise ValueError("temporal-pair cache entries must carry protocol stamps")
    if unstamped:
        logger.warning(
            f"{unstamped}/{len(merged)} episodes predate per-episode protocol stamps; "
            f"their contents cannot be checked against {expected!r}"
        )

    save_index(cache_dir, merged)
    write_frame_counts(cache_dir, merged)
    finalize_manifest(cache_dir, merged)
    logger.info(f"Merged {len(parts)} shards -> {len(merged)} episodes")


def write_frame_counts(cache_dir: Path, index: Dict[str, Dict]) -> None:
    """Publish `episode_key -> num_frames` so the dataset can enumerate windows
    without reopening every video."""
    save_json(
        Path(cache_dir) / FRAME_COUNTS_FILENAME,
        {key: int(entry["num_frames"]) for key, entry in index.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/robotwin.yaml")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument(
        "--vggt-checkpoint", type=str, default="pretrained_models/VGGT-1B"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="single frames or temporal pairs per VGGT call",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--limit-episodes", type=int, default=None, help="smoke-test on N episodes"
    )
    parser.add_argument(
        "--storage-dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument(
        "--queries-per-camera",
        type=int,
        default=32,
        help="queries pooled from each camera; num_queries = this * num_cameras",
    )
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument(
        "--protocol-id",
        type=str,
        default=None,
        help="required when any CLI flag moves the protocol off its defaults",
    )
    parser.add_argument(
        "--target-mode",
        choices=["single_frame", "joint_future_slot"],
        default=None,
        help="override cache construction; defaults to geometry_jepa.target_mode",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--merge-index", action="store_true", help="only merge existing shard indices"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.merge_index:
        merge_index(cache_dir)
        return

    config = OmegaConf.load(args.config)
    window_stride = int(
        config.common.video_action_freq_ratio * config.common.global_downsample_rate
    )
    configured_target = str(
        config.get("geometry_jepa", {}).get("target_mode", "single_frame")
    )
    target_mode = args.target_mode or (
        "joint_future_slot"
        if configured_target == "joint_future_slot"
        else "single_frame"
    )
    if target_mode == "joint_future_slot" and config.dataset.type != "robotwin":
        raise ValueError("joint_future_slot cache generation currently supports RobotWin only")

    protocol_kwargs = dict(
        teacher_revision=args.vggt_checkpoint,
        queries_per_camera=args.queries_per_camera,
        num_queries=args.queries_per_camera * len(TeacherProtocol.camera_keys),
        hidden_dim=args.hidden_dim,
        storage_dtype=args.storage_dtype,
        window_stride=window_stride,
    )
    explicit_protocol_id = args.protocol_id
    if target_mode == "joint_future_slot":
        horizons = tuple(int(k) for k in config.geometry_jepa.horizon_choices)
        protocol_kwargs.update(
            protocol_id=JOINT_FUTURE_SLOT_PROTOCOL_ID,
            target_construction=TEMPORAL_PAIR_TARGET,
            cache_layout=WINDOW_CACHE_LAYOUT,
            sequence_length=2,
            pair_order="current,future",
            selected_temporal_slot="future",
            static_baseline="repeat_current_pair",
            cached_horizons=horizons,
            slot_schema=("static",) + tuple(f"h{k}" for k in horizons),
            num_video_frames=int(config.common.num_video_frames),
        )
        if configured_target == "joint_future_slot":
            explicit_protocol_id = explicit_protocol_id or str(
                config.geometry_jepa.get(
                    "teacher_protocol_id", JOINT_FUTURE_SLOT_PROTOCOL_ID
                )
            )
        else:
            explicit_protocol_id = explicit_protocol_id or JOINT_FUTURE_SLOT_PROTOCOL_ID
    protocol = TeacherProtocol(**protocol_kwargs)
    protocol = assert_protocol_id_covers_overrides(
        protocol, explicit_id=explicit_protocol_id
    )
    logger.info(
        f"window_stride={window_stride} "
        f"(video_action_freq_ratio x global_downsample_rate); "
        f"only stride-aligned frames are encoded"
    )
    storage_dtype = np.dtype(args.storage_dtype)
    device = torch.device(args.device)

    episodes = build_episode_list(config)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    shard = [
        ep for i, ep in enumerate(episodes) if i % args.num_shards == args.shard_id
    ]
    logger.info(
        f"Shard {args.shard_id}/{args.num_shards}: {len(shard)}/{len(episodes)} episodes"
    )

    if args.shard_id == 0:
        cache_state = (
            {"complete": False}
            if protocol.cache_layout == WINDOW_CACHE_LAYOUT
            else {}
        )
        write_manifest(
            cache_dir,
            protocol,
            extra={
                "source_config": str(args.config),
                **manifest_dataset_fields(config),
                **cache_state,
            },
        )

    teacher = load_vggt(args.vggt_checkpoint, device=device)

    index: Dict[str, Dict] = {}
    index_path = cache_dir / f"index.part-{args.shard_id:03d}.json"
    if index_path.exists() and not args.overwrite:
        index = json.loads(index_path.read_text())

    failures = []
    for i, episode in enumerate(shard):
        key = episode["episode_key"]
        rel_file = f"{SHARD_DIRNAME}/{key}.npy"
        out_path = cache_dir / rel_file

        try:
            total_frames = get_video_frame_count(episode["video_path"])
        except Exception as exc:
            logger.error(f"[{key}] cannot read video: {exc}")
            if protocol.cache_layout == WINDOW_CACHE_LAYOUT:
                failures.append(key)
            continue

        entry = index.get(key, {})
        pair_fields_match = (
            protocol.cache_layout != WINDOW_CACHE_LAYOUT
            or (
                entry.get("num_windows")
                == len(temporal_pair_window_starts(total_frames, protocol))
                and entry.get("num_slots") == len(protocol.slot_schema)
            )
        )
        if (
            not args.overwrite
            and out_path.exists()
            and entry.get("num_frames") == total_frames
            # A shard written under another protocol has the same shape as a
            # valid one, so frame count alone cannot tell them apart.
            and entry.get("protocol_id") == protocol.protocol_id
            and pair_fields_match
        ):
            continue

        tmp_path = out_path.with_suffix(".tmp.npy")
        try:
            if protocol.cache_layout == WINDOW_CACHE_LAYOUT:
                num_rows = encode_temporal_pair_episode(
                    teacher,
                    episode["video_path"],
                    total_frames,
                    protocol,
                    args.batch_size,
                    storage_dtype,
                    device,
                    tmp_path,
                )
            else:
                latents = encode_episode(
                    teacher,
                    episode["video_path"],
                    total_frames,
                    protocol,
                    args.batch_size,
                    storage_dtype,
                    device,
                )
                np.save(tmp_path, latents)
                num_rows = int(latents.shape[0])
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            logger.error(f"[{key}] encoding failed: {exc}")
            if protocol.cache_layout == WINDOW_CACHE_LAYOUT:
                failures.append(key)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, out_path)

        index[key] = {
            "file": rel_file,
            "protocol_id": protocol.protocol_id,
            "num_frames": int(total_frames),
            "num_rows": num_rows,
            "video_path": episode["video_path"],
        }
        if protocol.cache_layout == WINDOW_CACHE_LAYOUT:
            index[key].update(
                num_windows=num_rows,
                num_slots=len(protocol.slot_schema),
            )
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True))

        if (i + 1) % 10 == 0 or i + 1 == len(shard):
            logger.info(f"[{args.shard_id}] {i + 1}/{len(shard)} episodes done")

    logger.info(f"Shard {args.shard_id} finished: {len(index)} episodes indexed")

    if failures:
        raise RuntimeError(
            f"temporal-pair cache generation failed for {len(failures)} episodes "
            f"(e.g. {failures[:3]}); the cache was not finalized"
        )

    if args.num_shards == 1:
        save_index(cache_dir, index)
        write_frame_counts(cache_dir, index)
        finalize_manifest(cache_dir, index)
        logger.info(f"Wrote {cache_dir / 'index.json'}")
    else:
        logger.info("Run with --merge-index once all shards are done.")


if __name__ == "__main__":
    main()
