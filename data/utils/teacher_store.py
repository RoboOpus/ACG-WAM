"""Read-only access to the pre-computed VGGT teacher latent cache.

The public interface follows ``LDJEPA_transfer.md`` section 5::

    store.get(sample_key, horizon) -> [M, D]

where ``sample_key`` names a *training window* as
``"<split>/<task>/<episode>@<start_frame>"`` and ``horizon`` counts sampled video
transitions inside that window. The target frame is therefore
``start_frame + horizon * window_stride``.

Because a teacher latent only depends on the frame, the storage backend is
deduplicated per frame: every horizon of every overlapping window resolves to the
same row. Layout on disk::

    <cache_dir>/
        manifest.json                        # TeacherProtocol, see vggt_teacher.py
        index.json                           # episode_key -> {file, num_frames, num_rows}
        frame_counts.json                    # episode_key -> num_frames
        shards/<split>/<task>/<episode>.npy  # [num_rows, M, D], row i == frame i*stride

Window starts are multiples of ``window_stride``, so every reachable target frame
is a multiple of it too and a shard holds ``ceil(num_frames / stride)`` rows.

``TemporalPairTeacherStore`` handles the separate Joint layout. Its episode
shards are ``[num_windows, static+horizons, M, D]`` because each target depends
on both the window start and the selected horizon.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

INDEX_FILENAME = "index.json"
FRAME_COUNTS_FILENAME = "frame_counts.json"
SHARD_DIRNAME = "shards"

logger = logging.getLogger(__name__)


class TeacherCacheError(RuntimeError):
    """Raised when a teacher target cannot be served.

    Callers must not swallow this: silently skipping samples would change the
    effective training distribution and desynchronise JEPA batches across ranks.
    """


def split_sample_key(sample_key: str) -> Tuple[str, int]:
    """``"clean/task/3@42"`` -> ``("clean/task/3", 42)``."""
    episode_key, _, start = sample_key.rpartition("@")
    if not episode_key or not start.isdigit():
        raise TeacherCacheError(
            f"malformed sample_key {sample_key!r}, expected '<episode_key>@<start_frame>'"
        )
    return episode_key, int(start)


def make_sample_key(episode_key: str, start_frame: int) -> str:
    return f"{episode_key}@{int(start_frame)}"


def shard_path(cache_dir: Path, episode_key: str) -> Path:
    return Path(cache_dir) / SHARD_DIRNAME / f"{episode_key}.npy"


def load_index(cache_dir: Path) -> Dict[str, Dict]:
    path = Path(cache_dir) / INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher cache index not found: {path}. "
            "Run tools/precompute_vggt_cache.py first."
        )
    return json.loads(path.read_text())


def save_json(path: Path, payload: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def save_index(cache_dir: Path, index: Dict[str, Dict]) -> Path:
    return save_json(Path(cache_dir) / INDEX_FILENAME, index)


class TeacherStore:
    """Memory-mapped teacher latent store with fail-fast validation."""

    def __init__(
        self,
        cache_dir: str,
        num_queries: int,
        hidden_dim: int,
        window_stride: int,
        max_open_files: int = 64,
    ):
        self.cache_dir = Path(cache_dir)
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.window_stride = int(window_stride)
        self.max_open_files = int(max_open_files)

        self.index = load_index(self.cache_dir)
        self._assert_index_matches_manifest()
        self._mmaps: "OrderedDict[str, np.memmap]" = OrderedDict()
        self._owner_pid = os.getpid()

    def _assert_index_matches_manifest(self) -> None:
        """Reject an index left behind by a different protocol.

        The manifest is published before the shards are encoded, so a re-run that
        dies before `--merge-index` leaves a new manifest next to the old
        `index.json`. Every shard has the same shape, so the per-entry stamp is
        the only thing that can tell the two protocols apart.
        """
        from data.utils.vggt_teacher import load_manifest

        expected = load_manifest(self.cache_dir).get("protocol_id")
        stamped = [entry.get("protocol_id") for entry in self.index.values()]
        mismatched = sorted(
            key
            for key, entry in self.index.items()
            if entry.get("protocol_id") not in (None, expected)
        )
        if mismatched:
            raise TeacherCacheError(
                f"{len(mismatched)}/{len(self.index)} cache entries were encoded under a "
                f"different protocol than the manifest's {expected!r} "
                f"(e.g. {mismatched[:3]}). Re-run tools/precompute_vggt_cache.py with "
                "--overwrite, then --merge-index."
            )
        if any(value is None for value in stamped):
            logger.warning(
                f"{self.cache_dir} predates per-episode protocol stamps; its contents "
                f"cannot be checked against the manifest's {expected!r}"
            )

    # -- introspection ---------------------------------------------------

    def __contains__(self, episode_key: str) -> bool:
        return episode_key in self.index

    def __len__(self) -> int:
        return len(self.index)

    def num_frames(self, episode_key: str) -> int:
        entry = self.index.get(episode_key)
        if entry is None:
            raise TeacherCacheError(f"episode not in teacher cache: {episode_key!r}")
        return int(entry["num_frames"])

    # -- internals -------------------------------------------------------

    def _open(self, episode_key: str) -> np.memmap:
        # DataLoader workers fork after construction; drop inherited handles.
        pid = os.getpid()
        if pid != self._owner_pid:
            self._mmaps = OrderedDict()
            self._owner_pid = pid

        mmap = self._mmaps.get(episode_key)
        if mmap is not None:
            self._mmaps.move_to_end(episode_key)
            return mmap

        entry = self.index.get(episode_key)
        if entry is None:
            raise TeacherCacheError(
                f"episode not in teacher cache: {episode_key!r}. "
                "Regenerate the cache or check the dataset split/task filters."
            )
        path = self.cache_dir / entry["file"]
        if not path.exists():
            raise TeacherCacheError(f"teacher cache shard missing: {path}")

        mmap = np.load(path, mmap_mode="r")
        if mmap.ndim != 3 or mmap.shape[1:] != (self.num_queries, self.hidden_dim):
            raise TeacherCacheError(
                f"teacher shard {path} has shape {mmap.shape}, expected "
                f"[*, {self.num_queries}, {self.hidden_dim}]"
            )

        self._mmaps[episode_key] = mmap
        while len(self._mmaps) > self.max_open_files:
            self._mmaps.popitem(last=False)
        return mmap

    def _rows(self, episode_key: str, frames: Sequence[int]) -> torch.Tensor:
        mmap = self._open(episode_key)
        rows = []
        for frame in frames:
            if frame % self.window_stride:
                raise TeacherCacheError(
                    f"frame {frame} of {episode_key!r} is not a multiple of the "
                    f"window stride {self.window_stride}; the cache only holds "
                    "stride-aligned frames"
                )
            row = frame // self.window_stride
            if row < 0 or row >= mmap.shape[0]:
                raise TeacherCacheError(
                    f"frame {frame} out of range for {episode_key!r} "
                    f"(cached rows: {mmap.shape[0]}, stride: {self.window_stride})"
                )
            rows.append(row)

        target = torch.from_numpy(np.ascontiguousarray(mmap[rows])).float()
        if not torch.isfinite(target).all():
            raise TeacherCacheError(
                f"teacher cache contains NaN/Inf for {episode_key!r} frames {list(frames)}"
            )
        return target

    # -- lookup ----------------------------------------------------------

    def get_horizons(self, sample_key: str, horizons: Sequence[int]) -> torch.Tensor:
        """Return ``[K, M, D]`` for several horizons of one training window."""
        episode_key, start_frame = split_sample_key(sample_key)
        if start_frame % self.window_stride:
            raise TeacherCacheError(
                f"window start {start_frame} in {sample_key!r} is not a multiple of "
                f"the window stride {self.window_stride}"
            )
        frames = [start_frame + int(k) * self.window_stride for k in horizons]
        return self._rows(episode_key, frames)

    def get(self, sample_key: str, horizon: int) -> torch.Tensor:
        """Return ``[M, D]`` for one ``(window, horizon)`` pair."""
        return self.get_horizons(sample_key, [horizon])[0]

    def get_batch(
        self, sample_keys: Sequence[str], horizons: Sequence[int]
    ) -> torch.Tensor:
        """Return ``[B, M, D]``."""
        if len(sample_keys) != len(horizons):
            raise ValueError("sample_keys and horizons must have equal length")
        return torch.stack(
            [self.get(key, int(k)) for key, k in zip(sample_keys, horizons)], dim=0
        )

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_mmaps"] = OrderedDict()
        return state


class TemporalPairTeacherStore:
    """Window-addressed store for jointly encoded temporal-pair targets.

    Each episode shard is ``[W, S, M, D]``. ``W`` follows the dataset's ordered
    stride-aligned window starts and ``S`` is ``(static, *cached_horizons)``.
    """

    def __init__(
        self,
        cache_dir: str,
        num_queries: int,
        hidden_dim: int,
        window_stride: int,
        cached_horizons: Sequence[int],
        num_video_frames: int,
        max_open_files: int = 64,
    ):
        from data.utils.vggt_teacher import (
            TEMPORAL_PAIR_TARGET,
            WINDOW_CACHE_LAYOUT,
            load_manifest,
        )

        self.cache_dir = Path(cache_dir)
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.window_stride = int(window_stride)
        self.manifest = load_manifest(self.cache_dir)
        self.cached_horizons = tuple(
            int(horizon) for horizon in self.manifest.get("cached_horizons", [])
        )
        missing_horizons = [
            int(horizon)
            for horizon in cached_horizons
            if int(horizon) not in self.cached_horizons
        ]
        if missing_horizons:
            raise TeacherCacheError(
                f"Requested teacher horizons {missing_horizons} are missing "
                f"from cache cached_horizons={list(self.cached_horizons)}"
            )
        self.num_video_frames = int(num_video_frames)
        self.max_open_files = int(max_open_files)
        self.slot_horizons = (0,) + self.cached_horizons
        self._horizon_to_slot = {
            horizon: slot for slot, horizon in enumerate(self.slot_horizons)
        }

        expected = {
            "target_construction": TEMPORAL_PAIR_TARGET,
            "cache_layout": WINDOW_CACHE_LAYOUT,
            "window_stride": self.window_stride,
            "num_queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "cached_horizons": list(self.cached_horizons),
            "num_video_frames": self.num_video_frames,
            "slot_schema": ["static"]
            + [f"h{horizon}" for horizon in self.cached_horizons],
            "complete": True,
        }
        mismatches = [
            f"{key}: cache={self.manifest.get(key)!r} expected={value!r}"
            for key, value in expected.items()
            if self.manifest.get(key) != value
        ]
        if mismatches:
            raise TeacherCacheError(
                "Temporal-pair cache manifest mismatch:\n  "
                + "\n  ".join(mismatches)
            )

        self.index = load_index(self.cache_dir)
        self._assert_index()
        self._mmaps: "OrderedDict[str, np.memmap]" = OrderedDict()
        self._owner_pid = os.getpid()

    def _expected_num_windows(self, num_frames: int) -> int:
        max_start = int(num_frames) - (self.num_video_frames * self.window_stride + 1)
        return 0 if max_start < 0 else max_start // self.window_stride + 1

    def _assert_index(self) -> None:
        protocol_id = self.manifest["protocol_id"]
        problems = []
        for episode_key, entry in self.index.items():
            expected_windows = self._expected_num_windows(entry["num_frames"])
            if entry.get("protocol_id") != protocol_id:
                problems.append(
                    f"{episode_key}: protocol_id={entry.get('protocol_id')!r}"
                )
            if int(entry.get("num_windows", -1)) != expected_windows:
                problems.append(
                    f"{episode_key}: num_windows={entry.get('num_windows')!r}, "
                    f"expected={expected_windows}"
                )
            if int(entry.get("num_slots", -1)) != len(self.slot_horizons):
                problems.append(
                    f"{episode_key}: num_slots={entry.get('num_slots')!r}, "
                    f"expected={len(self.slot_horizons)}"
                )
        if problems:
            raise TeacherCacheError(
                "Temporal-pair cache index mismatch:\n  "
                + "\n  ".join(problems[:8])
            )

    def __contains__(self, episode_key: str) -> bool:
        return episode_key in self.index

    def __len__(self) -> int:
        return len(self.index)

    def num_frames(self, episode_key: str) -> int:
        entry = self.index.get(episode_key)
        if entry is None:
            raise TeacherCacheError(f"episode not in teacher cache: {episode_key!r}")
        return int(entry["num_frames"])

    def _open(self, episode_key: str) -> np.memmap:
        pid = os.getpid()
        if pid != self._owner_pid:
            self._mmaps = OrderedDict()
            self._owner_pid = pid

        mmap = self._mmaps.get(episode_key)
        if mmap is not None:
            self._mmaps.move_to_end(episode_key)
            return mmap

        entry = self.index.get(episode_key)
        if entry is None:
            raise TeacherCacheError(
                f"episode not in teacher cache: {episode_key!r}. "
                "Regenerate the cache or check the dataset filters."
            )
        path = self.cache_dir / entry["file"]
        if not path.exists():
            raise TeacherCacheError(f"teacher cache shard missing: {path}")

        mmap = np.load(path, mmap_mode="r")
        expected_shape = (
            int(entry["num_windows"]),
            len(self.slot_horizons),
            self.num_queries,
            self.hidden_dim,
        )
        if mmap.shape != expected_shape:
            raise TeacherCacheError(
                f"temporal-pair shard {path} has shape {mmap.shape}, "
                f"expected {expected_shape}"
            )

        self._mmaps[episode_key] = mmap
        while len(self._mmaps) > self.max_open_files:
            self._mmaps.popitem(last=False)
        return mmap

    def get_horizons(self, sample_key: str, horizons: Sequence[int]) -> torch.Tensor:
        """Return ``[K, M, D]`` for static/horizon slots of one window."""
        episode_key, start_frame = split_sample_key(sample_key)
        if start_frame % self.window_stride:
            raise TeacherCacheError(
                f"window start {start_frame} in {sample_key!r} is not a multiple of "
                f"the window stride {self.window_stride}"
            )

        unsupported = [int(k) for k in horizons if int(k) not in self._horizon_to_slot]
        if unsupported:
            raise TeacherCacheError(
                f"temporal-pair cache has no horizons {unsupported}; available "
                f"slots are {list(self.slot_horizons)}"
            )

        mmap = self._open(episode_key)
        row = start_frame // self.window_stride
        if row < 0 or row >= mmap.shape[0]:
            raise TeacherCacheError(
                f"window start {start_frame} out of range for {episode_key!r} "
                f"(cached windows: {mmap.shape[0]}, stride: {self.window_stride})"
            )
        slots = [self._horizon_to_slot[int(k)] for k in horizons]
        target = torch.from_numpy(np.ascontiguousarray(mmap[row, slots])).float()
        if not torch.isfinite(target).all():
            raise TeacherCacheError(
                f"temporal-pair cache contains NaN/Inf for {sample_key!r} "
                f"horizons {list(horizons)}"
            )
        return target

    def get(self, sample_key: str, horizon: int) -> torch.Tensor:
        return self.get_horizons(sample_key, [horizon])[0]

    def get_batch(
        self, sample_keys: Sequence[str], horizons: Sequence[int]
    ) -> torch.Tensor:
        if len(sample_keys) != len(horizons):
            raise ValueError("sample_keys and horizons must have equal length")
        return torch.stack(
            [self.get(key, int(k)) for key, k in zip(sample_keys, horizons)], dim=0
        )

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_mmaps"] = OrderedDict()
        return state


def create_teacher_store(
    cache_dir: str,
    num_queries: int,
    hidden_dim: int,
    window_stride: int,
    cached_horizons: Sequence[int] = (),
    num_video_frames: int = 0,
):
    """Open the storage backend declared by the cache manifest."""
    from data.utils.vggt_teacher import (
        FRAME_CACHE_LAYOUT,
        WINDOW_CACHE_LAYOUT,
        load_manifest,
    )

    layout = load_manifest(Path(cache_dir)).get("cache_layout", FRAME_CACHE_LAYOUT)
    common = dict(
        cache_dir=cache_dir,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        window_stride=window_stride,
    )
    if layout == FRAME_CACHE_LAYOUT:
        return TeacherStore(**common)
    if layout == WINDOW_CACHE_LAYOUT:
        return TemporalPairTeacherStore(
            **common,
            cached_horizons=cached_horizons,
            num_video_frames=num_video_frames,
        )
    raise TeacherCacheError(f"unsupported teacher cache_layout: {layout!r}")
