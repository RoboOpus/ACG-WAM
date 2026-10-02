# Robotwin2 Dataset Loader for Motus
# Supports Robotwin2 data with video and action data from multiple tasks

import os
import random
import h5py
import numpy as np
import cv2
import json
import torch
import torch.utils.data as data
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Sequence, Tuple
import logging
from pathlib import Path
import warnings
from PIL import Image
import tempfile

# VLM processing imports
from utils.vlm_utils import preprocess_vlm_messages
from transformers import AutoProcessor

# Import image processing utilities
from data.utils.image_utils import (
    tensor_to_pil, apply_image_augmentation,
    load_video_frames, get_video_frame_count
)

# Geometry JEPA teacher latent cache
from data.utils.teacher_store import (
    FRAME_COUNTS_FILENAME,
    TeacherCacheError,
    create_teacher_store,
    make_sample_key,
    save_json,
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

class RobotWinTaskDataset(data.Dataset):
    """
    Dataset for RobotWin data with task-level organization and flexible sampling.
    
    Data structure:
    /share/dataset/preprocess/robotwin2/
    ├── clean/
    │   ├── adjust_bottle/
    │   │   ├── qpos/           # Robot position files (.pt)
    │   │   ├── videos/         # MP4 video files  
    │   │   └── umt5_wan/       # Pre-encoded language embeddings (.pt)
    │   ├── beat_block_hammer/
    │   └── ...
    └── randomized/
        ├── adjust_bottle/
        └── ...
    """
    
    def __init__(
        self,
        dataset_dir: str = "/share/dataset/preprocess/robotwin2/",
        data_mode: str = "clean",  # "clean", "randomized", or "both"
        task_mode: str = "multi",  # "single" or "multi" 
        task_name: Optional[str] = None,  # Required for single task mode
        task_names: Optional[Sequence[str]] = None,  # Task subset for multi task mode
        randomized_limit_per_task: Optional[int] = None,  # Limit randomized episodes per task (take first N)
        
        # Sampling parameters
        global_downsample_rate: int = 3,  # Global downsampling (e.g., 30Hz -> 10Hz)
        video_action_freq_ratio: int = 5,  # Video:Action frequency ratio  
        num_video_frames: int = 3,  # Number of video frames to predict
        
        # Standard parameters
        video_size: Tuple[int, int] = (320, 384),
        max_episodes: Optional[int] = None,
        episode_names: Optional[Sequence[str]] = None,
        upsample_rate: int = 1,  # For compatibility with H_RDT
        val: bool = False,
        image_aug: bool = False,
        
        # VLM processing parameters
        vlm_checkpoint_path: Optional[str] = None,  # Path to VLM model

        # Geometry JEPA teacher latent cache (see data/utils/vggt_teacher.py)
        teacher_cache_dir: Optional[str] = None,
        teacher_num_queries: int = 96,
        teacher_hidden_dim: int = 768,
        teacher_cached_horizons: Sequence[int] = (),

        # Deterministic window enumeration (required by the JEPA sample-key contract)
        deterministic_windows: bool = False,
        frame_count_cache: Optional[str] = None,
    ):
        """
        Initialize RobotWin dataset with flexible sampling.
        
        Args:
            dataset_dir: Root directory containing clean/ and randomized/ folders
            data_mode: Which data split to use ("clean", "randomized", or "both")
            task_mode: Single task or multi-task ("single" or "multi")
            task_name: Task name for single task mode (e.g., "adjust_bottle")
            task_names: Restrict multi task mode to this task subset. None keeps
                every task found under the split directories.
            
            global_downsample_rate: Global downsampling rate (e.g., 3 for 30Hz->10Hz)
            video_action_freq_ratio: Frequency ratio between video and action
            num_video_frames: Number of video frames to predict
            
            video_size: Target video resolution (H, W)
            max_episodes: Maximum number of episodes to load (for debugging)
            episode_names: Optional exact episode names to include.
            upsample_rate: Temporal data upsampling rate (for H_RDT compatibility)
            val: Whether this is validation set
            image_aug: Whether to apply image augmentation
            teacher_cache_dir: Root of the pre-computed VGGT teacher cache. When
                set, every sample additionally carries `teacher_latents`
                [num_video_frames + 1, M, D], indexed by horizon with row 0 the
                anchor at the window start.
            teacher_num_queries: Teacher query count M (must match the cache).
            teacher_hidden_dim: Teacher query width D (must match the cache).
            teacher_cached_horizons: Explicit temporal-pair slots. Empty means
                the legacy dense endpoint layout ``0..num_video_frames``.
            deterministic_windows: Enumerate every training window up front so
                that `__getitem__(idx)` is a pure function of `idx` and each
                window gets the stable key `<split>/<task>/<episode>@<start>`.
                Implied by `teacher_cache_dir`.
            frame_count_cache: JSON file caching `episode_key -> num_frames`, so
                enumeration does not have to reopen every video on start-up.
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_mode = data_mode
        self.task_mode = task_mode
        self.task_name = task_name
        self.task_names = None if task_names is None else {str(t) for t in task_names}
        self.randomized_limit_per_task = randomized_limit_per_task
        
        # Sampling parameters
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        
        # Calculate action sequence length
        self.action_chunk_size = num_video_frames * video_action_freq_ratio

        # Physical frame span of one video transition. horizon k targets
        # `start_frame + k * window_stride`, so aligning window starts to this
        # stride keeps every teacher target on the same grid.
        self.window_stride = video_action_freq_ratio * global_downsample_rate
        # Smallest episode length that fits a full window without clamping actions.
        self.min_episode_frames = self.action_chunk_size * global_downsample_rate + 1

        self.deterministic_windows = bool(deterministic_windows or teacher_cache_dir)
        self.frame_count_cache = frame_count_cache
        self.windows: List[Dict[str, Any]] = []
        self.sample_weights: List[float] = []
        
        # Standard parameters
        self.video_size = video_size
        self.max_episodes = max_episodes
        self.episode_names = (
            None if episode_names is None else {str(name) for name in episode_names}
        )
        self.upsample_rate = upsample_rate
        self.val = val
        self.image_aug = image_aug
        
        # Validate parameters
        if task_mode == "single" and not task_name:
            raise ValueError("Single task mode requires task_name parameter")
        
        assert data_mode in ["clean", "randomized", "both"], \
            f"data_mode must be 'clean', 'randomized', or 'both', got {data_mode}"
        
        # Initialize data structures
        if task_mode == "single":
            self.episode_files = []  # List of episode files for single task
        else:
            self.task_to_episodes = {}  # Task name -> episode files mapping
            self.task_weights = {}      # Task sampling weights
        
        self.total_episodes = 0
        
        logger.info(f"RobotWin dataset initialized:")
        logger.info(f"  Data mode: {data_mode}")
        logger.info(f"  Task mode: {task_mode}")
        if task_name:
            logger.info(f"  Task name: {task_name}")
        logger.info(f"  Global downsample rate: {global_downsample_rate}")
        logger.info(f"  Video:Action frequency ratio: {video_action_freq_ratio}")
        logger.info(f"  Action chunk size: {self.action_chunk_size}")
        logger.info(f"  Video frames to predict: {num_video_frames}")
        logger.info(f"  Total episodes: {self.total_episodes}")
        
        # Initialize VLM processor for complete VLM processing in dataset
        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor from {vlm_checkpoint_path}: {e}")
                logger.warning("VLM processing will be disabled for this dataset instance")
        else:
            logger.info("VLM checkpoint path not provided, VLM processing disabled")

        # Geometry JEPA teacher cache (fail fast: a missing/incompatible cache must
        # not silently disable the auxiliary loss)
        self.teacher_store = None
        self.teacher_cached_horizons = tuple(int(k) for k in teacher_cached_horizons)
        self.teacher_horizons = (
            (0,) + self.teacher_cached_horizons
            if self.teacher_cached_horizons
            else tuple(range(0, self.num_video_frames + 1))
        )
        if teacher_cache_dir:
            self.teacher_store = create_teacher_store(
                cache_dir=teacher_cache_dir,
                num_queries=teacher_num_queries,
                hidden_dim=teacher_hidden_dim,
                window_stride=self.window_stride,
                cached_horizons=self.teacher_cached_horizons,
                num_video_frames=self.num_video_frames,
            )
            logger.info(
                f"Teacher latent cache loaded from {teacher_cache_dir} "
                f"({len(self.teacher_store)} episodes)"
            )

        # Load dataset episodes
        self._load_episodes()

        if self.deterministic_windows:
            self._build_window_index(self.iter_episodes())
    
    def _limit_episodes_first_n(self, episodes: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        """
        Limit to the first N episodes based on sorted episode_name.
        Numeric names are sorted numerically; otherwise lexicographically.
        """
        if n is None or n <= 0 or not episodes:
            return episodes
        def _sort_key(ep: Dict[str, Any]):
            name = ep.get('episode_name', '')
            try:
                return (0, int(name))
            except Exception:
                return (1, str(name))
        episodes_sorted = sorted(episodes, key=_sort_key)
        return episodes_sorted[:n]
    
    def _scan_task_folder(self, task_path: Path) -> List[str]:
        """
        Scan a single task folder.
        
        Args:
            task_path: Path to task folder (e.g., .../clean/adjust_bottle)
            
        Returns:
            List of valid episode identifiers
        """
        qpos_dir = task_path / "qpos"
        videos_dir = task_path / "videos"
        umt5_dir = task_path / "umt5_wan"
        
        # Check if all required directories exist
        if not all([qpos_dir.exists(), videos_dir.exists(), umt5_dir.exists()]):
            logger.warning(f"Missing data directories in {task_path}")
            return []
        
        # Find valid episodes (those that have all three data types)
        valid_episodes = []
        
        # Get all qpos files as base (.pt format)
        qpos_files = list(qpos_dir.glob("*.pt"))
        
        for qpos_file in qpos_files:
            episode_name = qpos_file.stem
            
            # Check if corresponding video and language files exist
            video_file = videos_dir / f"{episode_name}.mp4"
            lang_file = umt5_dir / f"{episode_name}.pt"
            
            if video_file.exists() and lang_file.exists():
                # Store full paths
                episode_data = {
                    'episode_name': episode_name,
                    'task_name': task_path.name,
                    'split': task_path.parent.name,
                    # Stable identity of the episode, used as the teacher cache key.
                    # Invariant to shuffling, worker/rank count and batch size.
                    'episode_key': f"{task_path.parent.name}/{task_path.name}/{episode_name}",
                    'qpos_path': str(qpos_file),
                    'video_path': str(video_file),
                    'lang_path': str(lang_file),
                }
                valid_episodes.append(episode_data)

        if self.episode_names is not None:
            valid_episodes = [
                episode
                for episode in valid_episodes
                if episode['episode_name'] in self.episode_names
            ]
        
        logger.info(f"Task {task_path.name} ({task_path.parent.name}): Found {len(valid_episodes)} valid episodes")
        return valid_episodes
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Initialize dataset by scanning folders."""
        logger.info("Initializing dataset...")
        
        # Determine which data splits to scan
        if self.data_mode == "both":
            data_splits = ["clean", "randomized"]
        else:
            data_splits = [self.data_mode]
        
        all_episodes = []
        
        for split in data_splits:
            split_dir = self.dataset_dir / split
            
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            if self.task_mode == "single":
                # Single task mode: scan specific task
                task_dir = split_dir / self.task_name
                if task_dir.exists():
                    episodes = self._scan_task_folder(task_dir)
                    # If scanning randomized split, optionally limit to first N episodes per task
                    if split == "randomized" and self.randomized_limit_per_task is not None:
                        before = len(episodes)
                        episodes = self._limit_episodes_first_n(episodes, self.randomized_limit_per_task)
                        logger.info(f"Applied randomized per-task limit ({self.randomized_limit_per_task}) for {task_dir.name}: {before} -> {len(episodes)}")
                    all_episodes.extend(episodes)
                else:
                    logger.warning(f"Task directory not found: {task_dir}")
            
            else:
                # Multi task mode: scan all tasks
                task_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                if self.task_names is not None:
                    task_dirs = [d for d in task_dirs if d.name in self.task_names]
                
                for task_dir in task_dirs:
                    episodes = self._scan_task_folder(task_dir)
                    
                    # If scanning randomized split, optionally limit to first N episodes per task
                    if split == "randomized" and self.randomized_limit_per_task is not None:
                        before = len(episodes)
                        episodes = self._limit_episodes_first_n(episodes, self.randomized_limit_per_task)
                        logger.info(f"Applied randomized per-task limit ({self.randomized_limit_per_task}) for {task_dir.name}: {before} -> {len(episodes)}")
                    
                    # Group by task for multi-task sampling
                    task_name = task_dir.name
                    if task_name not in self.task_to_episodes:
                        self.task_to_episodes[task_name] = []
                    self.task_to_episodes[task_name].extend(episodes)
        
        if self.task_mode == "single":
            self.episode_files = all_episodes
            
            # Limit episodes if requested
            if self.max_episodes is not None:
                self.episode_files = self.episode_files[:self.max_episodes]
            
            self.total_episodes = len(self.episode_files)
            
            if self.total_episodes == 0:
                raise ValueError(f"No valid episodes found for task {self.task_name}")
        
        else:
            # Multi-task mode: calculate sampling weights
            if not self.task_to_episodes:
                raise ValueError("No valid episodes found for any task")
            
            # Equal weight for all tasks
            num_tasks = len(self.task_to_episodes)
            for task_name in self.task_to_episodes.keys():
                self.task_weights[task_name] = 1.0 / num_tasks
            
            self.total_episodes = sum(len(episodes) for episodes in self.task_to_episodes.values())
            
            logger.info(f"Multi-task dataset with {num_tasks} tasks:")
            for task_name, episodes in self.task_to_episodes.items():
                logger.info(f"  {task_name}: {len(episodes)} episodes")
        
        # Return all episodes for consistency with other datasets
        return all_episodes

    def _load_robot_data(self, qpos_path: str, action_indices: List[int], initial_state_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load robot position data.
        
        Args:
            qpos_path: Path to qpos .pt file
            action_indices: List of action frame indices
            initial_state_idx: Index for initial state (should match condition frame)
            
        Returns:
            - initial_state: Robot state at condition frame [state_dim]
            - action_sequence: Actions at specified indices [len(action_indices), action_dim]
        """
        qpos_data = torch.load(qpos_path, map_location='cpu')  # [T, feature_dim]
        
        # Get initial state at the condition frame index
        if initial_state_idx >= len(qpos_data):
            initial_state_idx = len(qpos_data) - 1
        initial_state = qpos_data[initial_state_idx].float()
        
        # Get action sequence at specified indices
        actions = []
        for idx in action_indices:
            if idx >= len(qpos_data):
                raise IndexError(f"Action index {idx} out of bounds for qpos data length {len(qpos_data)}")
            else:
                action = qpos_data[idx]
            actions.append(action)
        
        action_sequence = torch.stack(actions).float()
        
        # Normalize actions and initial state
        # action_sequence = self._normalize_actions(action_sequence)
        # initial_state = self._normalize_actions(initial_state.unsqueeze(0)).squeeze(0)  # Normalize state same way as actions
        
        return initial_state, action_sequence
    
    def _load_language_embedding(self, lang_path: str) -> tuple[torch.Tensor, int]:
        """Load pre-encoded language embedding and return the selected index."""
        try:
            embedding_data = torch.load(lang_path, map_location='cpu')
            
            # RobotWin embedding is always a list of tensors
            selected_idx = random.randint(0, len(embedding_data) - 1)
            embeddings = embedding_data[selected_idx]  # [seq_len, 4096]
            
            # Remove batch dimension if present
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)
            
            return embeddings, selected_idx
            
        except Exception as e:
            logger.error(f"Error loading language embedding from {lang_path}: {e}")
            raise
    
    def _load_text_instruction(self, task_name: str, episode_name: str, instruction_idx: int = None, split: Optional[str] = None) -> str:
        """Load text instruction for VLM processing.

        Args:
            task_name: Task name (e.g., "adjust_bottle")
            episode_name: Episode identifier (e.g., "429")
            instruction_idx: Optional index to select a deterministic instruction
            split: Dataset split to read from ("clean" or "randomized"). If None,
                   falls back to self.data_mode when it is not "both".
        """
        try:
            # Try to read from meta file first
            # Path structure: dataset_dir/split/task_name/metas/episode_name.txt
            if split is None:
                if self.data_mode in ["clean", "randomized"]:
                    split_to_use = self.data_mode
                else:
                    # Fallback to clean when split cannot be inferred
                    split_to_use = "clean"
            else:
                split_to_use = split

            meta_file = self.dataset_dir / split_to_use / task_name / "metas" / f"{episode_name}.txt"
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    # Filter out empty lines
                    instructions = [line.strip() for line in lines if line.strip()]
                
                if instructions:
                    if instruction_idx is not None and 0 <= instruction_idx < len(instructions):
                        # Use specific index to match language_embedding
                        return instructions[instruction_idx]
                    else:
                        # Random selection (fallback for old behavior)
                        import random
                        return random.choice(instructions)
                else:
                    # No fallback - raise error if no instructions found
                    raise ValueError(f"No instructions found in meta file for {task_name}/{episode_name}")
            else:
                raise FileNotFoundError(f"Meta file not found: {meta_file}")
                
        except Exception as e:
            logger.error(f"Failed to load text instruction for {task_name}/{episode_name}: {e}")
            raise
    
    def _sampling_indices_from_start(
        self, condition_frame_idx: int, total_frames: int
    ) -> Tuple[int, List[int], List[int]]:
        """Resolve one window start into video/action frame indices.

        Action i covers the transition to physical frame
        `condition_frame_idx + (i + 1) * global_downsample_rate`, and video frame
        i is the target of action `(i + 1) * video_action_freq_ratio - 1`. Hence
        horizon k (1-based) targets `condition_frame_idx + k * window_stride` and
        consumes the first `k * video_action_freq_ratio` actions.
        """
        action_indices = [
            min(
                condition_frame_idx + (i + 1) * self.global_downsample_rate,
                total_frames - 1,
            )
            for i in range(self.action_chunk_size)
        ]

        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            video_indices.append(
                action_indices[min(action_step, len(action_indices) - 1)]
            )

        return condition_frame_idx, video_indices, action_indices

    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """Pick a random window start and resolve it (non-deterministic mode)."""
        max_condition_idx = total_frames - self.min_episode_frames
        condition_frame_idx = (
            random.randint(0, max_condition_idx) if max_condition_idx >= 0 else 0
        )
        return self._sampling_indices_from_start(condition_frame_idx, total_frames)

    # -- deterministic window enumeration ---------------------------------

    def _load_frame_counts(self, episodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """`episode_key -> num_frames`, cached on disk to avoid reopening videos."""
        cache_path = Path(self.frame_count_cache) if self.frame_count_cache else None
        counts: Dict[str, int] = {}
        if cache_path is not None and cache_path.exists():
            counts = {k: int(v) for k, v in json.loads(cache_path.read_text()).items()}

        missing = [ep for ep in episodes if ep['episode_key'] not in counts]
        if missing:
            logger.info(f"Probing frame counts for {len(missing)} episodes...")
            with ThreadPoolExecutor(max_workers=32) as pool:
                probed = pool.map(
                    lambda ep: (ep['episode_key'], get_video_frame_count(ep['video_path'])),
                    missing,
                )
                counts.update(dict(probed))
            if cache_path is not None:
                save_json(cache_path, counts)
                logger.info(f"Cached frame counts to {cache_path}")

        return counts

    def _build_window_index(self, episodes: List[Dict[str, Any]]) -> None:
        """Enumerate every training window so `__getitem__` becomes deterministic.

        Window starts are multiples of `window_stride`, which keeps all teacher
        targets on the same frame grid. Episodes too short to fill a window are
        dropped rather than clamped, so padded actions never reach the predictor.
        """
        frame_counts = self._load_frame_counts(episodes)

        # Reproduce the previous sampling distribution (uniform over tasks, then
        # uniform over episodes) as per-window weights.
        episodes_per_task = {}
        for ep in episodes:
            episodes_per_task[ep['task_name']] = episodes_per_task.get(ep['task_name'], 0) + 1
        num_tasks = max(len(episodes_per_task), 1)

        self.windows = []
        self.sample_weights = []
        skipped = 0
        for ep in episodes:
            total_frames = frame_counts.get(ep['episode_key'], 0)
            max_start = total_frames - self.min_episode_frames
            if max_start < 0:
                skipped += 1
                continue
            starts = list(range(0, max_start + 1, self.window_stride))
            weight = 1.0 / (num_tasks * episodes_per_task[ep['task_name']] * len(starts))
            for start in starts:
                self.windows.append({
                    'episode': ep,
                    'start_frame': start,
                    'total_frames': total_frames,
                    'sample_key': make_sample_key(ep['episode_key'], start),
                })
                self.sample_weights.append(weight)

        if not self.windows:
            raise ValueError(
                f"No episode is long enough for a full window "
                f"(need >= {self.min_episode_frames} frames)"
            )
        logger.info(
            f"Enumerated {len(self.windows)} windows over {len(episodes) - skipped} "
            f"episodes (stride {self.window_stride}, skipped {skipped} too-short episodes)"
        )
    
    def __len__(self) -> int:
        """Number of enumerated windows, or an episode-proportional estimate."""
        if self.deterministic_windows:
            return len(self.windows)
        return self.total_episodes * 10  # Assume 100 samples per episode

    def iter_episodes(self) -> List[Dict[str, Any]]:
        """Return every discovered episode, sorted by `episode_key`.

        Used by tools/precompute_vggt_cache.py to enumerate the teacher cache.
        The order is deterministic and independent of the random training-window
        sampling done in `__getitem__`.
        """
        if self.task_mode == "single":
            episodes = list(self.episode_files)
        else:
            episodes = [ep for eps in self.task_to_episodes.values() for ep in eps]
        return sorted(episodes, key=lambda ep: ep['episode_key'])
    
    def _select_window(self, idx: int, attempt: int):
        """Resolve `idx` into `(episode, start_frame, total_frames, sample_key)`.

        In deterministic mode this is a pure function of `(idx, attempt)`, so a
        window keeps its key no matter how the DataLoader shuffles, how many
        workers run or which rank asks. Retries walk the index with a fixed
        stride instead of drawing a new random sample.
        """
        if self.deterministic_windows:
            window = self.windows[(idx + attempt * 7919) % len(self.windows)]
            return (
                window['episode'],
                window['start_frame'],
                window['total_frames'],
                window['sample_key'],
            )

        if self.task_mode == "single":
            episode_data = random.choice(self.episode_files) if self.episode_files else None
        else:
            task_name = random.choices(
                list(self.task_weights.keys()),
                weights=list(self.task_weights.values()),
                k=1,
            )[0] if self.task_to_episodes else None
            task_episodes = self.task_to_episodes.get(task_name, []) if task_name else []
            episode_data = random.choice(task_episodes) if task_episodes else None
        if episode_data is None:
            return None, None, None, None

        total_frames = get_video_frame_count(episode_data['video_path'])
        if total_frames < 2:
            return None, None, None, None
        max_start = total_frames - self.min_episode_frames
        start = random.randint(0, max_start) if max_start >= 0 else 0
        return (
            episode_data,
            start,
            total_frames,
            make_sample_key(episode_data['episode_key'], start),
        )

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a training sample.

        Args:
            idx: Window index (deterministic mode) or ignored (random mode)

        Returns:
            Dictionary containing training data
        """
        # Robust sampling with retries to avoid returning None (which breaks DDP sync)
        max_attempts = 8
        for attempt in range(max_attempts):
            episode_data, condition_frame_idx, total_frames, sample_key = self._select_window(
                idx, attempt
            )
            if episode_data is None:
                continue

            try:
                # Calculate sampling indices
                _, video_indices, action_indices = self._sampling_indices_from_start(
                    condition_frame_idx, total_frames
                )

                # Load frames and aligned robot/action data
                first_frame = load_video_frames(episode_data['video_path'], [condition_frame_idx], self.video_size)
                video_frames = load_video_frames(episode_data['video_path'], video_indices, self.video_size)
                initial_state, action_sequence = self._load_robot_data(episode_data['qpos_path'], action_indices, condition_frame_idx)
                language_embedding, instruction_idx = self._load_language_embedding(episode_data['lang_path'])

                # Infer split from paths
                inferred_split = None
                try:
                    for key in ('qpos_path', 'video_path', 'lang_path'):
                        parts = Path(episode_data[key]).parts
                        if 'clean' in parts:
                            inferred_split = 'clean'
                            break
                        if 'randomized' in parts:
                            inferred_split = 'randomized'
                            break
                except Exception:
                    inferred_split = None

                # Load raw text instruction for VLM processing using the same index
                text_instruction = self._load_text_instruction(
                    episode_data['task_name'],
                    episode_data['episode_name'],
                    instruction_idx,
                    split=inferred_split,
                )

                # Complete VLM processing in dataset
                vlm_inputs = None
                if self.vlm_processor is not None:
                    first_frame_pil = tensor_to_pil(first_frame.squeeze(0))
                    vlm_inputs = preprocess_vlm_messages(text_instruction, first_frame_pil, self.vlm_processor)

                sample = {
                    'first_frame': first_frame.squeeze(0),
                    'video_frames': video_frames,
                    'initial_state': initial_state,
                    'action_sequence': action_sequence,
                    'language_embedding': language_embedding,
                    'vlm_inputs': vlm_inputs,
                    # Transition identity: horizon k (1-based) targets video frame
                    # `video_frame_indices[k - 1]` == start_frame + k * window_stride
                    # and consumes the first `k * video_action_freq_ratio` actions.
                    'sample_key': sample_key,
                    'episode_key': episode_data['episode_key'],
                    'condition_frame_idx': condition_frame_idx,
                    'video_frame_indices': torch.tensor(video_indices, dtype=torch.long),
                }

                if self.teacher_store is not None:
                    sample['teacher_latents'] = self.teacher_store.get_horizons(
                        sample_key, self.teacher_horizons
                    )

                return sample

            except TeacherCacheError:
                # Never retry past a cache problem: skipping samples would change
                # the training distribution and hide a broken cache.
                raise
            except Exception as e:
                logger.warning(f"Retry due to sample error ({episode_data.get('episode_name','?')}): {e}")
                continue

        # If all attempts failed, let caller drop this sample (rare)
        return None