# Dataset Factory
# Simple factory to create different types of datasets

from pathlib import Path
from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
import torch
from torch.utils.data import Sampler


class WeightedDistributedSampler(Sampler):
    """Weighted sampling with replacement, sharded across ranks.

    Enumerating windows makes `__getitem__` deterministic but changes the
    sampling distribution: long episodes contribute more windows. Per-window
    weights restore the previous behaviour (uniform over tasks, then uniform over
    episodes) while keeping the sample keys stable.

    Every rank draws the same sequence from a seeded generator and then takes a
    disjoint slice of it, so no communication is needed. Sampling is with
    replacement (matching the previous `random.choice` behaviour), so two ranks
    may legitimately land on the same window.
    """

    def __init__(self, weights, num_replicas: int = 1, rank: int = 0, seed: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = len(self.weights) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        total = self.num_samples * self.num_replicas
        drawn = torch.multinomial(
            self.weights, total, replacement=True, generator=generator
        )
        return iter(drawn[self.rank :: self.num_replicas].tolist())


def apply_geometry_jepa_params(config: OmegaConf, params: Dict[str, Any]) -> None:
    """Attach the pre-computed VGGT teacher latent cache to a loader's kwargs.

    Shared by every loader that supports the JEPA contract, so the manifest check
    and the window/cache wiring cannot drift apart between embodiments.
    """
    geo_cfg = config.get('geometry_jepa', None)
    if geo_cfg is not None and geo_cfg.get('enable', False):
        cache_dir = geo_cfg.get('cache_dir', None)
        if not cache_dir:
            raise ValueError(
                "geometry_jepa.enable is true but geometry_jepa.cache_dir is not set"
            )
        from data.utils.teacher_store import FRAME_COUNTS_FILENAME
        from data.utils.vggt_teacher import (
            TEMPORAL_PAIR_TARGET,
            WINDOW_CACHE_LAYOUT,
            assert_manifest_matches,
            load_manifest,
        )

        num_queries = int(geo_cfg.get('num_queries', 96))
        hidden_dim = int(geo_cfg.get('hidden_dim', 768))
        target_mode = str(geo_cfg.get('target_mode', 'absolute'))
        window_stride = int(
            config.common.video_action_freq_ratio * config.common.global_downsample_rate
        )
        manifest_kwargs = {}
        if target_mode == 'joint_future_slot':
            if str(config.dataset.get('type', 'robotwin')) != 'robotwin':
                raise ValueError(
                    "geometry_jepa.target_mode='joint_future_slot' currently "
                    "supports dataset.type=robotwin only"
                )
            horizons = [int(k) for k in geo_cfg.get('horizon_choices', [])]
            cached_horizons = [
                int(horizon)
                for horizon in load_manifest(Path(cache_dir)).get('cached_horizons', [])
            ]
            missing_horizons = [horizon for horizon in horizons if horizon not in cached_horizons]
            if missing_horizons:
                raise ValueError(
                    f"Requested teacher horizons {missing_horizons} are missing "
                    f"from cache cached_horizons={cached_horizons}"
                )
            manifest_kwargs = {
                'target_construction': TEMPORAL_PAIR_TARGET,
                'cache_layout': WINDOW_CACHE_LAYOUT,
                'cached_horizons': cached_horizons,
                'num_video_frames': int(config.common.num_video_frames),
            }
            params['teacher_cached_horizons'] = horizons

        assert_manifest_matches(
            cache_dir,
            protocol_id=geo_cfg.get(
                'teacher_protocol_id',
                'vggt_tlayout_3view_percam_q32_d768_centered_v3',
            ),
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            window_stride=window_stride,
            **manifest_kwargs,
        )
        params['teacher_cache_dir'] = cache_dir
        params['teacher_num_queries'] = num_queries
        params['teacher_hidden_dim'] = hidden_dim
        # The teacher cache is keyed by window, so windows must be enumerable.
        params['deterministic_windows'] = True
        params['frame_count_cache'] = str(Path(cache_dir) / FRAME_COUNTS_FILENAME)
    else:
        if config.dataset.get('deterministic_windows', False):
            params['deterministic_windows'] = True
        if config.dataset.get('frame_count_cache', None):
            params['frame_count_cache'] = config.dataset.frame_count_cache


def create_dataset(config: OmegaConf, val: bool = False):
    """
    Create dataset based on config.
    
    Args:
        config: Configuration object
        val: Whether to create validation dataset
        
    Returns:
        Dataset instance
    """
    dataset_type = config.dataset.get('type', 'robotwin')  # Default to robotwin
    
    if dataset_type == 'robotwin':
        from .robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'task_names'):
            params['task_names'] = config.dataset.task_names
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        if hasattr(config.dataset, 'randomized_limit_per_task'):
            params['randomized_limit_per_task'] = config.dataset.randomized_limit_per_task
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Geometry JEPA: attach the pre-computed VGGT teacher latent cache
        apply_geometry_jepa_params(config, params)

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return RobotWinTaskDataset(**params)
    
    elif dataset_type == 'ac_one':
        from .ac_one.ac_one_dataset import ACOneDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return ACOneDataset(**params)

    elif dataset_type == 'latent_action':
        from .latent_action.latent_action_dataset import LatentActionDataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        if hasattr(config.dataset, 'dataset_dir'):
            dataset_dir = list(config.dataset.dataset_dir)
            params['dataset_dir'] = [str(p) for p in dataset_dir]
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Optional VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Optional additional params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return LatentActionDataset(**params)

    elif dataset_type == 'aloha_agilex_2':
        from .aloha_agilex_2.aloha_agilex2_dataset import AlohaAgilex2Dataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return AlohaAgilex2Dataset(**params)

    elif dataset_type == 'lerobot':
        from .lerobot.lerobot_dataset import LeRobotMotusDataset

        # Get all parameters from config
        params = {}

        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Geometry JEPA: attach the pre-computed VGGT teacher latent cache
        apply_geometry_jepa_params(config, params)

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        # Set validation flag
        params['val'] = val
        
        return LeRobotMotusDataset(**params)

    # Example: Add more dataset types here
    # elif dataset_type == 'bridge':
    #     from .bridge_dataset import BridgeDataset  
    #     return BridgeDataset(**params)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: robotwin, aloha_agilex_1, ac_one, aloha_agilex_2, table30")


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    
    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask
            
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    padded_embeddings = []
    
    for emb in language_embeddings:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    
    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Universal collate function for all datasets.
    
    Args:
        batch: List of sample dictionaries (may contain None)
        
    Returns:
        Batched dictionary or None if all samples are None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
    
    # Stack tensors（支持无 initial_state 的样本）
    first_frames = torch.stack([sample['first_frame'] for sample in batch])
    video_frames = torch.stack([sample['video_frames'] for sample in batch])
    action_sequences = torch.stack([sample['action_sequence'] for sample in batch])
    has_initial_state = all(('initial_state' in sample and sample['initial_state'] is not None) for sample in batch)
    initial_states = torch.stack([sample['initial_state'] for sample in batch]) if has_initial_state else None
    
    # Process VLM inputs with padding in collate_fn
    vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
    processed_vlm_inputs = None
    if vlm_inputs and all(vlm_input is not None for vlm_input in vlm_inputs):
        processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)
    
    # Process language embeddings with padding in collate_fn  
    language_embeddings = [sample.get('language_embedding') for sample in batch if 'language_embedding' in sample]
    processed_language_embeddings = None
    if language_embeddings and any(emb is not None for emb in language_embeddings):
        processed_language_embeddings = _process_language_embeddings_batch(language_embeddings)
    
    result = {
        'first_frame': first_frames,             # [B, C, H, W]
        'video_frames': video_frames,            # [B, F, C, H, W]
        'action_sequence': action_sequences,     # [B, F, D]
        'vlm_inputs': processed_vlm_inputs,
        'language_embedding': processed_language_embeddings,
    }

    if initial_states is not None:
        result['initial_state'] = initial_states

    # Geometry JEPA transition identity (present only for datasets that expose it)
    if all('episode_key' in sample for sample in batch):
        result['sample_key'] = [sample['sample_key'] for sample in batch]
        result['episode_key'] = [sample['episode_key'] for sample in batch]
        result['condition_frame_idx'] = torch.tensor(
            [sample['condition_frame_idx'] for sample in batch], dtype=torch.long
        )
        result['video_frame_indices'] = torch.stack(
            [sample['video_frame_indices'] for sample in batch]
        )

    if all('teacher_latents' in sample for sample in batch):
        # Legacy: dense 0..num_video_frames. Joint: static + configured horizons.
        result['teacher_latents'] = torch.stack(
            [sample['teacher_latents'] for sample in batch]
        )

    return result