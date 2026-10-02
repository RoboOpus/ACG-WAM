"""Contract tests that need the real 5B Motus model, GPU and checkpoints.

Covers the two checks from ``LDJEPA_transfer.md`` that cannot be verified on a
synthetic stand-in:

* section 15.2 -- the JEPA student tokens must not depend on future frames;
* section 12.3 -- deployment must be able to ignore the JEPA weights.

Run explicitly (needs ~40 GB of GPU memory and roughly two minutes to load)::

    pytest tests/test_real_model_contracts.py -q -s

Skipped automatically when CUDA or the pretrained checkpoints are unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
# train/train.py imports `sample` as a top-level module
sys.path.insert(1, str(ROOT / "train"))


def _load_train_module():
    """Import train/train.py directly; `train/` is a plain directory, not a package."""
    spec = importlib.util.spec_from_file_location(
        "motus_train_entrypoint", ROOT / "train" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

CONFIG_PATH = ROOT / "configs/robotwin.yaml"
_CONFIG = OmegaConf.load(CONFIG_PATH)
REQUIRED_PATHS = [
    Path(_CONFIG.model.wan.checkpoint_path),
    Path(_CONFIG.model.vlm.checkpoint_path),
]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not CONFIG_PATH.exists()
    or not all(p.exists() for p in REQUIRED_PATHS),
    reason="needs CUDA and the pretrained Motus backbones",
)


@pytest.fixture(scope="module")
def model_and_batch():
    from data.dataset import collate_fn, create_dataset

    create_model_and_optimizer = _load_train_module().create_model_and_optimizer

    config = OmegaConf.load(CONFIG_PATH)
    config.common.action_chunk_size = (
        config.common.num_video_frames * config.common.video_action_freq_ratio
    )
    config.dataset.task_mode = "single"
    config.dataset.task_name = "adjust_bottle"
    config.dataset.data_mode = "clean"
    config.dataset.max_episodes = 1
    config.training.batch_size = 1
    # Exercise the plain path: the leakage check does not need teacher targets.
    config.geometry_jepa.enable = False

    dataset = create_dataset(config)
    batch = collate_fn([dataset[0]])

    model, _optimizer, _scheduler = create_model_and_optimizer(config)
    model = model.cuda().eval()
    return model, batch


def _to_model(batch, model):
    dtype = model.dtype
    device = next(model.parameters()).device
    vlm_inputs = batch["vlm_inputs"]
    if vlm_inputs is not None:
        vlm_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in vlm_inputs.items()
        }
    return dict(
        first_frame=batch["first_frame"].to(device, dtype=dtype),
        video_frames=batch["video_frames"].to(device, dtype=dtype),
        state=batch["initial_state"].to(device, dtype=dtype),
        actions=batch["action_sequence"].to(device, dtype=dtype),
        language_embeddings=batch["language_embedding"].to(device, dtype=dtype),
        vlm_inputs=vlm_inputs,
    )


def test_student_tokens_do_not_see_the_future(model_and_batch, monkeypatch):
    """Section 15.2: fix the current frame, change the future, tokens must not move.

    The real `training_step` is driven end to end and the tokens are captured
    where the JEPA branch reads them, so this exercises the production path
    rather than a reimplementation of it.
    """
    model, batch = model_and_batch
    captured = []

    original = type(model.video_module).prepare_input

    def recording_prepare_input(self, noisy_video_latent):
        tokens = original(self, noisy_video_latent)
        captured.append(tokens.detach().float().cpu())
        return tokens

    monkeypatch.setattr(
        type(model.video_module), "prepare_input", recording_prepare_input
    )

    kwargs = _to_model(batch, model)
    future_a = kwargs["video_frames"]
    future_b = torch.rand_like(future_a)  # same condition frame, different future

    with torch.no_grad():
        model.training_step(**{**kwargs, "video_frames": future_a})
        model.training_step(**{**kwargs, "video_frames": future_b})

    assert len(captured) == 2
    tokens_a, tokens_b = captured
    per_frame = model.tokens_per_frame

    torch.testing.assert_close(
        tokens_a[:, :per_frame],
        tokens_b[:, :per_frame],
        rtol=0,
        atol=0,
        msg="student tokens changed when only the future frames changed",
    )
    assert not torch.allclose(tokens_a, tokens_b), (
        "the future tokens are identical too, so the test did not actually "
        "perturb the future"
    )


def test_deployment_can_ignore_the_jepa_weights(model_and_batch):
    """Section 12.3: a JEPA checkpoint must load into a JEPA-free model."""
    from models.geometry_jepa import GeometryJEPAConfig, GeometryJEPAModule

    model, _ = model_and_batch
    assert model.geometry_jepa is None, "fixture should build the plain model"

    jepa = GeometryJEPAModule(
        GeometryJEPAConfig(enable=True), student_num_tokens=model.tokens_per_frame
    )
    state_dict = dict(model.state_dict())
    jepa_keys = {f"geometry_jepa.{k}" for k in jepa.state_dict()}
    state_dict.update({k: torch.zeros(1) for k in jepa_keys})

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert set(unexpected) == jepa_keys, (
        "only the JEPA weights should be unexpected; "
        f"got {sorted(set(unexpected) - jepa_keys)[:5]}"
    )
    assert not missing, f"deployment model is missing weights: {missing[:5]}"
