"""Offline diagnostics for the Geometry JEPA branch.

Answers the questions that a single scalar loss cannot (LDJEPA_transfer.md 16,
17.3), on a trained checkpoint:

1. Is the per-token target well-conditioned, or are we fitting noise directions?
2. Does the predictor actually use the horizon label, or only the action chunk?
3. Which loss term produces the gradient reaching the student queries, and do the
   terms agree or fight?

`--student-source` decides where `q_t` comes from:

* ``random`` (default) is a synthetic probe. It runs anywhere, but only supports
  *relative* statements about the predictor as a function -- absolute losses and
  `skill` are far from their training values.
* ``backbone`` reproduces the training path for real: VAE encode, flow-matching
  noise, teacher-forced first latent frame, patch embedding, first
  ``tokens_per_frame`` tokens, adapter. It loads the VAE and the patch embedding
  only, never the 5B MoT stack, so it fits on CPU. Needs ``--config``.

Usage::

    python tools/diagnose_geometry_jepa.py \\
        --checkpoint checkpoints/s7_jepa/s7_jepa/checkpoint_step_3000

    python tools/diagnose_geometry_jepa.py \\
        --checkpoint checkpoints/s7_jepa/s7_jepa/checkpoint_step_3000 \\
        --student-source backbone --config /mnt/workspace/Tao/cache/s7_jepa.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bak"))

from data.utils.teacher_store import TeacherStore, make_sample_key  # noqa: E402
from models.geometry_jepa import GeometryJEPAConfig, GeometryJEPAModule  # noqa: E402


def load_branch(checkpoint: Path) -> tuple[GeometryJEPAModule, GeometryJEPAConfig, dict]:
    """Rebuild the branch from the checkpoint's own config + weights."""
    saved = json.loads((checkpoint / "config.json").read_text())
    section = saved["geometry_jepa"]
    fields = set(GeometryJEPAConfig.__dataclass_fields__)
    config = GeometryJEPAConfig(**{k: v for k, v in section.items() if k in fields})
    config.validate()

    weights = torch.load(
        checkpoint / "pytorch_model_0.bin", map_location="cpu", mmap=True, weights_only=True
    )
    state = {
        k.split("geometry_jepa.", 1)[1]: v.clone()
        for k, v in weights.items()
        if "geometry_jepa." in k
    }
    if not state:
        raise SystemExit(f"{checkpoint} carries no geometry_jepa weights")

    pos = state.get("adapter.pos_embed")
    module = GeometryJEPAModule(config, student_num_tokens=None if pos is None else pos.shape[1])
    module.load_state_dict(state, strict=True)
    module.eval()
    return module, config, saved


def sample_windows(store: TeacherStore, num_windows: int, max_horizon: int, seed: int):
    """Real `(anchor, targets)` pairs straight from the cache."""
    rng = random.Random(seed)
    keys = [k for k, v in store.index.items() if v["num_rows"] > max_horizon + 1]
    rows, sample_keys = [], []
    for _ in range(num_windows):
        episode = rng.choice(keys)
        last_start = store.index[episode]["num_rows"] - max_horizon - 1
        start = rng.randint(0, last_start) * store.window_stride
        sample_keys.append(make_sample_key(episode, start))
        rows.append(store.get_horizons(sample_keys[-1], range(0, max_horizon + 1)))
    return torch.stack(rows), sample_keys  # [B, K+1, M, D]


def clustered_se(values: torch.Tensor, episodes: List[str]) -> float:
    """Standard error that treats one episode, not one window, as the unit.

    Windows from the same episode overlap heavily, so the naive
    `std / sqrt(n_windows)` understates the uncertainty.
    """
    groups: Dict[str, List[float]] = {}
    for value, episode in zip(values.tolist(), episodes):
        groups.setdefault(episode, []).append(value)
    means = np.array([np.mean(v) for v in groups.values()])
    if len(means) < 2:
        return float("nan")
    return float(means.std(ddof=1) / np.sqrt(len(means)))


def load_patch_embedding(checkpoint: Path, dtype: torch.dtype) -> torch.nn.Conv3d:
    """The only piece of the 5B backbone the student path needs."""
    weights = torch.load(
        checkpoint / "pytorch_model_0.bin", map_location="cpu", mmap=True, weights_only=True
    )
    prefix = "video_model.wan_model.patch_embedding."
    weight, bias = weights[prefix + "weight"], weights[prefix + "bias"]
    conv = torch.nn.Conv3d(
        weight.shape[1], weight.shape[0], kernel_size=weight.shape[2:], stride=weight.shape[2:]
    )
    conv.load_state_dict({"weight": weight.clone().float(), "bias": bias.clone().float()})
    return conv.to(dtype).eval()


def backbone_student_tokens(
    checkpoint: Path,
    config_path: Path,
    num_windows: int,
    seed: int,
    device: torch.device,
    val: bool = False,
    chunk_size: int = 8,
):
    """Reproduce `Motus.training_step`'s student slice on real samples.

    Same order as the training loop (models/motus.py): VAE encode the
    condition frame plus the target frames, mix in flow-matching noise at a
    sampled sigma, overwrite latent frame 0 with the clean condition latent,
    patch embed, keep the first `tokens_per_frame` tokens.

    Encoding runs in chunks and the tokens land on the CPU: a full sweep of a
    few hundred windows does not fit in VRAM at once.
    """
    from omegaconf import OmegaConf
    from wan.modules.vae2_2 import Wan2_2_VAE
    from wan.utils.fm import FlowMatchScheduler

    from data.dataset import create_dataset

    config = OmegaConf.load(config_path)
    dataset = create_dataset(config, val=val)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_windows].tolist()
    samples = [s for s in (dataset[i] for i in indices) if s is not None]

    actions = torch.stack([s["action_sequence"] for s in samples]).float()
    latents = torch.stack([s["teacher_latents"] for s in samples])
    keys = [s["sample_key"] for s in samples]

    vae = Wan2_2_VAE(vae_pth=config.model.wan.vae_path, device=str(device))
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    patch_embedding = load_patch_embedding(checkpoint, dtype).to(device)

    scheduler = FlowMatchScheduler(
        shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000
    )
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    tokens = []
    for begin in range(0, len(samples), chunk_size):
        block = samples[begin : begin + chunk_size]
        first = torch.stack([s["first_frame"] for s in block]).to(device)
        frames = torch.stack([s["video_frames"] for s in block]).to(device)

        condition = (first * 2.0 - 1.0).unsqueeze(2)
        video = (frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            clean = vae.encode(torch.cat([condition, video], dim=2).to(dtype))
            condition_latent = vae.encode(condition.to(dtype))

            step = torch.randint(
                0, scheduler.num_train_timesteps, (clean.shape[0],), generator=generator
            )
            sigma = scheduler.sigmas[step].to(dtype=dtype, device=device).view(-1, 1, 1, 1, 1)
            noise = torch.randn(clean.shape, generator=generator).to(clean)
            noisy = clean * (1 - sigma) + noise * sigma
            noisy[:, :, 0:1] = condition_latent  # teacher forcing, keeps the slice leakage-free

            # The VAE returns float32; training casts here too (models/motus.py).
            patched = patch_embedding(noisy.to(dtype))
            tokens_per_frame = patched.shape[-1] * patched.shape[-2]
            flat = patched.flatten(2).transpose(1, 2)
            tokens.append(flat[:, :tokens_per_frame].float().cpu())
        del first, frames, condition, video, clean, condition_latent, noisy, patched

    return torch.cat(tokens), actions, latents, keys


def residual_norm_report(latents: torch.Tensor, horizons: List[int], queries_per_camera: int):
    """Per-token relative residual norm: is the direction of the target defined?"""
    print("\n[1] per-token relative residual norm  |du_i| / |u_i|")
    print(f"{'h':>3} {'p1':>7} {'p5':>7} {'p50':>7} {'<1%':>7} {'<5%':>7}  per-camera median")
    anchor = latents[:, 0]
    for k in horizons:
        delta = latents[:, k] - anchor
        ratio = delta.norm(dim=-1) / anchor.norm(dim=-1).clamp_min(1e-8)  # [B, M]
        flat = ratio.reshape(-1).numpy()
        q = np.percentile(flat, [1, 5, 50])
        cams = [
            float(ratio[:, c * queries_per_camera : (c + 1) * queries_per_camera].median())
            for c in range(ratio.shape[1] // queries_per_camera)
        ]
        print(
            f"{k:>3} {q[0]:7.4f} {q[1]:7.4f} {q[2]:7.4f} "
            f"{100 * (flat < 0.01).mean():6.2f}% {100 * (flat < 0.05).mean():6.2f}%  "
            + "  ".join(f"{c:.3f}" for c in cams)
        )


def _targets(module: GeometryJEPAModule, latents: torch.Tensor, k: int) -> torch.Tensor:
    target = latents[:, k]
    if module.config.target_mode == "residual":
        target = target - latents[:, 0]
    return target


def _per_sample_loss(
    module: GeometryJEPAModule, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Same weighting as `LDJEPALoss`, kept per sample so pairs can be compared."""
    config = module.config
    pred, target = pred.float(), target.float()
    loss = torch.zeros(pred.shape[0])
    if config.use_pool_cosine:
        loss = loss + config.alpha_cos * (
            1.0 - F.cosine_similarity(pred.mean(1), target.mean(1), dim=-1)
        )
    if config.use_token_mse:
        loss = loss + config.beta_mse * (pred - target).pow(2).mean(dim=(1, 2))
    if config.gamma_token_cos:
        loss = loss + config.gamma_token_cos * (
            1.0 - F.cosine_similarity(pred, target, dim=-1)
        ).mean(dim=1)
    return loss


def copy_baseline(module: GeometryJEPAModule, latents: torch.Tensor, k: int) -> torch.Tensor:
    """Per-sample score of the trivial predictor, for *this* horizon.

    The training log averages `geo_copy_loss` over whatever horizon each
    micro-batch drew, so it must not be used as the denominator of a per-horizon
    skill: the baseline itself moves with `k`.
    """
    anchor = latents[:, 0]
    trivial = torch.zeros_like(anchor) if module.config.target_mode == "residual" else anchor
    return _per_sample_loss(module, trivial, _targets(module, latents, k))


def horizon_ablation(
    module: GeometryJEPAModule,
    q_t: torch.Tensor,
    actions: torch.Tensor,
    latents: torch.Tensor,
    horizons: List[int],
    freq_ratio: int,
    seed: int,
    episodes: Optional[List[str]] = None,
):
    """Full counterfactual matrix over the horizon label, `q_t` and actions fixed.

    Row = the true horizon (which target and action chunk are used), column = the
    label handed to the predictor. Conditioning works when each row is minimised
    on its diagonal; a single random wrong label cannot show that, because how
    bad a wrong label is depends on which one was drawn.

    `sensitivity` is the relative L2 between the diagonal prediction and the
    worst off-diagonal one: `out_norm` is a LayerNorm, so comparing output
    *norms* would show nothing by construction.
    """
    print("\n[2] horizon-label counterfactual: rows = true horizon, cols = label given")
    print("    conditioning works iff every row is minimised on its diagonal (marked *)")
    header = " ".join(f"{'L=' + str(h):>9}" for h in horizons)
    print(f"{'true':>5} {header} {'skill':>8} {'sens':>7} {'diag gap+-SE':>18}")

    for k in horizons:
        target = _targets(module, latents, k)
        chunk = actions[:, : k * freq_ratio]
        batch = q_t.shape[0]
        with torch.no_grad():
            preds = {
                label: module.predictor(q_t, chunk, torch.full((batch,), label))
                for label in horizons
            }
            losses = {
                label: _per_sample_loss(module, pred, target) for label, pred in preds.items()
            }
            skill = 1.0 - losses[k].mean() / copy_baseline(module, latents, k).mean()
            sensitivity = max(
                float((preds[label] - preds[k]).norm() / preds[k].norm())
                for label in horizons
                if label != k
            )
            # Gap against the best wrong label: the conservative comparison.
            best_wrong = min((l for l in horizons if l != k), key=lambda l: losses[l].mean())
            gap = losses[best_wrong] - losses[k]

        se = (
            clustered_se(gap, episodes)
            if episodes is not None
            else float(gap.std(unbiased=True) / np.sqrt(len(gap)))
        )
        cells = " ".join(
            f"{float(losses[label].mean()):9.5f}" + ("*" if label == k else " ")
            for label in horizons
        )
        print(
            f"{k:>5} {cells} {float(skill):+8.4f} {sensitivity:7.4f} "
            f"{float(gap.mean()):+9.5f}+-{se:<7.5f}"
        )
    if episodes is not None:
        print(f"    SE clustered over {len(set(episodes))} episodes / {len(episodes)} windows")


def gradient_decomposition(
    module: GeometryJEPAModule,
    q_t: torch.Tensor,
    actions: torch.Tensor,
    latents: torch.Tensor,
    horizons: List[int],
    freq_ratio: int,
):
    """Split the gradient arriving at `q_t` into its three loss terms.

    A term that barely moves its own loss can still be useful if its gradient
    agrees with the others; it is only dead weight when the norm is negligible,
    and harmful when the cosine against the others is persistently negative.
    """
    print("\n[3] gradient at q_t per loss term  (weighted by alpha/beta/gamma)")
    config = module.config
    terms = {
        "pooled_cos": config.alpha_cos,
        "token_mse": config.beta_mse,
        "token_cos": config.gamma_token_cos,
    }
    print(f"{'h':>3} " + " ".join(f"{n:>11}" for n in terms) + "   cos(pool,tok) cos(pool,mse)")
    for k in horizons:
        target = _targets(module, latents, k)
        chunk = actions[:, : k * freq_ratio]
        horizon = torch.full((q_t.shape[0],), k)

        grads = {}
        for name, weight in terms.items():
            source = q_t.detach().clone().requires_grad_(True)
            pred = module.predictor(source, chunk, horizon).float()
            gold = target.float()
            if name == "pooled_cos":
                loss = (1.0 - F.cosine_similarity(pred.mean(1), gold.mean(1), dim=-1)).mean()
            elif name == "token_mse":
                loss = F.mse_loss(pred, gold)
            else:
                loss = (1.0 - F.cosine_similarity(pred, gold, dim=-1)).mean()
            (weight * loss).backward()
            grads[name] = source.grad.detach().flatten()

        def cos(a: str, b: str) -> float:
            return float(F.cosine_similarity(grads[a], grads[b], dim=0))

        print(
            f"{k:>3} "
            + " ".join(f"{float(g.norm()):11.3e}" for g in grads.values())
            + f"   {cos('pooled_cos', 'token_cos'):+12.3f} {cos('pooled_cos', 'token_mse'):+13.3f}"
        )


def prediction_scale(
    module: GeometryJEPAModule,
    q_t: torch.Tensor,
    actions: torch.Tensor,
    latents: torch.Tensor,
    horizons: List[int],
    freq_ratio: int,
):
    """`out_norm` is a LayerNorm with one learnable affine shared by all horizons.

    The head can therefore learn *a* scale, but not one per horizon, while the
    residual target grows with `k`. `s*` is the least-squares rescaling of the
    prediction for this horizon; how much MSE it recovers says whether a
    per-horizon output scale is worth adding.
    """
    print("\n[4] head scale vs target scale (per-token RMS) and the best per-horizon rescale")
    print(f"{'h':>3} {'pred':>9} {'target':>9} {'ratio':>7} {'s*':>7} {'mse':>9} {'mse@s*':>9}")
    for k in horizons:
        target = _targets(module, latents, k).float()
        with torch.no_grad():
            pred = module.predictor(
                q_t, actions[:, : k * freq_ratio], torch.full((q_t.shape[0],), k)
            ).float()
        p, t = float(pred.pow(2).mean().sqrt()), float(target.pow(2).mean().sqrt())
        scale = float((pred * target).sum() / pred.pow(2).sum())
        mse = float((pred - target).pow(2).mean())
        mse_scaled = float((scale * pred - target).pow(2).mean())
        print(
            f"{k:>3} {p:9.4f} {t:9.4f} {p / max(t, 1e-8):7.2f} {scale:7.3f} "
            f"{mse:9.4f} {mse_scaled:9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="defaults to the checkpoint's")
    parser.add_argument("--student-source", choices=("random", "backbone"), default="random")
    parser.add_argument("--config", type=Path, default=None, help="required by --student-source backbone")
    parser.add_argument("--val", action="store_true", help="sample the validation split")
    parser.add_argument("--device", default="cpu", help="only used by --student-source backbone")
    parser.add_argument("--num-windows", type=int, default=256)
    parser.add_argument(
        "--chunk-size", type=int, default=8, help="VAE encode batch for --student-source backbone"
    )
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    module, config, saved = load_branch(args.checkpoint)
    common = saved["common"]
    freq_ratio = int(common["video_action_freq_ratio"])
    stride = freq_ratio * int(common["global_downsample_rate"])
    cache_dir = args.cache_dir or Path(saved["geometry_jepa"]["cache_dir"])
    horizons = list(config.horizon_choices)

    print(f"checkpoint : {args.checkpoint}")
    print(f"target_mode: {config.target_mode}   gamma_token_cos: {config.gamma_token_cos}")
    print(f"student    : {args.student_source}" + ("  (val split)" if args.val else ""))

    generator = torch.Generator().manual_seed(args.seed)
    if args.student_source == "backbone":
        if args.config is None:
            raise SystemExit("--student-source backbone requires --config")
        student, actions, latents, keys = backbone_student_tokens(
            args.checkpoint,
            args.config,
            args.num_windows,
            args.seed,
            torch.device(args.device),
            val=args.val,
            chunk_size=args.chunk_size,
        )
        with torch.no_grad():
            q_t = module.adapter(student)
    else:
        store = TeacherStore(
            cache_dir,
            num_queries=config.num_queries,
            hidden_dim=config.hidden_dim,
            window_stride=stride,
        )
        latents, keys = sample_windows(store, args.num_windows, max(horizons), args.seed)
        q_t = torch.randn(
            latents.shape[0], config.num_queries, config.hidden_dim, generator=generator
        )
        actions = torch.randn(
            latents.shape[0], max(horizons) * freq_ratio, args.action_dim, generator=generator
        )
        print("    NOTE: q_t and the actions are synthetic; only relative readings apply.")

    episodes = [k.rpartition("@")[0] for k in keys]
    print(f"windows    : {latents.shape[0]}   horizons: {horizons}")

    queries_per_camera = int(json.loads((cache_dir / "manifest.json").read_text())["queries_per_camera"])
    residual_norm_report(latents, horizons, queries_per_camera)
    horizon_ablation(module, q_t, actions, latents, horizons, freq_ratio, args.seed, episodes)
    gradient_decomposition(module, q_t, actions, latents, horizons, freq_ratio)
    prediction_scale(module, q_t, actions, latents, horizons, freq_ratio)


if __name__ == "__main__":
    main()
