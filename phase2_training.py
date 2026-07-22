# -*- coding: utf-8 -*-
"""Canonical Phase-2 training and paired evaluation pipeline.

Key guarantees:
- deterministic/paired calibration and test examples;
- validation-based checkpoint selection;
- hard L2 projection (cap or equal-energy control);
- explicit separation of map/train/validation/test attack sets;
- actual attribution FAR, exact-message, recovery and regression metrics;
- optional real MP3/ClearerVoice/FACodec held-out adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from pesq import pesq
except ImportError:
    pesq = None
try:
    from pystoi import stoi as compute_stoi
except ImportError:
    compute_stoi = None
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment_utils import (
    align_audio_tensors,
    apply_cascade_attack,
    apply_eval_attack,
    apply_internal_attack as _internal_attack,
    compute_attribution_metrics,
    compute_attribution_per_sample,
    attribution_metrics_by_candidate_size,
    compute_logit_metrics,
    nan_summary,
    project_residual_l2,
    recovery_regression_metrics,
    save_json,
    set_global_seed,
    stable_int_hash,
    overlapping_attack_families,
    survival_heldout_leakage,
    HELDOUT_CODECS,
)
from phase1_attribution import compute_decoder_gradient_map, compute_decoder_utility_map
from survalign_p import (
    AlignMarkManager,
    DifferentiableDistortion,
    UnifiedSpeechDataset,
    bits_to_chunks,
    chunks_to_bits,
    compute_chunk_ce_loss,
    compute_si_sdr,
    compute_total_variation_loss,
    get_local_energy_masking_proxy,
    get_survival_map,
    istft_audio,
    minmax_per_sample,
    normalize_per_sample,
    stft_audio,
)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# Attacks train_gate() can run through with a differentiable forward pass (no argmax/
# discrete-codebook bottleneck blocking backprop). Kept as a module-level constant --
# not re-derived inline -- so it can't silently drift from --train_attacks' actual default
# again (it already has twice: masking/replacement/frame_shuffle were added to
# --train_attacks' default without updating this set, and highpass was added as a new
# differentiable attack without updating this set either).
TRAIN_GATE_SUPPORTED_ATTACKS = frozenset({
    "noise", "noise10db", "lowpass", "bandpass", "highpass", "resample",
    "speechtokenizer_nq6", "speechtokenizer_nq8", "strong_speechtokenizer",
    "spectral_proxy", "clean", "masking", "replacement", "frame_shuffle",
})


class SimplifiedSurvivalGate(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=16, gate_range=0.2, hard_mask=False):
        super().__init__()
        if gate_range <= 0 or gate_range >= 1:
            raise ValueError("gate_range must be in (0,1)")
        self.gate_range = float(gate_range)
        # multi-cascade-attack experiment Part D: when enabled, the guide channel
        # (normalized Survival Map, or a random map in the control-group variant) is
        # multiplied elementwise into the conv logits before the tanh/scale step, forcing
        # the guide to structurally shape the output scale map rather than merely being
        # available as one of several input channels the conv could learn to ignore.
        self.hard_mask = bool(hard_mask)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, feature_pack, residual_spec, guide_channel_index=2):
        logits = self.conv(feature_pack).squeeze(1)
        if self.hard_mask:
            # Per-sample min-max to [0,1] (not the z-normalized value already sitting in
            # feature_pack's guide channel), then rescaled to [0.5, 1.0] so a bin where the
            # guide is exactly 0 is *attenuated*, not completely zeroed -- a hard 0 would
            # kill gradient flow through that bin entirely and destabilize early training.
            guide = feature_pack[:, guide_channel_index]
            guide_norm = minmax_per_sample(guide)
            logits = logits * (0.5 + 0.5 * guide_norm)
        scale = 1.0 + self.gate_range * torch.tanh(logits)
        return residual_spec * scale, scale


def _resolve_survival_map(wav, wav_wm, distorter, args, context_seed, precomputed_survival):
    """Single call path for "reuse the cache if we have one, else compute it" -- both
    branches in build_guide_map() that need a survival map go through this, so a
    get_survival_map(..., args=args) call can no longer be updated in one branch and
    forgotten in the other (exactly what happened: the shuffled_survival branch had
    args=args, the proposed_gate/analytic_survival branch didn't, immediate
    AttributeError on the latter)."""
    if precomputed_survival is not None:
        return precomputed_survival
    return get_survival_map(
        wav,
        wav_wm,
        distorter,
        attack_names=args.survival_attack_names,
        base_seed=context_seed,
        quantile=args.survival_quantile,
        args=args,
    )


# Channel-ablation study (multi-cascade-attack experiment Part C): which of the Gate's 4
# input feature_pack channels (0=clean, 1=residual, 2=guide/Survival Map, 3=masking_map)
# are zeroed out before the Gate ever sees them. This only controls what information the
# Gate's CNN has access to -- SimplifiedSurvivalGate.forward() still multiplies its output
# scale against the real, unmasked `residual_spec` (see forward() above), so ablating a
# channel never changes what the gate's output is applied to, only what it was computed
# from.
CHANNEL_ABLATION_MASKS = {
    "full": frozenset(),                    # [1,2,3,4] whole thing: no masking (== proposed_gate as-is)
    "no_guide": frozenset({2}),              # [1,2,4] control: guide/Survival Map channel zeroed
    "no_residual": frozenset({1}),           # [1,3,4] alternative: residual channel zeroed
    "no_residual_no_guide": frozenset({1, 2}),  # [1,4] baseline: both residual and guide zeroed
}


def _apply_channel_ablation(feature_pack, channel_ablation):
    if channel_ablation not in CHANNEL_ABLATION_MASKS:
        raise ValueError(
            f"Unknown channel_ablation: {channel_ablation!r} "
            f"(expected one of {sorted(CHANNEL_ABLATION_MASKS)})"
        )
    masked_channels = CHANNEL_ABLATION_MASKS[channel_ablation]
    if not masked_channels:
        return feature_pack
    feature_pack = feature_pack.clone()
    for channel_index in masked_channels:
        feature_pack[:, channel_index] = 0.0
    return feature_pack


def build_guide_map(args, alignmark, distorter, wav, wav_wm, residual, context_seed, precomputed_survival=None):
    spec_clean = stft_audio(wav.squeeze(1), n_fft=256, hop_length=64)
    spec_wm = stft_audio(wav_wm.squeeze(1), n_fft=256, hop_length=64)
    residual_spec = spec_wm - spec_clean
    clean_feature = normalize_per_sample(torch.log1p(torch.abs(spec_clean)))
    residual_feature = normalize_per_sample(torch.log1p(torch.abs(residual_spec) + 1e-8))
    masking_map = get_local_energy_masking_proxy(wav)

    if args.mode == "random_gate":
        generator = torch.Generator(device=wav.device)
        generator.manual_seed(int(context_seed))
        guide = torch.rand(clean_feature.shape, generator=generator, device=wav.device, dtype=wav.dtype)
    elif args.mode == "energy_gate":
        guide = masking_map
    elif args.mode == "constant_gate":
        guide = torch.ones_like(clean_feature)
    elif args.mode == "shuffled_survival":
        guide = _resolve_survival_map(wav, wav_wm, distorter, args, context_seed, precomputed_survival)
        flat = guide.reshape(guide.shape[0], -1)
        shuffled = []
        for item in range(flat.shape[0]):
            generator = torch.Generator(device=wav.device)
            generator.manual_seed(stable_int_hash(context_seed, item))
            permutation = torch.randperm(flat.shape[1], generator=generator, device=wav.device)
            shuffled.append(flat[item, permutation])
        guide = torch.stack(shuffled, dim=0).reshape_as(guide)
    elif args.mode in {"proposed_gate", "analytic_survival"}:
        if args.map_type == "survival":
            guide = _resolve_survival_map(wav, wav_wm, distorter, args, context_seed, precomputed_survival)
        elif args.map_type == "gradient_saliency":
            with torch.enable_grad():
                guide = compute_decoder_gradient_map(alignmark, wav_wm, args.current_msg).detach()
        elif args.map_type == "codec_utility":
            with torch.enable_grad():
                guide = compute_decoder_utility_map(
                    alignmark,
                    wav,
                    residual,
                    args.current_msg,
                    distorter,
                    args.utility_attack_names,
                    base_seed=context_seed,
                ).detach()
        else:
            raise ValueError(f"Unknown map_type: {args.map_type}")
    else:
        guide = torch.zeros_like(clean_feature)

    guide_feature = normalize_per_sample(guide)
    feature_pack = torch.stack([clean_feature, residual_feature, guide_feature, masking_map], dim=1)
    feature_pack = _apply_channel_ablation(feature_pack, getattr(args, "channel_ablation", "full"))
    return feature_pack, residual_spec, guide, masking_map


def build_candidate(args, gate, alignmark, distorter, wav, msg, context_seed):
    wav_wm, residual = alignmark.embed(wav, msg)
    wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)
    baseline = wav_wm
    if args.mode == "baseline":
        return wav, baseline, baseline, residual, None
    if args.mode in {"uniform", "uniform_upper"}:
        # Deliberately violates the original energy budget; report only as an upper-bound reference.
        method = wav + residual * args.uniform_scale
        return wav, baseline, method, residual, None

    args.current_msg = msg
    feature_pack, residual_spec, guide, masking_map = build_guide_map(
        args, alignmark, distorter, wav, wav_wm, residual, context_seed
    )
    if args.mode == "analytic_survival":
        guide_01 = guide
        guide_01 = (guide_01 - guide_01.amin(dim=(1, 2), keepdim=True)) / (
            guide_01.amax(dim=(1, 2), keepdim=True) - guide_01.amin(dim=(1, 2), keepdim=True) + 1e-8
        )
        scale = 1.0 - args.gate_range + 2.0 * args.gate_range * guide_01
        gated_spec = residual_spec * scale
    else:
        if gate is None:
            raise ValueError(f"Mode {args.mode} requires a gate")
        gated_spec, scale = gate(feature_pack, residual_spec)
    gated_residual = istft_audio(gated_spec, length=wav.shape[-1], n_fft=256, hop_length=64)
    projected = project_residual_l2(
        gated_residual.unsqueeze(1), residual, mode=args.projection_mode
    ).squeeze(1)
    method = (wav.squeeze(1) + projected).unsqueeze(1)
    return wav, baseline, method, residual, scale


def validation_score(args, gate, alignmark, distorter, dataset_val, device):
    loader = _make_loader(dataset_val, args, shuffle=False)
    gate.eval()
    exact_values, ce_values, bit_values = [], [], []
    baseline_si_sdr, method_si_sdr, l2_ratios, clipping_ratios = [], [], [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.val_max_batches >= 0 and batch_index >= args.val_max_batches:
                break
            wav, msg, _ = batch
            wav, msg = wav.to(device), msg.to(device)
            aligned_wav, baseline, method, residual, _ = build_candidate(
                args, gate, alignmark, distorter, wav, msg, context_seed=args.seed + 500000 + batch_index
            )
            for item in range(aligned_wav.shape[0]):
                ref_np = aligned_wav[item].squeeze().cpu().numpy()
                baseline_si_sdr.append(compute_si_sdr(ref_np, baseline[item].squeeze().cpu().numpy()))
                method_si_sdr.append(compute_si_sdr(ref_np, method[item].squeeze().cpu().numpy()))
                base_energy = float(torch.sum(residual[item] ** 2).item()) + 1e-12
                l2_ratios.append(float(torch.sum((method[item] - aligned_wav[item]) ** 2).item()) / base_energy)
                clipping_ratios.append(float((method[item].abs() > 1.0).float().mean().item()))
            for attack_index, attack in enumerate(args.validation_attack_names):
                attacked = _internal_attack(method, attack, distorter, args.seed + 600000 + batch_index * 100 + attack_index)
                _, logits, _ = alignmark.decode(attacked)
                pred = chunks_to_bits(logits.argmax(dim=-1), 4)
                exact_values.extend(torch.all(pred == msg, dim=1).float().cpu().tolist())
                bit_values.extend((pred == msg).float().mean(dim=1).cpu().tolist())
                ce_values.append(float(compute_chunk_ce_loss(logits, msg).item()))
    if not exact_values:
        raise RuntimeError("Validation produced no examples")
    baseline_sdr_mean = float(np.mean(baseline_si_sdr))
    method_sdr_mean = float(np.mean(method_si_sdr))
    return {
        "exact_message_accuracy": float(np.mean(exact_values)),
        "bit_accuracy": float(np.mean(bit_values)),
        "ce": float(np.mean(ce_values)),
        "baseline_si_sdr": baseline_sdr_mean,
        "method_si_sdr": method_sdr_mean,
        "si_sdr_delta": method_sdr_mean - baseline_sdr_mean,
        "l2_ratio": float(np.mean(l2_ratios)),
        "clipping_ratio": float(np.mean(clipping_ratios)),
    }


def _make_loader(dataset, args, *, shuffle, generator=None):
    """DataLoader honoring the 3-B throughput knobs. persistent_workers needs workers>0."""
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
    if generator is not None:
        kwargs["generator"] = generator
    if shuffle:
        kwargs["drop_last"] = False
    return DataLoader(dataset, **kwargs)


def precompute_survival_cache(args, alignmark, distorter, dataset, device):
    """3-A: precompute the (gate-independent) Survival Map once per training sample.

    The Survival Map depends only on (clean, watermarked, attacks) — not on the Gate — so it is
    constant across epochs and safe to cache. We key by sample_id with an epoch-independent,
    sample-deterministic seed, which also removes the per-epoch seed drift the live path had.
    Returns {sample_id: cpu_tensor(F, T)}.
    """
    if args.map_type != "survival":
        raise ValueError("Survival-map caching only applies to --map_type survival.")
    loader = _make_loader(dataset, args, shuffle=False)
    cache: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Precompute Survival Map cache"):
            wav, msg, metadata = batch
            wav = wav.to(device)
            wav_wm, _ = alignmark.embed(wav, msg.to(device))
            wav_a, wav_wm_a, _ = align_audio_tensors(wav, wav_wm, wav_wm)
            sample_ids = [str(v) for v in metadata["sample_id"]]
            for item, sample_id in enumerate(sample_ids):
                # Sample-deterministic seed => identical map every epoch => cacheable.
                seed = stable_int_hash(args.seed, "survival_cache", sample_id)
                guide = get_survival_map(
                    wav_a[item:item + 1], wav_wm_a[item:item + 1], distorter,
                    attack_names=args.survival_attack_names, base_seed=seed,
                    quantile=args.survival_quantile, args=args,
                )
                cache[sample_id] = guide.squeeze(0).cpu()
    print(f"[cache] Survival Map cached for {len(cache)} samples.")
    return cache


def _gather_cached_survival(cache, sample_ids, device, dtype):
    """Stack per-sample cached maps into (B, F, T); None if any id is missing."""
    try:
        maps = [cache[str(sample_id)] for sample_id in sample_ids]
    except KeyError:
        return None
    return torch.stack(maps, dim=0).to(device=device, dtype=dtype)


def train_gate(args, device, alignmark, distorter, dataset_train, dataset_val, survival_cache=None):
    # --train_cascade replaces the per-attack independent-loop robust_loss below with a
    # single cascaded scenario (Time Jitter -> AWGN -> Bandpass applied in sequence to the
    # same candidate), so args.train_attack_names is unused in that mode -- skip its
    # validation accordingly.
    if not getattr(args, "train_cascade", False):
        if not args.train_attack_names:
            raise ValueError("At least one training attack is required")
        unsupported = set(args.train_attack_names) - TRAIN_GATE_SUPPORTED_ATTACKS
        if unsupported:
            raise ValueError(f"Non-differentiable/unknown train attacks: {sorted(unsupported)}")
    gate = SimplifiedSurvivalGate(
        in_channels=4, gate_range=args.gate_range, hard_mask=getattr(args, "hard_mask", False),
    ).to(device)
    optimizer = optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    use_cache = survival_cache is not None
    # Caching needs sample_ids to key on, so the training loader must return metadata.
    loader = _make_loader(dataset_train, args, shuffle=True, generator=generator)
    if len(loader) == 0:
        raise ValueError("Empty training loader")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    best = {"exact_message_accuracy": -1.0, "ce": float("inf")}

    # 3-C: AMP is opt-in and only active on CUDA. STFT/ISTFT run in fp32 regardless (complex ops
    # are not autocast-safe); autocast covers the conv/decoder matmuls where Tensor Cores help.
    amp_enabled = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for epoch in range(1, args.epochs + 1):
        gate.train()
        losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"Train {epoch}/{args.epochs}")):
            if len(batch) == 3:
                wav, msg, metadata = batch
                sample_ids = [str(v) for v in metadata["sample_id"]]
            else:
                wav, msg = batch
                sample_ids = None
            wav, msg = wav.to(device), msg.to(device)
            with torch.no_grad():
                wav_wm, residual = alignmark.embed(wav, msg)
                wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)
            args.current_msg = msg

            precomputed_survival = None
            if use_cache and sample_ids is not None:
                precomputed_survival = _gather_cached_survival(survival_cache, sample_ids, device, wav.dtype)

            optimizer.zero_grad(set_to_none=True)
            # alignmark.decode_logits_with_grad() backprops through wm_model.encoder, a
            # SEANetEncoder with an LSTM (SLSTM) submodule left in eval() mode (frozen
            # inference model). cudnn's RNN backward requires the module to have been in
            # train() mode at forward time; eval-mode + cudnn raises "cudnn RNN backward
            # can only be called in training mode". phase1_attribution.py's equivalent
            # gradient-map paths already work around this the same way (disable cudnn for
            # this block; SEANetEncoder has no BatchNorm/Dropout, so this has no other
            # numerical effect, just forces the non-cudnn RNN backward path).
            with torch.backends.cudnn.flags(enabled=False):
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    feature_pack, residual_spec, guide, masking_map = build_guide_map(
                        args,
                        alignmark,
                        distorter,
                        wav,
                        wav_wm,
                        residual,
                        context_seed=args.seed + epoch * 100000 + step,
                        precomputed_survival=precomputed_survival,
                    )
                    gated_spec, scale = gate(feature_pack, residual_spec)
                    gated_residual = istft_audio(gated_spec, length=wav.shape[-1], n_fft=256, hop_length=64)
                    projected = project_residual_l2(
                        gated_residual.unsqueeze(1), residual, mode=args.projection_mode
                    ).squeeze(1)
                    candidate = (wav.squeeze(1) + projected).unsqueeze(1)

                    if getattr(args, "train_cascade", False):
                        # Multi-cascaded-attack pilot: apply Time Jitter -> AWGN -> Bandpass
                        # in sequence to the *same* candidate (not independently averaged),
                        # matching a realistic multi-attack pipeline. AWGN SNR and jitter
                        # magnitude are randomized per batch (team request); the sampling
                        # generator is created with device=device and every random draw
                        # explicitly passes device=device too, to not repeat the GPU
                        # generator-device bug already fixed elsewhere (patch 3).
                        cascade_generator = torch.Generator(device=device)
                        cascade_generator.manual_seed(stable_int_hash(args.seed, "cascade", epoch, step))
                        snr_db = float(
                            8.0 + 4.0 * torch.rand(1, generator=cascade_generator, device=device).item()
                        )
                        jitter_ms = float(
                            0.5 + 1.0 * torch.rand(1, generator=cascade_generator, device=device).item()
                        )
                        stages = [
                            ("time_jitter", dict(max_shift_ms=jitter_ms)),
                            ("noise", dict(snr_db=snr_db)),
                            ("bandpass", dict(low_hz=200, high_hz=3500)),
                        ]
                        cascade_seed = args.seed + epoch * 1000000 + step * 100
                        attacked = apply_cascade_attack(candidate, stages, distorter, seed=cascade_seed)
                        _, logits = alignmark.decode_logits_with_grad(attacked)
                        robust_loss = compute_chunk_ce_loss(logits, msg)
                    else:
                        robust_loss = torch.zeros((), device=device)
                        for attack_index, attack in enumerate(args.train_attack_names):
                            attacked = _internal_attack(
                                candidate, attack, distorter, args.seed + epoch * 1000000 + step * 100 + attack_index
                            )
                            _, logits = alignmark.decode_logits_with_grad(attacked)
                            robust_loss = robust_loss + compute_chunk_ce_loss(logits, msg)
                        robust_loss = robust_loss / len(args.train_attack_names)

                    deviation_loss = torch.mean((scale - 1.0) ** 2)
                    # Penalize only positive amplification in perceptually exposed (low-energy) bins.
                    exposed_amplification = F.relu(scale - 1.0) * (1.0 - masking_map)
                    masking_loss = torch.mean(exposed_amplification**2)
                    tv_loss = compute_total_variation_loss(scale)
                    total = (
                        robust_loss
                        + args.lambda_dev * deviation_loss
                        + args.lambda_mask * masking_loss
                        + args.lambda_tv * tv_loss
                    )
                scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(total.item()))

        val = validation_score(args, gate, alignmark, distorter, dataset_val, device)
        print(
            f"[Epoch {epoch}] train_loss={np.mean(losses):.4f} "
            f"val_exact={val['exact_message_accuracy']:.4f} val_bit={val['bit_accuracy']:.4f} "
            f"val_ce={val['ce']:.4f} SI-SDR_delta={val['si_sdr_delta']:+.3f}dB "
            f"clip={val['clipping_ratio']:.6f}"
        )
        quality_ok = (
            val["si_sdr_delta"] >= args.min_validation_si_sdr_delta
            and val["clipping_ratio"] <= args.max_validation_clipping_ratio
        )
        improved = quality_ok and (
            val["exact_message_accuracy"] > best["exact_message_accuracy"]
            or (
                val["exact_message_accuracy"] == best["exact_message_accuracy"]
                and val["ce"] < best["ce"]
            )
        )
        if improved:
            best = val
            torch.save(
                {
                    "model_state_dict": gate.state_dict(),
                    "config": {k: v for k, v in vars(args).items() if k != "current_msg"},
                    "validation": val,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"[SAVE] {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            "No checkpoint satisfied the validation quality constraints. "
            "Relax --min_validation_si_sdr_delta/--max_validation_clipping_ratio or inspect training."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gate.load_state_dict(checkpoint["model_state_dict"])
    return gate, checkpoint_path


def _safe_pesq(reference, degraded):
    if pesq is None:
        return float("nan")
    try:
        return float(pesq(16000, reference, degraded, "wb"))
    except Exception:
        return float("nan")


def _safe_stoi(reference, degraded):
    if compute_stoi is None:
        return float("nan")
    try:
        return float(compute_stoi(reference, degraded, 16000, extended=False))
    except Exception:
        return float("nan")


def evaluate(args, device, alignmark, distorter, dataset_test, gate=None, checkpoint_path=""):
    """Evaluate baseline and method on exactly the same samples, messages and attacks."""
    loader = _make_loader(dataset_test, args, shuffle=False)
    if gate is not None:
        gate.eval()
    if not args.test_attack_names:
        raise ValueError("At least one test attack is required")

    targets = []
    sample_ids = []
    sample_metadata = []
    pred_store = {attack: {"baseline": [], "method": []} for attack in args.test_attack_names}
    logit_store = {
        attack: {
            "baseline": defaultdict(list),
            "method": defaultdict(list),
        }
        for attack in args.test_attack_names
    }
    fidelity = defaultdict(list)

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Evaluate")):
            if args.test_max_batches >= 0 and batch_index >= args.test_max_batches:
                break
            wav, msg, metadata = batch
            wav, msg = wav.to(device), msg.to(device)
            wav, baseline, method, residual, scale = build_candidate(
                args, gate, alignmark, distorter, wav, msg, context_seed=args.seed + 900000 + batch_index
            )
            targets.append(msg.cpu())
            batch_ids = [str(value) for value in metadata["sample_id"]]
            sample_ids.extend(batch_ids)
            for item in range(wav.shape[0]):
                sample_metadata.append({
                    "sample_id": batch_ids[item],
                    "file_path": str(metadata["file_path"][item]),
                    "speaker_id": str(metadata["speaker_id"][item]),
                    "crop_start": int(metadata["crop_start"][item]),
                    "valid_length": int(metadata["valid_length"][item]),
                })

                reference = wav[item].squeeze().cpu().numpy()
                baseline_np = baseline[item].squeeze().cpu().numpy()
                method_np = method[item].squeeze().cpu().numpy()
                for label, degraded in (("baseline", baseline_np), ("method", method_np)):
                    fidelity[f"{label}_pesq"].append(_safe_pesq(reference, degraded))
                    fidelity[f"{label}_stoi"].append(_safe_stoi(reference, degraded))
                    fidelity[f"{label}_si_sdr"].append(compute_si_sdr(reference, degraded))
                residual_energy = float(torch.sum(residual[item] ** 2).item()) + 1e-12
                fidelity["baseline_l2_ratio"].append(
                    float(torch.sum((baseline[item] - wav[item]) ** 2).item()) / residual_energy
                )
                fidelity["method_l2_ratio"].append(
                    float(torch.sum((method[item] - wav[item]) ** 2).item()) / residual_energy
                )
                fidelity["baseline_peak"].append(float(baseline[item].abs().max().item()))
                fidelity["method_peak"].append(float(method[item].abs().max().item()))
                fidelity["baseline_clipping_ratio"].append(
                    float((baseline[item].abs() > 1.0).float().mean().item())
                )
                fidelity["method_clipping_ratio"].append(
                    float((method[item].abs() > 1.0).float().mean().item())
                )

            target_chunks = torch.stack(bits_to_chunks(msg.long(), 4), dim=1).cpu()
            for attack_index, attack in enumerate(args.test_attack_names):
                # Stochastic attacks receive a file-specific seed, so results do not depend
                # on batch size or batch grouping. Baseline and method share that seed.
                if attack in {"noise", "noise10db", "spectral_proxy", "clearervoice"}:
                    attacked_baseline_parts = []
                    attacked_method_parts = []
                    for item, sample_id in enumerate(batch_ids):
                        item_seed = stable_int_hash(args.seed, sample_id, attack)
                        attacked_baseline_parts.append(
                            apply_eval_attack(baseline[item:item + 1], attack, distorter, item_seed, args)
                        )
                        attacked_method_parts.append(
                            apply_eval_attack(method[item:item + 1], attack, distorter, item_seed, args)
                        )
                    attacked_baseline = torch.cat(attacked_baseline_parts, dim=0)
                    attacked_method = torch.cat(attacked_method_parts, dim=0)
                else:
                    deterministic_seed = stable_int_hash(args.seed, attack)
                    attacked_baseline = apply_eval_attack(baseline, attack, distorter, deterministic_seed, args)
                    attacked_method = apply_eval_attack(method, attack, distorter, deterministic_seed, args)
                for label, attacked in (("baseline", attacked_baseline), ("method", attacked_method)):
                    _, logits, _ = alignmark.decode(attacked)
                    logits_cpu = logits.cpu()
                    pred = chunks_to_bits(logits_cpu.argmax(dim=-1), 4).cpu()
                    pred_store[attack][label].append(pred)
                    stats = compute_logit_metrics(logits_cpu, target_chunks)
                    for metric_name, values in stats.items():
                        logit_store[attack][label][metric_name].append(values.cpu())

    if not targets:
        raise RuntimeError("Test evaluation produced no samples")
    targets_tensor = torch.cat(targets, dim=0)
    if len(sample_ids) != targets_tensor.shape[0]:
        raise RuntimeError("Sample metadata and target counts do not match")

    summary = {
        "config": {k: v for k, v in vars(args).items() if k != "current_msg"},
        "checkpoint": checkpoint_path,
        "n_samples": int(targets_tensor.shape[0]),
        "fidelity": {key: nan_summary(values) for key, values in fidelity.items()},
        "attacks": {},
    }
    sample_rows = []

    def _bits_to_string(bits: torch.Tensor) -> str:
        return "".join(str(int(value)) for value in bits.tolist())

    for attack in args.test_attack_names:
        attack_payload = {}
        system_predictions = {}
        for system in ("baseline", "method"):
            predictions = torch.cat(pred_store[attack][system], dim=0)
            system_predictions[system] = predictions
            metrics = compute_attribution_metrics(predictions, targets_tensor)
            per_sample = compute_attribution_per_sample(predictions, targets_tensor)
            decoder_metrics = {
                metric_name: nan_summary(torch.cat(parts, dim=0).numpy().tolist())
                for metric_name, parts in logit_store[attack][system].items()
            }
            attack_payload[system] = metrics
            attack_payload[f"{system}_decoder"] = decoder_metrics
            attack_payload[f"{system}_by_candidate_size"] = attribution_metrics_by_candidate_size(
                predictions,
                targets_tensor,
                args.far_candidate_size_values,
                seed=stable_int_hash(args.seed, attack, system),
            )

            ce_values = torch.cat(logit_store[attack][system]["ce"], dim=0)
            min_margin_values = torch.cat(logit_store[attack][system]["min_logit_margin"], dim=0)
            mean_margin_values = torch.cat(logit_store[attack][system]["mean_logit_margin"], dim=0)
            entropy_values = torch.cat(logit_store[attack][system]["mean_entropy"], dim=0)
            for index in range(targets_tensor.shape[0]):
                target = targets_tensor[index]
                prediction = predictions[index]
                metadata = sample_metadata[index]
                bit_accuracy = float((prediction == target).float().mean().item())
                sample_rows.append({
                    **metadata,
                    "attack": attack,
                    "system": system,
                    "target_bits": _bits_to_string(target),
                    "predicted_bits": _bits_to_string(prediction),
                    "bit_accuracy": bit_accuracy,
                    "exact": int(torch.all(prediction == target).item()),
                    "true_hamming": int(per_sample["true_hamming"][index].item()),
                    "nearest_wrong_hamming": int(per_sample["nearest_wrong_hamming"][index].item()),
                    "attribution_margin": int(per_sample["attribution_margin"][index].item()),
                    "far_strict_failure": int(per_sample["strict_failure"][index].item()),
                    "far_lenient_failure": int(per_sample["lenient_failure"][index].item()),
                    "attribution_tie": int(per_sample["tie"][index].item()),
                    "ce": float(ce_values[index].item()),
                    "min_logit_margin": float(min_margin_values[index].item()),
                    "mean_logit_margin": float(mean_margin_values[index].item()),
                    "mean_entropy": float(entropy_values[index].item()),
                })

        comparison = recovery_regression_metrics(
            system_predictions["baseline"], system_predictions["method"], targets_tensor
        )
        attack_payload["paired_comparison"] = comparison
        summary["attacks"][attack] = attack_payload

    os.makedirs(args.results_dir, exist_ok=True)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    stem = f"{run_id}_{args.dataset_type}_{args.mode}_{args.map_type}"
    sample_path = os.path.join(args.results_dir, f"{stem}_samples.csv")
    with open(sample_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)
    summary_path = os.path.join(args.results_dir, f"{stem}_summary.json")
    save_json(summary_path, summary)

    long_path = os.path.join(args.results_dir, "phase2_results_long.csv")
    exists = os.path.exists(long_path)
    with open(long_path, "a", newline="", encoding="utf-8") as handle:
        fields = [
            "run_id", "dataset", "mode", "map_type", "seed", "attack", "system",
            "bit_accuracy", "exact_message_accuracy", "far_strict", "far_lenient",
            "tie_rate", "mean_attribution_margin", "min_codebook_hamming",
            "mean_nearest_codebook_hamming", "decoder_ce", "minimum_logit_margin",
            "mean_logit_margin", "mean_entropy", "recovery_rate", "regression_rate",
            "checkpoint", "train_attacks", "validation_attacks", "survival_attacks", "test_attacks",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for attack, payload in summary["attacks"].items():
            for system in ("baseline", "method"):
                metrics = payload[system]
                decoder = payload[f"{system}_decoder"]
                comparison = payload["paired_comparison"]
                writer.writerow({
                    "run_id": run_id,
                    "dataset": args.dataset_type,
                    "mode": args.mode,
                    "map_type": args.map_type,
                    "seed": args.seed,
                    "attack": attack,
                    "system": system,
                    "bit_accuracy": metrics["bit_accuracy"],
                    "exact_message_accuracy": metrics["exact_message_accuracy"],
                    "far_strict": metrics["far_strict"],
                    "far_lenient": metrics["far_lenient"],
                    "tie_rate": metrics["tie_rate"],
                    "mean_attribution_margin": metrics["mean_attribution_margin"],
                    "min_codebook_hamming": metrics["min_codebook_hamming"],
                    "mean_nearest_codebook_hamming": metrics["mean_nearest_codebook_hamming"],
                    "decoder_ce": decoder["ce"]["mean"],
                    "minimum_logit_margin": decoder["min_logit_margin"]["mean"],
                    "mean_logit_margin": decoder["mean_logit_margin"]["mean"],
                    "mean_entropy": decoder["mean_entropy"]["mean"],
                    "recovery_rate": comparison["recovery_rate"] if system == "method" else "",
                    "regression_rate": comparison["regression_rate"] if system == "method" else "",
                    "checkpoint": checkpoint_path,
                    "train_attacks": ",".join(args.train_attack_names),
                    "validation_attacks": ",".join(args.validation_attack_names),
                    "survival_attacks": ",".join(args.survival_attack_names),
                    "test_attacks": ",".join(args.test_attack_names),
                })

    print("\n[Evaluation summary]")
    for attack, payload in summary["attacks"].items():
        baseline_metrics = payload["baseline"]
        method_metrics = payload["method"]
        comparison = payload["paired_comparison"]
        print(
            f"{attack:<24} baseline Bit/Exact/FAR="
            f"{baseline_metrics['bit_accuracy']:.4f}/{baseline_metrics['exact_message_accuracy']:.4f}/"
            f"{baseline_metrics['far_strict']:.4f} | method="
            f"{method_metrics['bit_accuracy']:.4f}/{method_metrics['exact_message_accuracy']:.4f}/"
            f"{method_metrics['far_strict']:.4f} | recovery={comparison['recovery_rate']:.4f}, "
            f"regression={comparison['regression_rate']:.4f}"
        )
    print(f"Summary: {summary_path}")
    print(f"Samples: {sample_path}")


def validate_checkpoint_config(args, checkpoint_config):
    """Prevent silent evaluation with a Gate under incompatible feature semantics."""
    if not checkpoint_config:
        print("[WARNING] Checkpoint has no saved config; compatibility cannot be verified.")
        return
    critical_fields = ["mode", "map_type", "gate_range", "projection_mode", "latent_mode"]
    if args.map_type == "survival":
        critical_fields.append("survival_attack_names")
    if args.map_type == "codec_utility":
        critical_fields.append("utility_attack_names")
    mismatches = []
    current = vars(args)
    for field in critical_fields:
        if field not in checkpoint_config or field not in current:
            continue
        saved_value = checkpoint_config[field]
        current_value = current[field]
        if isinstance(saved_value, (list, tuple)) or isinstance(current_value, (list, tuple)):
            saved_value = list(saved_value)
            current_value = list(current_value)
        if saved_value != current_value:
            mismatches.append((field, saved_value, current_value))
    if mismatches:
        message = "\n".join(
            f" - {field}: checkpoint={saved!r}, current={current_value!r}"
            for field, saved, current_value in mismatches
        )
        if args.allow_checkpoint_config_mismatch:
            print("[WARNING] Incompatible checkpoint configuration was explicitly allowed:\n" + message)
        else:
            raise ValueError(
                "Checkpoint configuration does not match the current Gate semantics:\n"
                + message
                + "\nUse matching options or pass --allow_checkpoint_config_mismatch only for a deliberate ablation."
            )


def main():
    parser = argparse.ArgumentParser(description="SurvAlign-P canonical Phase-2 pipeline")
    parser.add_argument("--mode", required=True, choices=[
        "baseline", "uniform", "uniform_upper", "analytic_survival", "random_gate",
        "energy_gate", "constant_gate", "shuffled_survival", "proposed_gate",
    ])
    parser.add_argument("--map_type", default="survival", choices=["survival", "gradient_saliency", "codec_utility"])
    parser.add_argument("--dataset_type", default="librispeech", choices=["librispeech", "vctk", "ljspeech", "combined"])
    parser.add_argument("--dataset_name", default="train-clean-100")
    parser.add_argument("--combined_protocol", default="speaker_disjoint", choices=["speaker_disjoint", "paper"])
    parser.add_argument("--latent_mode", default="public_code", choices=["public_code", "unquantized"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate_range", type=float, default=0.2)
    parser.add_argument("--projection_mode", default="cap", choices=["cap", "equal"])
    parser.add_argument("--uniform_scale", type=float, default=1.1)
    parser.add_argument("--lambda_dev", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--lambda_tv", type=float, default=0.1)
    parser.add_argument("--min_validation_si_sdr_delta", type=float, default=-1e9)
    parser.add_argument("--max_validation_clipping_ratio", type=float, default=1.0)
    parser.add_argument("--survival_quantile", type=float, default=0.25)
    parser.add_argument("--survival_attacks", default="speechtokenizer_nq6,speechtokenizer_nq8,spectral_proxy")
    parser.add_argument("--utility_attacks", default="speechtokenizer_nq6,strong_speechtokenizer")
    parser.add_argument("--train_attacks", default="noise,lowpass,resample,speechtokenizer_nq6,spectral_proxy,masking,replacement,frame_shuffle")
    parser.add_argument("--train_cascade", action="store_true",
                        help="Replace the per-attack independent-loop robust_loss with a single "
                             "cascaded scenario (Time Jitter -> AWGN -> Bandpass applied in "
                             "sequence to the same candidate). --train_attacks is ignored when set.")
    parser.add_argument("--channel_ablation", default="full",
                        choices=sorted(CHANNEL_ABLATION_MASKS),
                        help="Zero out one or more of the Gate's 4 input feature_pack channels "
                             "(clean/residual/guide/masking_map) before the Gate sees them, to "
                             "isolate the Survival Map (guide) channel's contribution. Only "
                             "affects the Gate's input; its output scale is still applied to "
                             "the real, unmasked residual.")
    parser.add_argument("--hard_mask", action="store_true",
                        help="Structurally multiply the Gate's conv logits by the guide "
                             "channel (normalized to [0.5, 1.0], never fully zeroed) before "
                             "the tanh/scale step, instead of leaving it as just one of "
                             "several input channels the conv could learn to ignore. Combine "
                             "with --mode random_gate (instead of proposed_gate) to get the "
                             "random-map hard-masking control group, which isolates the "
                             "effect of the *mechanism* from the value of the Survival Map's "
                             "*information*.")
    parser.add_argument("--validation_attacks", default="bandpass,speechtokenizer_nq8")
    parser.add_argument("--test_attacks", default="clean,strong_speechtokenizer")
    parser.add_argument("--clearervoice_command", default="")
    parser.add_argument("--facodec_command", default="")
    parser.add_argument("--encodec_command", default="")
    parser.add_argument("--dac_command", default="")
    parser.add_argument("--vocos_command", default="")
    parser.add_argument("--hifigan_command", default="")
    parser.add_argument("--clearervoice_snr", type=float, default=10.0)
    parser.add_argument("--mp3_bitrate", default="64k")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--strict_heldout", action="store_true",
                        help="Fail when test attacks overlap map/train/validation attacks, or when "
                             "the Survival Map leaks a reserved cross-codec held-out codec.")
    parser.add_argument("--heldout_codecs", default=",".join(HELDOUT_CODECS),
                        help="Comma-separated codecs reserved as cross-codec held-out generalization "
                             "evidence; the Survival Map must never be built from these (incl. proxies).")
    parser.add_argument("--allow_checkpoint_config_mismatch", action="store_true")
    parser.add_argument("--load_weight", default="")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_name", default="")
    parser.add_argument("--val_max_batches", type=int, default=-1)
    parser.add_argument("--test_max_batches", type=int, default=-1)
    parser.add_argument("--far_candidate_sizes", default="100,300,600")
    parser.add_argument("--results_dir", default="results/phase2")
    parser.add_argument("--run_id", default="")
    # --- 3-B/3-C/3-A speed knobs (all default to the previous behavior; measure on GPU) ---
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (3-B).")
    parser.add_argument("--pin_memory", action="store_true", help="Pin host memory for faster H2D copies (3-B).")
    parser.add_argument("--persistent_workers", action="store_true",
                        help="Keep DataLoader workers alive across epochs; requires --num_workers > 0 (3-B).")
    parser.add_argument("--amp", action="store_true",
                        help="Enable CUDA mixed precision for the Gate/decoder path (3-C). No-op on CPU.")
    parser.add_argument("--cache_survival_map", action="store_true",
                        help="Precompute the Survival Map once per sample and reuse across epochs (3-A). "
                             "Only valid for --map_type survival.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    if args.mode == "uniform":
        args.mode = "uniform_upper"
    args.survival_attack_names = parse_csv_list(args.survival_attacks)
    args.utility_attack_names = parse_csv_list(args.utility_attacks)
    args.train_attack_names = parse_csv_list(args.train_attacks)
    args.validation_attack_names = parse_csv_list(args.validation_attacks)
    args.test_attack_names = parse_csv_list(args.test_attacks)
    args.far_candidate_size_values = [int(v) for v in parse_csv_list(args.far_candidate_sizes)]
    if not args.checkpoint_name:
        args.checkpoint_name = f"best_{args.dataset_type}_{args.mode}_{args.map_type}_seed{args.seed}.pth"

    # Held-out claims require no overlap with map, train, or validation attacks.
    overlap_sets = {
        "train_exact": sorted(set(args.test_attack_names) & set(args.train_attack_names)),
        "survival_map_exact": sorted(set(args.test_attack_names) & set(args.survival_attack_names)),
        "validation_exact": sorted(set(args.test_attack_names) & set(args.validation_attack_names)),
        "train_family": overlapping_attack_families(args.test_attack_names, args.train_attack_names),
        "survival_map_family": overlapping_attack_families(args.test_attack_names, args.survival_attack_names),
        "validation_family": overlapping_attack_families(args.test_attack_names, args.validation_attack_names),
    }
    overlap_messages = [f"{name}={values}" for name, values in overlap_sets.items() if values and values != ["clean"]]
    if overlap_messages:
        message = "Test-attack leakage detected: " + "; ".join(overlap_messages)
        if args.strict_heldout:
            raise ValueError(message)
        print(f"[WARNING] {message}. These results must not be described as held-out generalization.")

    # 1-A: the Survival Map feeds the Gate's input features, so it must never be built from a
    # codec that will later be presented as cross-codec held-out evidence (FACodec/ClearerVoice/
    # DAC/Vocos) — regardless of whether that codec appears in the *current* test set.
    heldout_codec_list = parse_csv_list(args.heldout_codecs)
    survival_leaks = survival_heldout_leakage(args.survival_attack_names, heldout_codec_list)
    if survival_leaks:
        leak_message = (
            "Survival-map leakage into reserved held-out codecs: "
            + "; ".join(f"{codec}<-{attacks}" for codec, attacks in sorted(survival_leaks.items()))
        )
        if args.strict_heldout:
            raise ValueError(leak_message)
        print(f"[WARNING] {leak_message}. The cross-codec generalization claim (C10) is invalid for this run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pay the one-time Encodec/Vocos/FACodec model-load cost up front, outside the timed
    # attack loop, whenever the in-process path (no explicit --encodec_command/
    # --vocos_command/--facodec_command override) will actually be used anywhere in this
    # run's attack sets.
    all_attack_names = {
        *args.survival_attack_names, *args.utility_attack_names, *args.train_attack_names,
        *args.validation_attack_names, *args.test_attack_names,
    }
    if ("encodec" in all_attack_names and not args.encodec_command) or (
        "vocos" in all_attack_names and not args.vocos_command
    ) or (
        "facodec" in all_attack_names and not args.facodec_command
    ):
        from inprocess_attacks import prewarm

        prewarm(device)
    alignmark = AlignMarkManager(device, latent_mode=args.latent_mode)
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    test_dataset = UnifiedSpeechDataset(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        split="test",
        seed=args.seed,
        return_metadata=True,
        combined_protocol=args.combined_protocol,
    )

    trainable_modes = {"random_gate", "energy_gate", "constant_gate", "shuffled_survival", "proposed_gate"}
    needs_training_data = args.mode in trainable_modes and not (args.test_only or args.load_weight)
    mode_uses_survival = args.mode == "shuffled_survival" or (
        args.mode == "proposed_gate" and args.map_type == "survival"
    )
    if args.cache_survival_map and not mode_uses_survival:
        raise ValueError("--cache_survival_map requires a survival-map mode (proposed_gate/shuffled_survival).")
    train_dataset = None
    val_dataset = None
    if needs_training_data:
        train_dataset = UnifiedSpeechDataset(
            dataset_type=args.dataset_type,
            dataset_name=args.dataset_name,
            split="train",
            seed=args.seed,
            # Caching keys on sample_id, so the training set must expose metadata.
            return_metadata=bool(args.cache_survival_map),
            combined_protocol=args.combined_protocol,
        )
        val_dataset = UnifiedSpeechDataset(
            dataset_type=args.dataset_type,
            dataset_name=args.dataset_name,
            split="calib",
            seed=args.seed,
            return_metadata=True,
            combined_protocol=args.combined_protocol,
        )

    gate = None
    checkpoint_path = ""
    if args.mode in trainable_modes:
        if args.test_only or args.load_weight:
            checkpoint_path = args.load_weight or os.path.join(args.checkpoint_dir, args.checkpoint_name)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Gate checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            validate_checkpoint_config(args, checkpoint.get("config", {}))
            gate = SimplifiedSurvivalGate(
                in_channels=4, gate_range=args.gate_range, hard_mask=getattr(args, "hard_mask", False),
            ).to(device)
            state = checkpoint.get("model_state_dict", checkpoint)
            gate.load_state_dict(state, strict=True)
        else:
            survival_cache = None
            if args.cache_survival_map:
                survival_cache = precompute_survival_cache(args, alignmark, distorter, train_dataset, device)
            gate, checkpoint_path = train_gate(
                args, device, alignmark, distorter, train_dataset, val_dataset, survival_cache=survival_cache
            )
    evaluate(args, device, alignmark, distorter, test_dataset, gate, checkpoint_path)


if __name__ == "__main__":
    main()
