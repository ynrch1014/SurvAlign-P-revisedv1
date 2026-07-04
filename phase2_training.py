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

from external_attacks import command_roundtrip_batch, ffmpeg_mp3_roundtrip_batch
from experiment_utils import (
    align_audio_tensors,
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
    normalize_per_sample,
    stft_audio,
)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class SimplifiedSurvivalGate(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=16, gate_range=0.2):
        super().__init__()
        if gate_range <= 0 or gate_range >= 1:
            raise ValueError("gate_range must be in (0,1)")
        self.gate_range = float(gate_range)
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

    def forward(self, feature_pack, residual_spec):
        logits = self.conv(feature_pack).squeeze(1)
        scale = 1.0 + self.gate_range * torch.tanh(logits)
        return residual_spec * scale, scale


def _internal_attack(wav, attack_name, distorter, seed):
    if attack_name in {"clean", "identity"}:
        return wav
    if attack_name == "noise":
        return distorter(wav, "noise", snr_db=20.0, seed=seed)
    if attack_name == "noise10db":
        return distorter(wav, "noise", snr_db=10.0, seed=seed)
    if attack_name == "lowpass":
        return distorter(wav, "lowpass", cutoff_hz=4000)
    if attack_name == "bandpass":
        return distorter(wav, "bandpass", low_hz=300, high_hz=3400)
    if attack_name == "resample":
        return distorter(wav, "resample", down_rate=2)
    if attack_name == "reconstruct_nq6":
        return distorter(wav, "reconstruct", n_q=6)
    if attack_name == "reconstruct_nq8":
        return distorter(wav, "reconstruct", n_q=8)
    if attack_name == "strong_speechtokenizer":
        return distorter(wav, "strong_speechtokenizer", n_q=2)
    if attack_name == "spectral_proxy":
        return distorter(wav, "spectral_proxy", cutoff_ratio=0.7, seed=seed)
    raise ValueError(f"Unknown internal attack: {attack_name}")


def apply_eval_attack(wav, attack_name, distorter, seed, args):
    internal = {
        "clean", "identity", "noise", "noise10db", "lowpass", "bandpass", "resample",
        "reconstruct_nq6", "reconstruct_nq8", "strong_speechtokenizer", "spectral_proxy",
    }
    if attack_name in internal:
        return _internal_attack(wav, attack_name, distorter, seed)
    if attack_name == "ffmpeg_mp3":
        return ffmpeg_mp3_roundtrip_batch(wav, sample_rate=16000, bitrate=args.mp3_bitrate)
    if attack_name in {"clearervoice", "clearervoice_only"}:
        if not args.clearervoice_command:
            raise ValueError(f"{attack_name} requested without --clearervoice_command")
        source = wav
        if attack_name == "clearervoice":
            source = distorter(wav, "noise", snr_db=args.clearervoice_snr, seed=seed)
        return command_roundtrip_batch(source, args.clearervoice_command, sample_rate=16000)
    if attack_name == "facodec":
        if not args.facodec_command:
            raise ValueError("facodec requested without --facodec_command")
        return command_roundtrip_batch(wav, args.facodec_command, sample_rate=16000)
    command_attr = {
        "encodec": "encodec_command",
        "dac": "dac_command",
        "vocos": "vocos_command",
        "hifigan": "hifigan_command",
    }.get(attack_name)
    if command_attr is not None:
        command = getattr(args, command_attr)
        if not command:
            raise ValueError(f"{attack_name} requested without --{command_attr}")
        return command_roundtrip_batch(wav, command, sample_rate=16000)
    raise ValueError(f"Unknown evaluation attack: {attack_name}")


def build_guide_map(args, alignmark, distorter, wav, wav_wm, residual, context_seed):
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
        guide = get_survival_map(
            wav,
            wav_wm,
            distorter,
            attack_names=args.survival_attack_names,
            base_seed=context_seed,
            quantile=args.survival_quantile,
        )
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
            guide = get_survival_map(
                wav,
                wav_wm,
                distorter,
                attack_names=args.survival_attack_names,
                base_seed=context_seed,
                quantile=args.survival_quantile,
            )
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
    loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
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


def train_gate(args, device, alignmark, distorter, dataset_train, dataset_val):
    if not args.train_attack_names:
        raise ValueError("At least one training attack is required")
    unsupported = set(args.train_attack_names) - {
        "noise", "noise10db", "lowpass", "bandpass", "resample", "reconstruct_nq6",
        "reconstruct_nq8", "strong_speechtokenizer", "spectral_proxy", "clean",
    }
    if unsupported:
        raise ValueError(f"Non-differentiable/unknown train attacks: {sorted(unsupported)}")
    gate = SimplifiedSurvivalGate(in_channels=4, gate_range=args.gate_range).to(device)
    optimizer = optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("Empty training loader")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    best = {"exact_message_accuracy": -1.0, "ce": float("inf")}

    for epoch in range(1, args.epochs + 1):
        gate.train()
        losses = []
        for step, (wav, msg) in enumerate(tqdm(loader, desc=f"Train {epoch}/{args.epochs}")):
            wav, msg = wav.to(device), msg.to(device)
            with torch.no_grad():
                wav_wm, residual = alignmark.embed(wav, msg)
                wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)
            args.current_msg = msg
            feature_pack, residual_spec, guide, masking_map = build_guide_map(
                args,
                alignmark,
                distorter,
                wav,
                wav_wm,
                residual,
                context_seed=args.seed + epoch * 100000 + step,
            )
            optimizer.zero_grad(set_to_none=True)
            gated_spec, scale = gate(feature_pack, residual_spec)
            gated_residual = istft_audio(gated_spec, length=wav.shape[-1], n_fft=256, hop_length=64)
            projected = project_residual_l2(
                gated_residual.unsqueeze(1), residual, mode=args.projection_mode
            ).squeeze(1)
            candidate = (wav.squeeze(1) + projected).unsqueeze(1)

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
            total.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=5.0)
            optimizer.step()
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
    loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
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
    parser.add_argument("--survival_attacks", default="reconstruct_nq6,reconstruct_nq8,spectral_proxy")
    parser.add_argument("--utility_attacks", default="reconstruct_nq6,strong_speechtokenizer")
    parser.add_argument("--train_attacks", default="noise,lowpass,resample,bandpass")
    parser.add_argument("--validation_attacks", default="bandpass,reconstruct_nq8")
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
                        help="Fail when test attacks overlap map/train/validation attacks.")
    parser.add_argument("--allow_checkpoint_config_mismatch", action="store_true")
    parser.add_argument("--load_weight", default="")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_name", default="")
    parser.add_argument("--val_max_batches", type=int, default=-1)
    parser.add_argument("--test_max_batches", type=int, default=-1)
    parser.add_argument("--far_candidate_sizes", default="100,300,600")
    parser.add_argument("--results_dir", default="results/phase2")
    parser.add_argument("--run_id", default="")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    train_dataset = None
    val_dataset = None
    if needs_training_data:
        train_dataset = UnifiedSpeechDataset(
            dataset_type=args.dataset_type,
            dataset_name=args.dataset_name,
            split="train",
            seed=args.seed,
            return_metadata=False,
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
            gate = SimplifiedSurvivalGate(in_channels=4, gate_range=args.gate_range).to(device)
            state = checkpoint.get("model_state_dict", checkpoint)
            gate.load_state_dict(state, strict=True)
        else:
            gate, checkpoint_path = train_gate(args, device, alignmark, distorter, train_dataset, val_dataset)
    evaluate(args, device, alignmark, distorter, test_dataset, gate, checkpoint_path)


if __name__ == "__main__":
    main()
