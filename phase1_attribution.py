# -*- coding: utf-8 -*-
"""Phase 1: controlled attribution and masking diagnostics.

This script is intentionally confirmatory: it controls retained residual energy,
re-evaluates masked residuals after attacks, includes residual/speech-energy baselines,
and reports direct paired tests between Survival and decoder-derived maps.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import permutation_test, wilcoxon
from torch.utils.data import DataLoader
from tqdm import tqdm

from external_attacks import command_roundtrip_batch, ffmpeg_mp3_roundtrip_batch
from experiment_utils import (
    align_audio_tensors,
    compute_attribution_metrics,
    compute_logit_metrics,
    exact_topk_mask,
    gaussian_kernel2d,
    project_residual_l2,
    retained_energy_ratio,
    save_json,
    set_global_seed,
    stable_int_hash,
    overlapping_attack_families,
)
from survalign_p import (
    AlignMarkManager,
    DifferentiableDistortion,
    UnifiedSpeechDataset,
    bits_to_chunks,
    chunks_to_bits,
    compute_chunk_ce_loss,
    frame_energy_vad,
    get_survival_map,
    istft_audio,
    minmax_per_sample,
    stft_audio,
)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def compute_decoder_gradient_map(alignmark, wav_wm, target_msg):
    """Legacy clean-input decoder gradient-magnitude saliency map."""
    with torch.backends.cudnn.flags(enabled=False):
        wav_var = wav_wm.detach().clone().requires_grad_(True)
        _, chunk_logits = alignmark.decode_logits_with_grad(wav_var)
        loss = compute_chunk_ce_loss(chunk_logits, target_msg)
        gradient = torch.autograd.grad(loss, wav_var, retain_graph=False, create_graph=False)[0]
    gradient_spec = stft_audio(gradient.squeeze(1), n_fft=256, hop_length=64)
    return minmax_per_sample(torch.abs(gradient_spec))


def _apply_internal_attack(wav, attack_name, distorter, seed):
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
    if attack_name in {
        "clean", "identity", "noise", "noise10db", "lowpass", "bandpass", "resample",
        "reconstruct_nq6", "reconstruct_nq8", "strong_speechtokenizer", "spectral_proxy",
    }:
        return _apply_internal_attack(wav, attack_name, distorter, seed)
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
            raise ValueError("facodec attack requested without --facodec_command")
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


def compute_decoder_utility_map(
    alignmark,
    wav,
    residual,
    target_msg,
    distorter,
    attack_names: Sequence[str],
    base_seed: int = 0,
):
    """Signed residual-scale utility: -d E[L_dec after attack] / d alpha(f,t)."""
    wav_2d = wav.squeeze(1)
    residual_2d = residual.squeeze(1)
    residual_spec = stft_audio(residual_2d, n_fft=256, hop_length=64).detach()
    alpha = torch.ones_like(residual_spec.real, requires_grad=True)
    scaled = residual_spec * alpha
    scaled_residual = istft_audio(scaled, length=wav_2d.shape[-1], n_fft=256, hop_length=64)
    candidate = (wav_2d + scaled_residual).unsqueeze(1)
    loss = torch.zeros((), device=wav.device)
    with torch.backends.cudnn.flags(enabled=False):
        for attack_index, attack_name in enumerate(attack_names):
            attacked = _apply_internal_attack(candidate, attack_name, distorter, base_seed + attack_index)
            _, logits = alignmark.decode_logits_with_grad(attacked)
            loss = loss + compute_chunk_ce_loss(logits, target_msg)
        loss = loss / len(attack_names)
        gradient = torch.autograd.grad(loss, alpha, retain_graph=False, create_graph=False)[0]
    utility = -gradient
    # Preserve sign before z-normalization; positive values mean increasing the residual helps locally.
    mean = utility.reshape(utility.shape[0], -1).mean(dim=1).view(-1, 1, 1)
    std = utility.reshape(utility.shape[0], -1).std(dim=1).view(-1, 1, 1)
    return (utility - mean) / (std + 1e-8)


def smooth_mask(mask: torch.Tensor, mode: str, kernel_size: int, sigma: float) -> torch.Tensor:
    if mode == "none":
        return mask
    if mode == "average":
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
        kernel = kernel / kernel.numel()
    elif mode == "gaussian":
        kernel = gaussian_kernel2d(kernel_size, sigma, device=mask.device, dtype=mask.dtype)
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")
    return F.conv2d(mask.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)


def valid_correlations(first: np.ndarray, second: np.ndarray, eps=1e-12):
    from scipy.stats import pearsonr, spearmanr
    if np.std(first) < eps or np.std(second) < eps:
        return None
    pearson = pearsonr(first, second).statistic
    spearman = spearmanr(first, second).statistic
    if not (np.isfinite(pearson) and np.isfinite(spearman)):
        return None
    return float(pearson), float(spearman)


def paired_statistics(a: Sequence[float], b: Sequence[float], seed: int) -> Dict[str, float]:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.size < 2:
        return {"mean_difference": float("nan"), "wilcoxon_p": float("nan"), "permutation_p": float("nan")}
    differences = a_arr - b_arr
    try:
        wilcoxon_p = float(wilcoxon(differences).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(differences.mean()))
    signs = rng.choice([-1.0, 1.0], size=(10000, differences.size))
    null = np.abs((signs * differences[None, :]).mean(axis=1))
    permutation_p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return {
        "mean_difference": float(differences.mean()),
        "wilcoxon_p": wilcoxon_p,
        "permutation_p": permutation_p,
    }


def _target_norm_reference(raw_residuals: Dict[str, torch.Tensor], original: torch.Tensor, target_mode: str):
    if target_mode == "baseline":
        return original
    norms = []
    for name, residual in raw_residuals.items():
        if name != "Full":
            norms.append(torch.linalg.vector_norm(residual, dim=-1))
    min_norm = torch.stack(norms, dim=0).min(dim=0).values
    original_norm = torch.linalg.vector_norm(original, dim=-1).clamp_min(1e-8)
    scale = (min_norm / original_norm).unsqueeze(-1)
    return original * scale


def main():
    parser = argparse.ArgumentParser(description="Controlled Phase-1 attribution diagnostics")
    parser.add_argument("--dataset_type", default="librispeech", choices=["librispeech", "vctk", "ljspeech", "combined"])
    parser.add_argument("--dataset_name", default="dev-clean")
    parser.add_argument("--split", default="test", choices=["calib", "test"])
    parser.add_argument("--combined_protocol", default="speaker_disjoint", choices=["speaker_disjoint", "paper"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_ratio", type=float, default=0.2)
    parser.add_argument("--random_repeats", type=int, default=20)
    parser.add_argument("--mask_smoothing", default="none", choices=["none", "average", "gaussian"])
    parser.add_argument("--smooth_kernel", type=int, default=5)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--energy_modes", default="natural,equal")
    parser.add_argument("--equal_energy_target", default="minimum", choices=["minimum", "baseline"])
    parser.add_argument("--survival_attacks", default="noise,lowpass,resample,reconstruct_nq6,spectral_proxy")
    parser.add_argument("--utility_attacks", default="reconstruct_nq6,strong_speechtokenizer")
    parser.add_argument("--eval_attacks", default="clean,bandpass,strong_speechtokenizer")
    parser.add_argument("--clearervoice_command", default="")
    parser.add_argument("--facodec_command", default="")
    parser.add_argument("--encodec_command", default="")
    parser.add_argument("--dac_command", default="")
    parser.add_argument("--vocos_command", default="")
    parser.add_argument("--hifigan_command", default="")
    parser.add_argument("--clearervoice_snr", type=float, default=10.0)
    parser.add_argument("--mp3_bitrate", default="64k")
    parser.add_argument("--latent_mode", default="public_code", choices=["public_code", "unquantized"])
    parser.add_argument("--strict_heldout", action="store_true",
                        help="Fail if evaluation attacks overlap Survival/utility-map attacks.")
    parser.add_argument("--results_dir", default="results/phase1_confirmatory")
    args = parser.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alignmark = AlignMarkManager(device, latent_mode=args.latent_mode)
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    dataset = UnifiedSpeechDataset(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
        return_metadata=True,
        combined_protocol=args.combined_protocol,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    survival_attacks = parse_csv_list(args.survival_attacks)
    utility_attacks = parse_csv_list(args.utility_attacks)
    eval_attacks = parse_csv_list(args.eval_attacks)
    energy_modes = parse_csv_list(args.energy_modes)
    overlap_sets = {
        "survival_map_exact": sorted(set(eval_attacks) & set(survival_attacks)),
        "utility_map_exact": sorted(set(eval_attacks) & set(utility_attacks)),
        "survival_map_family": overlapping_attack_families(eval_attacks, survival_attacks),
        "utility_map_family": overlapping_attack_families(eval_attacks, utility_attacks),
    }
    overlap_messages = [f"{name}={values}" for name, values in overlap_sets.items() if values and values != ["clean"]]
    if overlap_messages:
        message = "Evaluation-attack leakage detected: " + "; ".join(overlap_messages)
        if args.strict_heldout:
            raise ValueError(message)
        print(f"[WARNING] {message}. Do not describe these results as held-out generalization.")

    correlation_rows = []
    sample_rows = []
    targets_all = []
    predictions = defaultdict(list)  # (energy_mode, condition, attack, repeat) -> tensors
    per_sample_accuracy = defaultdict(list)
    retained_ratios = defaultdict(list)
    plot_saved = False
    processed = 0

    for batch_index, batch in enumerate(tqdm(loader, desc="Phase1")):
        wav, msg, metadata = batch
        if args.max_samples >= 0:
            remaining = args.max_samples - processed
            if remaining <= 0:
                break
            wav, msg = wav[:remaining], msg[:remaining]
        wav = wav.to(device)
        msg = msg.to(device)
        with torch.no_grad():
            wav_wm, residual = alignmark.embed(wav, msg)
        wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)
        batch_size = wav.shape[0]
        batch_sample_ids = [str(value) for value in metadata["sample_id"][:batch_size]]
        targets_all.append(msg.detach().cpu())

        survival = get_survival_map(
            wav,
            wav_wm,
            distorter,
            attack_names=survival_attacks,
            base_seed=args.seed + batch_index * 100,
        )
        gradient_saliency = compute_decoder_gradient_map(alignmark, wav_wm, msg).detach()
        with torch.enable_grad():
            utility = compute_decoder_utility_map(
                alignmark,
                wav,
                residual,
                msg,
                distorter,
                utility_attacks,
                base_seed=args.seed + batch_index * 1000,
            ).detach()

        residual_spec = stft_audio(residual.squeeze(1), n_fft=256, hop_length=64)
        residual_magnitude = torch.abs(residual_spec)
        speech_magnitude = torch.abs(stft_audio(wav.squeeze(1), n_fft=256, hop_length=64))
        vad = frame_energy_vad(wav, n_fft=256, hop_length=64)
        valid_support = residual_magnitude > torch.quantile(
            residual_magnitude.reshape(batch_size, -1), 0.05, dim=1
        ).view(-1, 1, 1)
        correlation_top_masks = {
            "survival": exact_topk_mask(survival, args.top_ratio, largest=True).bool(),
            "gradient_saliency": exact_topk_mask(gradient_saliency, args.top_ratio, largest=True).bool(),
            "codec_utility": exact_topk_mask(utility, args.top_ratio, largest=True).bool(),
        }

        for item in range(batch_size):
            support = valid_support[item].detach().cpu().numpy().astype(bool).reshape(-1)
            maps = {
                "survival": survival[item].detach().cpu().numpy().reshape(-1),
                "gradient_saliency": gradient_saliency[item].detach().cpu().numpy().reshape(-1),
                "codec_utility": utility[item].detach().cpu().numpy().reshape(-1),
            }
            for other_name in ("gradient_saliency", "codec_utility"):
                result = valid_correlations(maps["survival"][support], maps[other_name][support])
                if result is not None:
                    survival_top = correlation_top_masks["survival"][item].cpu().numpy().reshape(-1)
                    other_top = correlation_top_masks[other_name][item].cpu().numpy().reshape(-1)
                    intersection = np.logical_and(survival_top, other_top).sum()
                    union = np.logical_or(survival_top, other_top).sum()
                    correlation_rows.append({
                        "sample_index": processed + item,
                        "map_pair": f"survival_vs_{other_name}",
                        "pearson": result[0],
                        "spearman": result[1],
                        "topk_iou": float(intersection / max(1, union)),
                    })

        masks = {
            "Full": torch.ones_like(survival),
            "High-Survival": exact_topk_mask(survival, args.top_ratio, largest=True),
            "Low-Survival": exact_topk_mask(survival, args.top_ratio, largest=False),
            "High-Gradient-Saliency": exact_topk_mask(gradient_saliency, args.top_ratio, largest=True),
            "Low-Gradient-Saliency": exact_topk_mask(gradient_saliency, args.top_ratio, largest=False),
            "High-Codec-Utility": exact_topk_mask(utility, args.top_ratio, largest=True),
            "Residual-Energy": exact_topk_mask(residual_magnitude, args.top_ratio, largest=True),
            "Speech-Energy": exact_topk_mask(speech_magnitude, args.top_ratio, largest=True),
            "VAD": exact_topk_mask(vad, args.top_ratio, largest=True),
        }
        masks = {
            name: smooth_mask(mask, args.mask_smoothing, args.smooth_kernel, args.smooth_sigma)
            for name, mask in masks.items()
        }

        raw_residuals = {}
        for condition, mask in masks.items():
            masked_spec = residual_spec * mask
            raw = istft_audio(masked_spec, length=wav.shape[-1], n_fft=256, hop_length=64)
            raw_residuals[condition] = raw
            retained_ratios[condition].extend(retained_energy_ratio(raw, residual.squeeze(1)).cpu().tolist())

        random_raw = []
        for repeat in range(args.random_repeats):
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + batch_index * 10000 + repeat)
            random_values = torch.rand(survival.shape, generator=generator, device=device)
            random_mask = exact_topk_mask(random_values, args.top_ratio, largest=True)
            random_mask = smooth_mask(random_mask, args.mask_smoothing, args.smooth_kernel, args.smooth_sigma)
            random_raw.append(
                istft_audio(residual_spec * random_mask, length=wav.shape[-1], n_fft=256, hop_length=64)
            )
        if random_raw:
            retained_ratios["Random"].extend(
                torch.stack([retained_energy_ratio(r, residual.squeeze(1)) for r in random_raw]).mean(0).cpu().tolist()
            )

        target_reference = _target_norm_reference(raw_residuals, residual.squeeze(1), args.equal_energy_target)

        def evaluate_one(condition, raw_residual, repeat_index=0):
            for energy_mode in energy_modes:
                if energy_mode == "natural":
                    adjusted = raw_residual
                elif energy_mode == "equal":
                    adjusted = project_residual_l2(raw_residual, target_reference, mode="equal")
                else:
                    raise ValueError(f"Unknown energy mode: {energy_mode}")
                candidate = (wav.squeeze(1) + adjusted).unsqueeze(1)
                for attack_index, attack_name in enumerate(eval_attacks):
                    if attack_name in {"noise", "noise10db", "spectral_proxy", "clearervoice"}:
                        attacked_parts = []
                        for local_index, sample_id in enumerate(batch_sample_ids):
                            item_seed = stable_int_hash(args.seed, sample_id, attack_name)
                            attacked_parts.append(
                                apply_eval_attack(
                                    candidate[local_index:local_index + 1],
                                    attack_name,
                                    distorter,
                                    item_seed,
                                    args,
                                )
                            )
                        attacked = torch.cat(attacked_parts, dim=0)
                    else:
                        attacked = apply_eval_attack(
                            candidate, attack_name, distorter, stable_int_hash(args.seed, attack_name), args
                        )
                    _, logits, _ = alignmark.decode(attacked)
                    pred = chunks_to_bits(logits.argmax(dim=-1), 4).cpu()
                    key = (energy_mode, condition, attack_name, repeat_index)
                    predictions[key].append(pred)
                    acc = (pred == msg.cpu()).float().mean(dim=1)
                    per_sample_accuracy[key].extend(acc.tolist())
                    target_chunks = torch.stack(bits_to_chunks(msg.long(), 4), dim=1)
                    logit_stats = compute_logit_metrics(logits.cpu(), target_chunks.cpu())
                    for local_index in range(batch_size):
                        sample_rows.append({
                            "sample_index": processed + local_index,
                            "energy_mode": energy_mode,
                            "condition": condition,
                            "attack": attack_name,
                            "repeat": repeat_index,
                            "bit_accuracy": float(acc[local_index].item()),
                            "exact": int(torch.all(pred[local_index] == msg[local_index].cpu()).item()),
                            "ce": float(logit_stats["ce"][local_index].item()),
                            "min_logit_margin": float(logit_stats["min_logit_margin"][local_index].item()),
                        })

        for condition, raw in raw_residuals.items():
            evaluate_one(condition, raw, 0)
        for repeat, raw in enumerate(random_raw):
            evaluate_one("Random", raw, repeat)

        if not plot_saved:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].imshow(survival[0].cpu(), aspect="auto", origin="lower")
            axes[0].set_title("Survival prior")
            axes[1].imshow(gradient_saliency[0].cpu(), aspect="auto", origin="lower")
            axes[1].set_title("Clean gradient magnitude")
            axes[2].imshow(utility[0].cpu(), aspect="auto", origin="lower")
            axes[2].set_title("Codec-aware signed utility")
            fig.tight_layout()
            fig.savefig(os.path.join(args.results_dir, "map_comparison.png"), dpi=160)
            plt.close(fig)
            plot_saved = True
        processed += batch_size

    targets = torch.cat(targets_all, dim=0)
    summary = {
        "config": vars(args),
        "n_samples": int(targets.shape[0]),
        "correlations": {},
        "retained_energy_ratio": {},
        "conditions": {},
        "paired_tests": {},
    }
    for pair in sorted({row["map_pair"] for row in correlation_rows}):
        rows = [row for row in correlation_rows if row["map_pair"] == pair]
        summary["correlations"][pair] = {
            "pearson_mean": float(np.mean([r["pearson"] for r in rows])),
            "pearson_std": float(np.std([r["pearson"] for r in rows])),
            "spearman_mean": float(np.mean([r["spearman"] for r in rows])),
            "spearman_std": float(np.std([r["spearman"] for r in rows])),
            "topk_iou_mean": float(np.mean([r["topk_iou"] for r in rows])),
            "topk_iou_std": float(np.std([r["topk_iou"] for r in rows])),
        }
    for condition, values in retained_ratios.items():
        summary["retained_energy_ratio"][condition] = {
            "mean": float(np.mean(values)), "std": float(np.std(values))
        }

    for energy_mode in energy_modes:
        for attack_name in eval_attacks:
            conditions = sorted({key[1] for key in predictions if key[0] == energy_mode and key[2] == attack_name})
            for condition in conditions:
                repeat_keys = sorted(
                    [key for key in predictions if key[0] == energy_mode and key[1] == condition and key[2] == attack_name],
                    key=lambda x: x[3],
                )
                repeat_metrics = []
                for key in repeat_keys:
                    pred = torch.cat(predictions[key], dim=0)
                    repeat_metrics.append(compute_attribution_metrics(pred, targets))
                metric_names = repeat_metrics[0].keys()
                averaged = {
                    metric: float(np.mean([m[metric] for m in repeat_metrics]))
                    if isinstance(repeat_metrics[0][metric], (int, float)) else repeat_metrics[0][metric]
                    for metric in metric_names
                }
                summary["conditions"][f"{energy_mode}/{attack_name}/{condition}"] = averaged

            comparisons = [
                ("High-Survival", "High-Gradient-Saliency"),
                ("High-Survival", "High-Codec-Utility"),
                ("High-Survival", "Residual-Energy"),
                ("High-Survival", "Random"),
                ("High-Survival", "Low-Survival"),
            ]
            for left, right in comparisons:
                left_key = (energy_mode, left, attack_name, 0)
                if left_key not in per_sample_accuracy:
                    continue
                left_values = per_sample_accuracy[left_key]
                if right == "Random":
                    random_arrays = [
                        np.asarray(values)
                        for key, values in per_sample_accuracy.items()
                        if key[0] == energy_mode and key[1] == "Random" and key[2] == attack_name
                    ]
                    right_values = np.stack(random_arrays).mean(axis=0).tolist()
                else:
                    right_key = (energy_mode, right, attack_name, 0)
                    if right_key not in per_sample_accuracy:
                        continue
                    right_values = per_sample_accuracy[right_key]
                summary["paired_tests"][f"{energy_mode}/{attack_name}/{left}_vs_{right}"] = paired_statistics(
                    left_values, right_values, args.seed
                )

    with open(os.path.join(args.results_dir, "phase1_sample_results.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)
    with open(os.path.join(args.results_dir, "phase1_correlations.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "map_pair", "pearson", "spearman", "topk_iou"])
        writer.writeheader()
        writer.writerows(correlation_rows)
    save_json(os.path.join(args.results_dir, "phase1_summary.json"), summary)

    print("\n[Phase 1 completed]")
    print(f"Samples: {summary['n_samples']}")
    for key, value in summary["conditions"].items():
        print(
            f"{key:<65} BitAcc={value['bit_accuracy']:.4f} "
            f"Exact={value['exact_message_accuracy']:.4f} FAR(strict)={value['far_strict']:.4f}"
        )
    print(f"Results: {args.results_dir}")


if __name__ == "__main__":
    main()
