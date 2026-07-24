#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cascade vs Single-Attack Survival Map Empirical Comparison.

This script computes survival maps under three regimes:
  (A) Current: 10 single attacks, quantile=0.25
  (B) Cascade-augmented: 10 single attacks + 6 cascade combos, quantile=0.25
  (C) Cascade-only: 6 cascade combos only, quantile=0.25

Then compares the three maps visually and numerically.
"""
from __future__ import annotations

import os
import argparse
import itertools
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from survalign_p import (
    AlignMarkManager, DifferentiableDistortion, UnifiedSpeechDataset,
    stft_audio, align_audio_tensors, _apply_survival_attack_pair,
    minmax_per_sample,
)
from experiment_utils import stable_int_hash


def compute_survival_scores(
    clean, watermarked, distorter, attack_names, base_seed, args,
    n_fft=256, hop_length=64, residual_floor_quantile=0.05,
):
    """Compute raw per-attack survival scores (before quantile aggregation)."""
    with torch.no_grad():
        spec_clean = stft_audio(clean, n_fft=n_fft, hop_length=hop_length)
        spec_wm = stft_audio(watermarked, n_fft=n_fft, hop_length=hop_length)
        residual_mag = torch.abs(spec_wm - spec_clean)
        floor = torch.quantile(
            residual_mag.reshape(residual_mag.shape[0], -1),
            q=float(residual_floor_quantile), dim=1,
        ).view(-1, 1, 1)
        valid_support = residual_mag > floor
        residual_mag_safe = residual_mag.clamp_min(1e-8)

        scores = []
        for idx, attack_name in enumerate(attack_names):
            attacked_clean, attacked_wm = _apply_survival_attack_pair(
                clean, watermarked, distorter, attack_name,
                seed=int(base_seed) + idx, args=args,
            )
            attacked_clean, attacked_wm = align_audio_tensors(attacked_clean, attacked_wm)
            ac_spec = stft_audio(attacked_clean, n_fft=n_fft, hop_length=hop_length)
            aw_spec = stft_audio(attacked_wm, n_fft=n_fft, hop_length=hop_length)
            retained = torch.abs(aw_spec - ac_spec)
            recon_diff = torch.abs(ac_spec - spec_clean)
            retention = torch.clamp(retained / residual_mag_safe, 0.0, 1.0)
            dominance = retained / (retained + recon_diff + 1e-8)
            score = retention * dominance
            scores.append(torch.where(valid_support, score, torch.zeros_like(score)))
        return scores


def apply_cascade_survival(
    clean, watermarked, distorter, cascade_stages_list, base_seed, args,
    n_fft=256, hop_length=64, residual_floor_quantile=0.05,
):
    """Apply cascaded attack sequences and compute survival scores."""
    with torch.no_grad():
        spec_clean = stft_audio(clean, n_fft=n_fft, hop_length=hop_length)
        spec_wm = stft_audio(watermarked, n_fft=n_fft, hop_length=hop_length)
        residual_mag = torch.abs(spec_wm - spec_clean)
        floor = torch.quantile(
            residual_mag.reshape(residual_mag.shape[0], -1),
            q=float(residual_floor_quantile), dim=1,
        ).view(-1, 1, 1)
        valid_support = residual_mag > floor
        residual_mag_safe = residual_mag.clamp_min(1e-8)

        scores = []
        for cascade_idx, stages in enumerate(cascade_stages_list):
            seed = int(base_seed) + 100 + cascade_idx
            # Apply stages sequentially to both clean and watermarked
            att_clean = clean
            att_wm = watermarked
            for stage_idx, attack_name in enumerate(stages):
                stage_seed = seed + stage_idx * 1000
                ac, aw = _apply_survival_attack_pair(
                    att_clean, att_wm, distorter, attack_name,
                    seed=stage_seed, args=args,
                )
                att_clean, att_wm = align_audio_tensors(ac, aw)

            ac_spec = stft_audio(att_clean, n_fft=n_fft, hop_length=hop_length)
            aw_spec = stft_audio(att_wm, n_fft=n_fft, hop_length=hop_length)
            retained = torch.abs(aw_spec - ac_spec)
            recon_diff = torch.abs(ac_spec - spec_clean)
            retention = torch.clamp(retained / residual_mag_safe, 0.0, 1.0)
            dominance = retained / (retained + recon_diff + 1e-8)
            score = retention * dominance
            scores.append(torch.where(valid_support, score, torch.zeros_like(score)))
        return scores


def aggregate_survival(scores, quantile=0.25, smooth_kernel=5):
    stacked = torch.stack(scores, dim=0)
    survival = torch.quantile(stacked, q=float(quantile), dim=0)
    if smooth_kernel and int(smooth_kernel) > 1:
        k = int(smooth_kernel)
        survival = F.avg_pool2d(survival.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2).squeeze(1)
    return minmax_per_sample(survival)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="librispeech")
    parser.add_argument("--dataset_name", default="dev-clean")
    parser.add_argument("--combined_protocol", default="speaker_disjoint")
    parser.add_argument("--latent_mode", default="public_code")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_index", type=int, default=30)
    parser.add_argument("--survival_quantile", type=float, default=0.25)
    parser.add_argument("--mp3_bitrate", default="64k")
    parser.add_argument("--encodec_command", default="")
    parser.add_argument("--vocos_command", default="")
    parser.add_argument("--facodec_command", default="")
    parser.add_argument("--clearervoice_command", default="")
    parser.add_argument("--clearervoice_snr", type=float, default=10.0)
    parser.add_argument("--dac_command", default="")
    parser.add_argument("--hifigan_command", default="")
    parser.add_argument("--output_dir", default="outputs/cascade_comparison")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")

    # Single attacks
    single_attacks = [
        "replacement", "masking", "frame_shuffle",
        "lowpass", "bandpass", "highpass",
        "ffmpeg_mp3", "ffmpeg_aac", "encodec", "vocos",
    ]

    # Cascade combos (2-stage and 3-stage)
    cascade_combos = [
        # 2-stage cascades
        ("lowpass", "masking"),
        ("bandpass", "replacement"),
        ("encodec", "masking"),
        ("ffmpeg_mp3", "lowpass"),
        ("vocos", "highpass"),
        ("masking", "bandpass"),
        # 3-stage cascades
        ("lowpass", "masking", "replacement"),
        ("encodec", "bandpass", "masking"),
        ("ffmpeg_mp3", "highpass", "frame_shuffle"),
        ("vocos", "lowpass", "masking"),
    ]

    # Models
    from inprocess_attacks import prewarm
    prewarm(device)

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

    wav, msg, metadata = dataset[args.sample_index]
    wav = wav.unsqueeze(0).to(device)
    msg = msg.unsqueeze(0).to(device)

    wav_wm, residual = alignmark.embed(wav, msg)
    wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)
    clean = wav.squeeze(1)
    watermarked = wav_wm.squeeze(1)

    context_seed = stable_int_hash(args.seed, "cascade_exp", metadata["sample_id"])

    print("Computing single-attack scores...")
    single_scores = compute_survival_scores(
        clean, watermarked, distorter, single_attacks, context_seed, args,
    )
    for name, s in zip(single_attacks, single_scores):
        mean_val = float(s[s > 0].mean()) if (s > 0).any() else 0.0
        print(f"  {name:20s}: mean_score={mean_val:.4f}")

    print("\nComputing cascade-attack scores...")
    cascade_scores = apply_cascade_survival(
        clean, watermarked, distorter, cascade_combos, context_seed, args,
    )
    for combo, s in zip(cascade_combos, cascade_scores):
        name = " -> ".join(combo)
        mean_val = float(s[s > 0].mean()) if (s > 0).any() else 0.0
        print(f"  {name:40s}: mean_score={mean_val:.4f}")

    # --- Aggregate under three regimes ---
    q = args.survival_quantile

    map_A = aggregate_survival(single_scores, quantile=q)
    map_B = aggregate_survival(single_scores + cascade_scores, quantile=q)
    map_C = aggregate_survival(cascade_scores, quantile=q)

    arr_A = map_A[0].cpu().numpy()
    arr_B = map_B[0].cpu().numpy()
    arr_C = map_C[0].cpu().numpy()

    # Stats
    stats = {}
    for label, arr in [("A: Single Only", arr_A), ("B: Single+Cascade", arr_B), ("C: Cascade Only", arr_C)]:
        stats[label] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "low_fraction": float((arr < 0.3).mean()),
            "high_fraction": float((arr > 0.7).mean()),
        }
        print(f"\n[{label}]")
        for k, v in stats[label].items():
            print(f"  {k:20s}: {v:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axes[0, 0].imshow(arr_A, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[0, 0].set_title(f"(A) Single-Attack Only (10 attacks)\nmean={stats['A: Single Only']['mean']:.3f}, low<0.3: {stats['A: Single Only']['low_fraction']:.1%}")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(arr_B, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title(f"(B) Single + Cascade (10+10 attacks)\nmean={stats['B: Single+Cascade']['mean']:.3f}, low<0.3: {stats['B: Single+Cascade']['low_fraction']:.1%}")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(arr_C, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title(f"(C) Cascade-Only (10 cascade combos)\nmean={stats['C: Cascade Only']['mean']:.3f}, low<0.3: {stats['C: Cascade Only']['low_fraction']:.1%}")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Difference maps
    diff_AB = arr_B - arr_A
    diff_AC = arr_C - arr_A

    im3 = axes[1, 0].imshow(diff_AB, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title(f"(B) - (A) Difference\nmean_diff={diff_AB.mean():.4f}")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(diff_AC, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title(f"(C) - (A) Difference\nmean_diff={diff_AC.mean():.4f}")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Histogram overlay
    axes[1, 2].hist(arr_A.ravel(), bins=50, alpha=0.5, label="(A) Single", color="tab:blue", density=True)
    axes[1, 2].hist(arr_B.ravel(), bins=50, alpha=0.5, label="(B) Single+Cascade", color="tab:orange", density=True)
    axes[1, 2].hist(arr_C.ravel(), bins=50, alpha=0.5, label="(C) Cascade Only", color="tab:green", density=True)
    axes[1, 2].set_xlabel("Survival Map value")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].set_title("Pixel value distributions")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(
        f"Cascade vs Single-Attack Survival Map Comparison\n"
        f"sample={metadata['sample_id']} speaker={metadata['speaker_id']} quantile={q}",
        fontsize=14,
    )
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, f"cascade_comparison_{args.sample_index}.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
