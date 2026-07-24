#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Survival Map + Audio Sample Export + Attack Comparison Analysis.

Generates:
  1. Survival Map (single-attack based, quantile=0.25) — the ③ strategy
  2. Audio WAV files: clean, watermarked, and each attacked version
  3. Waveform + spectrogram comparison plots
  4. Per-attack survival score bar chart
  5. Summary statistics JSON
"""
from __future__ import annotations

import json
import os
import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr

from survalign_p import (
    AlignMarkManager, DifferentiableDistortion, UnifiedSpeechDataset,
    stft_audio, align_audio_tensors, _apply_survival_attack_pair,
    get_survival_map, minmax_per_sample,
)
from experiment_utils import stable_int_hash


SR = 16000  # sample rate


import scipy.io.wavfile as wavfile

def save_wav(tensor, path, sr=SR):
    """Save a 1-D or 2-D tensor as a WAV file using scipy."""
    wav_np = tensor.squeeze().cpu().numpy()
    wav_int16 = (np.clip(wav_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(path, sr, wav_int16)


def plot_waveform_spectrogram(wav_dict, sr, output_path, title=""):
    """Plot waveform (top) and spectrogram (bottom) for multiple audio signals."""
    n = len(wav_dict)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6), squeeze=False)

    for col, (label, wav) in enumerate(wav_dict.items()):
        wav_np = wav.squeeze().cpu().numpy()
        t = np.arange(len(wav_np)) / sr

        # Waveform
        axes[0, col].plot(t, wav_np, linewidth=0.3, color="tab:blue")
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].set_xlim(0, t[-1])
        axes[0, col].set_ylim(-1, 1)
        if col == 0:
            axes[0, col].set_ylabel("Amplitude")

        # Spectrogram
        axes[1, col].specgram(wav_np, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
        if col == 0:
            axes[1, col].set_ylabel("Freq (Hz)")
        axes[1, col].set_xlabel("Time (s)")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_survival_map_with_overlays(
    spec_clean, spec_wm, survival_map, output_path,
    n_fft=256, hop_length=64, title=""
):
    """Plot 3-channel guide map: Original, Residual, Survival Map."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    orig = spec_clean[0].cpu().numpy()
    residual = torch.abs(spec_wm - spec_clean)[0].cpu().numpy()
    smap = survival_map[0].cpu().numpy()

    im0 = axes[0].imshow(orig, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Original Spectrogram (Ch.1)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(residual, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Watermark Residual (Ch.2)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(smap, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Survival Map (Ch.3)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_attack_comparison_grid(attack_results, output_path, title=""):
    """Create a comprehensive grid showing per-attack spectrograms and scores."""
    n_attacks = len(attack_results)
    cols = min(4, n_attacks)
    rows = (n_attacks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (name, info) in enumerate(attack_results.items()):
        r, c = divmod(idx, cols)
        spec = info["attacked_wm_spec"][0].cpu().numpy()
        retention = info["mean_retention"]
        dominance = info["mean_dominance"]
        score = info["mean_score"]

        im = axes[r, c].imshow(
            20 * np.log10(spec + 1e-8), aspect="auto", origin="lower",
            cmap="magma",
        )
        axes[r, c].set_title(
            f"{name}\nret={retention:.3f} dom={dominance:.3f} score={score:.3f}",
            fontsize=8,
        )

    # Hide unused axes
    for idx in range(n_attacks, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_bar_chart(attack_results, output_path, title=""):
    """Bar chart of per-attack retention, dominance, and combined score."""
    names = list(attack_results.keys())
    retentions = [attack_results[n]["mean_retention"] for n in names]
    dominances = [attack_results[n]["mean_dominance"] for n in names]
    scores = [attack_results[n]["mean_score"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 5))
    ax.bar(x - width, retentions, width, label="Retention", color="tab:blue", alpha=0.8)
    ax.bar(x, dominances, width, label="Dominance", color="tab:orange", alpha=0.8)
    ax.bar(x + width, scores, width, label="Score (ret x dom)", color="tab:green", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_detailed_attack_scores(
    clean, watermarked, distorter, attack_names, base_seed, args,
    n_fft=256, hop_length=64, residual_floor_quantile=0.05,
):
    """Compute detailed per-attack survival scores with intermediate values."""
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

        results = {}
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
            masked_score = torch.where(valid_support, score, torch.zeros_like(score))

            valid_mask = valid_support[0]
            results[attack_name] = {
                "attacked_wm": attacked_wm,
                "attacked_clean": attacked_clean,
                "attacked_wm_spec": torch.abs(aw_spec),
                "mean_retention": float(retention[0][valid_mask].mean()),
                "mean_dominance": float(dominance[0][valid_mask].mean()),
                "mean_score": float(masked_score[0][valid_mask].mean()) if valid_mask.any() else 0.0,
            }
        return results


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
    parser.add_argument("--output_dir", default="outputs/survival_analysis")
    args = parser.parse_args()

    out = args.output_dir
    audio_dir = os.path.join(out, "audio")
    viz_dir = os.path.join(out, "viz")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    device = torch.device("cpu")

    attack_names = [
        "replacement", "masking", "frame_shuffle",
        "lowpass", "bandpass", "highpass",
        "ffmpeg_mp3", "ffmpeg_aac", "encodec", "vocos",
    ]

    from inprocess_attacks import prewarm
    prewarm(device)

    alignmark = AlignMarkManager(device, latent_mode=args.latent_mode)
    distorter = DifferentiableDistortion(sr=SR, vae=alignmark.vae).to(device)
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
    sample_id = metadata["sample_id"]
    speaker_id = metadata["speaker_id"]

    print(f"[INFO] Sample: {sample_id}, Speaker: {speaker_id}")

    # === 1. Embed watermark ===
    wav_wm, residual = alignmark.embed(wav, msg)
    wav, wav_wm, residual = align_audio_tensors(wav, wav_wm, residual)
    clean = wav.squeeze(1)
    watermarked = wav_wm.squeeze(1)

    # === 2. Save clean and watermarked audio ===
    clean_path = os.path.join(audio_dir, "01_clean.wav")
    wm_path = os.path.join(audio_dir, "02_watermarked.wav")
    save_wav(clean, clean_path)
    save_wav(watermarked, wm_path)
    print(f"[SAVED] {clean_path}")
    print(f"[SAVED] {wm_path}")

    # === 3. Compute Survival Map (strategy 3: single attacks only) ===
    survival_map = get_survival_map(
        wav, wav_wm, distorter,
        attack_names=tuple(attack_names),
        quantile=args.survival_quantile,
        args=args,
    )

    # === 4. Compute detailed per-attack scores and save attacked audio ===
    context_seed = stable_int_hash(args.seed, "analysis", sample_id)
    attack_results = compute_detailed_attack_scores(
        clean, watermarked, distorter, attack_names, context_seed, args,
    )

    audio_manifest = {
        "clean": clean_path,
        "watermarked": wm_path,
    }

    for idx, attack_name in enumerate(attack_names):
        attacked_wm = attack_results[attack_name]["attacked_wm"]
        attacked_path = os.path.join(audio_dir, f"{idx+3:02d}_attacked_{attack_name}.wav")
        save_wav(attacked_wm, attacked_path)
        audio_manifest[f"attacked_{attack_name}"] = attacked_path
        print(f"[SAVED] {attacked_path}")

    # === 5. Visualizations ===

    # 5a. Survival Map 3-channel view
    n_fft, hop_length = 256, 64
    spec_clean = stft_audio(clean, n_fft=n_fft, hop_length=hop_length)
    spec_wm = stft_audio(watermarked, n_fft=n_fft, hop_length=hop_length)

    survival_map_path = os.path.join(viz_dir, "survival_map_3channel.png")
    plot_survival_map_with_overlays(
        torch.abs(spec_clean), torch.abs(spec_wm), survival_map,
        survival_map_path,
        title=f"Survival Map (Strategy 3: Single-Attack, Q={args.survival_quantile})\n{sample_id} speaker={speaker_id}",
    )
    print(f"[SAVED] {survival_map_path}")

    # 5b. Waveform + spectrogram: clean vs watermarked
    core_wav_dict = {
        "Clean (Original)": clean,
        "Watermarked (No Attack)": watermarked,
        "Residual (x10 amplified)": (watermarked - clean) * 10,
    }
    core_path = os.path.join(viz_dir, "waveform_clean_vs_watermarked.png")
    plot_waveform_spectrogram(
        core_wav_dict, SR, core_path,
        title=f"Clean vs Watermarked -- {sample_id}",
    )
    print(f"[SAVED] {core_path}")

    # 5c. Waveform + spectrogram: attacked versions (split into 2 rows for readability)
    for batch_idx, batch_start in enumerate(range(0, len(attack_names), 5)):
        batch_attacks = attack_names[batch_start:batch_start + 5]
        atk_wav_dict = {}
        for name in batch_attacks:
            atk_wav_dict[name] = attack_results[name]["attacked_wm"]
        atk_path = os.path.join(viz_dir, f"waveform_attacked_batch{batch_idx}.png")
        plot_waveform_spectrogram(
            atk_wav_dict, SR, atk_path,
            title=f"Attacked Watermarked Audio (Batch {batch_idx+1}) -- {sample_id}",
        )
        print(f"[SAVED] {atk_path}")

    # 5d. Per-attack spectrogram grid
    grid_path = os.path.join(viz_dir, "attack_spectrogram_grid.png")
    plot_attack_comparison_grid(
        attack_results, grid_path,
        title=f"Per-Attack Spectrogram & Scores -- {sample_id}",
    )
    print(f"[SAVED] {grid_path}")

    # 5e. Score bar chart
    bar_path = os.path.join(viz_dir, "attack_score_barchart.png")
    plot_score_bar_chart(
        attack_results, bar_path,
        title=f"Per-Attack Retention / Dominance / Score -- {sample_id}",
    )
    print(f"[SAVED] {bar_path}")

    # === 6. Summary stats ===
    smap_np = survival_map[0].cpu().numpy()
    summary = {
        "sample_id": sample_id,
        "speaker_id": speaker_id,
        "strategy": "3: Single-attack Survival Map + train_cascade for Phase2",
        "survival_quantile": args.survival_quantile,
        "n_attacks": len(attack_names),
        "attack_names": attack_names,
        "survival_map_stats": {
            "mean": round(float(smap_np.mean()), 4),
            "std": round(float(smap_np.std()), 4),
            "median": round(float(np.median(smap_np)), 4),
            "low_fraction_lt03": round(float((smap_np < 0.3).mean()), 4),
            "high_fraction_gt07": round(float((smap_np > 0.7).mean()), 4),
        },
        "per_attack": {},
    }
    for name in attack_names:
        info = attack_results[name]
        summary["per_attack"][name] = {
            "mean_retention": round(info["mean_retention"], 4),
            "mean_dominance": round(info["mean_dominance"], 4),
            "mean_score": round(info["mean_score"], 4),
        }

    stats_path = os.path.join(out, "summary_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {stats_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE: {sample_id} (Speaker {speaker_id})")
    print(f"{'='*60}")
    print(f"Strategy: 3 Single-Attack Survival Map (Q={args.survival_quantile})")
    print(f"Attacks used: {len(attack_names)}")
    print(f"Survival Map -- mean={smap_np.mean():.4f}, std={smap_np.std():.4f}, "
          f"median={np.median(smap_np):.4f}")
    print(f"  Low (<0.3): {(smap_np < 0.3).mean():.1%}")
    print(f"  High (>0.7): {(smap_np > 0.7).mean():.1%}")
    print(f"\nPer-attack scores:")
    for name in attack_names:
        info = attack_results[name]
        print(f"  {name:20s}: ret={info['mean_retention']:.3f}  "
              f"dom={info['mean_dominance']:.3f}  score={info['mean_score']:.3f}")
    print(f"\nAudio files: {audio_dir}")
    print(f"Visualizations: {viz_dir}")


if __name__ == "__main__":
    main()
