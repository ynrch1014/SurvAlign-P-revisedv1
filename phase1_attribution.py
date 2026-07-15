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

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import permutation_test, wilcoxon
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment_utils import (
    align_audio_tensors,
    apply_eval_attack,
    apply_internal_attack as _apply_internal_attack,
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


def _single_sample_attack_loss(alignmark, wav_1, residual_spec_1, alpha_1, target_1,
                               distorter, attack_names, seed, length):
    """Mean post-attack decoder CE for one sample with residual scaled by alpha_1 (forward-only)."""
    scaled = residual_spec_1 * alpha_1
    scaled_residual = istft_audio(scaled, length=length, n_fft=256, hop_length=64)
    candidate = (wav_1 + scaled_residual).unsqueeze(1)  # (1, 1, T)
    loss = 0.0
    for attack_index, attack_name in enumerate(attack_names):
        attacked = _apply_internal_attack(candidate, attack_name, distorter, seed + attack_index)
        _, logits = alignmark.decode_logits_with_grad(attacked)
        loss = loss + compute_chunk_ce_loss(logits, target_1)
    return float((loss / len(attack_names)).item())


def compute_finite_difference_utility_topk(
    alignmark,
    wav,
    residual,
    target_msg,
    distorter,
    attack_names: Sequence[str],
    reference_map: torch.Tensor,
    num_bins: int = 32,
    epsilon: float = 0.05,
    base_seed: int = 0,
):
    """M3: true finite-difference codec-utility on the top-``num_bins`` bins of ``reference_map``.

    M2 (``compute_decoder_utility_map``) is a first-order (gradient) approximation of
    -dE[L_dec]/d alpha. M3 validates it by *directly* measuring the loss change from a real
    +-epsilon perturbation of alpha at individual bins, running the attack and decoder each time.
    This costs O(B * num_bins * |attacks| * 2) forward passes, so it is meant for a 20-30 sample
    subset only. Returns per-sample agreement between the M2 ranking and the measured M3 utility.
    """
    from scipy.stats import spearmanr

    wav_2d = wav.squeeze(1)
    residual_2d = residual.squeeze(1)
    residual_spec = stft_audio(residual_2d, n_fft=256, hop_length=64).detach()
    length = wav_2d.shape[-1]
    batch_size, freq, time = residual_spec.shape
    num_bins = int(min(num_bins, freq * time))

    per_sample = []
    with torch.no_grad():
        for item in range(batch_size):
            ref_flat = reference_map[item].reshape(-1)
            top_indices = torch.topk(ref_flat.abs(), k=num_bins, largest=True, sorted=False).indices
            spec_1 = residual_spec[item:item + 1]
            wav_1 = wav_2d[item:item + 1]
            target_1 = target_msg[item:item + 1]
            seed_item = int(base_seed) + item * 1000
            m2_values, m3_values = [], []
            for flat_idx in top_indices.tolist():
                f, t = divmod(flat_idx, time)
                alpha_plus = torch.ones_like(spec_1.real)
                alpha_minus = torch.ones_like(spec_1.real)
                alpha_plus[0, f, t] += epsilon
                alpha_minus[0, f, t] -= epsilon
                loss_plus = _single_sample_attack_loss(
                    alignmark, wav_1, spec_1, alpha_plus, target_1, distorter, attack_names, seed_item, length)
                loss_minus = _single_sample_attack_loss(
                    alignmark, wav_1, spec_1, alpha_minus, target_1, distorter, attack_names, seed_item, length)
                # utility = -dL/dalpha (positive => amplifying this bin helps the decoder).
                m3_values.append(-(loss_plus - loss_minus) / (2.0 * epsilon))
                m2_values.append(float(ref_flat[flat_idx].item()))
            m2_arr = np.asarray(m2_values)
            m3_arr = np.asarray(m3_values)
            entry = {"n_bins": len(m3_values)}
            if np.std(m2_arr) > 1e-12 and np.std(m3_arr) > 1e-12:
                rho = spearmanr(m2_arr, m3_arr).statistic
                entry["spearman_m2_m3"] = float(rho) if np.isfinite(rho) else float("nan")
            else:
                entry["spearman_m2_m3"] = float("nan")
            entry["sign_agreement"] = float(np.mean(np.sign(m2_arr) == np.sign(m3_arr)))
            entry["m3_mean"] = float(np.mean(m3_arr))
            per_sample.append(entry)
    return per_sample


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


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray, eps=1e-12):
    """x와 y의 Spearman 편상관(z를 통제)을 계산한다.

    검토 보고서 D-① 항목: M0(residual_placement)와 Survival Map의 단순 상관은
    "둘 다 음성 에너지가 큰 곳을 선호한다"는 혼입 변수(z = speech_magnitude)로
    부풀려지거나 상쇄될 수 있다. 순위 기반 편상관 공식을 그대로 적용한다:
        r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    """
    from scipy.stats import spearmanr
    if np.std(x) < eps or np.std(y) < eps or np.std(z) < eps:
        return None
    r_xy = spearmanr(x, y).statistic
    r_xz = spearmanr(x, z).statistic
    r_yz = spearmanr(y, z).statistic
    if not all(np.isfinite(v) for v in (r_xy, r_xz, r_yz)):
        return None
    denom = np.sqrt(max(1e-12, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
    if denom < eps:
        return None
    value = (r_xy - r_xz * r_yz) / denom
    if not np.isfinite(value):
        return None
    return float(np.clip(value, -1.0, 1.0))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1] via the paired difference sign balance.

    For paired samples we use the dominance of positive over negative differences,
    which equals the rank-biserial correlation of the paired test and is a robust,
    non-parametric effect size. Zero-differences (ties) contribute 0.
    """
    differences = a - b
    n = differences.size
    if n == 0:
        return float("nan")
    positives = int(np.sum(differences > 0))
    negatives = int(np.sum(differences < 0))
    return float((positives - negatives) / n)


def paired_statistics(a: Sequence[float], b: Sequence[float], seed: int) -> Dict[str, float]:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.size < 2:
        return {
            "mean_difference": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
            "wilcoxon_p": float("nan"), "wilcoxon_n_effective": 0,
            "permutation_p": float("nan"), "cliffs_delta": float("nan"), "n": int(a_arr.size),
        }
    differences = a_arr - b_arr
    # Effective n for Wilcoxon = number of non-zero paired differences (zeros are discarded).
    n_effective = int(np.sum(differences != 0))
    try:
        wilcoxon_p = float(wilcoxon(differences).pvalue)
    except ValueError:
        # Raised when every difference is zero; there is no evidence of a shift.
        wilcoxon_p = 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(differences.mean()))
    signs = rng.choice([-1.0, 1.0], size=(10000, differences.size))
    null = np.abs((signs * differences[None, :]).mean(axis=1))
    permutation_p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    # Bootstrap 95% CI of the mean paired difference (10k resamples, same rng stream).
    boot = rng.choice(differences, size=(10000, differences.size), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_p": wilcoxon_p,
        "wilcoxon_n_effective": n_effective,
        "permutation_p": permutation_p,
        "cliffs_delta": cliffs_delta(a_arr, b_arr),
        "n": int(a_arr.size),
    }


def holm_bonferroni(pvalues: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Holm–Bonferroni step-down correction over a family of hypotheses.

    Returns, per key, the adjusted p-value and a reject flag at alpha=0.05. NaN p-values
    are excluded from the family size so they do not deflate the correction of valid tests.
    """
    valid = {k: float(v) for k, v in pvalues.items() if np.isfinite(v)}
    m = len(valid)
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    result: Dict[str, Dict[str, float]] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * p)
        running_max = max(running_max, adjusted)  # enforce monotonicity of step-down
        result[key] = {"p_raw": p, "p_holm": running_max, "reject_0.05": bool(running_max < 0.05)}
    for key, p in pvalues.items():
        if key not in result:
            result[key] = {"p_raw": float(p), "p_holm": float("nan"), "reject_0.05": False}
    return result


def _target_norm_reference(
    raw_residuals: Dict[str, torch.Tensor],
    original: torch.Tensor,
    target_mode: str,
    fixed_fraction: Optional[float] = None,
):
    """Return the reference waveform whose L2 norm defines the equal-energy budget.

    target_mode:
      - "baseline":        target = full residual (no rescale).
      - "fixed_fraction":  target = fixed_fraction * ||full residual|| (default).
                           With fixed_fraction = sqrt(top_ratio), every condition is
                           equalized to exactly top_ratio of the full residual *energy*,
                           independent of which masked condition happened to be smallest.
      - "minimum":         legacy — target = smallest masked-condition norm. This couples
                           the budget to an arbitrary condition and is kept only for ablation.
    """
    if target_mode == "baseline":
        return original
    if target_mode == "fixed_fraction":
        if fixed_fraction is None:
            raise ValueError("fixed_fraction target requires a fraction value")
        return original * float(fixed_fraction)
    if target_mode == "minimum":
        norms = []
        for name, residual in raw_residuals.items():
            if name != "Full":
                norms.append(torch.linalg.vector_norm(residual, dim=-1))
        min_norm = torch.stack(norms, dim=0).min(dim=0).values
        original_norm = torch.linalg.vector_norm(original, dim=-1).clamp_min(1e-8)
        scale = (min_norm / original_norm).unsqueeze(-1)
        return original * scale
    raise ValueError(f"Unknown equal-energy target mode: {target_mode}")


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
    parser.add_argument("--equal_energy_target", default="fixed_fraction",
                        choices=["fixed_fraction", "minimum", "baseline"])
    parser.add_argument("--equal_energy_fraction", type=float, default=-1.0,
                        help="Fraction of the full residual norm used as the equal-energy target. "
                             "Default (-1) resolves to sqrt(top_ratio), i.e. top_ratio of the full energy.")
    parser.add_argument("--survival_attacks", default="noise,lowpass,resample,speechtokenizer_nq6,spectral_proxy")
    # 1-B: utility-map attacks must stay disjoint (by family) from the eval attacks used as
    # generalization evidence. strong_speechtokenizer was in BOTH defaults, so the default run
    # leaked its own held-out claim; keep only speechtokenizer_nq6 here.
    parser.add_argument("--utility_attacks", default="speechtokenizer_nq6")
    # Default eval attacks are now genuinely held-out from the survival/utility families
    # (frame_shuffle/replacement are absent from both), so a default run passes --strict_heldout.
    # To measure *codec* generalization, pass real held-out codecs explicitly, e.g.
    #   --eval_attacks clean,facodec --facodec_command "...", ensuring survival_attacks stay clean.
    parser.add_argument("--eval_attacks", default="clean,frame_shuffle,replacement")
    parser.add_argument("--clearervoice_command", default="")
    parser.add_argument("--facodec_command", default="")
    parser.add_argument("--encodec_command", default="")
    parser.add_argument("--dac_command", default="")
    parser.add_argument("--vocos_command", default="")
    parser.add_argument("--hifigan_command", default="")
    parser.add_argument("--clearervoice_snr", type=float, default=10.0)
    parser.add_argument("--mp3_bitrate", default="64k")
    parser.add_argument("--latent_mode", default="public_code", choices=["public_code", "unquantized"])
    parser.add_argument("--strict_heldout", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail (default) if evaluation attacks overlap Survival/utility-map attacks. "
                             "Pass --no-strict_heldout to only warn (e.g. for deliberate in-distribution ablations).")
    parser.add_argument("--results_dir", default="results/phase1_confirmatory")
    parser.add_argument("--run_m3_finite_difference", action="store_true",
                        help="Validate the M2 codec-utility map with true finite differences (M3) on a subset.")
    parser.add_argument("--m3_num_bins", type=int, default=32,
                        help="Number of top-|M2| bins per sample to probe with finite differences.")
    parser.add_argument("--m3_epsilon", type=float, default=0.05, help="Central finite-difference step on alpha.")
    parser.add_argument("--m3_max_samples", type=int, default=20, help="Subset size for the M3 finite-difference check.")
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
    # sqrt(top_ratio): keeping top_ratio of the bins and rescaling to this fraction of the
    # full residual norm makes the retained energy exactly top_ratio of the full energy.
    equal_energy_fraction = (
        float(np.sqrt(args.top_ratio)) if args.equal_energy_fraction < 0 else float(args.equal_energy_fraction)
    )
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
    equal_amplification = defaultdict(list)  # condition -> per-sample amplification under equal energy
    m3_rows = []  # 2-B: per-sample M2-vs-M3 finite-difference agreement
    m3_processed = 0
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

        # 2-B (M3): validate the first-order M2 map with a true finite-difference measurement on
        # a small subset. Gated behind a flag because it costs O(num_bins * |attacks| * 2) forwards.
        if args.run_m3_finite_difference and m3_processed < args.m3_max_samples:
            take = min(wav.shape[0], args.m3_max_samples - m3_processed)
            m3_rows.extend(
                compute_finite_difference_utility_topk(
                    alignmark, wav[:take], residual[:take], msg[:take], distorter, utility_attacks,
                    reference_map=utility[:take], num_bins=args.m3_num_bins,
                    epsilon=args.m3_epsilon, base_seed=args.seed + batch_index * 1000,
                )
            )
            m3_processed += take

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
            # M0: AlignMark가 "실제로 심어놓은" 에너지 분포(residual 크기) 자체.
            # 기존에는 Survival Map을 디코더의 사후 설명 신호(saliency/utility)와만
            # 비교했고, 정작 "실제 배치"와의 직접 비교가 빠져 있었다 (검토 보고서 B.1).
            "residual_placement": exact_topk_mask(residual_magnitude, args.top_ratio, largest=True).bool(),
        }

        for item in range(batch_size):
            support = valid_support[item].detach().cpu().numpy().astype(bool).reshape(-1)
            maps = {
                "survival": survival[item].detach().cpu().numpy().reshape(-1),
                "gradient_saliency": gradient_saliency[item].detach().cpu().numpy().reshape(-1),
                "codec_utility": utility[item].detach().cpu().numpy().reshape(-1),
                "residual_placement": residual_magnitude[item].detach().cpu().numpy().reshape(-1),
            }
            speech_support = speech_magnitude[item].detach().cpu().numpy().reshape(-1)
            for other_name in ("gradient_saliency", "codec_utility", "residual_placement"):
                result = valid_correlations(maps["survival"][support], maps[other_name][support])
                if result is not None:
                    survival_top = correlation_top_masks["survival"][item].cpu().numpy().reshape(-1)
                    other_top = correlation_top_masks[other_name][item].cpu().numpy().reshape(-1)
                    intersection = np.logical_and(survival_top, other_top).sum()
                    union = np.logical_or(survival_top, other_top).sum()
                    # 음성 에너지(speech_magnitude)를 통제한 편상관. Survival Map과
                    # residual_placement가 둘 다 "음성 구간을 선호한다"는 이유만으로
                    # 상관관계가 생기는 허위 상관(spurious correlation) 여부를 확인한다
                    # (검토 보고서 D-① 항목).
                    partial = partial_spearman(
                        maps["survival"][support], maps[other_name][support], speech_support[support]
                    )
                    correlation_rows.append({
                        "sample_index": processed + item,
                        "map_pair": f"survival_vs_{other_name}",
                        "pearson": result[0],
                        "spearman": result[1],
                        "topk_iou": float(intersection / max(1, union)),
                        "partial_spearman_ctrl_speech": partial if partial is not None else float("nan"),
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

        target_reference = _target_norm_reference(
            raw_residuals, residual.squeeze(1), args.equal_energy_target, equal_energy_fraction
        )

        def evaluate_one(condition, raw_residual, repeat_index=0):
            for energy_mode in energy_modes:
                if energy_mode == "natural":
                    adjusted = raw_residual
                elif energy_mode == "equal":
                    adjusted = project_residual_l2(raw_residual, target_reference, mode="equal")
                    # Log the per-sample amplification actually applied to reach the equal-energy
                    # budget (||adjusted|| / ||raw||). >1 means the masked residual was boosted.
                    raw_norm = torch.linalg.vector_norm(raw_residual, dim=-1).clamp_min(1e-12)
                    adj_norm = torch.linalg.vector_norm(adjusted, dim=-1)
                    equal_amplification[condition].extend((adj_norm / raw_norm).cpu().tolist())
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
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
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
        "equal_energy_amplification": {},
        "equal_energy_config": {
            "target": args.equal_energy_target,
            "fraction": equal_energy_fraction,
        },
        "conditions": {},
        "paired_tests": {},
    }
    for pair in sorted({row["map_pair"] for row in correlation_rows}):
        rows = [row for row in correlation_rows if row["map_pair"] == pair]
        partial_values = [r["partial_spearman_ctrl_speech"] for r in rows if np.isfinite(r["partial_spearman_ctrl_speech"])]
        summary["correlations"][pair] = {
            "pearson_mean": float(np.mean([r["pearson"] for r in rows])),
            "pearson_std": float(np.std([r["pearson"] for r in rows])),
            "spearman_mean": float(np.mean([r["spearman"] for r in rows])),
            "spearman_std": float(np.std([r["spearman"] for r in rows])),
            "topk_iou_mean": float(np.mean([r["topk_iou"] for r in rows])),
            "topk_iou_std": float(np.std([r["topk_iou"] for r in rows])),
            # 음성 에너지를 통제한 뒤에도 상관관계가 남는지 확인 (허위 상관 배제)
            "partial_spearman_ctrl_speech_mean": float(np.mean(partial_values)) if partial_values else float("nan"),
            "partial_spearman_ctrl_speech_std": float(np.std(partial_values)) if partial_values else float("nan"),
            "partial_spearman_n_valid": len(partial_values),
        }
    for condition, values in retained_ratios.items():
        summary["retained_energy_ratio"][condition] = {
            "mean": float(np.mean(values)), "std": float(np.std(values))
        }
    for condition, values in equal_amplification.items():
        summary["equal_energy_amplification"][condition] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "max": float(np.max(values)) if values else float("nan"),
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

    # Holm–Bonferroni correction across the whole family of paired comparisons.
    wilcoxon_family = {k: v["wilcoxon_p"] for k, v in summary["paired_tests"].items()}
    holm = holm_bonferroni(wilcoxon_family)
    for key, adjusted in holm.items():
        summary["paired_tests"][key]["wilcoxon_p_holm"] = adjusted["p_holm"]
        summary["paired_tests"][key]["wilcoxon_reject_holm_0.05"] = adjusted["reject_0.05"]
    summary["multiple_comparison_correction"] = {
        "method": "holm_bonferroni",
        "family_size": int(np.sum(np.isfinite(list(wilcoxon_family.values())))),
        "alpha": 0.05,
    }

    if args.run_m3_finite_difference and m3_rows:
        spearmans = [r["spearman_m2_m3"] for r in m3_rows if np.isfinite(r["spearman_m2_m3"])]
        summary["m3_finite_difference"] = {
            "n_samples": len(m3_rows),
            "num_bins": args.m3_num_bins,
            "epsilon": args.m3_epsilon,
            "spearman_m2_m3_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
            "spearman_m2_m3_std": float(np.std(spearmans)) if spearmans else float("nan"),
            "sign_agreement_mean": float(np.mean([r["sign_agreement"] for r in m3_rows])),
            "note": "First-order M2 vs true finite-difference M3 on top-|M2| bins; small subset, qualitative.",
        }

    with open(os.path.join(args.results_dir, "phase1_sample_results.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)
    with open(os.path.join(args.results_dir, "phase1_correlations.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "map_pair", "pearson", "spearman", "topk_iou", "partial_spearman_ctrl_speech"],
        )
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
