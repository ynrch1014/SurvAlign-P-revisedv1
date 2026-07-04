# -*- coding: utf-8 -*-
"""Shared utilities for reproducible SurvAlign-P experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def stable_int_hash(*parts: Any, modulo: int = 2**31 - 1) -> int:
    """Return a process-independent deterministic integer hash."""
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % modulo




def attack_family(name: str) -> str:
    """Map attack aliases/settings to a model or transformation family for leakage checks."""
    normalized = str(name).strip().lower()
    if normalized in {"clean", "identity"}:
        return "clean"
    if normalized in {"noise", "noise10db"}:
        return "awgn"
    if normalized in {"lowpass", "bandpass"}:
        return "linear_filter"
    if normalized == "resample":
        return "resampling"
    if normalized in {"reconstruct_nq6", "reconstruct_nq8", "strong_speechtokenizer", "facodec_proxy"}:
        return "speechtokenizer"
    if normalized in {"spectral_proxy", "mp3"}:
        return "spectral_proxy"
    if normalized == "ffmpeg_mp3":
        return "real_mp3"
    return normalized


def overlapping_attack_families(left: Sequence[str], right: Sequence[str]) -> List[str]:
    return sorted(set(map(attack_family, left)) & set(map(attack_family, right)))

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def integer_to_bits(value: int, nbits: int = 16) -> torch.Tensor:
    if value < 0 or value >= 2**nbits:
        raise ValueError(f"value must be in [0, {2**nbits}), got {value}")
    shifts = torch.arange(nbits - 1, -1, -1, dtype=torch.long)
    return ((value >> shifts) & 1).long()


def deterministic_unique_message(index: int, nbits: int = 16, offset: int = 0) -> torch.Tensor:
    """Generate an index-addressed unique message for evaluation sets."""
    capacity = 2**nbits
    value = index + offset
    if value >= capacity:
        raise ValueError(
            f"Evaluation set index {index} exceeds {nbits}-bit message capacity ({capacity})."
        )
    return integer_to_bits(value, nbits=nbits)


def align_audio_tensors(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Trim tensors to a common last-axis length and verify leading dimensions."""
    if not tensors:
        return tuple()
    min_len = min(int(t.shape[-1]) for t in tensors)
    aligned = tuple(t[..., :min_len] for t in tensors)
    lead = aligned[0].shape[:-1]
    for t in aligned[1:]:
        if t.shape[:-1] != lead:
            raise ValueError(f"Audio leading shapes do not match: {[x.shape for x in aligned]}")
    return aligned


def project_residual_l2(
    residual: torch.Tensor,
    reference: torch.Tensor,
    mode: str = "cap",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project waveform residual onto the reference L2 budget.

    mode='cap': ||residual|| <= ||reference||
    mode='equal': ||residual|| == ||reference|| (unless residual is numerically zero)
    """
    if residual.shape != reference.shape:
        raise ValueError(f"Residual/reference shape mismatch: {residual.shape} vs {reference.shape}")
    dims = tuple(range(1, residual.dim()))
    res_norm = torch.linalg.vector_norm(residual, ord=2, dim=dims, keepdim=True)
    ref_norm = torch.linalg.vector_norm(reference, ord=2, dim=dims, keepdim=True)
    ratio = ref_norm / (res_norm + eps)
    if mode == "cap":
        ratio = torch.minimum(torch.ones_like(ratio), ratio)
    elif mode == "equal":
        ratio = torch.where(res_norm > eps, ratio, torch.ones_like(ratio))
    else:
        raise ValueError(f"Unknown L2 projection mode: {mode}")
    return residual * ratio


def retained_energy_ratio(masked: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return per-sample energy ratio ||masked||^2 / ||reference||^2."""
    dims = tuple(range(1, masked.dim()))
    num = torch.sum(masked**2, dim=dims)
    den = torch.sum(reference**2, dim=dims) + eps
    return num / den


def exact_topk_mask(values: torch.Tensor, ratio: float, largest: bool = True) -> torch.Tensor:
    """Select exactly round(ratio * F*T) bins per sample."""
    if values.dim() != 3:
        raise ValueError(f"Expected (B,F,T), got {values.shape}")
    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1]")
    bsz, freq, time = values.shape
    flat = values.reshape(bsz, -1)
    k = max(1, int(round(flat.shape[1] * ratio)))
    indices = torch.topk(flat, k=k, dim=1, largest=largest, sorted=False).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)
    return mask.reshape(bsz, freq, time)


def gaussian_kernel2d(kernel_size: int = 5, sigma: float = 1.0, *, device=None, dtype=None) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd number")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def ensure_unique_messages(messages: torch.Tensor) -> None:
    if messages.dim() != 2:
        raise ValueError(f"Expected messages (N,Bits), got {messages.shape}")
    unique = torch.unique(messages, dim=0)
    if unique.shape[0] != messages.shape[0]:
        raise ValueError(
            f"Evaluation message codebook contains duplicates: {unique.shape[0]} unique / {messages.shape[0]} total."
        )


def _chunked_nearest_wrong_hamming(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute true and nearest-wrong Hamming distances without allocating huge N^2xB tensor."""
    predictions = predictions.bool().cpu()
    targets = targets.bool().cpu()
    n, nbits = targets.shape
    true_dist = torch.sum(predictions != targets, dim=1).long()
    nearest_wrong = torch.full((n,), nbits + 1, dtype=torch.long)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pred = predictions[start:end]
        distances = torch.sum(pred[:, None, :] != targets[None, :, :], dim=-1)
        local_rows = torch.arange(end - start)
        global_cols = torch.arange(start, end)
        distances[local_rows, global_cols] = nbits + 1
        nearest_wrong[start:end] = distances.min(dim=1).values
    return true_dist, nearest_wrong




def codebook_hamming_summary(targets: torch.Tensor, chunk_size: int = 256) -> Dict[str, float]:
    targets = targets.bool().cpu()
    n, nbits = targets.shape
    if n < 2:
        return {"min_codebook_hamming": float("nan"), "mean_nearest_codebook_hamming": float("nan")}
    nearest = torch.full((n,), nbits + 1, dtype=torch.long)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distances = torch.sum(targets[start:end, None, :] != targets[None, :, :], dim=-1)
        rows = torch.arange(end - start)
        cols = torch.arange(start, end)
        distances[rows, cols] = nbits + 1
        nearest[start:end] = distances.min(dim=1).values
    return {
        "min_codebook_hamming": float(nearest.min().item()),
        "mean_nearest_codebook_hamming": float(nearest.float().mean().item()),
    }



def compute_attribution_per_sample(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """Return per-sample Hamming attribution diagnostics on CPU."""
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predictions.shape} vs {targets.shape}")
    ensure_unique_messages(targets)
    predictions = predictions.long().cpu()
    targets = targets.long().cpu()
    true_dist, nearest_wrong = _chunked_nearest_wrong_hamming(predictions, targets, chunk_size)
    margin = nearest_wrong - true_dist
    return {
        "true_hamming": true_dist,
        "nearest_wrong_hamming": nearest_wrong,
        "attribution_margin": margin,
        "strict_failure": true_dist >= nearest_wrong,
        "lenient_failure": true_dist > nearest_wrong,
        "tie": true_dist == nearest_wrong,
    }

def compute_attribution_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 256,
) -> Dict[str, float]:
    """
    Compute bit/exact metrics and Feature-Aligned-style attribution FAR.

    FAR_strict counts ties as failures. FAR_lenient counts only a strictly closer wrong message.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predictions.shape} vs {targets.shape}")
    ensure_unique_messages(targets)
    predictions = predictions.long().cpu()
    targets = targets.long().cpu()
    bit_correct = (predictions == targets).float()
    hamming = torch.sum(predictions != targets, dim=1).float()
    exact = (hamming == 0)
    per_sample = compute_attribution_per_sample(predictions, targets, chunk_size=chunk_size)
    true_dist = per_sample["true_hamming"]
    nearest_wrong = per_sample["nearest_wrong_hamming"]
    margin = per_sample["attribution_margin"]
    strict_fail = per_sample["strict_failure"]
    lenient_fail = per_sample["lenient_failure"]
    tie = per_sample["tie"]
    codebook_summary = codebook_hamming_summary(targets, chunk_size=chunk_size)
    return {
        "bit_accuracy": float(bit_correct.mean().item()),
        "ber": float(1.0 - bit_correct.mean().item()),
        "exact_message_accuracy": float(exact.float().mean().item()),
        "mean_hamming_distance": float(hamming.mean().item()),
        "far_strict": float(strict_fail.float().mean().item()),
        "far_lenient": float(lenient_fail.float().mean().item()),
        "tie_rate": float(tie.float().mean().item()),
        "mean_attribution_margin": float(margin.float().mean().item()),
        "median_attribution_margin": float(margin.float().median().item()),
        "n_candidates": int(targets.shape[0]),
        **codebook_summary,
    }


def attribution_metrics_by_candidate_size(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    candidate_sizes: Sequence[int],
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate FAR on deterministic random candidate subsets of different sizes."""
    n = targets.shape[0]
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(n, generator=generator)
    sizes = sorted({int(size) for size in candidate_sizes if int(size) >= 2 and int(size) <= n} | {n})
    output = {}
    for size in sizes:
        indices = permutation[:size]
        output[str(size)] = compute_attribution_metrics(predictions[indices], targets[indices])
    return output


def compute_logit_metrics(chunk_logits: torch.Tensor, target_chunks: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return per-sample CE, minimum correct-vs-best-wrong margin and entropy."""
    if chunk_logits.dim() != 3:
        raise ValueError(f"Expected logits (B,Cchunks,Classes), got {chunk_logits.shape}")
    bsz, n_chunks, n_classes = chunk_logits.shape
    if target_chunks.shape != (bsz, n_chunks):
        raise ValueError(f"Target chunk shape mismatch: {target_chunks.shape}")
    log_probs = torch.log_softmax(chunk_logits, dim=-1)
    probs = torch.softmax(chunk_logits, dim=-1)
    gather = target_chunks.unsqueeze(-1)
    correct_logit = chunk_logits.gather(-1, gather).squeeze(-1)
    masked = chunk_logits.clone()
    masked.scatter_(-1, gather, float("-inf"))
    best_wrong = masked.max(dim=-1).values
    margins = correct_logit - best_wrong
    ce = -log_probs.gather(-1, gather).squeeze(-1).mean(dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean(dim=-1)
    return {
        "ce": ce,
        "min_logit_margin": margins.min(dim=-1).values,
        "mean_logit_margin": margins.mean(dim=-1),
        "mean_entropy": entropy,
    }


def recovery_regression_metrics(
    baseline_predictions: torch.Tensor,
    method_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    base_exact = torch.all(baseline_predictions.cpu() == targets.cpu(), dim=1)
    method_exact = torch.all(method_predictions.cpu() == targets.cpu(), dim=1)
    base_fail = ~base_exact
    base_success = base_exact
    recovered = base_fail & method_exact
    regressed = base_success & (~method_exact)
    return {
        "recovery_rate": float(recovered.sum().item() / max(1, base_fail.sum().item())),
        "regression_rate": float(regressed.sum().item() / max(1, base_success.sum().item())),
        "n_baseline_failures": int(base_fail.sum().item()),
        "n_baseline_successes": int(base_success.sum().item()),
        "n_recovered": int(recovered.sum().item()),
        "n_regressed": int(regressed.sum().item()),
    }


def nan_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    if not np.any(valid):
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "n_failed": int(len(array))}
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "n": int(valid.sum()),
        "n_failed": int((~valid).sum()),
    }


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
