# -*- coding: utf-8 -*-
"""Asset-free smoke tests for the revised experiment infrastructure."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from experiment_utils import (
    compute_attribution_metrics,
    integer_to_bits,
    project_residual_l2,
)
from phase2_training import SimplifiedSurvivalGate, build_candidate
from survalign_p import (
    DifferentiableDistortion, bits_to_chunks, chunks_to_bits, get_survival_map,
    _estimate_integer_shift, _apply_integer_shifts, compute_si_sdr,
)


class MockAlignMark:
    def embed(self, wav, msg):
        phase = torch.linspace(0, 20, wav.shape[-1], device=wav.device, dtype=wav.dtype)
        residual = 1e-3 * torch.sin(phase).view(1, 1, -1).expand_as(wav)
        return wav + residual, residual

    def decode_logits_with_grad(self, wav):
        batch = wav.shape[0]
        feature = wav.mean(dim=-1).reshape(batch, 1, 1)
        basis = torch.arange(16, device=wav.device, dtype=wav.dtype).reshape(1, 1, 16)
        logits = (feature * basis).expand(batch, 4, 16)
        return wav.mean(dim=1), logits

    @torch.no_grad()
    def decode(self, wav):
        frame, logits = self.decode_logits_with_grad(wav)
        return frame, logits, chunks_to_bits(logits.argmax(dim=-1), 4)


def main():
    targets = torch.stack([integer_to_bits(v) for v in [101, 903, 12291, 64001]])
    metrics = compute_attribution_metrics(targets.clone(), targets)
    assert metrics["exact_message_accuracy"] == 1.0
    assert metrics["far_strict"] == 0.0
    chunk_tensor = torch.stack(bits_to_chunks(targets, 4), dim=1)
    assert torch.equal(chunks_to_bits(chunk_tensor, 4), targets)

    reference = torch.randn(3, 1, 1000)
    candidate = torch.randn_like(reference) * 3
    capped = project_residual_l2(candidate, reference, mode="cap")
    assert torch.all(
        torch.linalg.vector_norm(capped, dim=(1, 2))
        <= torch.linalg.vector_norm(reference, dim=(1, 2)) + 1e-5
    )


    # Codec-output alignment must undo both positive and negative integer delays.
    impulse = torch.zeros(1, 100)
    impulse[0, 20] = 1.0
    delayed = torch.nn.functional.pad(impulse[0, :-5], (5, 0)).unsqueeze(0)
    shift = _estimate_integer_shift(impulse, delayed, max_shift=10)
    realigned = _apply_integer_shifts(delayed, shift)
    assert realigned.argmax().item() == impulse.argmax().item()

    # One-dimensional SI-SDR inputs must be interpreted as a single waveform.
    sdr = compute_si_sdr(impulse.squeeze(0).numpy(), impulse.squeeze(0).numpy())
    assert sdr > 60.0

    distorter = DifferentiableDistortion(sr=16000, vae=None)
    clean = torch.randn(2, 3200) * 0.01
    watermarked = clean + torch.randn_like(clean) * 1e-4
    survival = get_survival_map(
        clean,
        watermarked,
        distorter,
        attack_names=["noise", "lowpass", "resample", "spectral_proxy"],
    )
    assert survival.shape[0] == 2
    assert torch.isfinite(survival).all()

    args = SimpleNamespace(
        mode="proposed_gate",
        map_type="survival",
        gate_range=0.2,
        projection_mode="cap",
        uniform_scale=1.1,
        survival_attack_names=["noise", "lowpass", "resample", "spectral_proxy"],
        survival_quantile=0.25,
        utility_attack_names=["noise"],
        seed=42,
        current_msg=None,
    )
    wav = torch.randn(2, 1, 3200) * 0.01
    msg = torch.randint(0, 2, (2, 16))
    gate = SimplifiedSurvivalGate(in_channels=4, gate_range=0.2)
    output = build_candidate(args, gate, MockAlignMark(), distorter, wav, msg, context_seed=42)
    assert output[2].shape == wav.shape
    print("All asset-free smoke tests passed.")


if __name__ == "__main__":
    main()
