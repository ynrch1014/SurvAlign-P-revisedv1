# -*- coding: utf-8 -*-
"""Asset-free regression tests for the revision work (items 1-A..3-A).

Run: python test_revisions.py   (CPU only, no weights/datasets required)
"""

from __future__ import annotations

import numpy as np
import torch

import experiment_utils
import external_attacks
import inprocess_attacks
import phase1_attribution
import phase2_training
from experiment_utils import (
    heldout_codec_of, survival_heldout_leakage, HELDOUT_CODECS, project_residual_l2,
    INTERNAL_ATTACK_NAMES, apply_internal_attack, apply_eval_attack, attack_family,
)
from phase1_attribution import (
    _target_norm_reference, paired_statistics, holm_bonferroni, cliffs_delta,
    compute_finite_difference_utility_topk,
)
from phase1_experiment3_selfcheck import leave_one_attack_out_spearman, summarize
from phase2_training import _gather_cached_survival
from survalign_p import DifferentiableDistortion, stft_audio
from smoke_test import MockAlignMark


def test_heldout_leakage():  # 1-A
    assert heldout_codec_of("facodec_proxy") == "facodec"
    assert heldout_codec_of("clearervoice_only") == "clearervoice"
    assert heldout_codec_of("dac") == "dac"
    assert heldout_codec_of("vocos") == "vocos"
    assert heldout_codec_of("speechtokenizer_nq6") is None
    assert survival_heldout_leakage(["speechtokenizer_nq6", "spectral_proxy"]) == {}
    leaks = survival_heldout_leakage(["facodec_proxy", "noise", "vocos"])
    assert leaks == {"facodec": ["facodec_proxy"], "vocos": ["vocos"]}
    assert set(HELDOUT_CODECS) == {"facodec", "clearervoice", "dac", "vocos"}


def test_equal_energy_fixed_fraction():  # 1-C
    torch.manual_seed(0)
    original = torch.randn(4, 3200)
    fraction = float(np.sqrt(0.2))
    ref = _target_norm_reference({}, original, "fixed_fraction", fraction)
    got = torch.linalg.vector_norm(ref, dim=-1)
    want = fraction * torch.linalg.vector_norm(original, dim=-1)
    assert torch.allclose(got, want, atol=1e-5)
    # After equal projection any residual is rescaled to exactly the target norm.
    other = torch.randn(4, 3200) * 5.0
    projected = project_residual_l2(other, ref, mode="equal")
    assert torch.allclose(
        torch.linalg.vector_norm(projected, dim=-1),
        torch.linalg.vector_norm(ref, dim=-1), atol=1e-4,
    )


def test_paired_statistics_and_holm():  # 1-D
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.1, size=200)
    b = a - 0.05  # a is consistently larger
    b[:20] = a[:20]  # inject ties => effective n should drop
    stats = paired_statistics(a, b, seed=1)
    for key in ("mean_difference", "ci95_low", "ci95_high", "wilcoxon_p",
                "wilcoxon_n_effective", "permutation_p", "cliffs_delta", "n"):
        assert key in stats
    assert stats["wilcoxon_n_effective"] == 180  # 200 - 20 ties
    assert stats["ci95_low"] <= stats["mean_difference"] <= stats["ci95_high"]
    assert stats["cliffs_delta"] > 0.5  # a dominates b
    assert abs(cliffs_delta(a, a)) == 0.0

    holm = holm_bonferroni({"h1": 0.001, "h2": 0.04, "h3": 0.9, "h4": float("nan")})
    # Monotone non-decreasing in raw-p order, and adjusted >= raw.
    assert holm["h1"]["p_holm"] <= holm["h2"]["p_holm"] <= holm["h3"]["p_holm"]
    assert holm["h1"]["p_holm"] >= 0.001
    assert holm["h1"]["reject_0.05"] is True
    assert holm["h3"]["reject_0.05"] is False
    assert np.isnan(holm["h4"]["p_holm"])


def test_leave_one_attack_out():  # 2-A
    torch.manual_seed(0)
    dist = DifferentiableDistortion(sr=16000, vae=None)
    clean = torch.randn(3, 3200) * 0.02
    wm = clean + torch.randn_like(clean) * 3e-4
    res = leave_one_attack_out_spearman(
        clean.unsqueeze(1), wm.unsqueeze(1), dist,
        ["noise", "lowpass", "resample", "spectral_proxy"], base_seed=42,
    )
    assert set(res) == {"noise", "lowpass", "resample", "spectral_proxy"}
    summary = summarize(res)
    assert "_overall" in summary
    assert summary["_overall"]["n_valid"] > 0


def test_m3_finite_difference_runs():  # 2-B
    torch.manual_seed(0)
    dist = DifferentiableDistortion(sr=16000, vae=None)
    wav = torch.randn(2, 1, 3200) * 0.02
    residual = torch.randn(2, 1, 3200) * 1e-3
    msg = torch.randint(0, 2, (2, 16))
    ref = torch.abs(stft_audio(residual.squeeze(1), 256, 64))
    rows = compute_finite_difference_utility_topk(
        MockAlignMark(), wav, residual, msg, dist, ["noise", "lowpass"],
        reference_map=ref, num_bins=8, epsilon=0.05, base_seed=1,
    )
    assert len(rows) == 2
    for row in rows:
        assert row["n_bins"] == 8
        assert "spearman_m2_m3" in row and "sign_agreement" in row


def test_cache_gather():  # 3-A
    cache = {"a": torch.ones(5, 7), "b": torch.zeros(5, 7)}
    stacked = _gather_cached_survival(cache, ["a", "b"], torch.device("cpu"), torch.float32)
    assert stacked.shape == (2, 5, 7)
    assert _gather_cached_survival(cache, ["a", "missing"], torch.device("cpu"), torch.float32) is None


class _FakeArgs:
    """Minimal args stand-in for apply_eval_attack's external-adapter branches."""

    def __init__(self, **overrides):
        self.mp3_bitrate = "64k"
        self.clearervoice_command = None
        self.clearervoice_snr = 20.0
        self.facodec_command = None
        self.encodec_command = None
        self.dac_command = None
        self.vocos_command = None
        self.hifigan_command = None
        for key, value in overrides.items():
            setattr(self, key, value)


def test_phase1_phase2_share_attack_dispatch():  # refactor: dedup phase1/phase2 attack tables
    """phase1_attribution.py and phase2_training.py must resolve to the exact same
    experiment_utils dispatch functions (no re-forked copies), and the dispatch behavior
    itself (attack name coverage, family mapping) must match what both files had before
    the shared _apply_internal_attack/apply_eval_attack existed."""
    assert phase1_attribution._apply_internal_attack is experiment_utils.apply_internal_attack
    assert phase2_training._internal_attack is experiment_utils.apply_internal_attack
    assert phase1_attribution.apply_eval_attack is experiment_utils.apply_eval_attack
    assert phase2_training.apply_eval_attack is experiment_utils.apply_eval_attack

    # Attack name coverage must match the original hard-coded sets from both files.
    expected_internal = {
        "clean", "identity", "noise", "noise10db", "lowpass", "bandpass", "resample",
        "speechtokenizer_nq6", "speechtokenizer_nq8", "strong_speechtokenizer", "spectral_proxy",
        "masking", "replacement", "frame_shuffle",
    }
    assert set(INTERNAL_ATTACK_NAMES) == expected_internal

    # Unknown attack names still raise, exactly as before.
    dist = DifferentiableDistortion(sr=16000, vae=None)
    wav = torch.randn(1, 1, 3200) * 0.02
    try:
        apply_internal_attack(wav, "not_a_real_attack", dist, seed=0)
        assert False, "expected ValueError for unknown internal attack"
    except ValueError:
        pass
    try:
        apply_eval_attack(wav, "not_a_real_attack", dist, seed=0, args=_FakeArgs())
        assert False, "expected ValueError for unknown eval attack"
    except ValueError:
        pass

    # Missing command args must still raise for adapters with no in-process fallback
    # (facodec/clearervoice/dac/hifigan). encodec/vocos are exempt: with no override
    # command they now fall back to the in-process codec path instead of raising
    # (see test_encodec_vocos_inprocess_dispatch).
    for attack_name, attr in [
        ("facodec", "facodec_command"),
        ("clearervoice", "clearervoice_command"),
        ("dac", "dac_command"),
    ]:
        try:
            apply_eval_attack(wav, attack_name, dist, seed=0, args=_FakeArgs())
            assert False, f"expected ValueError for {attack_name} without --{attr}"
        except ValueError:
            pass

    # attack_family mapping is untouched by the refactor.
    assert attack_family("noise10db") == "awgn"
    assert attack_family("speechtokenizer_nq6") == "speechtokenizer"
    assert attack_family("facodec_proxy") == "speechtokenizer"
    assert attack_family("ffmpeg_mp3") == "real_mp3"

    # Same seed => identical output regardless of which module's alias is used
    # (they are the same function object, but this also pins actual numeric behavior).
    seed = 7
    out_via_p1 = phase1_attribution._apply_internal_attack(wav, "noise", dist, seed)
    out_via_p2 = phase2_training._internal_attack(wav, "noise", dist, seed)
    assert torch.equal(out_via_p1, out_via_p2)


class _MockEncodecModel:
    """Stand-in for `encodec.EncodecModel`. `decode()` is an identity pass-through of the
    original audio (stashed in the unused "scale" slot of the (codes, scale) tuple) so
    `encodec_roundtrip_batch`'s shape checks still work. `encode()` fabricates codes shaped
    (B, K, T) with K deliberately != any batch size used below (matching encodec's own
    documented "codes is [B, K, T]" convention), so a caller that mixes up the batch and
    codebook axes before calling Vocos's `codes_to_features` is caught below."""

    sample_rate = 24000
    load_count = 0
    num_codebooks = 8

    def __init__(self):
        type(self).load_count += 1

    def encode(self, wav):
        batch, _, time = wav.shape
        codes = torch.zeros(batch, self.num_codebooks, time, dtype=torch.long)
        return [(codes, wav)]

    def decode(self, encoded_frames):
        _, wav = encoded_frames[0]
        return wav


class _MockVocosModel:
    """Stand-in for `vocos.Vocos` that reproduces two real shape contracts so regressions
    fail here instead of only on a real GPU run:

    1. `codes_to_features` (vocos/pretrained.py) expects codes shaped (K, T) or (K, B, L)
       -- codebook count first. Passing raw encodec's (B, K, T) codes un-transposed makes
       the codebook axis look like the batch axis, so the returned "batch" size becomes K
       instead of the true B. This mock encodes only that shape contract (codes.shape[1]
       becomes the output's batch dim), not the real embedding-table math.
    2. `AdaLayerNorm` (vocos/modules.py) broadcasts `cond_embedding_id` against the whole
       batch: `scale = self.scale(cond_embedding_id)` has shape (len(bandwidth_id), dim),
       multiplied against features of shape (B, T, dim). Only a single shared id (shape
       (1,)) broadcasts correctly for any batch size; one id per sample (shape (B,)) does
       not, once B != T.
    """

    load_count = 0

    def __init__(self):
        type(self).load_count += 1

    def codes_to_features(self, codes):
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)  # (K, T) -> (K, 1, T)
        _, batch, time = codes.shape
        channels = 5
        return codes.new_zeros(batch, channels, time, dtype=torch.float32)

    def decode(self, features, bandwidth_id=None):
        x = features.transpose(1, 2)  # (B, C, T) -> (B, T, C), as VocosBackbone.forward does
        dim = x.shape[-1]
        scale = torch.ones(bandwidth_id.shape[0], dim)
        shift = torch.zeros(bandwidth_id.shape[0], dim)
        x = x * scale + shift
        return x.mean(dim=-1)


def test_encodec_vocos_inprocess_dispatch():  # in-process Encodec/Vocos (avoid per-sample subprocess reload)
    """Encodec/Vocos previously ran via tools/run_encodec.py|run_vocos.py through
    external_attacks.command_roundtrip_batch: one subprocess + full model reload per
    audio sample. inprocess_attacks.py must instead load each model exactly once per
    process and reuse it for every batch. This mocks the (unavailable here) encodec/vocos
    packages to verify the caching contract, tensor-shape handling, and that
    apply_eval_attack routes to the in-process path by default while still honoring an
    explicit --encodec_command/--vocos_command override."""
    inprocess_attacks._MODEL_CACHE.clear()
    _MockEncodecModel.load_count = 0
    _MockVocosModel.load_count = 0

    original_encodec_loader = inprocess_attacks._load_encodec_model
    original_vocos_loader = inprocess_attacks._load_vocos_model
    inprocess_attacks._load_encodec_model = lambda device: _MockEncodecModel()
    inprocess_attacks._load_vocos_model = lambda device: _MockVocosModel()
    try:
        device = torch.device("cpu")

        inprocess_attacks.prewarm(device)
        assert _MockEncodecModel.load_count == 1
        assert _MockVocosModel.load_count == 1

        wav_2d = torch.randn(3, 1600)
        out_2d = inprocess_attacks.encodec_roundtrip_batch(wav_2d, device=device, sample_rate=24000)
        assert out_2d.shape == wav_2d.shape

        wav_3d = wav_2d.unsqueeze(1)
        out_3d = inprocess_attacks.encodec_roundtrip_batch(wav_3d, device=device, sample_rate=24000)
        assert out_3d.shape == wav_3d.shape
        assert torch.allclose(out_3d.squeeze(1), out_2d)

        out_vocos = inprocess_attacks.vocos_roundtrip_batch(wav_2d, device=device, sample_rate=24000)
        assert out_vocos.shape == wav_2d.shape

        # Repeated calls must hit the cache, not reload the model.
        inprocess_attacks.encodec_roundtrip_batch(wav_2d, device=device, sample_rate=24000)
        inprocess_attacks.vocos_roundtrip_batch(wav_2d, device=device, sample_rate=24000)
        assert _MockEncodecModel.load_count == 1
        assert _MockVocosModel.load_count == 1

        # apply_eval_attack must default to the in-process path when no override command is set.
        out_via_dispatch = apply_eval_attack(wav_3d, "encodec", distorter=None, seed=0, args=_FakeArgs())
        assert out_via_dispatch.shape == wav_3d.shape
        assert _MockEncodecModel.load_count == 1

        out_via_dispatch_vocos = apply_eval_attack(wav_3d, "vocos", distorter=None, seed=0, args=_FakeArgs())
        assert out_via_dispatch_vocos.shape == wav_3d.shape
        assert _MockVocosModel.load_count == 1

        # An explicit override command must still take the old subprocess path (backward compat).
        original_command_roundtrip = external_attacks.command_roundtrip_batch
        seen_commands = {}

        def fake_command_roundtrip(wav, command, sample_rate=16000):
            seen_commands["command"] = command
            return wav

        external_attacks.command_roundtrip_batch = fake_command_roundtrip
        try:
            override_args = _FakeArgs(
                encodec_command="python tools/run_encodec.py --input {input} --output {output}"
            )
            apply_eval_attack(wav_3d, "encodec", distorter=None, seed=0, args=override_args)
            assert seen_commands.get("command") == override_args.encodec_command
        finally:
            external_attacks.command_roundtrip_batch = original_command_roundtrip
    finally:
        inprocess_attacks._load_encodec_model = original_encodec_loader
        inprocess_attacks._load_vocos_model = original_vocos_loader
        inprocess_attacks._MODEL_CACHE.clear()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"All {len(tests)} revision regression tests passed.")


if __name__ == "__main__":
    main()
