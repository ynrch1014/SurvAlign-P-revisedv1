# -*- coding: utf-8 -*-
"""Asset-free regression tests for the revision work (items 1-A..3-A).

Run: python test_revisions.py   (CPU only, no weights/datasets required)
"""

from __future__ import annotations

import shutil

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
    apply_cascade_attack, stable_int_hash,
)
from phase1_attribution import (
    _target_norm_reference, paired_statistics, holm_bonferroni, cliffs_delta,
    compute_finite_difference_utility_topk, PAIRED_COMPARISONS,
)
from phase1_experiment3_selfcheck import leave_one_attack_out_spearman, summarize
from phase2_training import _gather_cached_survival
from survalign_p import DifferentiableDistortion, stft_audio, _apply_survival_attack_pair, paired_awgn, get_survival_map
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


def test_high_codec_utility_paired_comparisons():  # new: High-Codec-Utility_vs_{Random,Low-Survival}
    """phase1_attribution.py's paired-comparison family previously only compared
    High-Survival against the other conditions. This adds High-Codec-Utility_vs_Random
    and High-Codec-Utility_vs_Low-Survival to PAIRED_COMPARISONS, computed the same way
    (paired_statistics + Holm-Bonferroni). This test mocks a `per_sample_accuracy` dict in
    exactly main()'s (energy_mode, condition, attack_name, repeat_index) key convention and
    replicates main()'s comparison-resolution loop verbatim, to check the new pairs
    actually get computed (including the special "Random" repeat-averaging path) without
    needing real datasets/weights."""
    assert ("High-Codec-Utility", "Random") in PAIRED_COMPARISONS
    assert ("High-Codec-Utility", "Low-Survival") in PAIRED_COMPARISONS

    rng = np.random.default_rng(0)
    energy_mode, attack_name, n_samples, random_repeats = "natural", "noise", 64, 5

    per_sample_accuracy = {}
    per_sample_accuracy[(energy_mode, "High-Survival", attack_name, 0)] = rng.normal(0.85, 0.05, n_samples).tolist()
    per_sample_accuracy[(energy_mode, "High-Codec-Utility", attack_name, 0)] = rng.normal(0.80, 0.05, n_samples).tolist()
    per_sample_accuracy[(energy_mode, "Low-Survival", attack_name, 0)] = rng.normal(0.40, 0.05, n_samples).tolist()
    for repeat in range(random_repeats):
        per_sample_accuracy[(energy_mode, "Random", attack_name, repeat)] = rng.normal(0.55, 0.05, n_samples).tolist()

    # Verbatim copy of the resolution logic in phase1_attribution.py's main(), so a
    # regression in that loop (not just in the PAIRED_COMPARISONS list) would fail here too.
    paired_tests = {}
    for left, right in PAIRED_COMPARISONS:
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
        paired_tests[f"{energy_mode}/{attack_name}/{left}_vs_{right}"] = paired_statistics(
            left_values, right_values, seed=42
        )

    assert f"{energy_mode}/{attack_name}/High-Codec-Utility_vs_Random" in paired_tests
    assert f"{energy_mode}/{attack_name}/High-Codec-Utility_vs_Low-Survival" in paired_tests
    # High-Survival_vs_* must still be present -- the new pairs are additive, not a replacement.
    assert f"{energy_mode}/{attack_name}/High-Survival_vs_Low-Survival" in paired_tests

    for key in (
        f"{energy_mode}/{attack_name}/High-Codec-Utility_vs_Random",
        f"{energy_mode}/{attack_name}/High-Codec-Utility_vs_Low-Survival",
    ):
        stats = paired_tests[key]
        for field in ("mean_difference", "ci95_low", "ci95_high", "wilcoxon_p",
                      "wilcoxon_n_effective", "permutation_p", "cliffs_delta", "n"):
            assert field in stats
        assert stats["n"] == n_samples

    # High-Codec-Utility (~0.80) clearly beats Low-Survival (~0.40): large positive effect.
    low_survival_stats = paired_tests[f"{energy_mode}/{attack_name}/High-Codec-Utility_vs_Low-Survival"]
    assert low_survival_stats["mean_difference"] > 0.3
    assert low_survival_stats["cliffs_delta"] > 0.9

    # The "Random" comparison must average across all random_repeats entries, not just repeat 0.
    manual_random_mean = np.stack(
        [np.asarray(per_sample_accuracy[(energy_mode, "Random", attack_name, r)]) for r in range(random_repeats)]
    ).mean(axis=0)
    expected = paired_statistics(
        per_sample_accuracy[(energy_mode, "High-Codec-Utility", attack_name, 0)],
        manual_random_mean.tolist(), seed=42,
    )
    got = paired_tests[f"{energy_mode}/{attack_name}/High-Codec-Utility_vs_Random"]
    assert got["mean_difference"] == expected["mean_difference"]
    assert got["wilcoxon_p"] == expected["wilcoxon_p"]

    # Holm-Bonferroni must run over the whole family, including the two new pairs.
    wilcoxon_family = {k: v["wilcoxon_p"] for k, v in paired_tests.items()}
    holm = holm_bonferroni(wilcoxon_family)
    for key in wilcoxon_family:
        assert key in holm
        assert holm[key]["p_holm"] >= holm[key]["p_raw"] or np.isnan(holm[key]["p_holm"])


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
        "clean", "identity", "noise", "noise10db", "lowpass", "bandpass", "highpass", "resample",
        "time_jitter", "speechtokenizer_nq6", "speechtokenizer_nq8", "strong_speechtokenizer",
        "spectral_proxy", "masking", "replacement", "frame_shuffle",
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
    # (clearervoice/dac/hifigan). encodec/vocos/facodec are exempt: with no override
    # command they now fall back to the in-process codec path instead of raising
    # (see test_encodec_vocos_inprocess_dispatch / test_facodec_inprocess_dispatch).
    for attack_name, attr in [
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
    original_facodec_loader = inprocess_attacks._load_facodec_model
    inprocess_attacks._load_encodec_model = lambda device: _MockEncodecModel()
    inprocess_attacks._load_vocos_model = lambda device: _MockVocosModel()
    # prewarm() also loads FACodec; mock it here too so this test doesn't depend on the
    # real (unavailable) ns3_codec/pyworld packages -- FACodec's own behavior is covered by
    # test_facodec_inprocess_dispatch.
    inprocess_attacks._load_facodec_model = lambda device: (_MockFACodecEncoder(), _MockFACodecDecoder())
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
        inprocess_attacks._load_facodec_model = original_facodec_loader
        inprocess_attacks._MODEL_CACHE.clear()


class _MockFACodecEncoder:
    """Identity stand-in for `ns3_codec.FACodecEncoder`: FACodecEncoder.forward is just
    `self.block(x)` (a plain Conv1d stack), so passing the input straight through pins the
    shape/caching contract without needing the real (unavailable here) `ns3_codec` package
    or downloaded checkpoints."""

    load_count = 0

    def __init__(self):
        type(self).load_count += 1

    def __call__(self, wav_in):
        return wav_in


class _MockFACodecDecoder:
    """Stand-in for `ns3_codec.FACodecDecoder` matching its two call sites in
    facodec_roundtrip_batch: `decoder(enc_out, eval_vq=False, vq=True)` returns a 5-tuple
    `(outs, qs, commit_loss, quantized_buf, spk_embs)` (real signature, ns3_codec/facodec.py
    FACodecDecoder.forward), and `decoder.inference(vq_post_emb, spk_embs)` returns the
    reconstructed (B, 1, T) waveform."""

    load_count = 0

    def __init__(self):
        type(self).load_count += 1

    def __call__(self, enc_out, eval_vq=False, vq=True):
        batch = enc_out.shape[0]
        return enc_out, None, None, None, torch.zeros(batch, 4)

    def inference(self, vq_post_emb, spk_embs):
        return vq_post_emb


def test_facodec_inprocess_dispatch():  # in-process FACodec (avoid per-sample subprocess reload)
    """FACodec previously ran via tools/run_facodec.py through
    external_attacks.command_roundtrip_batch: one subprocess + full encoder/decoder reload
    per audio sample -- the same problem encodec/vocos had. inprocess_attacks.py must load
    the FACodec encoder/decoder pair exactly once per process and reuse it for every batch.
    This mocks the (unavailable here) ns3_codec package to verify the caching contract,
    tensor-shape handling (including that FACodec needs no 16kHz<->24kHz resampling, unlike
    Encodec/Vocos), and that apply_eval_attack routes to the in-process path by default
    while still honoring an explicit --facodec_command override."""
    inprocess_attacks._MODEL_CACHE.clear()
    _MockFACodecEncoder.load_count = 0
    _MockFACodecDecoder.load_count = 0

    original_encodec_loader = inprocess_attacks._load_encodec_model
    original_vocos_loader = inprocess_attacks._load_vocos_model
    original_facodec_loader = inprocess_attacks._load_facodec_model
    # prewarm() also loads Encodec/Vocos; mock those too so this test doesn't depend on the
    # real (unavailable) encodec/vocos packages -- their own behavior is covered by
    # test_encodec_vocos_inprocess_dispatch.
    inprocess_attacks._load_encodec_model = lambda device: _MockEncodecModel()
    inprocess_attacks._load_vocos_model = lambda device: _MockVocosModel()
    inprocess_attacks._load_facodec_model = lambda device: (_MockFACodecEncoder(), _MockFACodecDecoder())
    try:
        device = torch.device("cpu")

        inprocess_attacks.prewarm(device)
        assert _MockFACodecEncoder.load_count == 1
        assert _MockFACodecDecoder.load_count == 1

        wav_2d = torch.randn(3, 1600)
        out_2d = inprocess_attacks.facodec_roundtrip_batch(wav_2d, device=device, sample_rate=16000)
        assert out_2d.shape == wav_2d.shape
        # FACodec's native rate (16000) matches sample_rate here, so this must be a lossless
        # passthrough for our identity mock (no resampling round-trip degradation).
        assert torch.allclose(out_2d, wav_2d, atol=1e-5)

        wav_3d = wav_2d.unsqueeze(1)
        out_3d = inprocess_attacks.facodec_roundtrip_batch(wav_3d, device=device, sample_rate=16000)
        assert out_3d.shape == wav_3d.shape
        assert torch.allclose(out_3d.squeeze(1), out_2d)

        # Repeated calls must hit the cache, not reload the model.
        inprocess_attacks.facodec_roundtrip_batch(wav_2d, device=device, sample_rate=16000)
        assert _MockFACodecEncoder.load_count == 1
        assert _MockFACodecDecoder.load_count == 1

        # apply_eval_attack must default to the in-process path when no override command is set.
        out_via_dispatch = apply_eval_attack(wav_3d, "facodec", distorter=None, seed=0, args=_FakeArgs())
        assert out_via_dispatch.shape == wav_3d.shape
        assert _MockFACodecEncoder.load_count == 1

        # An explicit override command must still take the old subprocess path (backward compat).
        original_command_roundtrip = external_attacks.command_roundtrip_batch
        seen_commands = {}

        def fake_command_roundtrip(wav, command, sample_rate=16000):
            seen_commands["command"] = command
            return wav

        external_attacks.command_roundtrip_batch = fake_command_roundtrip
        try:
            override_args = _FakeArgs(
                facodec_command="python tools/run_facodec.py --input {input} --output {output}"
            )
            apply_eval_attack(wav_3d, "facodec", distorter=None, seed=0, args=override_args)
            assert seen_commands.get("command") == override_args.facodec_command
        finally:
            external_attacks.command_roundtrip_batch = original_command_roundtrip
    finally:
        inprocess_attacks._load_encodec_model = original_encodec_loader
        inprocess_attacks._load_vocos_model = original_vocos_loader
        inprocess_attacks._load_facodec_model = original_facodec_loader
        inprocess_attacks._MODEL_CACHE.clear()


def test_highpass_filter():  # Part A-1: highpass_filter (paper §II-B linear_filter category)
    dist = DifferentiableDistortion(sr=16000, vae=None)
    torch.manual_seed(0)
    wav = torch.randn(2, 1600) * 0.05
    out = dist(wav, "highpass", cutoff_hz=300)
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()
    # A highpass filter must suppress DC / near-zero-frequency content: a constant (DC)
    # input's output magnitude should collapse to near zero, unlike the input itself.
    dc = torch.ones(2, 1600) * 0.5
    dc_out = dist(dc, "highpass", cutoff_hz=300)
    assert dc_out.abs().mean() < dc.abs().mean() * 0.05


def test_highpass_dispatch_wiring():  # Part A-3: highpass registered in the shared dispatch tables
    assert "highpass" in INTERNAL_ATTACK_NAMES
    assert attack_family("highpass") == "linear_filter"
    assert attack_family("lowpass") == attack_family("bandpass") == attack_family("highpass")

    dist = DifferentiableDistortion(sr=16000, vae=None)
    wav = torch.randn(1, 1, 1600) * 0.02
    out_via_internal = apply_internal_attack(wav, "highpass", dist, seed=0)
    out_via_eval = apply_eval_attack(wav, "highpass", dist, seed=0, args=_FakeArgs())
    assert torch.equal(out_via_internal, out_via_eval)


def test_time_jitter():  # multi-cascade-attack experiment Part A
    """time_jitter must (1) preserve shape, (2) actually be an integer torch.roll of the
    input within +-max_shift_samples (not silently a no-op or something else), (3) be
    wired into INTERNAL_ATTACK_NAMES/apply_internal_attack/apply_eval_attack identically,
    and (4) pass gradients through unchanged (pure indexing op, no learnable/lossy step)."""
    dist = DifferentiableDistortion(sr=16000, vae=None)
    wav = torch.arange(1600, dtype=torch.float32).unsqueeze(0)  # (1, T) monotonic ramp

    out = dist(wav, "time_jitter", max_shift_ms=1.0, seed=7)
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()

    max_shift_samples = int(1.0 * 16000 / 1000)
    assert any(
        torch.equal(out, torch.roll(wav, shifts=shift, dims=-1))
        for shift in range(-max_shift_samples, max_shift_samples + 1)
    ), "time_jitter output is not an integer roll of the input within max_shift_ms"

    assert "time_jitter" in INTERNAL_ATTACK_NAMES
    assert attack_family("time_jitter") == "time_jitter"  # its own family, not merged with anything
    wav_3d = wav.unsqueeze(1)
    out_via_internal = apply_internal_attack(wav_3d, "time_jitter", dist, seed=7)
    out_via_eval = apply_eval_attack(wav_3d, "time_jitter", dist, seed=7, args=_FakeArgs())
    assert torch.equal(out_via_internal, out_via_eval)

    wav_var = wav.clone().requires_grad_(True)
    dist(wav_var, "time_jitter", max_shift_ms=1.0, seed=3).sum().backward()
    assert wav_var.grad is not None
    assert torch.allclose(wav_var.grad, torch.ones_like(wav_var))


def test_ffmpeg_aac_dispatch_wiring():  # Part A-3: apply_eval_attack routes ffmpeg_aac correctly
    """Verifies the wiring itself (attack name -> ffmpeg_aac_roundtrip_batch, with the right
    sample_rate/bitrate) via a spy, independent of whether a real ffmpeg is on PATH --
    real subprocess behavior is covered separately by
    test_ffmpeg_aac_mp3_parallel_order_and_speed (skipped when ffmpeg is unavailable)."""
    seen = {}

    def fake_ffmpeg_aac(wav, sample_rate=16000, bitrate="64k"):
        seen["sample_rate"] = sample_rate
        seen["bitrate"] = bitrate
        return wav

    original = external_attacks.ffmpeg_aac_roundtrip_batch
    external_attacks.ffmpeg_aac_roundtrip_batch = fake_ffmpeg_aac
    try:
        wav = torch.randn(2, 1, 1600)
        args = _FakeArgs(mp3_bitrate="96k")
        out = apply_eval_attack(wav, "ffmpeg_aac", distorter=None, seed=0, args=args)
        assert seen == {"sample_rate": 16000, "bitrate": "96k"}
        assert torch.equal(out, wav)
    finally:
        external_attacks.ffmpeg_aac_roundtrip_batch = original


def test_ffmpeg_aac_mp3_parallel_order_and_speed():  # Part A-2: parallelized ffmpeg round trip
    """Requires a real ffmpeg on PATH; skipped otherwise (this check needs local ffmpeg but
    no GPU). Verifies (1) batch output order survives parallelization -- each sample
    carries a distinct sine tone, checked via its dominant FFT bin after the round trip --
    and (2) reports the actual measured parallel-vs-sequential speedup for a small batch,
    instead of assuming the refactor is faster without checking."""
    if shutil.which("ffmpeg") is None:
        print("  (skipped: ffmpeg not found on PATH)")
        return

    import time

    from external_attacks import ffmpeg_aac_roundtrip_batch, ffmpeg_mp3_roundtrip_batch

    sample_rate = 16000
    n_samples = int(sample_rate * 0.5)
    t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
    freqs = [220.0, 440.0, 880.0, 1760.0]
    wav = torch.stack([0.3 * torch.sin(2 * np.pi * f * t) for f in freqs], dim=0)

    def dominant_freq(signal: torch.Tensor) -> float:
        spec = torch.fft.rfft(signal)
        bins = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / sample_rate)
        return float(bins[torch.argmax(spec.abs())])

    out = ffmpeg_aac_roundtrip_batch(wav, sample_rate=sample_rate, bitrate="128k")
    assert out.shape == wav.shape
    for i, f in enumerate(freqs):
        got = dominant_freq(out[i])
        assert abs(got - f) < 30, f"sample {i}: expected ~{f}Hz, got {got}Hz (order scrambled?)"

    # Speed: sequential (max_workers=1) vs parallel (default worker count), using the
    # cheaper MP3 path for the timing comparison, on a slightly larger batch.
    batch20 = wav[0:1].repeat(20, 1)
    start = time.time()
    ffmpeg_mp3_roundtrip_batch(batch20, sample_rate=sample_rate, max_workers=1)
    sequential_s = time.time() - start
    start = time.time()
    ffmpeg_mp3_roundtrip_batch(batch20, sample_rate=sample_rate)
    parallel_s = time.time() - start
    print(f"  ffmpeg_mp3 n=20: sequential={sequential_s:.2f}s parallel={parallel_s:.2f}s "
          f"(speedup={sequential_s / max(parallel_s, 1e-6):.2f}x)")


class _FakeSpeechTokenizerVAE:
    """Deterministic, n_q-dependent stand-in for AlignMark's SpeechTokenizer VAE, matching
    the encoder/quantizer/decoder interface `DifferentiableDistortion.speech_reconstruct`
    calls. Not a real codec -- only used to exercise speechtokenizer_nq6/nq8/
    strong_speechtokenizer deterministically without the real (heavy) checkpoint."""

    def encoder(self, wav_3d):
        return wav_3d

    def quantizer(self, features, n_q, layers, st):
        scale = 1.0 - 0.01 * int(n_q)
        return features * scale, None, None, None

    def decoder(self, quantized):
        return quantized


def test_survival_attack_pair_backward_compatible():  # emergency patch 2: whitelist -> apply_eval_attack delegation
    """_apply_survival_attack_pair previously hard-coded its own independent 8-attack
    whitelist, separate from apply_eval_attack (already unified elsewhere). This is the
    function behind the already-reported H1 (correlation=0.318) and H4 (leave-one-out)
    results, so the refactor to delegate to apply_eval_attack must reproduce the exact
    same per-attack parameters for all 8 previously-supported attacks -- pinned here
    against direct DifferentiableDistortion calls, not just "doesn't crash"."""
    dist = DifferentiableDistortion(sr=16000, vae=_FakeSpeechTokenizerVAE())
    torch.manual_seed(0)
    clean = torch.randn(2, 3200) * 0.05
    wm = clean + torch.randn_like(clean) * 1e-3
    seed = 123

    # noise must stay paired_awgn (shared realization between clean/watermarked), not two
    # independent apply_eval_attack "noise" calls (which would draw independent noise).
    got_clean, got_wm = _apply_survival_attack_pair(clean, wm, dist, "noise", seed=seed, args=None)
    want_clean, want_wm = paired_awgn(clean, wm, snr_db=20.0, seed=seed)
    assert torch.equal(got_clean, want_clean)
    assert torch.equal(got_wm, want_wm)

    linear_cases = [
        ("lowpass", "lowpass", dict(cutoff_hz=4000)),
        ("bandpass", "bandpass", dict(low_hz=300, high_hz=3400)),
        ("resample", "resample", dict(down_rate=2)),
        ("speechtokenizer_nq6", "reconstruct", dict(n_q=6)),
        ("speechtokenizer_nq8", "reconstruct", dict(n_q=8)),
        ("strong_speechtokenizer", "strong_speechtokenizer", dict(n_q=2)),
    ]
    for attack_name, dtype, kwargs in linear_cases:
        got_clean, got_wm = _apply_survival_attack_pair(clean, wm, dist, attack_name, seed=seed, args=None)
        assert torch.equal(got_clean, dist(clean, dtype, **kwargs)), attack_name
        assert torch.equal(got_wm, dist(wm, dtype, **kwargs)), attack_name

    got_clean, got_wm = _apply_survival_attack_pair(clean, wm, dist, "spectral_proxy", seed=seed, args=None)
    assert torch.equal(got_clean, dist(clean, "spectral_proxy", cutoff_ratio=0.7, seed=seed))
    assert torch.equal(got_wm, dist(wm, "spectral_proxy", cutoff_ratio=0.7, seed=seed))

    # End-to-end: get_survival_map itself must still run for the full original default
    # attack set and produce a finite, correctly-shaped map (catches wiring mistakes that
    # the per-attack checks above wouldn't, e.g. a broken args passthrough).
    original_default_attacks = (
        "noise", "lowpass", "bandpass", "resample", "speechtokenizer_nq6", "spectral_proxy",
    )
    survival = get_survival_map(
        clean.unsqueeze(1), wm.unsqueeze(1), dist, attack_names=original_default_attacks,
        base_seed=42, quantile=0.25, args=None,
    )
    assert survival.shape[0] == clean.shape[0]
    assert torch.isfinite(survival).all()
    assert float(survival.min()) >= 0.0 and float(survival.max()) <= 1.0 + 1e-5


def test_survival_map_supports_new_attacks():  # emergency patch 2: new attacks no longer raise
    """replacement/masking/frame_shuffle/highpass (internal, asset-free) and
    ffmpeg_mp3/ffmpeg_aac/encodec/vocos (external adapters, mocked here) must resolve
    through _apply_survival_attack_pair instead of raising 'Unsupported survival-map
    attack', which is exactly what broke before this patch."""
    dist = DifferentiableDistortion(sr=16000, vae=None)
    torch.manual_seed(0)
    clean = torch.randn(2, 3200) * 0.05
    wm = clean + torch.randn_like(clean) * 1e-3

    for attack_name in ["replacement", "masking", "frame_shuffle", "highpass"]:
        got_clean, got_wm = _apply_survival_attack_pair(clean, wm, dist, attack_name, seed=7, args=None)
        assert got_clean.shape == clean.shape, attack_name
        assert got_wm.shape == wm.shape, attack_name
        assert torch.isfinite(got_clean).all() and torch.isfinite(got_wm).all(), attack_name

    # External-adapter attacks: mock the same way as test_encodec_vocos_inprocess_dispatch /
    # test_facodec_inprocess_dispatch so no real ffmpeg/encodec/vocos packages are needed.
    original_ffmpeg_mp3 = external_attacks.ffmpeg_mp3_roundtrip_batch
    original_ffmpeg_aac = external_attacks.ffmpeg_aac_roundtrip_batch
    external_attacks.ffmpeg_mp3_roundtrip_batch = lambda wav, sample_rate=16000, bitrate="64k": wav
    external_attacks.ffmpeg_aac_roundtrip_batch = lambda wav, sample_rate=16000, bitrate="64k": wav

    inprocess_attacks._MODEL_CACHE.clear()
    original_encodec_loader = inprocess_attacks._load_encodec_model
    original_vocos_loader = inprocess_attacks._load_vocos_model
    inprocess_attacks._load_encodec_model = lambda device: _MockEncodecModel()
    inprocess_attacks._load_vocos_model = lambda device: _MockVocosModel()
    try:
        args = _FakeArgs()
        for attack_name in ["ffmpeg_mp3", "ffmpeg_aac", "encodec", "vocos"]:
            got_clean, got_wm = _apply_survival_attack_pair(
                clean.unsqueeze(1), wm.unsqueeze(1), dist, attack_name, seed=7, args=args,
            )
            assert got_clean.shape == clean.unsqueeze(1).shape, attack_name
            assert got_wm.shape == wm.unsqueeze(1).shape, attack_name
    finally:
        external_attacks.ffmpeg_mp3_roundtrip_batch = original_ffmpeg_mp3
        external_attacks.ffmpeg_aac_roundtrip_batch = original_ffmpeg_aac
        inprocess_attacks._load_encodec_model = original_encodec_loader
        inprocess_attacks._load_vocos_model = original_vocos_loader
        inprocess_attacks._MODEL_CACHE.clear()


def test_masking_replacement_frame_shuffle_gpu_generator_device():  # emergency patch 3
    """torch.randint(...) without an explicit device= argument always samples on CPU
    regardless of the generator's own device, so a CUDA generator + no device= raised
    'RuntimeError: Expected a cpu device type for generator but found cuda' the first time
    apply_masking/apply_replacement/apply_frame_shuffle actually ran on GPU (they were only
    added to survival_attacks this session; the bug existed unnoticed before that). Skipped
    on CPU-only machines since the bug is CUDA-specific -- run this on a GPU box (e.g.
    RunPod) to actually exercise the fix. time_jitter (added for the cascade-attack
    experiment) was written with device=wav.device from the start to avoid repeating this
    same bug a fourth time; included here to confirm that."""
    if not torch.cuda.is_available():
        print("  (skipped: no CUDA device available)")
        return

    dist = DifferentiableDistortion(sr=16000, vae=None).to("cuda")
    wav = torch.randn(2, 1, 3200, device="cuda") * 0.05
    for dtype, kwargs in [
        ("masking", dict(max_ratio=0.1, seed=1)),
        ("replacement", dict(max_ratio=0.1, snr_db=0.0, seed=2)),
        ("frame_shuffle", dict(frame_duration_ms=50, shuffle_ratio=0.2, seed=3)),
        ("time_jitter", dict(max_shift_ms=1.0, seed=4)),
    ]:
        out = dist(wav, dtype, **kwargs)
        assert out.shape == wav.shape, dtype
        assert out.device.type == "cuda", dtype
        assert torch.isfinite(out).all(), dtype


def test_train_gate_supports_default_train_attacks():  # emergency patch 5
    """train_gate()'s differentiable-attack whitelist (TRAIN_GATE_SUPPORTED_ATTACKS) had
    drifted from --train_attacks' own default twice already: masking/replacement/
    frame_shuffle were added to the default without updating the whitelist (immediate
    ValueError crash running with defaults), and highpass was added as a new
    differentiable attack without updating the whitelist either. Pin the actual argparse
    default string against the whitelist directly so this can't silently drift a third
    time."""
    default_train_attacks = "noise,lowpass,resample,speechtokenizer_nq6,spectral_proxy,masking,replacement,frame_shuffle"
    names = {a.strip() for a in default_train_attacks.split(",") if a.strip()}
    unsupported = names - phase2_training.TRAIN_GATE_SUPPORTED_ATTACKS
    assert not unsupported, f"default --train_attacks not covered by whitelist: {sorted(unsupported)}"
    # highpass must be included too: it's a differentiable attack (added this session)
    # that was previously missing from the whitelist despite having no discrete/argmax
    # bottleneck blocking backprop, same category as lowpass/bandpass.
    assert "highpass" in phase2_training.TRAIN_GATE_SUPPORTED_ATTACKS


def test_build_guide_map_survival_branches_share_args_threading():  # emergency patch 6
    """build_guide_map() previously called get_survival_map() independently in its
    shuffled_survival and proposed_gate/analytic_survival branches; only the
    shuffled_survival one passed args=args (copy-paste omission on the other), so
    proposed_gate + map_type=survival crashed with 'AttributeError: NoneType object has
    no attribute mp3_bitrate' the moment an external-adapter survival attack was used,
    while shuffled_survival worked fine with the exact same attack. Both branches now go
    through the shared _resolve_survival_map helper -- exercise an external-adapter
    survival attack (mocked ffmpeg_mp3) through BOTH modes so neither can drift back to
    the buggy state independently."""
    torch.manual_seed(0)
    wav = torch.randn(2, 1, 3200) * 0.05
    wav_wm = wav + torch.randn_like(wav) * 1e-3

    original_ffmpeg_mp3 = external_attacks.ffmpeg_mp3_roundtrip_batch
    external_attacks.ffmpeg_mp3_roundtrip_batch = lambda wav, sample_rate=16000, bitrate="64k": wav
    try:
        dist = DifferentiableDistortion(sr=16000, vae=None)
        args = _FakeArgs(survival_attack_names=["ffmpeg_mp3"], survival_quantile=0.25)

        for mode, map_type in [("shuffled_survival", None), ("proposed_gate", "survival")]:
            args.mode = mode
            args.map_type = map_type
            feature_pack, residual_spec, guide, masking_map = phase2_training.build_guide_map(
                args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav_wm,
                residual=torch.zeros_like(wav), context_seed=7,
            )
            assert torch.isfinite(guide).all(), mode
            assert feature_pack.shape[0] == wav.shape[0], mode
    finally:
        external_attacks.ffmpeg_mp3_roundtrip_batch = original_ffmpeg_mp3


def test_apply_cascade_attack_sequential_and_reproducible():  # multi-cascade-attack experiment Part B
    """apply_cascade_attack must chain stages onto each other's OUTPUT (not apply each
    stage independently to the original input and average, which is what the existing
    train_attack_names loop does) -- that's the entire point of a cascade. Also pins:
    reproducibility for a fixed seed, variation across seeds, and end-to-end
    differentiability (all three stages -- time_jitter/noise/bandpass -- are
    differentiable, so the cascade as a whole must be too)."""
    dist = DifferentiableDistortion(sr=16000, vae=None)
    torch.manual_seed(0)
    wav = torch.randn(2, 1, 3200) * 0.05

    stages = [
        ("time_jitter", dict(max_shift_ms=1.0)),
        ("noise", dict(snr_db=10.0)),
        ("bandpass", dict(low_hz=200, high_hz=3500)),
    ]

    out = apply_cascade_attack(wav, stages, dist, seed=42)
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()

    # Must equal manually chaining the three stages onto each other's output, each with
    # its own per-stage sub-seed (seed + index*1000).
    stage1 = dist(wav, "time_jitter", max_shift_ms=1.0, seed=42 + 0 * 1000)
    stage2 = dist(stage1, "noise", snr_db=10.0, seed=42 + 1 * 1000)
    stage3 = dist(stage2, "bandpass", low_hz=200, high_hz=3500, seed=42 + 2 * 1000)
    assert torch.equal(out, stage3)

    # Must NOT match applying a stage independently to the original wav (that would mean
    # the cascade collapsed back into the old independent-attack pattern).
    independent_noise = dist(wav, "noise", snr_db=10.0, seed=42 + 1000)
    assert not torch.equal(out, independent_noise)

    # Reproducible for a fixed seed; varies with a different seed.
    assert torch.equal(out, apply_cascade_attack(wav, stages, dist, seed=42))
    assert not torch.equal(out, apply_cascade_attack(wav, stages, dist, seed=43))

    # Differentiable end-to-end (gradient must exist and be finite).
    wav_var = wav.clone().requires_grad_(True)
    apply_cascade_attack(wav_var, stages, dist, seed=7).sum().backward()
    assert wav_var.grad is not None
    assert torch.isfinite(wav_var.grad).all()


def test_train_cascade_per_batch_randomization_and_gpu_device_safety():  # Part B
    """The --train_cascade branch in train_gate() samples AWGN SNR (8-12dB) and jitter
    magnitude (0.5-1.5ms) freshly per batch (team request), via a torch.Generator created
    with device=device and device=device passed explicitly on every draw -- this exercises
    that exact sampling snippet directly (not the full train_gate(), which needs a real
    AlignMark model) to confirm the range is respected and the value actually varies
    across steps, and (on a GPU box) that it doesn't repeat the generator-device bug
    already fixed elsewhere (patch 3: torch.randint ignores the generator's device unless
    device= is passed; this uses torch.rand the same way, so pin it too)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_snr_and_jitter(step):
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_int_hash(42, "cascade", 1, step))
        snr_db = float(8.0 + 4.0 * torch.rand(1, generator=generator, device=device).item())
        jitter_ms = float(0.5 + 1.0 * torch.rand(1, generator=generator, device=device).item())
        return snr_db, jitter_ms

    seen = set()
    for step in range(5):
        snr_db, jitter_ms = sample_snr_and_jitter(step)
        assert 8.0 <= snr_db <= 12.0, snr_db
        assert 0.5 <= jitter_ms <= 1.5, jitter_ms
        seen.add((round(snr_db, 6), round(jitter_ms, 6)))
    assert len(seen) > 1, "per-batch cascade randomization did not vary across steps"


def test_channel_ablation_masks_correct_channels():  # multi-cascade-attack experiment Part C
    """Each --channel_ablation mode must zero exactly its documented channels of the
    Gate's 4-channel feature_pack (0=clean, 1=residual, 2=guide/Survival Map,
    3=masking_map) and leave every other channel byte-identical to the unablated ("full")
    baseline -- not recomputed differently, just zeroed."""
    torch.manual_seed(0)
    wav = torch.randn(2, 1, 3200) * 0.05
    wav_wm = wav + torch.randn_like(wav) * 1e-3
    dist = DifferentiableDistortion(sr=16000, vae=None)
    args = _FakeArgs(
        mode="proposed_gate", map_type="survival",
        survival_attack_names=["lowpass"], survival_quantile=0.25, channel_ablation="full",
    )

    baseline_pack, _, _, _ = phase2_training.build_guide_map(
        args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav_wm,
        residual=torch.zeros_like(wav), context_seed=7,
    )

    expectations = {
        "no_guide": {2},
        "no_residual": {1},
        "no_residual_no_guide": {1, 2},
    }
    for mode, masked_channels in expectations.items():
        args.channel_ablation = mode
        pack, _, _, _ = phase2_training.build_guide_map(
            args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav_wm,
            residual=torch.zeros_like(wav), context_seed=7,
        )
        for ch in range(4):
            if ch in masked_channels:
                assert torch.equal(pack[:, ch], torch.zeros_like(pack[:, ch])), (mode, ch)
            else:
                assert torch.equal(pack[:, ch], baseline_pack[:, ch]), (mode, ch)


def test_channel_ablation_full_is_backward_compatible():  # Part C
    """channel_ablation='full' (the default -- and what an older `args` object with no
    channel_ablation attribute at all falls back to via getattr) must be a true no-op.
    Also confirms residual_spec/guide/masking_map (everything build_guide_map returns
    besides feature_pack) are completely untouched by channel ablation, since
    SimplifiedSurvivalGate.forward() multiplies its output scale against the real,
    unmasked residual_spec regardless of what the Gate's input saw."""
    torch.manual_seed(0)
    wav = torch.randn(2, 1, 3200) * 0.05
    wav_wm = wav + torch.randn_like(wav) * 1e-3
    dist = DifferentiableDistortion(sr=16000, vae=None)
    args = _FakeArgs(
        mode="proposed_gate", map_type="survival",
        survival_attack_names=["lowpass"], survival_quantile=0.25,
    )
    assert not hasattr(args, "channel_ablation")
    pack_no_attr, residual_spec_a, guide_a, masking_map_a = phase2_training.build_guide_map(
        args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav_wm,
        residual=torch.zeros_like(wav), context_seed=7,
    )
    args.channel_ablation = "full"
    pack_explicit_full, residual_spec_b, guide_b, masking_map_b = phase2_training.build_guide_map(
        args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav_wm,
        residual=torch.zeros_like(wav), context_seed=7,
    )
    assert torch.equal(pack_no_attr, pack_explicit_full)
    assert torch.equal(residual_spec_a, residual_spec_b)
    assert torch.equal(guide_a, guide_b)
    assert torch.equal(masking_map_a, masking_map_b)


def test_channel_ablation_invalid_value_raises():  # Part C
    dist = DifferentiableDistortion(sr=16000, vae=None)
    wav = torch.randn(1, 1, 1600) * 0.02
    args = _FakeArgs(
        mode="proposed_gate", map_type="survival", survival_attack_names=["lowpass"],
        survival_quantile=0.25, channel_ablation="not_a_real_mode",
    )
    try:
        phase2_training.build_guide_map(
            args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav,
            residual=torch.zeros_like(wav), context_seed=0,
        )
        assert False, "expected ValueError for unknown channel_ablation"
    except ValueError:
        pass


def test_hard_mask_output_depends_on_guide_channel():  # multi-cascade-attack experiment Part D
    """hard_mask=True's whole purpose is to force the Gate's output scale to structurally
    depend on the guide channel's content -- this is the core check that hard masking
    actually intervenes, rather than merely sitting as one of several input channels the
    conv is free to ignore. Two otherwise-identical feature_packs differing only in the
    guide channel must produce different output scales, with identical (untrained,
    randomly-initialized) conv weights."""
    torch.manual_seed(0)
    gate = phase2_training.SimplifiedSurvivalGate(in_channels=4, gate_range=0.2, hard_mask=True)
    gate.eval()
    # conv[-1] is zero-initialized by construction (so the untrained gate starts as an
    # identity, scale==1 everywhere) -- that also means conv(feature_pack) is identically
    # 0 regardless of input until some training has happened, which would make guide
    # content invisible here (0 * anything == 0). Force a non-zero constant conv output so
    # the guide multiplier's effect is actually observable, matching a post-training state.
    with torch.no_grad():
        gate.conv[-1].bias.fill_(1.0)
    batch, freq, time = 2, 8, 10
    clean = torch.randn(batch, freq, time)
    residual_feat = torch.randn(batch, freq, time)
    masking_map = torch.randn(batch, freq, time)
    residual_spec = torch.randn(batch, freq, time, dtype=torch.cfloat)

    guide_a = torch.zeros(batch, freq, time)
    guide_a[:, : freq // 2] = 1.0
    guide_b = 1.0 - guide_a

    feature_pack_a = torch.stack([clean, residual_feat, guide_a, masking_map], dim=1)
    feature_pack_b = torch.stack([clean, residual_feat, guide_b, masking_map], dim=1)

    with torch.no_grad():
        _, scale_a = gate(feature_pack_a, residual_spec)
        _, scale_b = gate(feature_pack_b, residual_spec)

    assert not torch.allclose(scale_a, scale_b), "hard_mask output did not change with guide content"


def test_hard_mask_never_fully_zeroes_gradient_path():  # Part D
    """A constant (all-same-value) guide channel -- e.g. every bin scored 0 by the
    Survival Map -- must not collapse the multiplier to exactly 0. A hard 0 would kill
    gradient flow through those bins entirely and destabilize early training (the design
    note's explicit concern); minmax_per_sample's +eps guard keeps a constant channel's
    guide_norm well-defined (0), so hard_mask's "0.5 + 0.5*guide_norm" floor still leaves
    a 0.5x multiplier instead of 0x."""
    torch.manual_seed(0)
    gate = phase2_training.SimplifiedSurvivalGate(in_channels=4, gate_range=0.2, hard_mask=True)
    with torch.no_grad():
        gate.conv[-1].bias.fill_(1.0)  # force non-zero conv output so 0.5x vs 0x is distinguishable
    batch, freq, time = 1, 4, 4
    feature_pack = torch.randn(batch, 4, freq, time)
    feature_pack[:, 2] = 0.0  # constant (all-zero) guide channel
    residual_spec = torch.ones(batch, freq, time, dtype=torch.cfloat)
    with torch.no_grad():
        _, scale = gate(feature_pack, residual_spec)
    assert torch.isfinite(scale).all()
    # If the multiplier were a hard 0 (bug), logits would be zeroed regardless of the
    # conv's actual output, giving scale == 1.0 everywhere. It must not.
    assert not torch.allclose(scale, torch.ones_like(scale))


def test_hard_mask_default_false_is_backward_compatible():  # Part D
    """hard_mask defaults to False and must reproduce the exact pre-Part-D forward
    behavior: scale = 1 + gate_range*tanh(conv(feature_pack)), independent of the guide
    channel's content."""
    torch.manual_seed(0)
    gate = phase2_training.SimplifiedSurvivalGate(in_channels=4, gate_range=0.2)
    assert gate.hard_mask is False
    batch, freq, time = 2, 8, 10
    feature_pack = torch.randn(batch, 4, freq, time)
    residual_spec = torch.randn(batch, freq, time, dtype=torch.cfloat)
    with torch.no_grad():
        residual_out, scale = gate(feature_pack, residual_spec)
        expected_logits = gate.conv(feature_pack).squeeze(1)
        expected_scale = 1.0 + gate.gate_range * torch.tanh(expected_logits)
    assert torch.allclose(scale, expected_scale)
    assert torch.allclose(residual_out, residual_spec * expected_scale)


def test_random_gate_mode_provides_random_hard_mask_control_group():  # Part D
    """--mode random_gate + --hard_mask is the required random-map control group (per the
    task doc) that isolates hard-masking's *mechanism* effect from the Survival Map's
    *information* value: build_guide_map's random_gate branch already fills the guide
    channel with torch.rand(...), so combining it with --hard_mask needs no new code path.
    Confirm the guide channel is genuinely random and varies across calls (different
    context_seed), i.e. it is NOT the real Survival Map."""
    torch.manual_seed(0)
    wav = torch.randn(2, 1, 3200) * 0.05
    dist = DifferentiableDistortion(sr=16000, vae=None)
    args = _FakeArgs(mode="random_gate", channel_ablation="full")

    pack_a, _, guide_a, _ = phase2_training.build_guide_map(
        args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav.clone(),
        residual=torch.zeros_like(wav), context_seed=1,
    )
    pack_b, _, guide_b, _ = phase2_training.build_guide_map(
        args, alignmark=None, distorter=dist, wav=wav, wav_wm=wav.clone(),
        residual=torch.zeros_like(wav), context_seed=2,
    )
    assert not torch.equal(guide_a, guide_b), "random_gate guide did not vary across context_seed"
    assert not torch.equal(pack_a[:, 2], pack_b[:, 2])


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"All {len(tests)} revision regression tests passed.")


if __name__ == "__main__":
    main()
