# -*- coding: utf-8 -*-
"""Asset-free regression tests for the revision work (items 1-A..3-A).

Run: python test_revisions.py   (CPU only, no weights/datasets required)
"""

from __future__ import annotations

import numpy as np
import torch

import experiment_utils
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

    # Missing command args must still raise (facodec/clearervoice/encodec/etc guards preserved).
    for attack_name, attr in [
        ("facodec", "facodec_command"),
        ("clearervoice", "clearervoice_command"),
        ("encodec", "encodec_command"),
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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"All {len(tests)} revision regression tests passed.")


if __name__ == "__main__":
    main()
