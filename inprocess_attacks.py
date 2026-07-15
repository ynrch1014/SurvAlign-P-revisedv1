# -*- coding: utf-8 -*-
"""In-process Encodec/Vocos attack adapters.

`tools/run_encodec.py` / `tools/run_vocos.py`, invoked once per audio sample via
`external_attacks.command_roundtrip_batch` (a fresh subprocess + file I/O + full model
reload every single call), are correct but wasteful: the Encodec/Vocos weights get
reloaded from disk on every sample instead of once per process.

This module keeps the same codecs but loads each model exactly once per process (cached
in `_MODEL_CACHE`, keyed by (model name, device)) and operates directly on batched
tensors, with no subprocess or temporary files. `encodec_roundtrip_batch`/
`vocos_roundtrip_batch` are drop-in, batch-tensor equivalents of the old per-sample
file-based round trip. `prewarm(device)` lets a caller pay the one-time model-load cost
up front, outside of a timed attack loop.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torchaudio.functional as AF

from experiment_utils import align_audio_tensors

_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def _load_encodec_model(device) -> object:
    from encodec import EncodecModel

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    model.eval()
    return model


def _load_vocos_model(device) -> object:
    from vocos import Vocos

    model = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)
    model.eval()
    return model


def _get_model(name: str, device, loader) -> object:
    key = (name, str(device))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = loader(device)
        _MODEL_CACHE[key] = model
    return model


def _get_encodec_model(device) -> object:
    return _get_model("encodec", device, _load_encodec_model)


def _get_vocos_model(device) -> object:
    return _get_model("vocos", device, _load_vocos_model)


def prewarm(device) -> None:
    """Load the Encodec and Vocos weights ahead of time.

    Calling this before the attack loop moves the (one-time) model-load latency out of
    the timed region; without it, the first `encodec`/`vocos` attack call in a run pays
    that cost instead. Safe to call more than once (a no-op after the first call per
    device).
    """
    _get_encodec_model(device)
    _get_vocos_model(device)


def _split_batch_channel(wav: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    was_3d = wav.dim() == 3
    wav_2d = wav.squeeze(1) if was_3d else wav
    if wav_2d.dim() != 2:
        raise ValueError(f"Expected (B,T) or (B,1,T), got {tuple(wav.shape)}")
    return wav_2d, was_3d


def _restore_batch_channel(wav_2d: torch.Tensor, was_3d: bool) -> torch.Tensor:
    return wav_2d.unsqueeze(1) if was_3d else wav_2d


def encodec_roundtrip_batch(wav: torch.Tensor, device, sample_rate: int = 16000) -> torch.Tensor:
    """Encode/decode `wav` through Encodec (24kHz, 6kbps) in-process, batched.

    `wav` is (B, T) or (B, 1, T) at `sample_rate`; the model itself runs at its native
    24kHz, matching `tools/run_encodec.py`. Output is resampled back to `sample_rate` and
    trimmed to the input length via `align_audio_tensors`, mirroring the
    file-based `command_roundtrip_batch` convention.
    """
    model = _get_encodec_model(device)
    wav_2d, was_3d = _split_batch_channel(wav)
    wav_2d = wav_2d.to(device)

    codec_sr = int(model.sample_rate)
    wav_resampled = AF.resample(wav_2d, sample_rate, codec_sr) if sample_rate != codec_sr else wav_2d
    wav_in = wav_resampled.unsqueeze(1)  # (B, 1, T') channel dim expected by Encodec

    with torch.no_grad():
        encoded_frames = model.encode(wav_in)
        decoded = model.decode(encoded_frames)  # (B, 1, T'')

    decoded_2d = decoded.squeeze(1)
    decoded_back = AF.resample(decoded_2d, codec_sr, sample_rate) if sample_rate != codec_sr else decoded_2d

    _, decoded_aligned = align_audio_tensors(wav_2d, decoded_back.to(wav_2d.dtype))
    return _restore_batch_channel(decoded_aligned.to(wav.device), was_3d)


def vocos_roundtrip_batch(wav: torch.Tensor, device, sample_rate: int = 16000) -> torch.Tensor:
    """Encode with Encodec then decode with Vocos (charactr/vocos-encodec-24khz), batched.

    Mirrors `tools/run_vocos.py` but reuses the cached Encodec/Vocos models and processes
    the whole batch directly, with no per-sample subprocess/file I/O.
    """
    encodec_model = _get_encodec_model(device)
    vocos_model = _get_vocos_model(device)
    wav_2d, was_3d = _split_batch_channel(wav)
    wav_2d = wav_2d.to(device)

    codec_sr = int(encodec_model.sample_rate)
    wav_resampled = AF.resample(wav_2d, sample_rate, codec_sr) if sample_rate != codec_sr else wav_2d
    wav_in = wav_resampled.unsqueeze(1)

    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav_in)
        # encodec.EncodecModel.encode() returns codes shaped (B, K, T) ("codes is [B, K, T]",
        # encodec/model.py) -- batch first. vocos.Vocos.codes_to_features expects the
        # opposite convention, (K, T) or (K, B, T) -- codebook count first (see its docstring
        # and the charactr/vocos README example, `codes_to_features(torch.randint(..., size=
        # (8, 200)))`, i.e. (K, L) with no leading batch dim at all for a single item).
        # Without this transpose, codes_to_features silently sums over the true batch
        # dimension instead of the codebook dimension, corrupting the output and (whenever
        # num_codebooks != batch_size) leaving the result with the wrong "batch" size.
        codes = encoded_frames[0][0].transpose(0, 1)  # (B, K, T) -> (K, B, T)
        features = vocos_model.codes_to_features(codes)
        # Vocos's AdaLayerNorm (vocos/modules.py) broadcasts `cond_embedding_id` against the
        # whole batch: `scale = self.scale(bandwidth_id)` has shape (len(bandwidth_id), dim),
        # multiplied against features of shape (B, T, dim). A single shared id (shape (1,))
        # broadcasts correctly for any batch size; one id per sample (shape (B,)) does not --
        # it collides with the T dimension once B != T (e.g. RuntimeError: size of tensor a
        # (T) must match size of tensor b (B) at non-singleton dimension 1). Vocos has no
        # per-sample bandwidth selection API, so one shared id for the whole batch is correct.
        bandwidth_id = torch.tensor([2], device=device)
        decoded = vocos_model.decode(features, bandwidth_id=bandwidth_id)  # (B, T'')

    decoded_back = AF.resample(decoded, codec_sr, sample_rate) if sample_rate != codec_sr else decoded

    _, decoded_aligned = align_audio_tensors(wav_2d, decoded_back.to(wav_2d.dtype))
    return _restore_batch_channel(decoded_aligned.to(wav.device), was_3d)
