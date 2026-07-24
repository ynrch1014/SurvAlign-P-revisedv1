# -*- coding: utf-8 -*-
"""In-process Encodec/Vocos/FACodec attack adapters.

`tools/run_encodec.py` / `tools/run_vocos.py` / `tools/run_facodec.py`, invoked once per
audio sample via `external_attacks.command_roundtrip_batch` (a fresh subprocess + file I/O
+ full model reload every single call), are correct but wasteful: the codec weights get
reloaded from disk on every sample instead of once per process.

This module keeps the same codecs but loads each model exactly once per process (cached
in `_MODEL_CACHE`, keyed by (model name, device)) and operates directly on batched
tensors, with no subprocess or temporary files. `encodec_roundtrip_batch`/
`vocos_roundtrip_batch`/`facodec_roundtrip_batch` are drop-in, batch-tensor equivalents of
the old per-sample file-based round trip. `prewarm(device)` lets a caller pay the one-time
model-load cost up front, outside of a timed attack loop.
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


def _load_facodec_model(device) -> Tuple[object, object]:
    import os
    import sys

    facodec_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "facodec_lib")
    if facodec_lib_dir not in sys.path:
        sys.path.append(facodec_lib_dir)
    from ns3_codec import FACodecEncoder, FACodecDecoder
    from huggingface_hub import hf_hub_download

    encoder = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
    decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )
    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


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


def _get_facodec_model(device) -> Tuple[object, object]:
    return _get_model("facodec", device, _load_facodec_model)


def prewarm(device) -> None:
    """Load the Encodec, Vocos, and FACodec weights ahead of time."""
    for loader in (_get_encodec_model, _get_vocos_model, _get_facodec_model):
        try:
            loader(device)
        except Exception:
            pass


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


def facodec_roundtrip_batch(wav: torch.Tensor, device, sample_rate: int = 16000) -> torch.Tensor:
    """Encode/decode `wav` through FACodec (amphion/naturalspeech3_facodec) in-process, batched.

    Mirrors `tools/run_facodec.py`, which already operates natively at 16kHz (unlike
    Encodec/Vocos's native 24kHz), but reuses the cached encoder/decoder and processes the
    whole batch directly instead of spawning a subprocess (with a full model reload) per
    sample.
    """
    encoder, decoder = _get_facodec_model(device)
    wav_2d, was_3d = _split_batch_channel(wav)
    wav_2d = wav_2d.to(device)

    codec_sr = 16000  # FACodec's native/trained sample rate (see tools/run_facodec.py)
    wav_resampled = AF.resample(wav_2d, sample_rate, codec_sr) if sample_rate != codec_sr else wav_2d
    wav_in = wav_resampled.unsqueeze(1)  # (B, 1, T') channel dim expected by FACodecEncoder

    with torch.no_grad():
        enc_out = encoder(wav_in)
        vq_post_emb, _, _, _, spk_embs = decoder(enc_out, eval_vq=False, vq=True)
        recon_wav = decoder.inference(vq_post_emb, spk_embs)  # (B, 1, T'')

    decoded_2d = recon_wav.squeeze(1)
    decoded_back = AF.resample(decoded_2d, codec_sr, sample_rate) if sample_rate != codec_sr else decoded_2d

    _, decoded_aligned = align_audio_tensors(wav_2d, decoded_back.to(wav_2d.dtype))
    return _restore_batch_channel(decoded_aligned.to(wav.device), was_3d)
