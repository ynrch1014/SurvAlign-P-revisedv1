# -*- coding: utf-8 -*-
"""Optional held-out attack adapters (ffmpeg, ClearerVoice, FACodec, etc.)."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch

from experiment_utils import align_audio_tensors


def _run_command_template(command_template: str, input_path: str, output_path: str) -> None:
    if "{input}" not in command_template or "{output}" not in command_template:
        raise ValueError("External command must contain both {input} and {output} placeholders.")
    command = command_template.format(input=input_path, output=output_path)
    result = subprocess.run(
        shlex.split(command, posix=(os.name != "nt")),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"External attack command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"External attack did not create output: {output_path}")


def command_roundtrip_batch(
    wav: torch.Tensor,
    command_template: str,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Apply an arbitrary file-based command independently to each waveform."""
    was_3d = wav.dim() == 3
    wav_2d = wav.squeeze(1) if was_3d else wav
    if wav_2d.dim() != 2:
        raise ValueError(f"Expected (B,T) or (B,1,T), got {wav.shape}")
    outputs = []
    with tempfile.TemporaryDirectory(prefix="survalign_attack_") as temp_dir:
        for index, sample in enumerate(wav_2d):
            input_path = os.path.join(temp_dir, f"input_{index}.wav")
            output_path = os.path.join(temp_dir, f"output_{index}.wav")
            sf.write(input_path, sample.detach().cpu().numpy(), sample_rate, subtype="PCM_16")
            _run_command_template(command_template, input_path, output_path)
            output, sr = sf.read(output_path, dtype="float32")
            if output.ndim > 1:
                output = output.mean(axis=1)
            output_t = torch.from_numpy(np.asarray(output)).to(sample.device, sample.dtype)
            if sr != sample_rate:
                import torchaudio.functional as AF
                output_t = AF.resample(output_t.unsqueeze(0), sr, sample_rate).squeeze(0)
            sample_aligned, output_aligned = align_audio_tensors(sample, output_t)
            if output_aligned.shape[-1] < sample.shape[-1]:
                output_aligned = torch.nn.functional.pad(output_aligned, (0, sample.shape[-1] - output_aligned.shape[-1]))
            outputs.append(output_aligned[..., : sample.shape[-1]])
    stacked = torch.stack(outputs, dim=0)
    return stacked.unsqueeze(1) if was_3d else stacked


def ffmpeg_mp3_roundtrip_batch(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> torch.Tensor:
    """Apply a real ffmpeg MP3 encode/decode round trip."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg was not found on PATH; cannot run real MP3 evaluation.")
    # ffmpeg needs separate encode and decode commands, so implement explicitly.
    was_3d = wav.dim() == 3
    wav_2d = wav.squeeze(1) if was_3d else wav
    outputs = []
    with tempfile.TemporaryDirectory(prefix="survalign_mp3_") as temp_dir:
        for index, sample in enumerate(wav_2d):
            input_path = os.path.join(temp_dir, f"input_{index}.wav")
            mp3_path = os.path.join(temp_dir, f"compressed_{index}.mp3")
            output_path = os.path.join(temp_dir, f"output_{index}.wav")
            sf.write(input_path, sample.detach().cpu().numpy(), sample_rate, subtype="PCM_16")
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", input_path,
                 "-codec:a", "libmp3lame", "-b:a", bitrate, mp3_path],
                check=True,
            )
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", mp3_path,
                 "-ar", str(sample_rate), "-ac", "1", output_path],
                check=True,
            )
            output, _ = sf.read(output_path, dtype="float32")
            if output.ndim > 1:
                output = output.mean(axis=1)
            output_t = torch.from_numpy(np.asarray(output)).to(sample.device, sample.dtype)
            _, output_t = align_audio_tensors(sample, output_t)
            if output_t.shape[-1] < sample.shape[-1]:
                output_t = torch.nn.functional.pad(output_t, (0, sample.shape[-1] - output_t.shape[-1]))
            outputs.append(output_t[..., : sample.shape[-1]])
    stacked = torch.stack(outputs, dim=0)
    return stacked.unsqueeze(1) if was_3d else stacked
