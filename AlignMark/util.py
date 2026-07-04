import base64
import struct
from typing import List

import numpy as np
import torch


def hamming_distance(preds, message):
    return (preds != message).sum().item()


def random_message(nbits: int, batch_size: int) -> torch.Tensor:
    if nbits == 0:
        return torch.tensor([])
    return torch.randint(0, 2, (batch_size, nbits))


def bits_to_chunks(bits: torch.Tensor, nchunk_size: int) -> List[torch.Tensor]:
    batch_size, nbits = bits.shape
    nchunks = nbits // nchunk_size
    chunk_values = []
    for i in range(nchunks):
        chunk_bits = bits[:, i * nchunk_size : (i + 1) * nchunk_size]
        chunk_val = torch.zeros(batch_size, dtype=torch.long, device=bits.device)
        for bit_idx in range(nchunk_size):
            chunk_val += (chunk_bits[:, bit_idx].long() << bit_idx)
        chunk_values.append(chunk_val)
    return chunk_values


def chunks_to_bits(chunk_values, nchunk_size: int) -> torch.Tensor:
    """Convert chunk class values to bits.

    Accepts either the original list-of-(B,) representation or a tensor of shape
    ``(B, n_chunks)``. Supporting the tensor form prevents accidentally iterating
    over the batch dimension when converting ``logits.argmax(dim=-1)``.
    """
    if isinstance(chunk_values, torch.Tensor):
        if chunk_values.dim() == 1:
            chunk_values = chunk_values.unsqueeze(1)
        if chunk_values.dim() != 2:
            raise ValueError(
                f"Tensor chunk_values must have shape (B,n_chunks), got {chunk_values.shape}"
            )
        chunk_values = list(chunk_values.unbind(dim=1))
    elif not isinstance(chunk_values, (list, tuple)):
        raise TypeError(
            "chunk_values must be a tensor (B,n_chunks) or a list/tuple of (B,) tensors"
        )

    bit_chunks = []
    for chunk_val in chunk_values:
        if chunk_val.dim() != 1:
            raise ValueError(f"Each chunk tensor must have shape (B,), got {chunk_val.shape}")
        chunk_bits = []
        for bit_idx in range(nchunk_size):
            bit = (chunk_val.long() >> bit_idx) & 1
            chunk_bits.append(bit.unsqueeze(-1))
        bit_chunks.append(torch.cat(chunk_bits, dim=-1))
    if not bit_chunks:
        raise ValueError("chunk_values must contain at least one chunk")
    return torch.cat(bit_chunks, dim=-1)


def tensor_to_base64(audio_tensor, sample_rate=16000):
    audio_tensor = audio_tensor.cpu()
    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.squeeze(0)
    audio_numpy = audio_tensor.numpy()
    audio_numpy = np.clip(audio_numpy, -1.0, 1.0)
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(audio_int16.tobytes()),
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        len(audio_int16.tobytes()),
    )
    wav_data = header + audio_int16.tobytes()
    base64_str = base64.b64encode(wav_data).decode("utf-8")
    return base64_str


