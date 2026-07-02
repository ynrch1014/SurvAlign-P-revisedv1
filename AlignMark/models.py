from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechtokenizer.modules.seanet import SEANetEncoder
from util import chunks_to_bits


class SkipGatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.skip_connection = c_in == c_out and (stride == 1 if isinstance(stride, int) else all(s == 1 for s in stride))
        self.has_skip_proj = (c_in == c_out) and not self.skip_connection

    def forward(self, x):
        conv_output = self.conv(x)
        gated_output = torch.sigmoid(self.gate(x))
        output = conv_output * gated_output
        if self.skip_connection:
            output.add_(x)
        elif self.has_skip_proj:
            skip_output = F.adaptive_avg_pool2d(x, output.shape[2:])
            output.add_(skip_output)
        return output


class WatermarkDecoder(nn.Module):
    def __init__(self, input_channels, nbits, nchunk_size, hidden_dim, d_model):
        super().__init__()
        self.nchunk_size = nchunk_size
        assert nbits % nchunk_size == 0
        self.nbits = nbits
        self.d_model = d_model
        self.nchunks = nbits // nchunk_size

        self.proj = nn.Linear(input_channels, d_model)
        self.detect_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, 1, kernel_size=3, padding=1),
        )

        self.message_encoder = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=(5, 3), stride=(2, 1), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(2, 1), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(2, 1), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=(5, 3), stride=(1, 1), padding=(0, 1)),
        ])

        self.d_proj_16 = nn.Linear(126, 1)
        self.d_proj_32 = nn.Linear(61, 1)
        self.d_proj_64 = nn.Linear(29, 1)
        self.d_proj_128 = nn.Linear(13, 1)
        self.d_proj_256 = nn.Linear(5, 1)
        self.d_projs = [self.d_proj_16, self.d_proj_32, self.d_proj_64, self.d_proj_128, self.d_proj_256]
        self.message_head = nn.Sequential(
            nn.Linear(512 + 256 + 128 + 64 + 32 + 16, 1024), nn.GELU(), nn.Linear(1024, 2**nchunk_size * self.nchunks),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.proj(x.transpose(-1, -2)).transpose(-1, -2)
        frame_logits = self.detect_encoder(x).squeeze(1)
        temperature = 2.0
        frame_weights = torch.sigmoid(frame_logits / temperature).clamp(min=0.01, max=1.0)

        current_feat = x.unsqueeze(1)
        multi_scale_features = []
        d_proj_idx = 0
        for layer in self.message_encoder:
            current_feat = layer(current_feat)
            if isinstance(layer, nn.Conv2d):
                B, C, D, T = current_feat.shape
                if D != 1:
                    feat_trans = current_feat.permute(0, 1, 3, 2)
                    feat_flat = feat_trans.reshape(-1, D)
                    d_proj = self.d_projs[d_proj_idx]
                    feat_proj = d_proj(feat_flat).reshape(B, C, T, 1).squeeze(-1)
                else:
                    feat_proj = current_feat.squeeze(2)
                multi_scale_features.append(feat_proj)
                d_proj_idx += 1

        combined_features = torch.cat(multi_scale_features, dim=1).permute(0, 2, 1)
        time_logits = self.message_head(combined_features)
        weighted_logits = time_logits * frame_weights.unsqueeze(-1)
        weight_sum = frame_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        avg_logits = weighted_logits.sum(dim=1) / weight_sum
        chunk_logits = avg_logits.reshape(batch_size, self.nchunks, 2 ** self.nchunk_size)
        return frame_logits, chunk_logits

    def detect_watermark(self, x):
        frame_logits, chunk_logits = self.forward(x)
        chunk_probs = F.softmax(chunk_logits, dim=-1)
        chunk_indices = torch.argmax(chunk_probs, dim=-1)
        chunk_values = [chunk_indices[:, i] for i in range(self.nchunks)]
        binary_message = chunks_to_bits(chunk_values, self.nchunk_size)
        return frame_logits, chunk_logits, binary_message


class WatermarkEmbedder(nn.Module):
    def __init__(self, nbits, input_dim, hidden_dim, d_model):
        super().__init__()
        self.nbits = nbits
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, d_model)
        self.msg_embedding = nn.Embedding(2 * nbits, d_model)
        self.conv_layers = nn.Sequential(
            SkipGatedBlock(1 + 1, hidden_dim, kernel_size=3, stride=1, padding=1),
            SkipGatedBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            SkipGatedBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            SkipGatedBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            SkipGatedBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            SkipGatedBlock(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
        )
        self.out_proj = nn.Conv1d(d_model, input_dim, kernel_size=1)

    def embed_bits(self, message):
        idx = message + torch.arange(self.nbits, device=message.device) * 2
        emb = self.msg_embedding(idx)
        return emb.sum(dim=1).unsqueeze(1)

    def forward(self, hidden, msg):
        seq_len = hidden.shape[-1]
        hidden_orig = hidden
        hidden = self.proj(hidden.transpose(-1, -2)).transpose(-1, -2)
        msg_emb = self.embed_bits(msg)
        if hidden.dim() == 3:
            hidden = hidden.unsqueeze(1)
        combined_input = torch.cat([hidden, msg_emb.unsqueeze(-1).expand(-1, -1, -1, seq_len)], dim=1)
        x = self.conv_layers(combined_input)
        output = x.squeeze(1)
        output = self.out_proj(output)
        output = hidden_orig + output
        return output


class WatermarkModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nbits = config.nbits
        self.nfft = config.wm_mb.nfft
        self.sr = config.wm_mb.sr
        self.latent_dim = 1024

        self.encoder = SEANetEncoder(
            n_filters=64,
            dimension=self.latent_dim,
            ratios=[8, 5, 4, 2],
            lstm=2,
            dilation_base=2,
            residual_kernel_size=3,
            n_residual_layers=1,
            activation="ELU",
            bidirectional=True,
        )

        self.embedder = WatermarkEmbedder(
            nbits=config.nbits, input_dim=self.latent_dim, hidden_dim=32, d_model=256
        )
        self.detector = WatermarkDecoder(
            self.latent_dim, config.nbits, nchunk_size=config.wm_mb.nchunk_size, hidden_dim=32, d_model=256
        )

    def decode_watermark(self, x: torch.Tensor) -> Tuple[Any, ...]:
        embedding = self.encoder(x)
        frame_logits, chunk_logits, binary_message = self.detector.detect_watermark(embedding)
        return embedding, (frame_logits, chunk_logits), binary_message

    def forward(self, feat: torch.Tensor, message: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat_wm = self.embedder(feat, message)
        return feat_wm


class AudioFusionModel(nn.Module):
    def __init__(self, n_fft=256, hop_length=64, win_length=256, hidden_dim=64, nbits=16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.hidden_dim = hidden_dim
        self.weight_net = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)), nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, 2, kernel_size=(3, 3), padding=(1, 1)), nn.Sigmoid(),
        )

    def forward(self, wav_orig, wav_wm):
        wav_orig = wav_orig.detach()
        stft_orig = torch.stft(
            wav_orig.squeeze(1), self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(wav_orig.device), return_complex=True
        )
        real_orig, imag_orig = stft_orig.real, stft_orig.imag
        stft_wm = torch.stft(
            wav_wm.squeeze(1), self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(wav_wm.device), return_complex=True
        )
        real_wm, imag_wm = stft_wm.real, stft_wm.imag
        complex_input = torch.stack([real_orig, imag_orig, real_wm, imag_wm], dim=1)
        alpha_weights = self.weight_net(complex_input)
        alpha_real = alpha_weights[:, 0, :, :]
        alpha_imag = alpha_weights[:, 1, :, :]
        real_fused = real_orig * alpha_real + real_wm * (1 - alpha_real)
        imag_fused = imag_orig * alpha_imag + imag_wm * (1 - alpha_imag)
        final_stft = torch.complex(real_fused, imag_fused)
        wav_fused = torch.istft(
            final_stft.to(wav_orig.device), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(wav_orig.device), length=wav_orig.shape[-1]
        ).unsqueeze(1)
        return wav_fused


