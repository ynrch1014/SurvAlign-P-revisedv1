import os
import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T

from base import WatermarkBase
from util import random_message, hamming_distance


def load_audio(path, target_sr=None):
    """soundfile로 오디오 로드 후 torch tensor 반환."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, T)
    else:
        data = data.T  # (channels, T)
    waveform = torch.from_numpy(data)
    if target_sr is not None and sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform, sr


def save_audio(path, waveform, sr):
    """torch tensor를 soundfile로 저장."""
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)  # (channels, T)
    if waveform.dim() == 2:
        data = waveform.cpu().numpy().T  # (T, channels)
    else:
        data = waveform.cpu().numpy()
    sf.write(path, data, sr)


class WatermarkInferencer(WatermarkBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.eval()
        self.vae.eval()

    @torch.no_grad()
    def embed(self, input_path: str, output_path: str, message=None):
        waveform, sample_rate = load_audio(input_path, target_sr=self.sample_rate)

        if message is None:
            message = random_message(self.cfg.nbits, batch_size=1).to(self.device)
        elif isinstance(message, str):
            message = torch.tensor([[int(bit) for bit in message]]).to(self.device)

        waveform = waveform.unsqueeze(1).to(self.device)
        feat = self.vae.encode(waveform)
        feat_wm = self.model(feat, message)
        wav_wm = self.vae.decode(feat_wm)
        min_length = min(waveform.size(-1), wav_wm.size(-1))
        wav = waveform[..., :min_length]
        wav_wm = wav_wm[..., :min_length]
        wav_wm = self.fusion_model(wav, wav_wm)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_audio(output_path, wav_wm.squeeze(1).cpu(), self.sample_rate)

        return {
            "watermarked_path": output_path,
            "message": message.squeeze(0).detach().cpu().numpy().tolist(),
        }

    @torch.no_grad()
    def decode(self, input_path: str, label=None):
        waveform, sample_rate = load_audio(input_path, target_sr=self.sample_rate)
        wav = waveform.unsqueeze(1).to(self.device)

        _, _, preds = self.model.decode_watermark(wav)
        preds = preds.squeeze(1).cpu()
        result = {
            "predicted_message": preds.numpy().tolist(),
        }

        if label is not None:
            if isinstance(label, str):
                label_tensor = torch.tensor([[int(bit) for bit in label]])
            elif isinstance(label, (list, tuple)):
                label_tensor = torch.tensor([list(label)])
            elif isinstance(label, torch.Tensor):
                label_tensor = label.unsqueeze(0) if label.dim() == 1 else label
            else:
                label_tensor = None
            if label_tensor is not None:
                hd = hamming_distance(preds, label_tensor)
                result["hamming_distance"] = int(hd)
                result["label"] = label_tensor.numpy().tolist()

        return result



