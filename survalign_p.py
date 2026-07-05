# -*- coding: utf-8 -*-
"""
Title: SurvAlign-P Research Training and Evaluation Engine (2nd Revision)
Author: 정연재 (SKKU URP)
Description:
    SurvAlign-P의 공통 데이터, distortion, Survival Map, AlignMark wrapper 및 legacy 실험 구성요소.
    canonical 연구 파이프라인은 phase1_attribution.py와 phase2_training.py를 사용합니다.
    - reconstruction forward에는 실제 codec 출력을 사용하고, discrete 경로 backward에는 identity STE를 적용
    - residual retention과 residual-dominance 점수의 하위 분위수로 Survival Map 구성
    - LibriSpeech/VCTK 화자 격리 및 LJSpeech 파일 격리 분할 지원
    - 공통 Hann-window STFT/ISTFT와 sample-wise 정규화 적용
"""

import os
import sys
import tarfile
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from types import SimpleNamespace
import warnings

from experiment_utils import (
    stable_int_hash, integer_to_bits, align_audio_tensors,
    project_residual_l2,
)

# =======================================================
# 0. AlignMark 경로 설정 및 패키지 검증
# =======================================================
ALIGNMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AlignMark")
sys.path.insert(0, ALIGNMARK_DIR)

try:
    from pesq import pesq
except ImportError:
    pesq = None
try:
    from pystoi import stoi as compute_stoi
except ImportError:
    compute_stoi = None
try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    roc_auc_score = None
    roc_curve = None

# AlignMark 내부 모듈 임포트
try:
    from models import WatermarkModel, AudioFusionModel
    from speechtokenizer import SpeechTokenizer
    from util import bits_to_chunks, chunks_to_bits, random_message
except ImportError as e:
    raise ImportError(
        f"AlignMark 모듈 임포트 실패: {e}. "
        f"AlignMark 디렉토리가 {ALIGNMARK_DIR}에 존재하는지 확인하세요."
    )


# =======================================================
# 0.1 공통 STFT / ISTFT Hann Window 헬퍼 함수
# =======================================================
def stft_audio(wav, n_fft=256, hop_length=64):
    """AlignMark 융합 모듈과 정합되는 공통 Hann Window STFT."""
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True
    )


def istft_audio(spec, length, n_fft=256, hop_length=64):
    """AlignMark 융합 모듈과 정합되는 공통 Hann Window ISTFT."""
    window = torch.hann_window(n_fft, device=spec.device, dtype=spec.real.dtype)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        length=length
    )


# =======================================================
# 1. 실제 데이터셋 (화자 기준 격리 분할 & Resampler 캐싱)
# =======================================================
class RealLibriSpeechDataset(Dataset):
    """Backward-compatible LibriSpeech wrapper using the unified deterministic split logic."""

    def __init__(
        self,
        dataset_name="dev-clean",
        download=True,
        segment_len=32000,
        split="train",
        seed=42,
        return_metadata=False,
    ):
        self._dataset = UnifiedSpeechDataset(
            dataset_type="librispeech",
            dataset_name=dataset_name,
            download=download,
            segment_len=segment_len,
            split=split,
            seed=seed,
            return_metadata=return_metadata,
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


class UnifiedSpeechDataset(Dataset):
    """Unified audio dataset with reproducible train/calibration/test behavior.

    Training uses random crops/messages. Calibration and test splits use deterministic
    file-specific crops and an index-addressed unique 16-bit message codebook, enabling
    paired comparisons and attribution-FAR evaluation across separate runs.

    ``combined_protocol='paper'`` reproduces the file-level 200-per-dataset test split.
    ``combined_protocol='speaker_disjoint'`` uses speaker-disjoint LibriSpeech/VCTK
    splits and an index split for single-speaker LJSpeech.
    """

    def __init__(
        self,
        dataset_type="librispeech",
        dataset_name="dev-clean",
        download=True,
        segment_len=32000,
        split="train",
        seed=42,
        return_metadata=False,
        combined_protocol="speaker_disjoint",
        nbits=16,
    ):
        self.dataset_type = dataset_type.lower()
        self.dataset_name = dataset_name
        self.segment_len = int(segment_len)
        self.sample_rate = 16000
        self.split = split
        self.seed = int(seed)
        self.return_metadata = bool(return_metadata)
        self.combined_protocol = combined_protocol
        self.nbits = int(nbits)
        self.resamplers = {}
        self.files = []
        self.speaker_by_file = {}

        if split not in {"train", "calib", "test"}:
            raise ValueError(f"split must be train/calib/test, got {split}")
        if self.dataset_type == "librispeech":
            self._init_librispeech(download)
        elif self.dataset_type == "vctk":
            self._init_vctk(download)
        elif self.dataset_type == "ljspeech":
            self._init_ljspeech(download)
        elif self.dataset_type == "combined":
            self._init_combined(download)
        else:
            raise ValueError(
                f"Unsupported dataset type: {self.dataset_type}. "
                "Choose librispeech, vctk, ljspeech, or combined."
            )
        self.files = sorted(self.files)
        if not self.files:
            raise FileNotFoundError(f"No files found for {self.dataset_type}/{self.split}")
        if self.split != "train" and len(self.files) > 2**self.nbits:
            raise ValueError(
                f"Evaluation split has {len(self.files)} samples, exceeding the {self.nbits}-bit unique codebook."
            )
        self.eval_message_values = None
        if self.split != "train":
            # Random-looking but deterministic unique codebook. Sequential binary indices
            # would have unrealistically small Hamming distances and bias attribution FAR.
            codebook_seed = stable_int_hash(self.seed, self.dataset_type, self.dataset_name, self.split)
            rng = np.random.RandomState(codebook_seed)
            self.eval_message_values = rng.permutation(2**self.nbits)[: len(self.files)]

    @staticmethod
    def _audio_files(root):
        if not os.path.exists(root):
            return []
        return sorted(
            os.path.join(dp, fn)
            for dp, _, names in os.walk(root)
            for fn in names
            if fn.lower().endswith((".flac", ".wav"))
        )

    def _init_librispeech(self, download):
        self.data_dir = f"./data/LibriSpeech/{self.dataset_name}"
        if download and (not os.path.exists(self.data_dir) or not os.listdir(self.data_dir)):
            self._download_librispeech()
        files = self._audio_files(self.data_dir)
        pairs = [(f, os.path.basename(f).split("-")[0]) for f in files]
        self.files = self._split_by_speaker(pairs)

    def _init_vctk(self, download):
        self.data_dir = "./data/VCTK"
        candidates = [
            os.path.join(self.data_dir, "VCTK-Corpus-0.92", "wav48_silence_trimmed"),
            os.path.join(self.data_dir, "wav48_silence_trimmed"),
            os.path.join(self.data_dir, "wav48"),
        ]
        audio_dir = next((d for d in candidates if os.path.exists(d)), candidates[0])
        if download and not os.path.exists(audio_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            try:
                torchaudio.datasets.VCTK_092(root=self.data_dir, download=True)
            except Exception as exc:
                warnings.warn(f"VCTK automatic download failed: {exc}")
            audio_dir = next((d for d in candidates if os.path.exists(d)), audio_dir)
        files = self._audio_files(audio_dir)
        pairs = [(f, os.path.basename(f).split("_")[0]) for f in files]
        self.files = self._split_by_speaker(pairs)

    def _init_ljspeech(self, download):
        self.data_dir = "./data/LJSpeech"
        candidates = [
            os.path.join(self.data_dir, "LJSpeech-1.1", "wavs"),
            os.path.join(self.data_dir, "wavs"),
        ]
        audio_dir = next((d for d in candidates if os.path.exists(d)), candidates[0])
        if download and not os.path.exists(audio_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            try:
                torchaudio.datasets.LJSPEECH(root=self.data_dir, download=True)
            except Exception as exc:
                warnings.warn(f"LJSpeech automatic download failed: {exc}")
            audio_dir = next((d for d in candidates if os.path.exists(d)), audio_dir)
        files = self._audio_files(audio_dir)
        self.speaker_by_file.update({f: "LJSpeech_single_speaker" for f in files})
        self.files = self._split_by_index(files)

    def _init_combined(self, download):
        # Discover each corpus first.
        lib_dir = f"./data/LibriSpeech/{self.dataset_name}"
        if download and (not os.path.exists(lib_dir) or not os.listdir(lib_dir)):
            self._download_librispeech()
        lib_files = self._audio_files(lib_dir)
        lib_pairs = [(f, f"libri:{os.path.basename(f).split('-')[0]}") for f in lib_files]

        vctk_candidates = [
            "./data/VCTK/VCTK-Corpus-0.92/wav48_silence_trimmed",
            "./data/VCTK/wav48_silence_trimmed",
            "./data/VCTK/wav48",
        ]
        vctk_dir = next((d for d in vctk_candidates if os.path.exists(d)), vctk_candidates[0])
        if download and not os.path.exists(vctk_dir):
            os.makedirs("./data/VCTK", exist_ok=True)
            try:
                torchaudio.datasets.VCTK_092(root="./data/VCTK", download=True)
            except Exception as exc:
                warnings.warn(f"VCTK automatic download failed: {exc}")
            vctk_dir = next((d for d in vctk_candidates if os.path.exists(d)), vctk_dir)
        vctk_files = self._audio_files(vctk_dir)
        vctk_pairs = [(f, f"vctk:{os.path.basename(f).split('_')[0]}") for f in vctk_files]

        lj_candidates = ["./data/LJSpeech/LJSpeech-1.1/wavs", "./data/LJSpeech/wavs"]
        lj_dir = next((d for d in lj_candidates if os.path.exists(d)), lj_candidates[0])
        if download and not os.path.exists(lj_dir):
            os.makedirs("./data/LJSpeech", exist_ok=True)
            try:
                torchaudio.datasets.LJSPEECH(root="./data/LJSpeech", download=True)
            except Exception as exc:
                warnings.warn(f"LJSpeech automatic download failed: {exc}")
            lj_dir = next((d for d in lj_candidates if os.path.exists(d)), lj_dir)
        lj_files = self._audio_files(lj_dir)
        self.speaker_by_file.update({f: "lj:LJSpeech_single_speaker" for f in lj_files})

        if self.combined_protocol == "paper":
            self.files = (
                self._paper_file_split([p[0] for p in lib_pairs])
                + self._paper_file_split([p[0] for p in vctk_pairs])
                + self._paper_file_split(lj_files)
            )
            for path, spk in lib_pairs + vctk_pairs:
                self.speaker_by_file[path] = spk
            print(
                f"[DATASET] COMBINED/{self.split}: paper-style file split, "
                f"speaker leakage may be present; files={len(self.files)}"
            )
        elif self.combined_protocol == "speaker_disjoint":
            self.files = self._split_by_speaker(lib_pairs) + self._split_by_speaker(vctk_pairs) + self._split_by_index(lj_files)
            print(
                f"[DATASET] COMBINED/{self.split}: speaker-disjoint for LibriSpeech/VCTK; "
                f"LJSpeech remains file-disjoint only; files={len(self.files)}"
            )
        else:
            raise ValueError("combined_protocol must be 'paper' or 'speaker_disjoint'")

    def _split_by_speaker(self, pairs):
        pairs = sorted(pairs, key=lambda x: x[0])
        for path, speaker in pairs:
            self.speaker_by_file[path] = speaker
        speakers = sorted({speaker for _, speaker in pairs})
        rng = np.random.RandomState(self.seed)
        rng.shuffle(speakers)
        n = len(speakers)
        n_train = int(n * 0.8)
        n_calib = int(n * 0.1)
        split_sets = {
            "train": set(speakers[:n_train]),
            "calib": set(speakers[n_train:n_train + n_calib]),
            "test": set(speakers[n_train + n_calib:]),
        }
        selected = [path for path, speaker in pairs if speaker in split_sets[self.split]]
        print(
            f"[DATASET] {self.dataset_type.upper()} {self.split.upper()} "
            f"speakers={len(split_sets[self.split])}, files={len(selected)}"
        )
        return selected

    def _split_by_index(self, files):
        files = sorted(files)
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(files))
        rng.shuffle(indices)
        n = len(files)
        n_train = int(n * 0.8)
        n_calib = int(n * 0.1)
        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "calib":
            selected = indices[n_train:n_train + n_calib]
        else:
            selected = indices[n_train + n_calib:]
        return [files[i] for i in selected]

    def _paper_file_split(self, files):
        files = sorted(files)
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(files))
        rng.shuffle(indices)
        test = indices[: min(200, len(indices))]
        remaining = indices[min(200, len(indices)):]
        n_calib = int(len(remaining) * 0.1)
        if self.split == "test":
            selected = test
        elif self.split == "calib":
            selected = remaining[:n_calib]
        else:
            selected = remaining[n_calib:]
        return [files[i] for i in selected]

    def _download_librispeech(self):
        tar_name = f"{self.dataset_name}.tar.gz"
        tar_path = f"./data/{tar_name}"
        os.makedirs("./data", exist_ok=True)
        min_size = 300000000 if self.dataset_name == "dev-clean" else 6000000000
        if not os.path.exists(tar_path) or os.path.getsize(tar_path) < min_size:
            url = f"https://www.openslr.org/resources/12/{tar_name}"
            urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path="./data/")
        if os.path.exists(tar_path):
            os.remove(tar_path)

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path):
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        wav = torch.from_numpy(data)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                import torchaudio.transforms as T
                self.resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self.resamplers[sr](wav)
        return wav

    def __getitem__(self, idx):
        path = self.files[idx]
        wav = self._load_audio(path)
        original_length = int(wav.shape[-1])
        if wav.shape[-1] > self.segment_len:
            max_start = wav.shape[-1] - self.segment_len
            if self.split == "train":
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = stable_int_hash(path, self.split, self.seed, modulo=max_start + 1)
            wav = wav[:, start:start + self.segment_len]
            valid_length = self.segment_len
        else:
            start = 0
            valid_length = int(wav.shape[-1])
            wav = F.pad(wav, (0, self.segment_len - wav.shape[-1]))

        if self.split == "train":
            msg = torch.randint(0, 2, (self.nbits,), dtype=torch.long)
        else:
            msg = integer_to_bits(int(self.eval_message_values[idx]), nbits=self.nbits)

        if not self.return_metadata:
            return wav, msg
        metadata = {
            "sample_id": f"{self.dataset_type}:{self.split}:{idx}",
            "file_path": path,
            "speaker_id": self.speaker_by_file.get(path, "unknown"),
            "crop_start": int(start),
            "valid_length": int(valid_length),
            "original_length": int(original_length),
            "split": self.split,
        }
        return wav, msg, metadata


# =======================================================
# 2. 미분 가능한 채널 및 Reconstruction 왜곡
# =======================================================
class DifferentiableDistortion(nn.Module):
    """Differentiable training distortions and explicitly named proxy attacks.

    Real MP3, ClearerVoice and FACodec are held-out file-based attacks implemented in
    ``external_attacks.py``. The proxy attacks below must not be reported as those real models.
    """

    def __init__(self, sr=16000, vae=None):
        super().__init__()
        self.sr = int(sr)
        self.vae = vae

    @staticmethod
    def _randn_like(wav, seed=None):
        if seed is None:
            return torch.randn_like(wav)
        generator = torch.Generator(device=wav.device)
        generator.manual_seed(int(seed))
        return torch.randn(wav.shape, device=wav.device, dtype=wav.dtype, generator=generator)

    def add_awgn(self, wav, snr_db=20.0, seed=None):
        noise = self._randn_like(wav, seed=seed)
        wav_pwr = torch.sum(wav ** 2, dim=-1, keepdim=True)
        noise_pwr = torch.sum(noise ** 2, dim=-1, keepdim=True)
        scale = torch.sqrt(wav_pwr / (noise_pwr * (10 ** (snr_db / 10)) + 1e-8))
        return wav + scale * noise

    def lowpass_filter(self, wav, cutoff_hz=4000):
        n_taps = 101
        t = torch.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=wav.dtype, device=wav.device)
        fc = cutoff_hz / self.sr
        kernel = 2 * fc * torch.sinc(2 * fc * t)
        kernel = kernel * torch.hann_window(n_taps, device=wav.device, dtype=wav.dtype)
        kernel = kernel / (kernel.sum() + 1e-8)
        kernel = kernel.view(1, 1, -1)
        wav_3d = wav.unsqueeze(1) if wav.dim() == 2 else wav
        filtered = F.conv1d(wav_3d, kernel, padding=n_taps // 2)
        return filtered.squeeze(1) if wav.dim() == 2 else filtered

    def bandpass_filter(self, wav, low_hz=300, high_hz=3400):
        n_taps = 101
        t = torch.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=wav.dtype, device=wav.device)
        window = torch.hann_window(n_taps, device=wav.device, dtype=wav.dtype)
        fc_low = low_hz / self.sr
        lp_low = 2 * fc_low * torch.sinc(2 * fc_low * t) * window
        lp_low = lp_low / (lp_low.sum() + 1e-8)
        hp = -lp_low
        hp[n_taps // 2] += 1.0
        fc_high = high_hz / self.sr
        lp_high = 2 * fc_high * torch.sinc(2 * fc_high * t) * window
        lp_high = lp_high / (lp_high.sum() + 1e-8)
        was_2d = wav.dim() == 2
        wav_3d = wav.unsqueeze(1) if was_2d else wav
        filtered = F.conv1d(wav_3d, hp.view(1, 1, -1), padding=n_taps // 2)
        filtered = F.conv1d(filtered, lp_high.view(1, 1, -1), padding=n_taps // 2)
        return filtered.squeeze(1) if was_2d else filtered

    def resample_distortion(self, wav, down_rate=2):
        was_2d = wav.dim() == 2
        wav_3d = wav.unsqueeze(1) if was_2d else wav
        original_length = wav_3d.shape[-1]
        downsampled = F.interpolate(
            wav_3d, size=max(1, original_length // int(down_rate)), mode="linear", align_corners=False
        )
        upsampled = F.interpolate(downsampled, size=original_length, mode="linear", align_corners=False)
        return upsampled.squeeze(1) if was_2d else upsampled

    def speech_reconstruct(self, wav, n_q=8):
        """SpeechTokenizer forward with identity STE for the discrete quantizer path."""
        if self.vae is None:
            raise RuntimeError("SpeechTokenizer reconstruction requested, but no VAE/tokenizer is loaded.")
        was_2d = wav.dim() == 2
        wav_3d = wav.unsqueeze(1) if was_2d else wav
        with torch.no_grad():
            features = self.vae.encoder(wav_3d)
            quantized, _, _, _ = self.vae.quantizer(
                features, n_q=int(n_q), layers=list(range(int(n_q))), st=0
            )
            reconstructed = self.vae.decoder(quantized)
            wav_trim, rec_trim = align_audio_tensors(wav_3d, reconstructed)
            if rec_trim.shape[-1] < wav_3d.shape[-1]:
                rec_trim = F.pad(rec_trim, (0, wav_3d.shape[-1] - rec_trim.shape[-1]))
            rec_raw = rec_trim[..., : wav_3d.shape[-1]]
        # Forward uses reconstructed audio; backward approximates codec Jacobian as identity.
        rec_ste = wav_3d + (rec_raw - wav_3d).detach()
        return rec_ste.squeeze(1) if was_2d else rec_ste

    def spectral_compression_proxy(self, wav, cutoff_ratio=0.7, noise_scale=0.002, seed=None):
        """Differentiable spectral-compression proxy; this is not real MP3."""
        was_2d = wav.dim() == 2
        wav_2d = wav if was_2d else wav.squeeze(1)
        spec = stft_audio(wav_2d, n_fft=256, hop_length=64)
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)
        cutoff_bin = int(magnitude.shape[1] * float(cutoff_ratio))
        mask = torch.ones_like(magnitude)
        mask[:, cutoff_bin:] = 0.1
        masked = magnitude * mask
        noise = self._randn_like(masked, seed=seed) * float(noise_scale)
        reconstructed = torch.polar(torch.clamp(masked + noise, min=1e-8), phase)
        wav_rec = istft_audio(reconstructed, length=wav_2d.shape[-1], n_fft=256, hop_length=64)
        return wav_rec if was_2d else wav_rec.unsqueeze(1)

    def apply_masking(self, wav, max_ratio=0.1, seed=None):
        was_2d = wav.dim() == 2
        wav_3d = wav.unsqueeze(1) if was_2d else wav
        B, C, T = wav_3d.shape
        
        generator = torch.Generator(device=wav.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()

        mask = torch.ones_like(wav_3d)
        mask_len = int(T * max_ratio)
        
        for i in range(B):
            start = torch.randint(0, max(1, T - mask_len), (1,), generator=generator).item()
            mask[i, :, start:start + mask_len] = 0.0
            
        masked_wav = wav_3d * mask
        return masked_wav.squeeze(1) if was_2d else masked_wav

    def apply_replacement(self, wav, max_ratio=0.1, snr_db=0.0, seed=None):
        was_2d = wav.dim() == 2
        wav_3d = wav.unsqueeze(1) if was_2d else wav
        B, C, T = wav_3d.shape
        
        generator = torch.Generator(device=wav.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()

        replaced_wav = wav_3d.clone()
        replace_len = int(T * max_ratio)
        
        for i in range(B):
            start = torch.randint(0, max(1, T - replace_len), (1,), generator=generator).item()
            segment = wav_3d[i, :, start:start + replace_len]
            noise = torch.randn_like(segment, generator=generator)
            
            sig_pwr = torch.sum(segment ** 2, dim=-1, keepdim=True)
            noise_pwr = torch.sum(noise ** 2, dim=-1, keepdim=True)
            scale = torch.sqrt(sig_pwr / (noise_pwr * (10 ** (snr_db / 10)) + 1e-8))
            
            replaced_wav[i, :, start:start + replace_len] = scale * noise
            
        return replaced_wav.squeeze(1) if was_2d else replaced_wav

    def apply_frame_shuffle(self, wav, frame_duration_ms=50, shuffle_ratio=0.2, seed=None):
        was_2d = wav.dim() == 2
        wav_3d = wav.unsqueeze(1) if was_2d else wav
        B, C, T = wav_3d.shape
        
        generator = torch.Generator(device=wav.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()

        frame_size = int(self.sr * (frame_duration_ms / 1000.0))
        if frame_size == 0 or T < frame_size:
            return wav_3d.squeeze(1) if was_2d else wav_3d
            
        n_frames = T // frame_size
        shuffle_frames = max(2, int(n_frames * shuffle_ratio))
        
        shuffled_wav = wav_3d.clone()
        for i in range(B):
            start_frame = torch.randint(0, max(1, n_frames - shuffle_frames), (1,), generator=generator).item()
            frame_indices = torch.arange(start_frame, start_frame + shuffle_frames, device=wav.device)
            
            # Shuffle indices
            perm = torch.randperm(shuffle_frames, generator=generator, device=wav.device)
            shuffled_indices = frame_indices[perm]
            
            # Reconstruct the shuffled segment
            temp = torch.zeros((C, shuffle_frames * frame_size), device=wav.device, dtype=wav.dtype)
            for j in range(shuffle_frames):
                shuff_idx = shuffled_indices[j]
                temp[:, j*frame_size:(j+1)*frame_size] = wav_3d[i, :, shuff_idx*frame_size:(shuff_idx+1)*frame_size]
                
            shuffled_wav[i, :, start_frame*frame_size:(start_frame+shuffle_frames)*frame_size] = temp
            
        return shuffled_wav.squeeze(1) if was_2d else shuffled_wav

    def forward(self, wav, dtype="noise", **kwargs):
        seed = kwargs.get("seed")
        if dtype == "noise":
            return self.add_awgn(wav, snr_db=kwargs.get("snr_db", 20.0), seed=seed)
        if dtype == "masking":
            return self.apply_masking(wav, max_ratio=kwargs.get("max_ratio", 0.1), seed=seed)
        if dtype == "replacement":
            return self.apply_replacement(wav, max_ratio=kwargs.get("max_ratio", 0.1), snr_db=kwargs.get("snr_db", 0.0), seed=seed)
        if dtype == "frame_shuffle":
            return self.apply_frame_shuffle(wav, frame_duration_ms=kwargs.get("frame_duration_ms", 50), shuffle_ratio=kwargs.get("shuffle_ratio", 0.2), seed=seed)
        if dtype == "lowpass":
            return self.lowpass_filter(wav, cutoff_hz=kwargs.get("cutoff_hz", 4000))
        if dtype == "bandpass":
            return self.bandpass_filter(
                wav, low_hz=kwargs.get("low_hz", 300), high_hz=kwargs.get("high_hz", 3400)
            )
        if dtype == "resample":
            return self.resample_distortion(wav, down_rate=kwargs.get("down_rate", 2))
        if dtype in {"reconstruct", "speechtokenizer"}:
            return self.speech_reconstruct(wav, n_q=kwargs.get("n_q", 8))
        if dtype in {"strong_speechtokenizer", "facodec_proxy"}:
            if dtype == "facodec_proxy":
                warnings.warn(
                    "'facodec_proxy' is deprecated and is not FACodec; use 'strong_speechtokenizer'.",
                    DeprecationWarning,
                )
            return self.speech_reconstruct(wav, n_q=kwargs.get("n_q", 2))
        if dtype in {"spectral_proxy", "mp3"}:
            if dtype == "mp3":
                warnings.warn(
                    "'mp3' is a spectral proxy, not real MP3; use 'spectral_proxy'.",
                    DeprecationWarning,
                )
            return self.spectral_compression_proxy(
                wav,
                cutoff_ratio=kwargs.get("cutoff_ratio", 0.7),
                noise_scale=kwargs.get("noise_scale", 0.002),
                seed=seed,
            )
        if dtype in {"identity", "clean"}:
            return wav
        raise ValueError(f"Unknown differentiable distortion type: {dtype}")


# =======================================================
# 2.1 Paired AWGN 및 스케일 공유 함수
# =======================================================
def paired_awgn(clean, watermarked, snr_db, seed=None):
    """Inject exactly the same deterministic AWGN realization into a clean/WM pair."""
    if clean.shape != watermarked.shape:
        raise ValueError(f"Paired AWGN shape mismatch: {clean.shape} vs {watermarked.shape}")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=clean.device)
        generator.manual_seed(int(seed))
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    clean_power = clean.pow(2).mean(dim=-1, keepdim=True)
    noise_power = noise.pow(2).mean(dim=-1, keepdim=True)
    scale = torch.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10)) + 1e-8))
    shared_noise = scale * noise
    return clean + shared_noise, watermarked + shared_noise


def normalize_per_sample(x, eps=1e-8):
    """Sample-wise z-normalization for feature maps."""
    mean = x.reshape(x.shape[0], -1).mean(dim=-1).view(-1, 1, 1)
    std = x.reshape(x.shape[0], -1).std(dim=-1).view(-1, 1, 1)
    return (x - mean) / (std + eps)


def minmax_per_sample(x, eps=1e-8):
    minimum = x.reshape(x.shape[0], -1).min(dim=-1).values.view(-1, 1, 1)
    maximum = x.reshape(x.shape[0], -1).max(dim=-1).values.view(-1, 1, 1)
    return (x - minimum) / (maximum - minimum + eps)


def get_local_energy_masking_proxy(wav, n_fft=256, hop_length=64):
    """Local spectral-energy proxy. This is not a full psychoacoustic masking model."""
    with torch.no_grad():
        wav_2d = wav.squeeze(1) if wav.dim() == 3 else wav
        magnitude = torch.abs(stft_audio(wav_2d, n_fft=n_fft, hop_length=hop_length))
        log_magnitude = torch.log10(magnitude + 1e-5)
        kernel = torch.ones(1, 1, 5, 5, device=wav.device, dtype=wav.dtype) / 25.0
        smoothed = F.conv2d(log_magnitude.unsqueeze(1), kernel, padding=2).squeeze(1)
        return minmax_per_sample(smoothed)


def _estimate_integer_shift(reference, candidate, max_shift=64):
    """Estimate one integer sample shift per item using normalized cross-correlation."""
    shifts = []
    for ref, cand in zip(reference, candidate):
        best_shift, best_score = 0, float("-inf")
        for shift in range(-max_shift, max_shift + 1):
            if shift < 0:
                r = ref[-shift:]
                c = cand[: cand.shape[-1] + shift]
            elif shift > 0:
                r = ref[: ref.shape[-1] - shift]
                c = cand[shift:]
            else:
                r, c = ref, cand
            if r.numel() < 16:
                continue
            score = torch.sum(r * c) / (torch.linalg.vector_norm(r) * torch.linalg.vector_norm(c) + 1e-8)
            score_value = float(score.item())
            if score_value > best_score:
                best_score, best_shift = score_value, shift
        shifts.append(best_shift)
    return shifts


def _apply_integer_shifts(wav, shifts):
    """Undo candidate delays/advances estimated by ``_estimate_integer_shift``."""
    aligned = []
    length = wav.shape[-1]
    for sample, shift in zip(wav, shifts):
        # Positive shift means the candidate is delayed relative to the reference,
        # so advance it. Negative shift means it is early, so delay it.
        if shift < 0:
            amount = -int(shift)
            shifted = F.pad(sample, (amount, 0))[..., :length]
        elif shift > 0:
            amount = int(shift)
            shifted = F.pad(sample[..., amount:], (0, amount))
        else:
            shifted = sample
        aligned.append(shifted[..., :length])
    return torch.stack(aligned, dim=0)


def _apply_survival_attack_pair(clean, watermarked, distorter, attack_name, seed):
    if attack_name == "noise":
        return paired_awgn(clean, watermarked, snr_db=20.0, seed=seed)
    if attack_name == "lowpass":
        return distorter(clean, "lowpass", cutoff_hz=4000), distorter(watermarked, "lowpass", cutoff_hz=4000)
    if attack_name == "bandpass":
        return (
            distorter(clean, "bandpass", low_hz=300, high_hz=3400),
            distorter(watermarked, "bandpass", low_hz=300, high_hz=3400),
        )
    if attack_name == "resample":
        return distorter(clean, "resample", down_rate=2), distorter(watermarked, "resample", down_rate=2)
    if attack_name == "speechtokenizer_nq6":
        return distorter(clean, "reconstruct", n_q=6), distorter(watermarked, "reconstruct", n_q=6)
    if attack_name == "speechtokenizer_nq8":
        return distorter(clean, "reconstruct", n_q=8), distorter(watermarked, "reconstruct", n_q=8)
    if attack_name == "strong_speechtokenizer":
        return (
            distorter(clean, "strong_speechtokenizer", n_q=2),
            distorter(watermarked, "strong_speechtokenizer", n_q=2),
        )
    if attack_name == "spectral_proxy":
        return (
            distorter(clean, "spectral_proxy", cutoff_ratio=0.7, seed=seed),
            distorter(watermarked, "spectral_proxy", cutoff_ratio=0.7, seed=seed),
        )
    raise ValueError(f"Unsupported survival-map attack: {attack_name}")


def get_survival_map(
    wav_clean,
    wav_wm,
    distorter,
    n_fft=256,
    hop_length=64,
    attack_names=None,
    quantile=0.25,
    base_seed=42,
    smooth_kernel=5,
    residual_floor_quantile=0.05,
    align_outputs=True,
    max_alignment_shift=64,
):
    """Compute an attack-derived physical survival prior.

    The score combines residual retention and residual dominance. It does not use the
    watermark decoder. ``attack_names`` must be kept separate from held-out evaluation
    attacks when making generalization claims.
    """
    if attack_names is None:
        attack_names = ("noise", "lowpass", "bandpass", "resample", "speechtokenizer_nq6", "spectral_proxy")
    if not attack_names:
        raise ValueError("At least one survival-map attack is required.")
    with torch.no_grad():
        clean = wav_clean.squeeze(1) if wav_clean.dim() == 3 else wav_clean
        watermarked = wav_wm.squeeze(1) if wav_wm.dim() == 3 else wav_wm
        clean, watermarked = align_audio_tensors(clean, watermarked)
        spec_clean = stft_audio(clean, n_fft=n_fft, hop_length=hop_length)
        spec_wm = stft_audio(watermarked, n_fft=n_fft, hop_length=hop_length)
        residual_mag = torch.abs(spec_wm - spec_clean)
        floor = torch.quantile(
            residual_mag.reshape(residual_mag.shape[0], -1),
            q=float(residual_floor_quantile),
            dim=1,
        ).view(-1, 1, 1)
        valid_support = residual_mag > floor
        residual_mag_safe = residual_mag.clamp_min(1e-8)

        attack_scores = []
        for attack_index, attack_name in enumerate(attack_names):
            attacked_clean, attacked_wm = _apply_survival_attack_pair(
                clean, watermarked, distorter, attack_name, seed=int(base_seed) + attack_index
            )
            attacked_clean, attacked_wm = align_audio_tensors(attacked_clean, attacked_wm)
            if align_outputs and attack_name.startswith(("reconstruct", "strong_speechtokenizer")):
                shifts = _estimate_integer_shift(clean, attacked_clean, max_shift=max_alignment_shift)
                attacked_clean = _apply_integer_shifts(attacked_clean, shifts)
                attacked_wm = _apply_integer_shifts(attacked_wm, shifts)

            attacked_clean_spec = stft_audio(attacked_clean, n_fft=n_fft, hop_length=hop_length)
            attacked_wm_spec = stft_audio(attacked_wm, n_fft=n_fft, hop_length=hop_length)
            retained_residual = torch.abs(attacked_wm_spec - attacked_clean_spec)
            reconstruction_difference = torch.abs(attacked_clean_spec - spec_clean)
            retention = torch.clamp(retained_residual / residual_mag_safe, 0.0, 1.0)
            dominance = retained_residual / (retained_residual + reconstruction_difference + 1e-8)
            score = retention * dominance
            attack_scores.append(torch.where(valid_support, score, torch.zeros_like(score)))

        stacked = torch.stack(attack_scores, dim=0)
        survival = torch.quantile(stacked, q=float(quantile), dim=0)
        if smooth_kernel and int(smooth_kernel) > 1:
            k = int(smooth_kernel)
            if k % 2 == 0:
                raise ValueError("smooth_kernel must be odd")
            survival = F.avg_pool2d(survival.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2).squeeze(1)
        return minmax_per_sample(survival)


def frame_energy_vad(wav, n_fft=256, hop_length=64, threshold_ratio=0.15):
    """STFT 에너지를 기반으로 dynamic 시간-주파수 VAD 맵을 리턴 (비발화/무음 무분별 잔차 융합 차단)."""
    with torch.no_grad():
        if wav.dim() == 3:
            wav_2d = wav.squeeze(1)
        else:
            wav_2d = wav
        mag = torch.abs(stft_audio(wav_2d, n_fft=n_fft, hop_length=hop_length))
        frame_energy = torch.mean(mag, dim=1)  # (B, T_stft)
        
        max_energy = frame_energy.max(dim=-1, keepdim=True)[0]
        threshold = max_energy * threshold_ratio + 1e-6
        vad_mask = (frame_energy > threshold).float()
        return vad_mask.unsqueeze(1).expand(-1, n_fft // 2 + 1, -1)


def get_energy_based_vad_prior(wav_2d, n_fft=256, hop_length=64, eps=1e-8):
    """
    STFT frame power의 상대 dB를 [0, 1] soft score로 변환한 prior.
    """
    with torch.no_grad():
        mag = torch.abs(stft_audio(wav_2d, n_fft=n_fft, hop_length=hop_length))  # (B, F, T_stft)
        frame_power = torch.sum(mag ** 2, dim=1)  # (B, T_stft)
        max_power = frame_power.max(dim=-1, keepdim=True)[0]  # (B, 1)
        relative_db = 10 * torch.log10(frame_power / (max_power + eps) + eps)  # (B, T_stft)
        # -30dB 이하면 0.0, 0dB면 1.0으로 매핑
        soft_score = torch.clamp((relative_db + 30.0) / 30.0, 0.0, 1.0)
        return soft_score.unsqueeze(1).expand(-1, mag.shape[1], -1)


def get_residual_prior(r0_complex, eps=1e-8):
    """
    Baseline residual이 존재한 위치를 0-1로 표현한 prior.
    """
    with torch.no_grad():
        r0_mag = torch.abs(r0_complex)  # (B, F, T_stft)
        max_r0 = r0_mag.reshape(r0_mag.shape[0], -1).max(dim=-1)[0].view(-1, 1, 1)  # (B, 1, 1)
        r0_prior = r0_mag / (max_r0 + eps)
        return r0_prior


# =======================================================
# 4. AlignMark Manager (올바른 API 래핑 및 미분 백프로퍼게이션 확보)
# =======================================================
class AlignMarkManager:
    """AlignMark 모델의 올바른 로드 및 추론 인터페이스."""

    def __init__(self, device, alignmark_dir=None, latent_mode="public_code"):
        if alignmark_dir is None:
            alignmark_dir = ALIGNMARK_DIR
        self.device = device
        self.alignmark_dir = alignmark_dir
        if latent_mode not in {"public_code", "unquantized"}:
            raise ValueError("latent_mode must be public_code or unquantized")
        self.latent_mode = latent_mode

        self.cfg = SimpleNamespace(
            device=device,
            local_rank=None,
            sample_rate=16000,
            nbits=16,
            wm_mb=SimpleNamespace(nfft=256, sr=16000, nchunk_size=4),
        )

        config_path = os.path.join(
            alignmark_dir, "speechtokenizer", "pretrained_model",
            "speechtokenizer_hubert_avg_config.json"
        )
        ckpt_path = os.path.join(
            alignmark_dir, "speechtokenizer", "pretrained_model",
            "SpeechTokenizer.pt"
        )
        weight_path = os.path.join(alignmark_dir, "weight.pth")
        missing_assets = [
            path for path in (config_path, ckpt_path, weight_path)
            if not os.path.exists(path)
        ]
        if missing_assets:
            missing_list = "\n - ".join(missing_assets)
            raise FileNotFoundError(
                "Missing required AlignMark assets:\n"
                f" - {missing_list}\n"
                "Place the pretrained files at the documented paths before running experiments."
            )

        self.vae = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.wm_model = WatermarkModel(self.cfg).to(device)
        self.fusion = AudioFusionModel(
            n_fft=256, hop_length=64, win_length=256, hidden_dim=64, nbits=16
        ).to(device)

        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        wm_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
        self.wm_model.load_state_dict(wm_dict, strict=True)
        fusion_dict = {k.replace("module.", ""): v for k, v in checkpoint["fusion_state_dict"].items()}
        self.fusion.load_state_dict(fusion_dict, strict=True)

        self.wm_model.eval()
        self.fusion.eval()
        for p in self.wm_model.parameters():
            p.requires_grad = False
        for p in self.fusion.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(self, wav, msg):
        """Embed a watermark using either the public-code or unquantized codec latent path."""
        if self.latent_mode == "unquantized":
            feat = self.vae.encoder(wav)
        else:
            feat = self.vae.encode(wav)
        feat_wm = self.wm_model(feat, msg)
        wav_wm = self.vae.decode(feat_wm)
        wav_trim, wav_wm_trim = align_audio_tensors(wav, wav_wm)
        wav_fused = self.fusion(wav_trim, wav_wm_trim)
        wav_trim, wav_fused = align_audio_tensors(wav_trim, wav_fused)
        residual = wav_fused - wav_trim
        if wav_fused.shape != residual.shape or wav_trim.shape != residual.shape:
            raise RuntimeError("AlignMark embedding produced inconsistent waveform lengths.")
        return wav_fused, residual

    @torch.no_grad()
    def decode(self, wav):
        """
        워터마크 디코딩.

        Returns:
            frame_logits: (B, T_frames) 프레임별 워터마크 존재 로짓
            chunk_logits: (B, nchunks=4, 16) 메시지 청크 로짓
            binary_message: (B, 16) 예측된 이진 메시지
        """
        embedding, (frame_logits, chunk_logits), binary_msg = \
            self.wm_model.decode_watermark(wav)
        return frame_logits, chunk_logits, binary_msg

    def decode_logits_with_grad(self, wav):
        """
        입력 파형(wav)에 대해 역전파가 가능한 미분 가능 디코더 로직.
        이 경로에는 양자화 병목이 없어 입력 파형에서 encoder와 detector까지 autograd가 연결됩니다.
        """
        embedding = self.wm_model.encoder(wav)
        frame_logits, chunk_logits = self.wm_model.detector(embedding)
        return frame_logits, chunk_logits


# =======================================================
# 5. Survival Gate 네트워크 (초기 마스크 1.0 강제)
# =======================================================
class SurvivalGate(nn.Module):
    """STFT 도메인에서 워터마크 잔차를 주파수-시간별로 조절하는 Gate."""

    def __init__(self, in_channels=6, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            # batch size가 작으므로 GroupNorm을 적용해 학습 시 안정성 극대화
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        
        # 마지막 Conv 가중치와 편향을 0으로 강제 초기화하여 초기 출력이 정확히 1.0에서 시작되도록 보장
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, feature_pack, R0_complex):
        """
        Args:
            feature_pack: (B, 6, F, T_stft) 6채널 정규화 feature map
            R0_complex: (B, F, T_stft) 복소수 잔차 스펙트로그램

        Returns:
            R_gated: (B, F, T_stft) 게이트된 복소수 잔차
            gate_scale: (B, F, T_stft) 게이트 스케일 [0.8, 1.2]
        """
        logits = self.conv(feature_pack).squeeze(1)  # (B, F, T_stft)
        # 1.0 근처에서 안정적으로 제어되도록 tanh 기반 스케일 제약 적용 (학술적 제약)
        gate_scale = 1.0 + 0.2 * torch.tanh(logits)
        R_gated = R0_complex * gate_scale
        return R_gated, gate_scale


# =======================================================
# 6. Presence Head (Message-decoding Evidence 기반)
# =======================================================
class PresenceHead(nn.Module):
    """
    디코딩된 logits (64차원) 및 Shannon Entropy (4차원), Margin(top1-top2, 4차원), Max Probability (4차원)를
    입력받아 워터마크 유무를 이진 분류하는 판별기 (총 76차원).
    """

    def __init__(self, n_features=76):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, chunk_logits):
        """
        Args:
            chunk_logits: (B, 4, 16) 디코딩 결과 로짓

        Returns:
            prob: (B,) 워터마크 탐지 확률
        """
        # Flat logits: (B, 64)
        flat_logits = chunk_logits.reshape(chunk_logits.shape[0], -1)
        
        # Decoding evidence 피처 추출 (Entropy & Margin & Max Probability)
        probs = F.softmax(chunk_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B, 4)
        
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]         # (B, 4)
        max_prob = sorted_probs[:, :, 0]                              # (B, 4)
        
        # 12차원 또는 76차원 입력 피처 병합
        if self.mlp[0].in_features == 12:
            features = torch.cat([entropy, margin, max_prob], dim=-1) # (B, 12)
        else:
            features = torch.cat([flat_logits, entropy, margin, max_prob], dim=-1) # (B, 76)
            
        return self.mlp(features).squeeze(-1)


# =======================================================
# 7. Loss 함수 및 SI-SDR 계산식
# =======================================================
def compute_chunk_ce_loss(chunk_logits, target_msg, nchunk_size=4):
    """AlignMark의 chunk-based classification에 맞는 CrossEntropy Loss."""
    target_chunks = bits_to_chunks(target_msg.long(), nchunk_size)  # List of (B,)
    target = torch.stack(target_chunks, dim=1)  # (B, nchunks)
    num_classes = 2 ** nchunk_size
    loss = F.cross_entropy(
        chunk_logits.reshape(-1, num_classes),
        target.reshape(-1),
    )
    return loss


def compute_ber(pred_bits, target_bits):
    """Bit Error Rate 계산."""
    return (pred_bits != target_bits).float().mean().item()


def compute_si_sdr(reference, estimated, eps=1e-8, zero_mean=True):
    """Compute standard scale-invariant SDR for one waveform or a batch.

    One-dimensional inputs are treated as a single waveform rather than as a batch of
    scalar samples. By default, each waveform is centered before projection, matching
    the conventional SI-SDR definition used by common evaluation toolkits.
    """
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference)
    if isinstance(estimated, np.ndarray):
        estimated = torch.from_numpy(estimated)

    reference = reference.float()
    estimated = estimated.float()
    if reference.dim() == 1:
        reference = reference.unsqueeze(0)
    elif reference.dim() > 2:
        reference = reference.reshape(reference.shape[0], -1)
    if estimated.dim() == 1:
        estimated = estimated.unsqueeze(0)
    elif estimated.dim() > 2:
        estimated = estimated.reshape(estimated.shape[0], -1)
    if reference.shape != estimated.shape:
        raise ValueError(f"SI-SDR inputs must have matching shapes, got {reference.shape} and {estimated.shape}")
    if zero_mean:
        reference = reference - reference.mean(dim=-1, keepdim=True)
        estimated = estimated - estimated.mean(dim=-1, keepdim=True)

    dot_product = torch.sum(reference * estimated, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    scaled_ref = (dot_product / ref_energy) * reference
    noise = estimated - scaled_ref
    scaled_ref_energy = torch.sum(scaled_ref ** 2, dim=-1, keepdim=True)
    noise_energy = torch.sum(noise ** 2, dim=-1, keepdim=True) + eps
    sdr = 10 * torch.log10((scaled_ref_energy + eps) / noise_energy)
    return torch.mean(sdr).item()

def compute_total_variation_loss(gate_scale):
    """게이트 맵의 인접 격자(시간/주파수) 간의 불연속성을 직접 제어하여 급격한 변이를 억제."""
    diff_t = torch.abs(gate_scale[:, :, 1:] - gate_scale[:, :, :-1])
    diff_f = torch.abs(gate_scale[:, 1:, :] - gate_scale[:, :-1, :])
    return torch.mean(diff_t) + torch.mean(diff_f)


def compute_survival_routing_loss(r_gated_complex, survival_map, masking_map, vad_map, eps=1e-8):
    """
    워터마크 에너지가 왜곡에 살아남고 지각적으로 안전하며(p_map이 큼 = q_safe), 음성 활성 영역(vad) 대역으로 유기적으로 배분되도록 유도.
    잔차 크기 크기에 종속되지 않도록 정규화(Normalization)합니다.
    """
    q_route = (survival_map * masking_map * vad_map).clamp(0.0, 1.0)
    residual_mag = torch.abs(r_gated_complex)
    
    numerator = torch.sum((1.0 - q_route) * residual_mag)
    denominator = torch.sum(residual_mag) + eps
    return numerator / denominator


# =======================================================
# 8. 학습 파이프라인
# =======================================================
def run_training(args):
    warnings.warn(
        "survalign_p.py contains the legacy 6-channel/presence pipeline. "
        "Use phase2_training.py for the canonical paper experiments.",
        DeprecationWarning,
    )
    print("=" * 60)
    print("[START] SurvAlign-P URP 2차 수정 훈련 파이프라인 시작")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    # ---- 모델 로드 ----
    print("\n[초기화] AlignMark 모델 로드...")
    alignmark = AlignMarkManager(device)

    # ---- 데이터 로드 (화자 격리 분할 적용) ----
    dataset_type = getattr(args, 'dataset_type', 'librispeech')
    print(f"\n[초기화] 데이터셋 ({dataset_type} / {args.dataset_name}) 로드...")
    train_dataset = UnifiedSpeechDataset(dataset_type=dataset_type, dataset_name=args.dataset_name, download=True, split="train")
    calib_dataset = UnifiedSpeechDataset(dataset_type=dataset_type, dataset_name=args.dataset_name, download=False, split="calib")
    test_dataset = UnifiedSpeechDataset(dataset_type=dataset_type, dataset_name=args.dataset_name, download=False, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    calib_loader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    distorter = DifferentiableDistortion(vae=alignmark.vae).to(device)

    # STFT 파라미터
    n_fft = 256
    hop_length = 64
    freq_bins = n_fft // 2 + 1  # 129

    # ---- 모델 선언 ----
    # 6채널 입력 스택: wav_mag_normalized, r0_mag_normalized, s_map_smoothed, p_map, vad_prior, res_prior
    gate = SurvivalGate(in_channels=6, hidden_dim=32).to(device)
    presence = PresenceHead(n_features=76).to(device)

    opt_gate = optim.AdamW(gate.parameters(), lr=args.lr_gate)
    opt_presence = optim.AdamW(presence.parameters(), lr=args.lr_presence)

    # positive/negative 왜곡 선택적 적용 헬퍼 (dist_mode에 따라 6가지 중 선택)
    def apply_dist_by_mode(w, mode, seed):
        w_2d = w.squeeze(1)
        if mode == 0:
            return distorter(w_2d, "noise", snr_db=20.0, seed=seed).unsqueeze(1)
        elif mode == 1:
            return distorter(w_2d, "lowpass", cutoff_hz=4000).unsqueeze(1)
        elif mode == 2:
            return distorter(w_2d, "bandpass", low_hz=300, high_hz=3400).unsqueeze(1)
        elif mode == 3:
            return distorter(w_2d, "resample", down_rate=2).unsqueeze(1)
        elif mode == 4:
            return distorter(w_2d, "reconstruct", n_q=6).unsqueeze(1)
        else:  # mode == 5
            return distorter(w_2d, "spectral_proxy", cutoff_ratio=0.7, seed=seed).unsqueeze(1)

    # ================================================================
    # STAGE 1: Survival Gate 학습 (실제 디코더 직접 역전파)
    # ================================================================
    print("\n" + "=" * 60)
    print("[Stage 1] Survival Gate 학습 (디코더 직접 역전파 - VQ 없음)")
    print("  목적: 워터마크 잔차의 주파수-시간별 가중치를 학습하여")
    print("        왜곡 내성을 높이면서 청각 품질을 유지")
    print("=" * 60)

    gate.train()
    n_gate_steps = args.n_gate_steps

    train_iter = iter(train_loader)
    for step in range(n_gate_steps):
        try:
            wav, msg = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            wav, msg = next(train_iter)

        wav, msg = wav.to(device), msg.to(device)
        opt_gate.zero_grad()

        # a. AlignMark에서 원본 잔차 R0 추출 (no grad)
        with torch.no_grad():
            _, r0 = alignmark.embed(wav, msg)
            r0_2d = r0.squeeze(1)  # (B, T)

        # b. STFT 도메인으로 변환
        wav_2d = wav.squeeze(1)
        r0_complex = stft_audio(r0_2d, n_fft=n_fft, hop_length=hop_length)  # (B, F, T_stft)
        wav_mag = torch.abs(stft_audio(wav_2d, n_fft=n_fft, hop_length=hop_length))

        # c. Feature maps 및 VAD 구축 (no grad)
        p_map = get_local_energy_masking_proxy(wav, n_fft=n_fft, hop_length=hop_length)
        wav_wm_approx = wav_2d + r0_2d
        s_map = get_survival_map(wav_2d, wav_wm_approx, distorter, n_fft=n_fft, hop_length=hop_length)
        vad_map = frame_energy_vad(wav_2d, n_fft=n_fft, hop_length=hop_length)

        # d. 피처 전처리 (log1p & sample-wise 정규화)
        wav_feature = normalize_per_sample(torch.log1p(wav_mag))
        residual_feature = normalize_per_sample(torch.log1p(torch.abs(r0_complex)))
        vad_prior = get_energy_based_vad_prior(wav_2d, n_fft=n_fft, hop_length=hop_length)
        res_prior = get_residual_prior(r0_complex)

        features = torch.stack([
            wav_feature,
            residual_feature,
            s_map,
            p_map,
            vad_prior,
            res_prior,
        ], dim=1)  # (B, 6, F, T_stft)

        # e. Gate 적용 (Autograd 역전파 그래프 시작)
        r_gated_complex, gate_scale = gate(features, r0_complex)

        # f. 시간 도메인 복원
        r_gated = istft_audio(r_gated_complex, length=wav_2d.shape[-1], n_fft=n_fft, hop_length=hop_length)  # (B, T)

        # Waveform L2 Norm Projection (Baseline 에너지를 초과하지 않도록 엄격 규제)
        norm_r0 = torch.norm(r0_2d, p=2, dim=-1, keepdim=True) + 1e-8
        norm_gated = torch.norm(r_gated, p=2, dim=-1, keepdim=True) + 1e-8
        scale_factor = torch.minimum(
            torch.tensor(1.0, device=r_gated.device),
            norm_r0 / norm_gated
        )
        r_gated_final = r_gated * scale_factor

        wav_final = wav_2d + r_gated_final  # (B, T)
        
        # g. 동일한 공격 적용 (Stochastic Attack 시드 일관성)
        dist_mode = step % 6
        attack_seed = 42 + step
        wav_distorted = apply_dist_by_mode(wav_final.unsqueeze(1), dist_mode, attack_seed)

        # h. 디코더를 직접 통과하여 BER Loss 계산 (이미 미분 가능!)
        _, chunk_logits = alignmark.decode_logits_with_grad(wav_distorted)

        loss_rob = compute_chunk_ce_loss(chunk_logits, msg)

        # i. 청각 패널티: 지각 민감 영역에서의 과도한 증폭 방지 (조용한 영역(p_mag이 0에 가깝움) 벌점)
        loss_saf = torch.mean(
            torch.clamp((gate_scale - 1.0) * (1.0 - p_map), min=0.0) ** 2
        )

        # j. Deviation 패널티: Gate가 1.0에서 과도하게 벗어나지 않도록
        loss_dev = torch.mean((gate_scale - 1.0) ** 2)

        # k. 평활화 및 라우팅 추가 손실 연산 (VAD 및 p_map 정방향 융합)
        loss_tv = compute_total_variation_loss(gate_scale)
        loss_route = compute_survival_routing_loss(r_gated_complex, s_map, p_map, vad_map)

        # l. Waveform Clipping 방지 손실 추가 (|wav_final| > 1.0 규제)
        loss_clip = torch.relu(torch.abs(wav_final) - 1.0).pow(2).mean()

        loss_total = loss_rob + 1.5 * loss_saf + 0.5 * loss_dev + 0.1 * loss_tv + 2.0 * loss_route + 10.0 * loss_clip
        loss_total.backward()
        opt_gate.step()

        if (step + 1) % 5 == 0 or step == 0:
            dist_name = ["AWGN", "Lowpass", "Bandpass", "Resample", "Reconstruct", "MP3Proxy"][dist_mode]
            print(
                f"  Step {step + 1}/{n_gate_steps} | 왜곡: {dist_name} | "
                f"Total: {loss_total.item():.4f} "
                f"(Rob: {loss_rob.item():.4f}, Saf: {loss_saf.item():.4f}, Route: {loss_route.item():.4f}, Clip: {loss_clip.item():.4f})"
            )

    print("[SUCCESS] Survival Gate 학습 완료\n")

    # ================================================================
    # STAGE 2: Presence Head 학습 (왜곡 가해진 Positive/Negative)
    # ================================================================
    print("=" * 60)
    print("[Stage 2] Presence Head 학습 (왜곡 환경 하의 디코딩 에비던스)")
    print("=" * 60)

    presence.train()
    bce = nn.BCELoss()
    n_presence_steps = args.n_presence_steps

    train_iter = iter(train_loader)
    for step in range(n_presence_steps):
        try:
            wav, msg = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            wav, msg = next(train_iter)

        wav, msg = wav.to(device), msg.to(device)
        opt_presence.zero_grad()

        # 학습 시 정직하게 distorted positive/negative를 인풋으로 주입해 노이즈 오탐 방지 (6종 왜곡 순환 주입)
        dist_mode = step % 6
        attack_seed = 100 + step

        with torch.no_grad():
            # Positive: 워터마크 있는 오디오 (Baseline 및 Gated 혼합 기용해 일반화 향상)
            wav_wm_base, r0 = alignmark.embed(wav, msg)
            
            # Gated 파형 생성
            r0_2d = r0.squeeze(1)
            wav_2d = wav.squeeze(1)
            r0_complex = stft_audio(r0_2d, n_fft=n_fft, hop_length=hop_length)
            wav_mag = torch.abs(stft_audio(wav_2d, n_fft=n_fft, hop_length=hop_length))
            p_map = get_local_energy_masking_proxy(wav, n_fft=n_fft, hop_length=hop_length)
            s_map = get_survival_map(wav_2d, wav_2d + r0_2d, distorter, n_fft=n_fft, hop_length=hop_length)

            wav_feature = normalize_per_sample(torch.log1p(wav_mag))
            residual_feature = normalize_per_sample(torch.log1p(torch.abs(r0_complex)))
            vad_prior = get_energy_based_vad_prior(wav_2d, n_fft=n_fft, hop_length=hop_length)
            res_prior = get_residual_prior(r0_complex)
            features = torch.stack([wav_feature, residual_feature, s_map, p_map, vad_prior, res_prior], dim=1)

            r_gated_complex, _ = gate(features, r0_complex)
            r_gated = istft_audio(r_gated_complex, length=wav_2d.shape[-1], n_fft=n_fft, hop_length=hop_length)
            
            norm_r0 = torch.norm(r0_2d, p=2, dim=-1, keepdim=True) + 1e-8
            norm_gated = torch.norm(r_gated, p=2, dim=-1, keepdim=True) + 1e-8
            scale_factor = torch.minimum(torch.tensor(1.0, device=r_gated.device), norm_r0 / norm_gated)
            wav_gated_final = (wav_2d + r_gated * scale_factor).unsqueeze(1)

            # Positive 왜곡 적용 (Gated와 Base를 고르게 주입)
            if step % 2 == 0:
                pos_dist = apply_dist_by_mode(wav_gated_final, dist_mode, attack_seed)
            else:
                pos_dist = apply_dist_by_mode(wav_wm_base, dist_mode, attack_seed)
            _, pos_chunk_logits, _ = alignmark.decode(pos_dist)

            # Negative 왜곡 적용
            neg_dist = apply_dist_by_mode(wav, dist_mode, attack_seed)
            _, neg_chunk_logits, _ = alignmark.decode(neg_dist)

        inputs = torch.cat([pos_chunk_logits, neg_chunk_logits], dim=0) # (2B, 4, 16)
        labels = torch.cat([
            torch.ones(wav.shape[0], device=device),
            torch.zeros(wav.shape[0], device=device),
        ])

        probs = presence(inputs)
        loss = bce(probs, labels)
        loss.backward()
        opt_presence.step()

        if (step + 1) % 5 == 0 or step == 0:
            acc = ((probs > 0.5).float() == labels).float().mean()
            print(f"  Step {step + 1}/{n_presence_steps} | BCE: {loss.item():.4f} | Acc: {acc.item():.2%}")

    print("[SUCCESS] Presence Head 학습 완료\n")

    # ================================================================
    # STAGE 2.1: Threshold Calibration (독립된 calib 화자 세트 사용)
    # ================================================================
    print("=" * 60)
    print("[Stage 2.1] Presence Threshold Calibration (Calibration 화자 세트)")
    print("  목적: 명목상 FPR <= 1% 수준을 달성하는 임계값(tau_p) 결정 (6종 왜곡 대상)")
    print("=" * 60)

    presence.eval()
    calib_negative_scores = []
    
    with torch.no_grad():
        for step, (wav, msg) in enumerate(calib_loader):
            wav = wav.to(device)
            # Calibration 화자들의 다양한 오디오 왜곡 적용 (Negative)
            dist_mode = step % 6
            neg_dist = apply_dist_by_mode(wav, dist_mode, 500+step)
            _, neg_chunk_logits, _ = alignmark.decode(neg_dist)
            neg_probs = presence(neg_chunk_logits)
            calib_negative_scores.extend(neg_probs.cpu().numpy().tolist())

    # Calibration negative score의 99%ile로 임계값 결정해 데이터 누수 격리
    tau_p = np.quantile(calib_negative_scores, 0.99)
    print(f"[CALIBRATION] Calibration Negative Scores 개수: {len(calib_negative_scores)}")
    print(f"[CALIBRATION] 결정된 Presence 임계값 (tau_p at nominal FPR<=1%): {tau_p:.4f}")

    # ---- 체크포인트 저장 ----
    print("\n[CHECKPOINT] 모델 가중치 및 임계값(tau_p) 저장 중...")
    gate_ckpt_path = "./gate_checkpoint.pth"
    presence_ckpt_path = "./presence_checkpoint.pth"
    
    torch.save({
        "model_state_dict": gate.state_dict(),
        "optimizer_state_dict": opt_gate.state_dict(),
    }, gate_ckpt_path)
    print(f"  - Survival Gate 가중치 저장 완료: {gate_ckpt_path}")
    
    torch.save({
        "model_state_dict": presence.state_dict(),
        "optimizer_state_dict": opt_presence.state_dict(),
        "tau_p": float(tau_p),
    }, presence_ckpt_path)
    print(f"  - Presence Head 가중치 및 임계값 저장 완료: {presence_ckpt_path}")

    # ================================================================
    # STAGE 3: 종합 평가 (격리된 test 화자 세트를 사용해 1:1 성능 대조)
    # ================================================================
    print("\n" + "=" * 60)
    print("[Stage 3] 종합 평가 (독립된 Test 화자 세트 1:1 비교)")
    print("=" * 60)

    gate.eval()

    test_positive_scores = []
    test_negative_scores = []
    
    results = {
        "Clean": {"base_ber": [], "gated_ber": []},
        "AWGN (20dB)": {"base_ber": [], "gated_ber": []},
        "Lowpass (4kHz)": {"base_ber": [], "gated_ber": []},
        "Bandpass (300-3400Hz)": {"base_ber": [], "gated_ber": []},
        "Resample (2x down)": {"base_ber": [], "gated_ber": []},
        "Reconstruct (RVQ 6)": {"base_ber": [], "gated_ber": []},
        "Spectral Compression Proxy (70%)": {"base_ber": [], "gated_ber": []}
    }
    
    pesq_base_scores, pesq_gated_scores = [], []
    stoi_base_scores, stoi_gated_scores = [], []
    sdr_base_scores, sdr_gated_scores = [], []
    
    n_eval_steps = args.n_eval_steps

    with torch.no_grad():
        for step, (wav, msg) in enumerate(test_loader):
            if step >= n_eval_steps:
                break

            wav, msg = wav.to(device), msg.to(device)
            B = wav.shape[0]

            # Gated & Baseline 1:1 융합 생성
            wav_wm_base, r0 = alignmark.embed(wav, msg)
            
            r0_2d = r0.squeeze(1)
            wav_2d = wav.squeeze(1)
            r0_complex = stft_audio(r0_2d, n_fft=n_fft, hop_length=hop_length)
            wav_mag = torch.abs(stft_audio(wav_2d, n_fft=n_fft, hop_length=hop_length))
            p_map = get_local_energy_masking_proxy(wav, n_fft=n_fft, hop_length=hop_length)
            s_map = get_survival_map(wav_2d, wav_2d + r0_2d, distorter, n_fft=n_fft, hop_length=hop_length)

            wav_feature = normalize_per_sample(torch.log1p(wav_mag))
            residual_feature = normalize_per_sample(torch.log1p(torch.abs(r0_complex)))
            vad_prior = get_energy_based_vad_prior(wav_2d, n_fft=n_fft, hop_length=hop_length)
            res_prior = get_residual_prior(r0_complex)
            features = torch.stack([wav_feature, residual_feature, s_map, p_map, vad_prior, res_prior], dim=1)

            r_gated_complex, _ = gate(features, r0_complex)
            r_gated = istft_audio(r_gated_complex, length=wav_2d.shape[-1], n_fft=n_fft, hop_length=hop_length)
            
            norm_r0 = torch.norm(r0_2d, p=2, dim=-1, keepdim=True) + 1e-8
            norm_gated = torch.norm(r_gated, p=2, dim=-1, keepdim=True) + 1e-8
            scale_factor = torch.minimum(torch.tensor(1.0, device=r_gated.device), norm_r0 / norm_gated)
            r_gated_final = r_gated * scale_factor

            wav_gated = (wav_2d + r_gated_final).unsqueeze(1)

            # 동일한 stochastic attack 시드 일치 공유 적용
            attack_seed = 2000 + step
            
            distortions = {
                "Clean": lambda w: w,
                "AWGN (20dB)": lambda w: distorter(w.squeeze(1), "noise", snr_db=20.0, seed=attack_seed).unsqueeze(1),
                "Lowpass (4kHz)": lambda w: distorter(w.squeeze(1), "lowpass", cutoff_hz=4000).unsqueeze(1),
                "Bandpass (300-3400Hz)": lambda w: distorter(w.squeeze(1), "bandpass", low_hz=300, high_hz=3400).unsqueeze(1),
                "Resample (2x down)": lambda w: distorter(w.squeeze(1), "resample", down_rate=2).unsqueeze(1),
                "Reconstruct (RVQ 6)": lambda w: distorter(w.squeeze(1), "reconstruct", n_q=6).unsqueeze(1),
                "Spectral Compression Proxy (70%)": lambda w: distorter(w.squeeze(1), "spectral_proxy", cutoff_ratio=0.7, seed=attack_seed).unsqueeze(1)
            }
            
            for dist_name, dist_fn in distortions.items():
                w_base_dist = dist_fn(wav_wm_base)
                _, _, pred_base = alignmark.decode(w_base_dist)
                ber_base_per_sample = (pred_base != msg).float().mean(dim=1).cpu().tolist()
                results[dist_name]["base_ber"].extend(ber_base_per_sample)
                
                w_gated_dist = dist_fn(wav_gated)
                _, _, pred_gated = alignmark.decode(w_gated_dist)
                ber_gated_per_sample = (pred_gated != msg).float().mean(dim=1).cpu().tolist()
                results[dist_name]["gated_ber"].extend(ber_gated_per_sample)

            # Test 화자 Presence 검증용 positive/negative 왜곡 수집
            dist_mode_test = step % 6
            pos_dist_test = apply_dist_by_mode(wav_gated, dist_mode_test, attack_seed)
            _, pos_chunk_logits, _ = alignmark.decode(pos_dist_test)
            p_pos = presence(pos_chunk_logits)
            test_positive_scores.extend(p_pos.cpu().numpy().tolist())

            neg_dist_test = apply_dist_by_mode(wav, dist_mode_test, attack_seed)
            _, neg_chunk_logits, _ = alignmark.decode(neg_dist_test)
            p_neg = presence(neg_chunk_logits)
            test_negative_scores.extend(p_neg.cpu().numpy().tolist())

            # 4) 오디오 품질 1:1 비교
            for i in range(B):
                ref = wav_2d[i].cpu().numpy()
                base_deg = wav_wm_base[i, 0].cpu().numpy()
                gated_deg = wav_gated[i, 0].cpu().numpy()
                
                # PESQ WB
                if pesq is not None:
                    try:
                        pesq_base_scores.append(pesq(16000, ref, base_deg, "wb"))
                    except Exception:
                        pass
                    try:
                        pesq_gated_scores.append(pesq(16000, ref, gated_deg, "wb"))
                    except Exception:
                        pass
                if compute_stoi is not None:
                    try:
                        stoi_base_scores.append(compute_stoi(ref, base_deg, 16000, extended=False))
                    except Exception:
                        pass
                    try:
                        stoi_gated_scores.append(compute_stoi(ref, gated_deg, 16000, extended=False))
                    except Exception:
                        pass
                
                # SI-SDR
                sdr_base_scores.append(compute_si_sdr(ref[np.newaxis, :], base_deg[np.newaxis, :]))
                sdr_gated_scores.append(compute_si_sdr(ref[np.newaxis, :], gated_deg[np.newaxis, :]))

    # 통계 도출 함수 (평균 ± 95% 신뢰구간)
    def compute_stats(values):
        if not values:
            return float("nan"), float("nan")
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        se = std / np.sqrt(len(values))
        ci = 1.96 * se  # 95% Confidence Interval
        return mean, ci

    # ---- 결과 출력 및 정리 ----
    print("\n" + "=" * 60)
    print("[REPORT] 종합 리포트")
    print("=" * 60)
    print(f"| {'왜곡 조건 (Distortion)':<35} | {'기본형 (AlignMark)':<25} | {'제안형 (SurvAlign-P)':<25} | {'개선도':<8} |")
    print(f"|{'-'*37}|{'-'*27}|{'-'*27}|{'-'*10}|")
    
    for dist_name in results.keys():
        b_mean, b_ci = compute_stats(results[dist_name]["base_ber"])
        g_mean, g_ci = compute_stats(results[dist_name]["gated_ber"])
        diff = b_mean - g_mean
        b_str = f"{b_mean:.4f} (±{b_ci:.4f})"
        g_str = f"{g_mean:.4f} (±{g_ci:.4f})"
        print(f"| {dist_name:<35} | {b_str:<25} | {g_str:<25} | {diff:+.4f} |")
        
    print(f"|{'-'*37}|{'-'*27}|{'-'*27}|{'-'*10}|")

    # AUC 및 FPR/TPR 탐지력 검증
    test_positive_scores = np.array(test_positive_scores)
    test_negative_scores = np.array(test_negative_scores)
    all_scores = np.concatenate([test_positive_scores, test_negative_scores])
    all_labels = np.concatenate([np.ones_like(test_positive_scores), np.zeros_like(test_negative_scores)])
    
    if roc_auc_score is not None and len(all_scores) >= 2 and len(set(all_labels)) > 1:
        test_auc = roc_auc_score(all_labels, all_scores)
        actual_test_fpr = np.mean(test_negative_scores >= tau_p)
        actual_test_tpr = np.mean(test_positive_scores >= tau_p)
        print(f"\n[INFO] Detection AUROC on Test Speakers: {test_auc:.4f}")
        print(f"[INFO] Test FPR (at Nominal 1% Calibrated Threshold): {actual_test_fpr:.2%}")
        print(f"[INFO] Test TPR (at Nominal 1% Calibrated Threshold): {actual_test_tpr:.2%}")

    print("\n[INFO] 1:1 오디오 품질 비교 지표 (평균 ± 95% 신뢰구간):")
    pesq_b_m, pesq_b_c = compute_stats(pesq_base_scores)
    pesq_g_m, pesq_g_c = compute_stats(pesq_gated_scores)
    stoi_b_m, stoi_b_c = compute_stats(stoi_base_scores)
    stoi_g_m, stoi_g_c = compute_stats(stoi_gated_scores)
    sdr_b_m, sdr_b_c = compute_stats(sdr_base_scores)
    sdr_g_m, sdr_g_c = compute_stats(sdr_gated_scores)

    print(f"  - PESQ WB | 기본형: {pesq_b_m:.3f} (±{pesq_b_c:.3f}) vs 제안형: {pesq_g_m:.3f} (±{pesq_g_c:.3f})")
    print(f"  - STOI    | 기본형: {stoi_b_m:.3f} (±{stoi_b_c:.3f}) vs 제안형: {stoi_g_m:.3f} (±{stoi_g_c:.3f})")
    print(f"  - SI-SDR   | 기본형: {sdr_b_m:.2f} (±{sdr_b_c:.2f}) dB vs 제안형: {sdr_g_m:.2f} (±{sdr_g_c:.2f}) dB")

    print("\n[SUCCESS] SurvAlign-P 전체 시뮬레이션 완료!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SurvAlign-P Research Training and Evaluation Engine")
    parser.add_argument("--dataset_type", type=str, default="librispeech",
                        choices=["librispeech", "vctk", "ljspeech", "combined"],
                        help="Dataset type to use (librispeech, vctk, ljspeech, combined)")
    parser.add_argument("--dataset_name", type=str, default="dev-clean",
                        help="LibriSpeech subset name (e.g., dev-clean, train-clean-100)")
    parser.add_argument("--n_gate_steps", type=int, default=1000, help="Steps for Survival Gate training")
    parser.add_argument("--n_presence_steps", type=int, default=500, help="Steps for Presence Head training")
    parser.add_argument("--n_eval_steps", type=int, default=100, help="Steps for final evaluation")
    parser.add_argument("--sanity_check", action="store_true", help="Run a quick end-to-end sanity check")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training/evaluation")
    parser.add_argument("--lr_gate", type=float, default=1e-4, help="Learning rate for Survival Gate")
    parser.add_argument("--lr_presence", type=float, default=5e-4, help="Learning rate for Presence Head")
    
    args = parser.parse_args()
    
    if args.sanity_check:
        args.n_gate_steps = 5
        args.n_presence_steps = 5
        args.n_eval_steps = 2
        print("[SANITY CHECK] Sanity check flag detected: setting n_gate_steps=5, n_presence_steps=5, n_eval_steps=2.")

    run_training(args)


# Final Version for URP
