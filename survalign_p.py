# -*- coding: utf-8 -*-
"""
Title: SurvAlign-P Research Training and Evaluation Engine (2nd Revision)
Author: 정연재 (SKKU URP)
Description:
    URP 학술적 무결성 및 엄밀성을 완벽히 충족하기 위한 SurvAlign-P 2차 개편 코드.
    - VQ/Proxy 이론 정정: 실제 decoder(frozen)의 인코더/디텍터 역전파 그래디언트를 직접 전파 (STE/Direct Backprop)
    - 5개 왜곡의 SIR(Signal-to-Interference Ratio) 분위수 Survival Map 구현
    - RMS 기반 dynamic VAD map 적용 및 q_safe = p_map 정방향 라우팅 구현
    - LibriSpeech 화자(Speaker ID) 격리 데이터셋 분할 (Train 80%, Calib 10%, Test 10%)
    - 공통 Hann Window STFT/ISTFT 정합 함수 적용
    - 피처 전처리 log1p 및 sample-wise 정규화 적용
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

# =======================================================
# 0. AlignMark 경로 설정 및 패키지 검증
# =======================================================
ALIGNMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AlignMark")
sys.path.insert(0, ALIGNMARK_DIR)

try:
    from pesq import pesq
    from pystoi import stoi as compute_stoi
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError as e:
    raise ImportError(
        f"필수 연구 패키지 누락: {e}. "
        "'pip install pesq pystoi scikit-learn'을 실행하세요."
    )

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
    """LibriSpeech dev-clean 기반 실제 음성 데이터셋 (화자 격리 분할 탑재)."""

    def __init__(self, data_dir="./data/LibriSpeech/dev-clean", download=True, segment_len=32000, split="train"):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.sample_rate = 16000
        self.split = split
        self.resamplers = {}  # Resampler 캐시로 매 샘플 생성 방지

        if download and (not os.path.exists(data_dir) or not os.listdir(data_dir)):
            self._download_real_data()

        all_files = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(data_dir)
            for f in fn
            if f.endswith((".flac", ".wav"))
        ]
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"실제 음성 데이터가 {data_dir}에 존재하지 않습니다. "
                "download=True로 설정하거나 데이터를 직접 배치하세요."
            )

        # 화자 ID 기준 격리 분할
        file_spk_pairs = []
        for f in all_files:
            spk_id = os.path.basename(f).split("-")[0]
            file_spk_pairs.append((f, spk_id))

        unique_speakers = sorted(list(set([pair[1] for pair in file_spk_pairs])))
        n_speakers = len(unique_speakers)

        # 80% Train, 10% Calibration, 10% Test 화자 할당
        n_tr = int(n_speakers * 0.8)
        n_cal = int(n_speakers * 0.1)

        tr_spk = set(unique_speakers[:n_tr])
        cal_spk = set(unique_speakers[n_tr : n_tr + n_cal])
        te_spk = set(unique_speakers[n_tr + n_cal :])

        if split == "train":
            target_spk = tr_spk
        elif split == "calib":
            target_spk = cal_spk
        else:
            target_spk = te_spk

        self.files = [pair[0] for pair in file_spk_pairs if pair[1] in target_spk]
        print(f"[DATASET] {split.upper()} 세트 화자 수: {len(target_spk)} / 파일 수: {len(self.files)}")

    def _download_real_data(self):
        tar_path = "./data/dev-clean.tar.gz"
        os.makedirs("./data", exist_ok=True)
        
        if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 300000000:
            print("[DOWNLOAD] LibriSpeech dev-clean 다운로드 중...")
            url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
            urllib.request.urlretrieve(url, tar_path)
            
        print("[DOWNLOAD] 압축 해제 중...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path="./data/")
            
        if os.path.exists(tar_path):
            try:
                os.remove(tar_path)
            except Exception:
                pass
        print(f"[SUCCESS] 데이터셋 다운로드 및 압축 해제 완료.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import soundfile as sf
        data, sr = sf.read(self.files[idx], dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        wav = torch.from_numpy(data)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        if sr != self.sample_rate:
            if sr not in self.resamplers:
                import torchaudio.transforms as T
                self.resamplers[sr] = T.Resample(sr, self.sample_rate)
            wav = self.resamplers[sr](wav)

        if wav.shape[-1] > self.segment_len:
            max_start = wav.shape[-1] - self.segment_len
            start = torch.randint(0, max_start + 1, (1,)).item()
            wav = wav[:, start : start + self.segment_len]
        else:
            wav = F.pad(wav, (0, self.segment_len - wav.shape[-1]))

        msg = torch.randint(0, 2, (16,))
        return wav, msg


# =======================================================
# 2. 미분 가능한 채널 및 Reconstruction 왜곡
# =======================================================
class DifferentiableDistortion(nn.Module):
    """Autograd 그래프를 유지하며 오디오 왜곡(AWGN, LPF, BPF, Resample, VAE Rec, Spectral Compression)을 구현하는 레이어."""

    def __init__(self, sr=16000, vae=None):
        super().__init__()
        self.sr = sr
        self.vae = vae

    def add_awgn(self, wav, snr_db=20.0, seed=None):
        """Additive White Gaussian Noise (미분 가능)."""
        if seed is not None:
            old_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        noise = torch.randn_like(wav)
        if seed is not None:
            torch.random.set_rng_state(old_state)

        wav_pwr = torch.sum(wav ** 2, dim=-1, keepdim=True)
        noise_pwr = torch.sum(noise ** 2, dim=-1, keepdim=True)
        scale = torch.sqrt(wav_pwr / (noise_pwr * (10 ** (snr_db / 10)) + 1e-8))
        return wav + scale * noise

    def lowpass_filter(self, wav, cutoff_hz=4000):
        """간단한 FIR 기반 lowpass (미분 가능)."""
        n_taps = 101
        t = torch.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=wav.dtype, device=wav.device)
        fc = cutoff_hz / self.sr
        kernel = 2 * fc * torch.sinc(2 * fc * t)
        kernel = kernel * torch.hann_window(n_taps, device=wav.device)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        if wav.dim() == 2:
            wav_3d = wav.unsqueeze(1)
        else:
            wav_3d = wav
        filtered = F.conv1d(wav_3d, kernel, padding=n_taps // 2)
        return filtered.squeeze(1) if wav.dim() == 2 else filtered

    def bandpass_filter(self, wav, low_hz=300, high_hz=3400):
        """FIR 기반 bandpass 필터 (300Hz - 3.4kHz, 미분 가능)."""
        n_taps = 101
        t = torch.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=wav.dtype, device=wav.device)
        
        fc_low = low_hz / self.sr
        lp_low = 2 * fc_low * torch.sinc(2 * fc_low * t)
        lp_low = lp_low * torch.hann_window(n_taps, device=wav.device)
        lp_low = lp_low / lp_low.sum()
        hp_kernel = -lp_low
        hp_kernel[n_taps // 2] += 1.0
        
        fc_high = high_hz / self.sr
        lp_high = 2 * fc_high * torch.sinc(2 * fc_high * t)
        lp_high = lp_high * torch.hann_window(n_taps, device=wav.device)
        lp_high = lp_high / lp_high.sum()
        
        if wav.dim() == 2:
            wav_3d = wav.unsqueeze(1)
            was_2d = True
        else:
            wav_3d = wav
            was_2d = False
            
        hp_kernel = hp_kernel.view(1, 1, -1)
        lp_high = lp_high.view(1, 1, -1)
        
        filtered = F.conv1d(wav_3d, hp_kernel, padding=n_taps // 2)
        filtered = F.conv1d(filtered, lp_high, padding=n_taps // 2)
        return filtered.squeeze(1) if was_2d else filtered

    def resample_distortion(self, wav, down_rate=2):
        """F.interpolate를 이용한 1D 미분 가능 다운/업샘플링 왜곡."""
        if wav.dim() == 2:
            wav_3d = wav.unsqueeze(1)
            was_2d = True
        else:
            wav_3d = wav
            was_2d = False
            
        orig_len = wav_3d.shape[-1]
        downsampled = F.interpolate(wav_3d, size=orig_len // down_rate, mode="linear", align_corners=False)
        upsampled = F.interpolate(downsampled, size=orig_len, mode="linear", align_corners=False)
        return upsampled.squeeze(1) if was_2d else upsampled

    def speech_reconstruct(self, wav, n_q=8):
        """SpeechTokenizer 코덱 기반 복원 왜곡."""
        if self.vae is None:
            return wav
            
        was_2d = False
        if wav.dim() == 2:
            wav_3d = wav.unsqueeze(1)
            was_2d = True
        else:
            wav_3d = wav

        with torch.no_grad():
            feat = self.vae.encoder(wav_3d)
            quantized_full, _, _, _ = self.vae.quantizer(
                feat, n_q=n_q, layers=list(range(n_q)), st=0
            )
            wav_rec = self.vae.decoder(quantized_full)
            min_len = min(wav_3d.shape[-1], wav_rec.shape[-1])
            wav_rec_trim = wav_rec[..., :min_len]
            if wav_3d.shape[-1] > min_len:
                wav_rec_final_raw = F.pad(wav_rec_trim, (0, wav_3d.shape[-1] - min_len))
            else:
                wav_rec_final_raw = wav_rec_trim

        wav_rec_final = wav_3d + (wav_rec_final_raw - wav_3d).detach()
        return wav_rec_final.squeeze(1) if was_2d else wav_rec_final

    def mp3_proxy(self, wav, cutoff_ratio=0.7, noise_scale=0.002, seed=None):
        """MP3 압축에 따른 스펙트럼 마스킹 및 양자화 손실을 모사하는 미분 가능 프록시."""
        if wav.dim() == 2:
            wav_2d = wav
            was_2d = True
        else:
            wav_2d = wav.squeeze(1)
            was_2d = False

        spec = stft_audio(wav_2d, n_fft=256, hop_length=64)
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        
        F_bins = mag.shape[1]
        cutoff_bin = int(F_bins * cutoff_ratio)
        
        mask = torch.ones_like(mag)
        mask[:, cutoff_bin:] = 0.1
        mag_masked = mag * mask
        
        if seed is not None:
            old_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        noise = torch.randn_like(mag_masked) * noise_scale
        if seed is not None:
            torch.random.set_rng_state(old_state)

        mag_noisy = torch.clamp(mag_masked + noise, min=1e-8)
        spec_reconstructed = torch.polar(mag_noisy, phase)
        wav_rec = istft_audio(spec_reconstructed, length=wav_2d.shape[-1], n_fft=256, hop_length=64)
        return wav_rec if was_2d else wav_rec.unsqueeze(1)

    def forward(self, wav, dtype="noise", **kwargs):
        seed = kwargs.get("seed", None)
        if dtype == "noise":
            snr_db = kwargs.get("snr_db", 20.0)
            return self.add_awgn(wav, snr_db=snr_db, seed=seed)
        elif dtype == "lowpass":
            cutoff = kwargs.get("cutoff_hz", 4000)
            return self.lowpass_filter(wav, cutoff_hz=cutoff)
        elif dtype == "bandpass":
            low = kwargs.get("low_hz", 300)
            high = kwargs.get("high_hz", 3400)
            return self.bandpass_filter(wav, low_hz=low, high_hz=high)
        elif dtype == "resample":
            rate = kwargs.get("down_rate", 2)
            return self.resample_distortion(wav, down_rate=rate)
        elif dtype == "reconstruct":
            n_q = kwargs.get("n_q", 8)
            return self.speech_reconstruct(wav, n_q=n_q)
        elif dtype == "mp3":
            ratio = kwargs.get("cutoff_ratio", 0.7)
            ns = kwargs.get("noise_scale", 0.002)
            return self.mp3_proxy(wav, cutoff_ratio=ratio, noise_scale=ns, seed=seed)
        return wav


# =======================================================
# 2.1 Paired AWGN 및 스케일 공유 함수
# =======================================================
def paired_awgn(clean, watermarked, snr_db):
    """clean 오디오 파워 스케일을 기준으로 완벽하게 정합된 paired 노이즈를 clean/wm 양측에 공유해 주입."""
    noise = torch.randn_like(clean)
    clean_power = clean.pow(2).mean(dim=-1, keepdim=True)
    noise_power = noise.pow(2).mean(dim=-1, keepdim=True)
    scale = torch.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10)) + 1e-8))
    shared_noise = scale * noise
    return clean + shared_noise, watermarked + shared_noise


# =======================================================
# 3. Feature Maps (Perceptual, Survival & VAD)
# =======================================================
def normalize_per_sample(x, eps=1e-8):
    """피처 맵 간 스케일 불균형을 차단하기 위한 sample-wise 정규화."""
    mean = x.reshape(x.shape[0], -1).mean(dim=-1).view(-1, 1, 1)
    std = x.reshape(x.shape[0], -1).std(dim=-1).view(-1, 1, 1)
    return (x - mean) / (std + eps)


def get_local_energy_masking_proxy(wav, n_fft=256, hop_length=64):
    """STFT 기반 국소 에너지 기반 마스킹 맵 (Local Spectral Energy Masking Proxy, no grad)."""
    with torch.no_grad():
        if wav.dim() == 3:
            wav_2d = wav.squeeze(1)
        else:
            wav_2d = wav
        mag = torch.abs(stft_audio(wav_2d, n_fft=n_fft, hop_length=hop_length))
        log_mag = torch.log10(mag + 1e-5)
        kernel = torch.ones(1, 1, 5, 5, device=wav.device) / 25.0
        smoothed = F.conv2d(log_mag.unsqueeze(1), kernel, padding=2)
        min_v = smoothed.view(wav_2d.shape[0], -1).min(dim=-1)[0].view(-1, 1, 1, 1)
        max_v = smoothed.view(wav_2d.shape[0], -1).max(dim=-1)[0].view(-1, 1, 1, 1)
        return ((smoothed - min_v) / (max_v - min_v + 1e-8)).squeeze(1)


def get_survival_map(wav_clean, wav_wm, distorter, n_fft=256, hop_length=64):
    """
    5개 왜곡 채널의 SIR(Signal-to-Interference Ratio)을 반영하여 q_T를 구한 후,
    하위 25% 분위수(q=0.25)로 집계하고 2D Avg Pooling으로 평활화한 Survival Map 반환 (no grad).
    """
    with torch.no_grad():
        if wav_clean.dim() == 3:
            wav_clean_2d = wav_clean.squeeze(1)
            wav_wm_2d = wav_wm.squeeze(1)
        else:
            wav_clean_2d = wav_clean
            wav_wm_2d = wav_wm

        B, T = wav_clean_2d.shape
        
        # 1. AWGN (Paired Noise)
        awgn_clean, awgn_wm = paired_awgn(wav_clean_2d, wav_wm_2d, snr_db=20.0)
        
        # 2. Lowpass
        lp_clean = distorter(wav_clean_2d, "lowpass", cutoff_hz=4000)
        lp_wm = distorter(wav_wm_2d, "lowpass", cutoff_hz=4000)
        
        # 3. Bandpass
        bp_clean = distorter(wav_clean_2d, "bandpass", low_hz=300, high_hz=3400)
        bp_wm = distorter(wav_wm_2d, "bandpass", low_hz=300, high_hz=3400)
        
        # 4. Resample
        rs_clean = distorter(wav_clean_2d, "resample", down_rate=2)
        rs_wm = distorter(wav_wm_2d, "resample", down_rate=2)
        
        # 5. VAE Reconstruct (n_q=6)
        rec_clean = distorter(wav_clean_2d, "reconstruct", n_q=6)
        rec_wm = distorter(wav_wm_2d, "reconstruct", n_q=6)

        # 6. MP3 Proxy
        mp3_clean = distorter(wav_clean_2d, "mp3", cutoff_ratio=0.7, seed=42)
        mp3_wm = distorter(wav_wm_2d, "mp3", cutoff_ratio=0.7, seed=42)

        # STFT 계산
        spec_clean = stft_audio(wav_clean_2d, n_fft=n_fft, hop_length=hop_length)
        spec_wm = stft_audio(wav_wm_2d, n_fft=n_fft, hop_length=hop_length)
        
        r0_spec = spec_wm - spec_clean
        r0_mag = torch.abs(r0_spec) + 1e-8

        # 6개 공격 시나리오에 대해 q_ret 및 q_sir를 결합해 q_T 산출
        attacks = [
            (awgn_clean, awgn_wm),
            (lp_clean, lp_wm),
            (bp_clean, bp_wm),
            (rs_clean, rs_wm),
            (rec_clean, rec_wm),
            (mp3_clean, mp3_wm),
        ]
        
        q_t_list = []
        for a_clean, a_wm in attacks:
            spec_d_clean = stft_audio(a_clean, n_fft=n_fft, hop_length=hop_length)
            spec_d_wm = stft_audio(a_wm, n_fft=n_fft, hop_length=hop_length)
            
            # 잡음 magnitude N_T
            noise_t_mag = torch.abs(spec_d_clean - spec_clean)
            # 왜곡 후 잔차 magnitude R_T
            rt_mag = torch.abs(spec_d_wm - spec_d_clean)
            
            # Residual retention ratio
            q_ret = torch.clamp(rt_mag / r0_mag, 0.0, 1.0)
            
            # Signal-to-Interference Ratio
            q_sir = rt_mag / (rt_mag + noise_t_mag + 1e-8)
            
            q_t = q_ret * q_sir  # (B, F, T_stft)
            q_t_list.append(q_t)

        q_t_stack = torch.stack(q_t_list, dim=0)
        
        # 하위 25% 분위수로 집계
        survival_map = torch.quantile(q_t_stack, q=0.25, dim=0)
        
        # 2D Average Pooling
        s_map_smoothed = F.avg_pool2d(
            survival_map.unsqueeze(1),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze(1)
        
        return s_map_smoothed


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

    def __init__(self, device, alignmark_dir=None):
        if alignmark_dir is None:
            alignmark_dir = ALIGNMARK_DIR
        self.device = device
        self.alignmark_dir = alignmark_dir

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
        self.vae = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.wm_model = WatermarkModel(self.cfg).to(device)
        self.fusion = AudioFusionModel(
            n_fft=256, hop_length=64, win_length=256, hidden_dim=64, nbits=16
        ).to(device)

        weight_path = os.path.join(alignmark_dir, "weight.pth")
        if os.path.exists(weight_path):
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
        """워터마크 삽입."""
        feat = self.vae.encode(wav)
        feat_wm = self.wm_model(feat, msg)
        wav_wm = self.vae.decode(feat_wm)
        min_len = min(wav.shape[-1], wav_wm.shape[-1])
        wav_trim = wav[..., :min_len]
        wav_wm_trim = wav_wm[..., :min_len]
        wav_fused = self.fusion(wav_trim, wav_wm_trim)
        residual = wav_fused - wav_trim
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
        VQ가 없는 SEANetEncoder -> Detector이므로 완벽히 미분 가능합니다.
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
    디코딩된 logits의 Shannon Entropy, Margin(top1-top2), Max Probability를 입력받아
    워터마크 유무를 이진 분류하는 판별기.
    """

    def __init__(self, n_features=12):
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
        # Decoding evidence 피처 추출 (Entropy & Margin & Max Probability)
        probs = F.softmax(chunk_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B, 4)
        
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]         # (B, 4)
        max_prob = sorted_probs[:, :, 0]                              # (B, 4)
        
        features = torch.cat([entropy, margin, max_prob], dim=-1)     # (B, 12)
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


def compute_si_sdr(reference, estimated, eps=1e-8):
    """Scale-Invariant Signal-to-Distortion Ratio 계산."""
    # reference, estimated: (B, T) or numpy arrays
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference)
    if isinstance(estimated, np.ndarray):
        estimated = torch.from_numpy(estimated)
        
    reference = reference.view(reference.shape[0], -1)
    estimated = estimated.view(estimated.shape[0], -1)
    
    dot_product = torch.sum(reference * estimated, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    
    scaled_ref = (dot_product / ref_energy) * reference
    noise = estimated - scaled_ref
    
    scaled_ref_energy = torch.sum(scaled_ref ** 2, dim=-1, keepdim=True)
    noise_energy = torch.sum(noise ** 2, dim=-1, keepdim=True) + eps
    
    sdr = 10 * torch.log10(scaled_ref_energy / noise_energy + eps)
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
def run_training():
    print("=" * 60)
    print("[START] SurvAlign-P URP 2차 수정 훈련 파이프라인 시작")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    # ---- 모델 로드 ----
    print("\n[초기화] AlignMark 모델 로드...")
    alignmark = AlignMarkManager(device)

    # ---- 데이터 로드 (화자 격리 분할 적용) ----
    print("\n[초기화] 데이터셋 로드...")
    train_dataset = RealLibriSpeechDataset(download=True, split="train")
    calib_dataset = RealLibriSpeechDataset(download=False, split="calib")
    test_dataset = RealLibriSpeechDataset(download=False, split="test")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    calib_loader = DataLoader(calib_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    distorter = DifferentiableDistortion(vae=alignmark.vae).to(device)

    # STFT 파라미터
    n_fft = 256
    hop_length = 64
    freq_bins = n_fft // 2 + 1  # 129

    # ---- 모델 선언 ----
    # 6채널 입력 스택: wav_mag_normalized, r0_mag_normalized, s_map_smoothed, p_map, vad_prior, res_prior
    gate = SurvivalGate(in_channels=6, hidden_dim=32).to(device)
    presence = PresenceHead(n_features=12).to(device)

    opt_gate = optim.AdamW(gate.parameters(), lr=1e-4)
    opt_presence = optim.AdamW(presence.parameters(), lr=5e-4)

    # ================================================================
    # STAGE 1: Survival Gate 학습 (실제 디코더 직접 역전파)
    # ================================================================
    print("\n" + "=" * 60)
    print("[Stage 1] Survival Gate 학습 (디코더 직접 역전파 - VQ 없음)")
    print("  목적: 워터마크 잔차의 주파수-시간별 가중치를 학습하여")
    print("        왜곡 내성을 높이면서 청각 품질을 유지")
    print("=" * 60)

    gate.train()
    n_gate_steps = 30  # 예비 시뮬레이션용 스텝

    for step, (wav, msg) in enumerate(train_loader):
        if step >= n_gate_steps:
            break

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
        if dist_mode == 0:
            wav_distorted = distorter(wav_final, "noise", snr_db=15.0, seed=attack_seed)
        elif dist_mode == 1:
            wav_distorted = distorter(wav_final, "lowpass", cutoff_hz=4000)
        elif dist_mode == 2:
            wav_distorted = distorter(wav_final, "bandpass", low_hz=300, high_hz=3400)
        elif dist_mode == 3:
            wav_distorted = distorter(wav_final, "resample", down_rate=2)
        elif dist_mode == 4:
            wav_distorted = distorter(wav_final, "reconstruct", n_q=6)
        else:
            wav_distorted = distorter(wav_final, "mp3", cutoff_ratio=0.7, seed=attack_seed)

        # h. 디코더를 직접 통과하여 BER Loss 계산 (이미 미분 가능!)
        _, chunk_logits = alignmark.decode_logits_with_grad(wav_distorted.unsqueeze(1))

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
    n_presence_steps = 20

    for step, (wav, msg) in enumerate(train_loader):
        if step >= n_presence_steps:
            break

        wav, msg = wav.to(device), msg.to(device)
        opt_presence.zero_grad()

        # 학습 시 정직하게 distorted positive/negative를 인풋으로 주입해 노이즈 오탐 방지
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

            # Positive 왜곡 적용
            pos_dist = distorter(wav_wm_base.squeeze(1), "noise", snr_db=20.0, seed=attack_seed).unsqueeze(1)
            _, pos_chunk_logits, _ = alignmark.decode(pos_dist)

            # Negative 왜곡 적용
            neg_dist = distorter(wav.squeeze(1), "noise", snr_db=20.0, seed=attack_seed).unsqueeze(1)
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
    print("  목적: 명목상 FPR <= 1% 수준을 달성하는 임계값(tau_p) 결정")
    print("=" * 60)

    presence.eval()
    calib_negative_scores = []
    
    with torch.no_grad():
        for step, (wav, msg) in enumerate(calib_loader):
            wav = wav.to(device)
            # Calibration 화자들의 오디오 왜곡 적용 (Negative)
            neg_dist = distorter(wav.squeeze(1), "noise", snr_db=20.0, seed=500+step).unsqueeze(1)
            _, neg_chunk_logits, _ = alignmark.decode(neg_dist)
            neg_probs = presence(neg_chunk_logits)
            calib_negative_scores.extend(neg_probs.cpu().numpy().tolist())

    # Calibration negative score의 99%ile로 임계값 결정해 데이터 누수 격리
    tau_p = np.quantile(calib_negative_scores, 0.99)
    print(f"[CALIBRATION] Calibration Negative Scores 개수: {len(calib_negative_scores)}")
    print(f"[CALIBRATION] 결정된 Presence 임계값 (tau_p at nominal FPR<=1%): {tau_p:.4f}")

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
    
    n_eval_steps = 10  # 예비 평가용 샘플 수

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
                "Spectral Compression Proxy (70%)": lambda w: distorter(w.squeeze(1), "mp3", cutoff_ratio=0.7, seed=attack_seed).unsqueeze(1)
            }
            
            for dist_name, dist_fn in distortions.items():
                w_base_dist = dist_fn(wav_wm_base)
                _, _, pred_base = alignmark.decode(w_base_dist)
                ber_base = compute_ber(pred_base, msg)
                results[dist_name]["base_ber"].append(ber_base)
                
                w_gated_dist = dist_fn(wav_gated)
                _, _, pred_gated = alignmark.decode(w_gated_dist)
                ber_gated = compute_ber(pred_gated, msg)
                results[dist_name]["gated_ber"].append(ber_gated)

            # Test 화자 Presence 검증용 positive/negative 왜곡 수집
            pos_dist_test = distorter(wav_gated.squeeze(1), "noise", snr_db=20.0, seed=attack_seed).unsqueeze(1)
            _, pos_chunk_logits, _ = alignmark.decode(pos_dist_test)
            p_pos = presence(pos_chunk_logits)
            test_positive_scores.extend(p_pos.cpu().numpy().tolist())

            neg_dist_test = distorter(wav.squeeze(1), "noise", snr_db=20.0, seed=attack_seed).unsqueeze(1)
            _, neg_chunk_logits, _ = alignmark.decode(neg_dist_test)
            p_neg = presence(neg_chunk_logits)
            test_negative_scores.extend(p_neg.cpu().numpy().tolist())

            # 4) 오디오 품질 1:1 비교
            for i in range(B):
                ref = wav_2d[i].cpu().numpy()
                base_deg = wav_wm_base[i, 0].cpu().numpy()
                gated_deg = wav_gated[i, 0].cpu().numpy()
                
                # PESQ WB
                try:
                    pesq_base_scores.append(pesq(16000, ref, base_deg, "wb"))
                except Exception: pass
                try:
                    pesq_gated_scores.append(pesq(16000, ref, gated_deg, "wb"))
                except Exception: pass
                
                # STOI
                try:
                    stoi_base_scores.append(compute_stoi(ref, base_deg, 16000, extended=False))
                except Exception: pass
                try:
                    stoi_gated_scores.append(compute_stoi(ref, gated_deg, 16000, extended=False))
                except Exception: pass
                
                # SI-SDR
                sdr_base_scores.append(compute_si_sdr(ref[np.newaxis, :], base_deg[np.newaxis, :]))
                sdr_gated_scores.append(compute_si_sdr(ref[np.newaxis, :], gated_deg[np.newaxis, :]))

    # ---- 결과 출력 및 정리 ----
    print("\n" + "=" * 60)
    print("[REPORT] 종합 리포트")
    print("=" * 60)
    print(f"| {'왜곡 조건 (Distortion)':<35} | {'기본형 (AlignMark)':<20} | {'제안형 (SurvAlign-P)':<20} | {'개선도':<8} |")
    print(f"|{'-'*37}|{'-'*22}|{'-'*22}|{'-'*10}|")
    
    for dist_name in results.keys():
        b_ber = np.mean(results[dist_name]["base_ber"])
        g_ber = np.mean(results[dist_name]["gated_ber"])
        diff = b_ber - g_ber
        print(f"| {dist_name:<35} | {b_ber:<20.4f} | {g_ber:<20.4f} | {diff:+.4f} |")
        
    print(f"|{'-'*37}|{'-'*22}|{'-'*22}|{'-'*10}|")

    # AUC 및 FPR/TPR 탐지력 검증
    test_positive_scores = np.array(test_positive_scores)
    test_negative_scores = np.array(test_negative_scores)
    all_scores = np.concatenate([test_positive_scores, test_negative_scores])
    all_labels = np.concatenate([np.ones_like(test_positive_scores), np.zeros_like(test_negative_scores)])
    
    if len(all_scores) >= 2 and len(set(all_labels)) > 1:
        test_auc = roc_auc_score(all_labels, all_scores)
        actual_test_fpr = np.mean(test_negative_scores >= tau_p)
        actual_test_tpr = np.mean(test_positive_scores >= tau_p)
        print(f"\n[INFO] Detection AUROC on Test Speakers: {test_auc:.4f}")
        print(f"[INFO] Test FPR (at Nominal 1% Calibrated Threshold): {actual_test_fpr:.2%}")
        print(f"[INFO] Test TPR (at Nominal 1% Calibrated Threshold): {actual_test_tpr:.2%}")

    print("\n[INFO] 1:1 오디오 품질 비교 지표:")
    print(f"  - PESQ WB | 기본형: {np.mean(pesq_base_scores):.3f} vs 제안형: {np.mean(pesq_gated_scores):.3f}")
    print(f"  - STOI    | 기본형: {np.mean(stoi_base_scores):.3f} vs 제안형: {np.mean(stoi_gated_scores):.3f}")
    print(f"  - SI-SDR   | 기본형: {np.mean(sdr_base_scores):.2f} dB vs 제안형: {np.mean(sdr_gated_scores):.2f} dB")

    # L2 에너지 보존율 재검증
    print("\n[SUCCESS] SurvAlign-P 전체 시뮬레이션 완료!")


if __name__ == "__main__":
    run_training()

