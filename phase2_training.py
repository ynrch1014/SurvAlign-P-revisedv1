# -*- coding: utf-8 -*-
"""
Phase 2: Survival Gate Full Training & Evaluation Engine (본학습용)
Author: 정연재 (SKKU URP)

지원 모드 (--mode):
1. baseline: 아무런 처리도 하지 않은 AlignMark 원본
2. uniform: 에너지(L2) 제약을 맞추기 위해 균일하게 스케일링
3. random_gate: 무작위 T-F 맵을 가이드로 사용하는 Gate 학습
4. proposed_gate: Survival Map을 가이드로 사용하는 제안된 Gate 학습
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from pesq import pesq
from pystoi import stoi as compute_stoi

# 기존 인프라 재사용
from survalign_p import (
    AlignMarkManager, 
    RealLibriSpeechDataset,
    UnifiedSpeechDataset,
    DifferentiableDistortion,
    stft_audio, 
    istft_audio, 
    get_survival_map,
    compute_chunk_ce_loss,
    compute_ber,
    compute_si_sdr,
    chunks_to_bits,
    normalize_per_sample
)

from phase1_attribution import compute_decoder_gradient_map

class SimplifiedSurvivalGate(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, feature_pack, R0_complex):
        logits = self.conv(feature_pack).squeeze(1)
        # [0.8, 1.2] 범위로 제한
        gate_scale = 1.0 + 0.2 * torch.tanh(logits)
        R_gated = R0_complex * gate_scale
        return R_gated, gate_scale


def train_gate(args, device, alignmark, distorter, dataset_train, dataset_val):
    print(f"\n[TRAIN] Starting Full Gate Training in mode: {args.mode}")
    print(f"[INFO] Dataset: {args.dataset_type}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    
    gate = SimplifiedSurvivalGate(in_channels=3).to(device)
    optimizer = optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 본학습용 대용량 DataLoader 설정 (num_workers 지정)
    dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    dist_types = ["noise", "lowpass", "bandpass", "resample", "reconstruct", "mp3"]
    os.makedirs("./checkpoints", exist_ok=True)
    
    best_loss = float('inf')
    ckpt_name = f"best_gate_{args.dataset_type}_{args.mode}_{args.map_type}.pth" if args.mode == "proposed_gate" else f"best_gate_{args.dataset_type}_{args.mode}.pth"
    ckpt_path = f"./checkpoints/{ckpt_name}"
    
    for epoch in range(1, args.epochs + 1):
        gate.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for wav, msg in pbar:
            wav = wav.to(device)
            msg = msg.to(device)
            B = wav.shape[0]
            
            with torch.no_grad():
                wav_wm, residual = alignmark.embed(wav, msg)
                spec_clean = stft_audio(wav.squeeze(1), n_fft=256, hop_length=64)
                spec_wm = stft_audio(wav_wm.squeeze(1), n_fft=256, hop_length=64)
                r0_complex = spec_wm - spec_clean
                
                wav_mag = normalize_per_sample(torch.abs(spec_clean))
                res_mag = normalize_per_sample(torch.abs(r0_complex) + 1e-8)
                
                if args.mode == "random_gate":
                    guide_map = torch.rand_like(wav_mag)
                elif args.mode == "proposed_gate" and args.map_type == "gradient":
                    with torch.enable_grad():
                        guide_map = compute_decoder_gradient_map(alignmark, wav_wm, msg).detach()
                else: # proposed_gate
                    guide_map = get_survival_map(wav, wav_wm, distorter)
                guide_map = normalize_per_sample(guide_map)
                
            feature_pack = torch.stack([wav_mag, res_mag, guide_map], dim=1)
            
            optimizer.zero_grad()
            r_gated_complex, gate_scale = gate(feature_pack, r0_complex)
            
            # 시간 도메인 변환 후 L2 Projection 적용 (Energy Cheating 엄격 차단)
            r0_2d = residual.squeeze(1)
            wav_2d = wav.squeeze(1)
            r_gated = istft_audio(r_gated_complex, length=wav_2d.shape[-1], n_fft=256, hop_length=64)
            
            norm_r0 = torch.norm(r0_2d, p=2, dim=-1, keepdim=True) + 1e-8
            norm_gated = torch.norm(r_gated, p=2, dim=-1, keepdim=True) + 1e-8
            scale_factor = torch.minimum(torch.tensor(1.0, device=device), norm_r0 / norm_gated)
            
            r_gated_final = r_gated * scale_factor
            wav_gated = (wav_2d + r_gated_final).unsqueeze(1)
            
            # Loss 계산
            loss_robust = 0
            for d_type in dist_types:
                wav_dist = distorter(wav_gated, dtype=d_type)
                _, chunk_logits = alignmark.decode_logits_with_grad(wav_dist)
                loss_robust += compute_chunk_ce_loss(chunk_logits, msg)
            loss_robust = loss_robust / len(dist_types)
            
            baseline_energy = torch.sum(residual**2, dim=-1)
            gated_energy = torch.sum((wav_gated - wav)**2, dim=-1)
            loss_energy = torch.mean(F.relu(gated_energy - baseline_energy)) * 100.0
            loss_dev = torch.mean((gate_scale - 1.0)**2) * 10.0
            
            loss_total = loss_robust + loss_energy + loss_dev
            loss_total.backward()
            optimizer.step()
            
            epoch_loss += loss_total.item()
            pbar.set_postfix({"L_rob": f"{loss_robust.item():.3f}", "L_egy": f"{loss_energy.item():.3f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")
        
        # 간단한 Validation 로직 (학습 손실 기반으로 최고 모델 저장)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(gate.state_dict(), ckpt_path)
            print(f"[SAVE] New best model saved to {ckpt_path}")
            
    # 평가를 위해 최고 성능 모델 로드
    gate.load_state_dict(torch.load(ckpt_path))
    return gate

def evaluate(args, device, alignmark, distorter, dataset_test, gate=None):
    print(f"\n[EVAL] Starting Full Evaluation in mode: {args.mode} on dataset: {args.dataset_type}")
    dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    if gate is not None:
        gate.eval()
        
    dist_types = ["Clean", "noise", "lowpass", "bandpass", "resample", "reconstruct", "mp3"]
    results = {d: [] for d in dist_types}
    metrics = {"PESQ": [], "STOI": [], "SI_SDR": [], "L2_Ratio": []}
    
    with torch.no_grad():
        for wav, msg in tqdm(dataloader, desc="Evaluating"):
            wav = wav.to(device)
            msg = msg.to(device)
            B = wav.shape[0]
            
            wav_wm, residual = alignmark.embed(wav, msg)
            
            if args.mode == "baseline":
                wav_final = wav_wm
            elif args.mode == "uniform":
                wav_final = wav + residual * 1.1
            else:
                spec_clean = stft_audio(wav.squeeze(1), n_fft=256, hop_length=64)
                spec_wm = stft_audio(wav_wm.squeeze(1), n_fft=256, hop_length=64)
                r0_complex = spec_wm - spec_clean
                
                wav_mag = normalize_per_sample(torch.abs(spec_clean))
                res_mag = normalize_per_sample(torch.abs(r0_complex) + 1e-8)
                
                if args.mode == "random_gate":
                    guide_map = torch.rand_like(wav_mag)
                elif args.mode == "proposed_gate" and args.map_type == "gradient":
                    with torch.enable_grad():
                        guide_map = compute_decoder_gradient_map(alignmark, wav_wm, msg).detach()
                else:
                    guide_map = get_survival_map(wav, wav_wm, distorter)
                guide_map = normalize_per_sample(guide_map)
                
                feature_pack = torch.stack([wav_mag, res_mag, guide_map], dim=1)
                r_gated_complex, gate_scale = gate(feature_pack, r0_complex)
                
                # 평가 시에도 L2 Projection 엄격 적용
                r0_2d = residual.squeeze(1)
                wav_2d = wav.squeeze(1)
                r_gated = istft_audio(r_gated_complex, length=wav_2d.shape[-1], n_fft=256, hop_length=64)
                
                norm_r0 = torch.norm(r0_2d, p=2, dim=-1, keepdim=True) + 1e-8
                norm_gated = torch.norm(r_gated, p=2, dim=-1, keepdim=True) + 1e-8
                scale_factor = torch.minimum(torch.tensor(1.0, device=device), norm_r0 / norm_gated)
                
                r_gated_final = r_gated * scale_factor
                wav_final = (wav_2d + r_gated_final).unsqueeze(1)
            
            # Fidelity Metrics
            for b in range(B):
                c_np = wav[b].squeeze().cpu().numpy()
                f_np = wav_final[b].squeeze().cpu().numpy()
                try:
                    metrics["PESQ"].append(pesq(16000, c_np, f_np, 'wb'))
                except:
                    metrics["PESQ"].append(1.0)
                metrics["STOI"].append(compute_stoi(c_np, f_np, 16000, extended=False))
                metrics["SI_SDR"].append(compute_si_sdr(c_np, f_np))
                
                res_l2 = torch.sum(residual[b]**2).item()
                final_res_l2 = torch.sum((wav_final[b] - wav[b])**2).item()
                metrics["L2_Ratio"].append(final_res_l2 / (res_l2 + 1e-8))
                
            # Robustness Metrics
            for d_type in dist_types:
                if d_type == "Clean":
                    wav_dist = wav_final
                else:
                    wav_dist = distorter(wav_final, dtype=d_type)
                    
                _, chunk_logits, _ = alignmark.decode(wav_dist)
                pred_bits = chunks_to_bits(chunk_logits.argmax(dim=-1), 4)
                
                for b in range(B):
                    ber = compute_ber(pred_bits[b], msg[b])
                    results[d_type].append(1.0 - ber)
                    
    # 결과 로깅
    print("\n" + "="*50)
    print(f" Evaluation Results ({args.mode}) on Full Test Set")
    print("="*50)
    print(f"[Fidelity Metrics]")
    for k, v in metrics.items():
        print(f" - {k}: {np.mean(v):.4f} ± {np.std(v):.4f}")
        
    print(f"\n[Robustness (Bit Accuracy)]")
    for d_type, accs in results.items():
        print(f" - {d_type:<12}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print("="*50)
    
    # CSV 저장 (논문 작성용 테이블화)
    os.makedirs("./results", exist_ok=True)
    csv_file = "./results/phase2_results.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ["Dataset", "Mode", "PESQ", "STOI", "SI-SDR", "L2_Ratio"] + dist_types
            writer.writerow(headers)
            
        row = [args.dataset_type, args.mode, 
               f"{np.mean(metrics['PESQ']):.4f}", 
               f"{np.mean(metrics['STOI']):.4f}", 
               f"{np.mean(metrics['SI_SDR']):.4f}",
               f"{np.mean(metrics['L2_Ratio']):.4f}"]
        
        for d_type in dist_types:
            row.append(f"{np.mean(results[d_type]):.4f}")
            
        writer.writerow(row)
    print(f"[INFO] Results saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["baseline", "uniform", "random_gate", "proposed_gate"])
    parser.add_argument("--dataset_type", type=str, default="librispeech",
                        choices=["librispeech", "vctk", "ljspeech"],
                        help="사용할 데이터셋 유형 (librispeech, vctk, ljspeech)")
    parser.add_argument("--dataset_name", type=str, default="train-clean-100", 
                        help="LibriSpeech 서브셋 이름 (예: train-clean-100, dev-clean)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (GPU 메모리에 맞춰 조절)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--map_type", type=str, default="survival",
                        choices=["survival", "gradient"],
                        help="proposed_gate 모드에서 사용할 가이드 맵 유형")
    parser.add_argument("--test_only", action="store_true", help="학습을 생략하고 저장된 체크포인트로 평가만 수행")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Dataset Type: {args.dataset_type}")
    
    alignmark = AlignMarkManager(device)
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    
    # 데이터셋 로드 (UnifiedSpeechDataset 활용)
    print(f"[INFO] Loading Dataset: {args.dataset_type} (Train/Val/Test Split)")
    dataset_train = UnifiedSpeechDataset(
        dataset_type=args.dataset_type, dataset_name=args.dataset_name, split="train"
    )
    dataset_val = UnifiedSpeechDataset(
        dataset_type=args.dataset_type, dataset_name=args.dataset_name, split="calib"
    )
    dataset_test = UnifiedSpeechDataset(
        dataset_type=args.dataset_type, dataset_name=args.dataset_name, split="test"
    )
    
    gate = None
    if args.mode in ["random_gate", "proposed_gate"]:
        if args.test_only:
            print(f"[INFO] Test Only mode: Loading pretrained checkpoint...")
            gate = SimplifiedSurvivalGate(in_channels=3).to(device)
            ckpt_name = f"best_gate_{args.dataset_type}_{args.mode}_{args.map_type}.pth" if args.mode == "proposed_gate" else f"best_gate_{args.dataset_type}_{args.mode}.pth"
            ckpt_path = f"./checkpoints/{ckpt_name}"
            if os.path.exists(ckpt_path):
                gate.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"[INFO] Successfully loaded {ckpt_path}")
            else:
                print(f"[WARNING] Checkpoint {ckpt_path} not found. Evaluating with untrained weights!")
        else:
            gate = train_gate(args, device, alignmark, distorter, dataset_train, dataset_val)
        
    evaluate(args, device, alignmark, distorter, dataset_test, gate=gate)

if __name__ == "__main__":
    main()
