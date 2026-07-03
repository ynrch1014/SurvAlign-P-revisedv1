# -*- coding: utf-8 -*-
"""
Phase 1: Attribution Correlation Analysis (Paper-Ready Version)
Author: 정연재 (SKKU URP)

본 스크립트는 SurvAlign-P의 핵심 가설인 "물리적 신호 보존량(Survival Map)이 
실제 Decoder가 필요로 하는 유틸리티(Decoder Gradient Map)와 일치하는가?"를
전체 테스트 데이터셋에서 대규모로 통계적 검증합니다.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, ttest_rel
import matplotlib.pyplot as plt
from tqdm import tqdm

# 기존 코드 인프라 재사용
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
    chunks_to_bits
)

def compute_decoder_gradient_map(alignmark, wav_wm, target_msg):
    wav_wm_grad = wav_wm.clone().detach()
    wav_wm_grad.requires_grad_(True)
    
    _, chunk_logits = alignmark.decode_logits_with_grad(wav_wm_grad)
    loss = compute_chunk_ce_loss(chunk_logits, target_msg)
    loss.backward()
    
    grad_stft = stft_audio(wav_wm_grad.grad.squeeze(1), n_fft=256, hop_length=64)
    grad_mag = torch.abs(grad_stft)
    
    b_min = grad_mag.reshape(grad_mag.shape[0], -1).min(dim=-1)[0].reshape(-1, 1, 1)
    b_max = grad_mag.reshape(grad_mag.shape[0], -1).max(dim=-1)[0].reshape(-1, 1, 1)
    decoder_map = (grad_mag - b_min) / (b_max - b_min + 1e-8)
    
    return decoder_map

def get_top_k_mask(map_tensor, top_ratio=0.2):
    B, F, T = map_tensor.shape
    masks = []
    for b in range(B):
        flat = map_tensor[b].cpu().numpy().flatten()
        k = max(1, int(len(flat) * top_ratio))
        threshold = np.partition(flat, -k)[-k]
        masks.append((map_tensor[b] >= threshold).float())
    return torch.stack(masks, dim=0).to(map_tensor.device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="librispeech",
                        choices=["librispeech", "vctk", "ljspeech"],
                        help="사용할 데이터셋 유형 (librispeech, vctk, ljspeech)")
    parser.add_argument("--dataset_name", type=str, default="train-clean-100", help="LibriSpeech 서브셋 (예: train-clean-100)")
    parser.add_argument("--split", type=str, default="test", help="검증에 사용할 분할 (test 또는 calib)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1, help="분석할 최대 샘플 수 (-1: 전체 데이터셋)")
    args = parser.parse_args()

    print("="*60)
    print("Phase 1: Attribution Correlation Analysis (Full Dataset)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    alignmark = AlignMarkManager(device)
    distorter = DifferentiableDistortion(sr=16000, vae=alignmark.vae).to(device)
    
    print(f"[INFO] Loading Dataset: {args.dataset_type} ({args.split} split)...")
    dataset = UnifiedSpeechDataset(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        split=args.split
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    processed_samples = 0
    all_pearson = []
    all_spearman = []
    all_iou = []
    
    masking_results = {
        "Full (Baseline)": [],
        "High-Survival (Top 20%)": [],
        "Low-Survival (Bottom 20%)": [],
        "High-Gradient (Top 20%)": [],
        "Low-Gradient (Bottom 20%)": [],
        "Random 20%": []
    }
    
    print("\n[INFO] Starting Analysis...")
    saved_plot = False
    
    for wav, msg in tqdm(dataloader, desc="Phase 1 Analysis"):
        if args.max_samples != -1 and processed_samples >= args.max_samples:
            break
            
        wav = wav.to(device)
        msg = msg.to(device)
        B = wav.shape[0]
        
        wav_wm, residual = alignmark.embed(wav, msg)
        
        survival_map = get_survival_map(wav, wav_wm, distorter)
        s_min = survival_map.reshape(B, -1).min(dim=-1)[0].reshape(-1, 1, 1)
        s_max = survival_map.reshape(B, -1).max(dim=-1)[0].reshape(-1, 1, 1)
        survival_map = (survival_map - s_min) / (s_max - s_min + 1e-8)
        
        decoder_map = compute_decoder_gradient_map(alignmark, wav_wm, msg)
        
        for b in range(B):
            s_flat = survival_map[b].cpu().numpy().flatten()
            d_flat = decoder_map[b].cpu().detach().numpy().flatten()
            
            r, _ = pearsonr(s_flat, d_flat)
            rho, _ = spearmanr(s_flat, d_flat)
            all_pearson.append(r)
            all_spearman.append(rho)
            
            s_top20_mask = s_flat >= np.percentile(s_flat, 80)
            d_top20_mask = d_flat >= np.percentile(d_flat, 80)
            intersection = np.logical_and(s_top20_mask, d_top20_mask).sum()
            union = np.logical_or(s_top20_mask, d_top20_mask).sum()
            iou = intersection / (union + 1e-8)
            all_iou.append(iou)
        
        res_spec = stft_audio(residual.squeeze(1), n_fft=256, hop_length=64)
        
        # Soft mask generation via Gaussian blur
        kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0
        
        mask_high_surv = get_top_k_mask(survival_map, top_ratio=0.2).unsqueeze(1)
        mask_low_surv = (1.0 - get_top_k_mask(survival_map, top_ratio=0.8)).unsqueeze(1)
        mask_high_grad = get_top_k_mask(decoder_map, top_ratio=0.2).unsqueeze(1)
        mask_low_grad = (1.0 - get_top_k_mask(decoder_map, top_ratio=0.8)).unsqueeze(1)
        mask_rand = (torch.rand_like(survival_map) >= 0.8).float().unsqueeze(1)
        
        mask_high_surv = F.conv2d(mask_high_surv, kernel, padding=2).squeeze(1)
        mask_low_surv = F.conv2d(mask_low_surv, kernel, padding=2).squeeze(1)
        mask_high_grad = F.conv2d(mask_high_grad, kernel, padding=2).squeeze(1)
        mask_low_grad = F.conv2d(mask_low_grad, kernel, padding=2).squeeze(1)
        mask_rand = F.conv2d(mask_rand, kernel, padding=2).squeeze(1)
        
        conditions = {
            "Full (Baseline)": torch.ones_like(mask_high_surv),
            "High-Survival (Top 20%)": mask_high_surv,
            "Low-Survival (Bottom 20%)": mask_low_surv,
            "High-Gradient (Top 20%)": mask_high_grad,
            "Low-Gradient (Bottom 20%)": mask_low_grad,
            "Random 20%": mask_rand
        }
        
        for cond_name, mask in conditions.items():
            masked_res_spec = res_spec * mask.to(device)
            masked_res_wav = istft_audio(masked_res_spec, length=wav.shape[-1], n_fft=256, hop_length=64)
            if wav.dim() == 3:
                masked_res_wav = masked_res_wav.unsqueeze(1)
            wav_masked = wav + masked_res_wav
            
            _, chunk_logits, _ = alignmark.decode(wav_masked)
            pred_bits = chunks_to_bits(chunk_logits.argmax(dim=-1), 4)
            
            for b in range(B):
                ber = compute_ber(pred_bits[b], msg[b])
                masking_results[cond_name].append(ber)
                
        processed_samples += B
        
        # 첫 배치에 대해서만 시각화 플롯 저장
        if not saved_plot:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title("Survival Map")
            plt.imshow(survival_map[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.title("Decoder Gradient Map")
            plt.imshow(decoder_map[0].cpu().detach().numpy(), aspect='auto', origin='lower', cmap='magma')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.title("High-Survival (Green) vs High-Gradient (Red)")
            rgb = np.zeros((*survival_map[0].shape, 3))
            rgb[..., 1] = mask_high_surv[0].cpu().numpy() * 0.7
            rgb[..., 0] = mask_high_grad[0].cpu().numpy() * 0.7
            rgb[..., 2] = mask_high_surv[0].cpu().numpy() * mask_high_grad[0].cpu().numpy()
            plt.imshow(rgb, aspect='auto', origin='lower')
            
            os.makedirs("results", exist_ok=True)
            plt.tight_layout()
            plt.savefig("results/phase1_map_comparison.png")
            plt.close()
            saved_plot = True

    print("\n" + "="*60)
    print(" Phase 1 Full Analysis Results ")
    print("="*60)
    
    print(f"[1] Correlation Metrics (N={len(all_pearson)})")
    r_mean, r_std = np.mean(all_pearson), np.std(all_pearson)
    rho_mean, rho_std = np.mean(all_spearman), np.std(all_spearman)
    iou_mean, iou_std = np.mean(all_iou), np.std(all_iou)
    print(f" - Pearson r:       {r_mean:.4f} ± {r_std:.4f}")
    print(f" - Spearman rho:    {rho_mean:.4f} ± {rho_std:.4f}")
    print(f" - Top-20% IoU:     {iou_mean:.4f} ± {iou_std:.4f}")
    
    print("\n[2] Masking Ablation (Causal Verification)")
    print("  Condition                     | Mean BER  | Bit Acc")
    print("  ------------------------------+-----------+----------")
    for cond_name, bers in masking_results.items():
        mean_ber = np.mean(bers)
        acc = 1.0 - mean_ber
        print(f"  {cond_name:<30}| {mean_ber:.4f}    | {acc:.4f}")
        
    t_stat, p_val = ttest_rel(masking_results["High-Survival (Top 20%)"], masking_results["Low-Survival (Bottom 20%)"])
    print(f"\n[3] Branching Decision (Paired t-test)")
    print(f" - High vs Low Survival BER p-value: {p_val:.4e}")
    if p_val < 0.05:
        print(" - Conclusion: SIGNIFICANT CAUSALITY DETECTED. Proceed to Phase 2.")
    else:
        print(" - Conclusion: NO SIGNIFICANT CAUSALITY. Branch to Phase 2B might lack justification.")
        
    # 파일에 기록
    summary_file = f"results/phase1_summary_{args.dataset_type}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset_type}\n")
        f.write(f"N={len(all_pearson)}\n")
        f.write(f"Pearson r: {r_mean:.4f} ± {r_std:.4f}\n")
        f.write(f"Spearman rho: {rho_mean:.4f} ± {rho_std:.4f}\n")
        f.write(f"Top-20% IoU: {iou_mean:.4f} ± {iou_std:.4f}\n\n")
        for cond_name, bers in masking_results.items():
            f.write(f"{cond_name}: Bit Acc {1.0 - np.mean(bers):.4f}\n")
            
        f.write(f"\nBranching p-value: {p_val:.4e}\n")
            
    print(f"\n[INFO] Results saved to {summary_file} and results/phase1_map_comparison.png")
    print("="*60)

if __name__ == "__main__":
    main()
