# -*- coding: utf-8 -*-
"""Burst Error Analysis for SpeechTokenizer / Neural Codecs.

This script demonstrates empirically that codec attacks cause clustered (burst) 
errors in specific Time-Frequency (T-F) regions, rather than i.i.d random bit errors.
It extracts the physical Destruction Map of the codec and shows how our Survival Map
accurately predicts these regions, unlike the Decoder's Utility Map.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from phase1_attribution import compute_decoder_utility_map, _apply_internal_attack
from survalign_p import (
    AlignMarkManager,
    DifferentiableDistortion,
    UnifiedSpeechDataset,
    stft_audio,
    normalize_per_sample,
)
from experiment_utils import set_global_seed


def compute_pearson(map_a, map_b):
    """Compute Pearson correlation between two 2D tensors."""
    a_flat = map_a.detach().cpu().numpy().flatten()
    b_flat = map_b.detach().cpu().numpy().flatten()
    corr, _ = pearsonr(a_flat, b_flat)
    return corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_type", type=str, default="librispeech")
    parser.add_argument("--save_path", type=str, default="results/burst_error_heatmaps.png")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate for statistics.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device)

    print("Loading Models...")
    alignmark = AlignMarkManager(device=device, latent_mode="public_code")
    distorter = DifferentiableDistortion(sr=16000).to(device)
    
    print("Loading Sample Audio...")
    dataset = UnifiedSpeechDataset(dataset_type=args.dataset_type, split="test", download=False)
    
    if len(dataset) == 0:
        raise ValueError("Dataset empty! Check your data directory.")
        
    num_eval = min(args.num_samples, len(dataset))
    print(f"Evaluating on {num_eval} samples for statistical analysis...")
    
    corr_survival_list = []
    corr_utility_list = []
    
    import tqdm
    from survalign_p import get_survival_map

    print("Running evaluation...")
    for i in tqdm.tqdm(range(num_eval)):
        wav, _, _ = dataset[i]
        wav = wav.unsqueeze(0).to(device)  # (1, 1, T)

        # Generate random 16-bit message
        msg = torch.randint(0, 2, (1, 16), dtype=torch.float32, device=device)

        wav_wm, residual = alignmark.embed(wav, msg)
        
        utility_map = compute_decoder_utility_map(
            alignmark, wav, residual, msg, distorter, attack_names=["speechtokenizer_nq6"]
        )
        
        survival_map = get_survival_map(
            wav, wav_wm, distorter, attack_names=["speechtokenizer_nq6"], base_seed=args.seed + i
        )
        
        with torch.no_grad():
            wav_attacked = _apply_internal_attack(wav_wm, "speechtokenizer_nq6", distorter, args.seed + i)
        
        # Calculate Destruction Map (What the codec actually destroyed)
        spec_wm = torch.abs(stft_audio(wav_wm.squeeze(1)))
        spec_atk = torch.abs(stft_audio(wav_attacked.squeeze(1)))
        
        # Absolute destruction energy
        destruction_map = torch.abs(spec_wm - spec_atk)
        destruction_map = normalize_per_sample(destruction_map)
        
        # Reshape maps for plotting and analysis
        sm = survival_map[0].detach().cpu().numpy()
        um = utility_map[0].detach().cpu().numpy()
        dm = destruction_map[0].detach().cpu().numpy()

        # The survival map predicts where energy is KEPT. 
        # So 1 - Survival Map predicts where energy is DESTROYED.
        sm_inverse = 1.0 - sm
        
        corr_survival = compute_pearson(torch.tensor(sm_inverse), torch.tensor(dm))
        corr_utility = compute_pearson(torch.tensor(um), torch.tensor(dm))
        
        corr_survival_list.append(corr_survival)
        corr_utility_list.append(corr_utility)
        
        if i == 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Spectrogram (Original Watermarked)
            im0 = axes[0].imshow(np.log1p(spec_wm[0].detach().cpu().numpy()), origin='lower', aspect='auto', cmap='magma')
            axes[0].set_title("Watermarked Spectrogram")
            fig.colorbar(im0, ax=axes[0])
            
            # Actual Destruction Map (Ground Truth)
            im1 = axes[1].imshow(dm, origin='lower', aspect='auto', cmap='Reds')
            axes[1].set_title("Actual Codec Destruction Map\n(Burst Errors)")
            fig.colorbar(im1, ax=axes[1])
            
            # Predicted Destruction (1 - Survival Map)
            im2 = axes[2].imshow(sm_inverse, origin='lower', aspect='auto', cmap='Blues')
            axes[2].set_title(f"Predicted Destruction (Inverse Survival)\nCorr: {corr_survival:.3f}")
            fig.colorbar(im2, ax=axes[2])
            
            # Decoder Utility Map
            im3 = axes[3].imshow(um, origin='lower', aspect='auto', cmap='Greens')
            axes[3].set_title(f"Decoder Utility Map\nCorr: {corr_utility:.3f}")
            fig.colorbar(im3, ax=axes[3])
            
            plt.tight_layout()
            plt.savefig(args.save_path, dpi=300)
            print(f"\n✅ Plot for sample 0 saved to {args.save_path}")
            plt.close(fig)

    mean_survival = np.mean(corr_survival_list)
    std_survival = np.std(corr_survival_list)
    mean_utility = np.mean(corr_utility_list)
    std_utility = np.std(corr_utility_list)

    print("\n==================================================")
    print(f"[ Quantitative Results (N={num_eval}) ]")
    print(f"Correlation (Predicted Destruction via Survival Map vs Actual Destruction): {mean_survival:.4f} ± {std_survival:.4f}")
    print(f"Correlation (Decoder Utility vs Actual Destruction): {mean_utility:.4f} ± {std_utility:.4f}")
    print("==================================================")


if __name__ == "__main__":
    main()
