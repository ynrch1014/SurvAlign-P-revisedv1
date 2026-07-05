import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from survalign_p import AlignMarkManager, RealLibriSpeechDataset, UnifiedSpeechDataset
from experiment_utils import deterministic_unique_message, set_global_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate watermark detection specificity")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of audio samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset_type", type=str, default="librispeech", help="Dataset to use")
    parser.add_argument("--dataset_name", type=str, default="dev-clean", help="Split name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--target_fprs", type=str, default="0.001,0.01,0.05", help="Comma-separated target FPRs")
    return parser.parse_args()

def compute_confidence(chunk_logits):
    """
    Compute confidence metrics (Top-1 margin and Negative Entropy) from logits without knowing the target.
    chunk_logits: (B, n_chunks=4, n_classes=16)
    Returns:
        margin: (B,) Mean Top-1 vs Top-2 margin over chunks
        neg_entropy: (B,) Mean Negative Entropy over chunks (higher = more confident)
    """
    probs = torch.softmax(chunk_logits, dim=-1)
    log_probs = torch.log_softmax(chunk_logits, dim=-1)
    
    # Entropy calculation
    entropy = -(probs * log_probs).sum(dim=-1) # (B, n_chunks)
    mean_entropy = entropy.mean(dim=-1) # (B,)
    neg_entropy = -mean_entropy
    
    # Top-1 Margin calculation
    top2_logits = torch.topk(chunk_logits, k=2, dim=-1).values # (B, n_chunks, 2)
    top1 = top2_logits[:, :, 0]
    top2 = top2_logits[:, :, 1]
    margin = (top1 - top2).mean(dim=-1) # (B,)
    
    return margin, neg_entropy

def find_threshold_and_recall(y_true, y_score, target_fpr):
    """Find operating threshold for a target FPR and return the TPR (Recall)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find the largest threshold where FPR <= target_fpr
    # Since thresholds are sorted descending, we want the first one where fpr <= target_fpr
    idx = np.where(fpr <= target_fpr)[0][-1]
    return thresholds[idx], tpr[idx], fpr[idx]

def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load AlignMark Baseline Model
    print("Loading AlignMark Manager...")
    manager = AlignMarkManager(device=device, latent_mode="public_code")

    # Load Dataset
    print(f"Loading {args.dataset_type}/{args.dataset_name} dataset...")
    # Note: Using UnifiedSpeechDataset logic.
    dataset = UnifiedSpeechDataset(
        dataset_type=args.dataset_type,
        dataset_name=args.dataset_name,
        download=True,
        segment_len=32000,
        split="test" # use test split or val split to mimic eval
    )
    num_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating on {num_samples} samples.")
    
    # Subset the dataset
    subset_indices = list(range(num_samples))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    clean_margins = []
    clean_neg_entropies = []
    clean_frame_scores = []
    
    watermarked_margins = []
    watermarked_neg_entropies = []
    watermarked_frame_scores = []

    print("Running inference...")
    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            wav = batch[0].to(device)
            bsz = wav.shape[0]
            
            # Create messages for watermarking
            messages = []
            for i in range(bsz):
                messages.append(deterministic_unique_message(sample_idx + i, offset=1000))
            msg_tensor = torch.stack(messages).to(device)
            sample_idx += bsz

            # 1. Clean Audio Path
            frame_logits_clean, chunk_logits_clean, _ = manager.decode(wav)
            margin_clean, neg_ent_clean = compute_confidence(chunk_logits_clean)
            frame_score_clean = frame_logits_clean.mean(dim=1)
            
            clean_margins.extend(margin_clean.cpu().numpy().tolist())
            clean_neg_entropies.extend(neg_ent_clean.cpu().numpy().tolist())
            clean_frame_scores.extend(frame_score_clean.cpu().numpy().tolist())
            
            # 2. Watermarked Audio Path
            wav_wm, _ = manager.embed(wav, msg_tensor)
            frame_logits_wm, chunk_logits_wm, _ = manager.decode(wav_wm)
            margin_wm, neg_ent_wm = compute_confidence(chunk_logits_wm)
            frame_score_wm = frame_logits_wm.mean(dim=1)
            
            watermarked_margins.extend(margin_wm.cpu().numpy().tolist())
            watermarked_neg_entropies.extend(neg_ent_wm.cpu().numpy().tolist())
            watermarked_frame_scores.extend(frame_score_wm.cpu().numpy().tolist())

    y_true = np.array([0] * num_samples + [1] * num_samples)
    y_margin = np.array(clean_margins + watermarked_margins)
    y_neg_ent = np.array(clean_neg_entropies + watermarked_neg_entropies)
    y_frame = np.array(clean_frame_scores + watermarked_frame_scores)

    # 1. Calculate AUC
    auc_margin = roc_auc_score(y_true, y_margin)
    auc_neg_ent = roc_auc_score(y_true, y_neg_ent)
    auc_frame = roc_auc_score(y_true, y_frame)
    
    print("\n" + "="*50)
    print("DETECTION SPECIFICITY RESULTS")
    print("="*50)
    print(f"ROC-AUC (Logit Margin): {auc_margin:.4f}")
    print(f"ROC-AUC (-Entropy):     {auc_neg_ent:.4f}")
    print(f"ROC-AUC (Frame Logits): {auc_frame:.4f}")
    
    target_fprs = [float(f) for f in args.target_fprs.split(",")]
    
    results = {
        "num_samples": num_samples,
        "auc_margin": float(auc_margin),
        "auc_neg_entropy": float(auc_neg_ent),
        "auc_frame_logits": float(auc_frame),
        "thresholds_margin": {},
        "thresholds_neg_entropy": {},
        "thresholds_frame_logits": {}
    }
    
    print("\n--- Operating Thresholds (Logit Margin) ---")
    for t_fpr in target_fprs:
        thresh, tpr, actual_fpr = find_threshold_and_recall(y_true, y_margin, t_fpr)
        print(f"Target FPR: {t_fpr*100:5.2f}% | Thresh: {thresh:6.2f} | Actual FPR: {actual_fpr*100:5.2f}% | TPR (Recall): {tpr*100:5.2f}%")
        results["thresholds_margin"][f"fpr_{t_fpr}"] = {
            "threshold": float(thresh),
            "actual_fpr": float(actual_fpr),
            "tpr": float(tpr)
        }

    print("\n--- Operating Thresholds (-Entropy) ---")
    for t_fpr in target_fprs:
        thresh, tpr, actual_fpr = find_threshold_and_recall(y_true, y_neg_ent, t_fpr)
        print(f"Target FPR: {t_fpr*100:5.2f}% | Thresh: {thresh:6.2f} | Actual FPR: {actual_fpr*100:5.2f}% | TPR (Recall): {tpr*100:5.2f}%")
        results["thresholds_neg_entropy"][f"fpr_{t_fpr}"] = {
            "threshold": float(thresh),
            "actual_fpr": float(actual_fpr),
            "tpr": float(tpr)
        }

    print("\n--- Operating Thresholds (Frame Logits) ---")
    for t_fpr in target_fprs:
        thresh, tpr, actual_fpr = find_threshold_and_recall(y_true, y_frame, t_fpr)
        print(f"Target FPR: {t_fpr*100:5.2f}% | Thresh: {thresh:6.2f} | Actual FPR: {actual_fpr*100:5.2f}% | TPR (Recall): {tpr*100:5.2f}%")
        results["thresholds_frame_logits"][f"fpr_{t_fpr}"] = {
            "threshold": float(thresh),
            "actual_fpr": float(actual_fpr),
            "tpr": float(tpr)
        }

    print("\n--- Compound FAR Logic ---")
    print("Compound FAR = Detection FPR * Conditional Decoding FAR")
    print("Example: If Conditional FAR = 1e-4 and Detection FPR = 0.01 (1%),")
    print("Then the true operational False Positive Rate is 1e-6.")
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/detection_specificity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {out_path}\n")

if __name__ == "__main__":
    main()
