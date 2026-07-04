import os
import subprocess
import csv
from scipy import stats
import numpy as np

seeds = [42, 43, 44]
mode = "proposed_gate"
map_type = "survival"
dataset_name = "dev-clean"

print("Starting Multi-Seed Evaluation for Statistical Significance...")

for s in seeds:
    print(f"\n[Running Seed {s}]")
    cmd = [
        "python", "phase2_training.py",
        "--mode", mode,
        "--map_type", map_type,
        "--dataset_type", "librispeech",
        "--dataset_name", dataset_name,
        "--epochs", "5",
        "--projection_mode", "equal",
        "--train_attacks", "noise,lowpass,resample,reconstruct_nq6",
        "--validation_attacks", "bandpass,reconstruct_nq8",
        "--test_attacks", "ffmpeg_mp3",
        "--strict_heldout",
        "--seed", str(s),
        "--run_id", f"stat_sig_seed_{s}"
    ]
    subprocess.run(cmd, check=True)

print("\nAll seeds completed. Gathering results and performing Paired T-test...")

long_path = "results/phase2/phase2_results_long.csv"

run_ids = set([f"stat_sig_seed_{s}" for s in seeds])

print("\n--- RESULTS SUMMARY ---")
attack = "ffmpeg_mp3"
print(f"Attack: {attack}")

baseline_accs = []
method_accs = []
baseline_exacts = []
method_exacts = []

if os.path.exists(long_path):
    with open(long_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["run_id"] in run_ids and row["attack"] == attack:
                sys = row["system"]
                acc = float(row["bit_accuracy"])
                exact = float(row["exact_message_accuracy"])
                
                if sys == "baseline":
                    baseline_accs.append(acc)
                    baseline_exacts.append(exact)
                elif sys == "method":
                    method_accs.append(acc)
                    method_exacts.append(exact)

b_acc_mean, b_acc_std = np.mean(baseline_accs)*100, np.std(baseline_accs)*100
m_acc_mean, m_acc_std = np.mean(method_accs)*100, np.std(method_accs)*100
b_exact_mean, b_exact_std = np.mean(baseline_exacts)*100, np.std(baseline_exacts)*100
m_exact_mean, m_exact_std = np.mean(method_exacts)*100, np.std(method_exacts)*100

print(f"Bit Accuracy   | Baseline: {b_acc_mean:.2f}% +- {b_acc_std:.2f}%  | Proposed: {m_acc_mean:.2f}% +- {m_acc_std:.2f}%")
print(f"Exact Match    | Baseline: {b_exact_mean:.2f}% +- {b_exact_std:.2f}%  | Proposed: {m_exact_mean:.2f}% +- {m_exact_std:.2f}%")

print("\n--- PAIRED T-TEST (Sample-level) ---")
all_b_exact = []
all_m_exact = []

for s in seeds:
    stem = f"stat_sig_seed_{s}_librispeech_{mode}_{map_type}"
    sample_path = f"results/phase2/{stem}_samples.csv"
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["attack"] == attack:
                    sys = row["system"]
                    ex = float(row["exact"])
                    if sys == "baseline":
                        all_b_exact.append(ex)
                    elif sys == "method":
                        all_m_exact.append(ex)

if len(all_b_exact) > 0:
    t_stat, p_val = stats.ttest_rel(all_m_exact, all_b_exact)
    print(f"Total samples evaluated across seeds: {len(all_b_exact)}")
    print(f"Paired t-test t-statistic: {t_stat:.4f}")
    print(f"Paired t-test p-value: {p_val:.2e}")
    if p_val < 0.05:
        print("SIGNIFICANT: The proposed method statistically significantly improves Exact Match Accuracy.")
    else:
        print("NOT SIGNIFICANT at alpha=0.05.")
else:
    print("Could not find sample files for t-test.")
