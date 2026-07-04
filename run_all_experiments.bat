@echo off
setlocal
set COMMON=--dataset_type librispeech --dataset_name train-clean-100 --epochs 5 --projection_mode equal --test_attacks ffmpeg_mp3 --strict_heldout

echo ============================================================
echo  SurvAlign-P Full Experiment Suite (strict held-out)
echo ============================================================

echo [1/8] Baseline (AlignMark original)
python phase2_training.py --mode baseline %COMMON% || goto :fail

echo [2/8] Uniform energy upper bound (1.1x, not fair comparison)
python phase2_training.py --mode uniform_upper %COMMON% || goto :fail

echo [3/8] Analytic Survival gate (decoder-free, no training)
python phase2_training.py --mode analytic_survival --map_type survival %COMMON% || goto :fail

echo [4/8] Constant-map gate (ablation: does any gate help?)
python phase2_training.py --mode constant_gate %COMMON% || goto :fail

echo [5/8] Random-map gate (ablation: is map information needed?)
python phase2_training.py --mode random_gate %COMMON% || goto :fail

echo [6/8] Shuffled-Survival gate (ablation: does spatial structure matter?)
python phase2_training.py --mode shuffled_survival %COMMON% || goto :fail

echo [7/8] Proposed Survival gate (main method)
python phase2_training.py --mode proposed_gate --map_type survival %COMMON% || goto :fail

echo [8/8] Codec-utility gate (decoder-derived alternative)
python phase2_training.py --mode proposed_gate --map_type codec_utility %COMMON% || goto :fail

echo ============================================================
echo  All experiments completed successfully.
echo  Check results\phase2\phase2_results_long.csv
echo ============================================================
pause
exit /b 0

:fail
echo ============================================================
echo  Experiment failed. See error above.
echo ============================================================
pause
exit /b 1
