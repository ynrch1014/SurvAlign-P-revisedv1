@echo off
echo =========================================================
echo SurvAlign-P Phase 2: Evaluation-Only Runner (3 Datasets)
echo =========================================================
echo.

REM =========================================================
REM 1. LibriSpeech
REM =========================================================
echo ###################################################
echo # Dataset 1/3: LibriSpeech
echo ###################################################

echo [1/15] Testing LibriSpeech - Baseline...
python phase2_training.py --mode baseline --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [2/15] Testing LibriSpeech - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [3/15] Testing LibriSpeech - Random Gate...
python phase2_training.py --mode random_gate --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [4/15] Testing LibriSpeech - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [5/15] Testing LibriSpeech - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type librispeech --dataset_name train-clean-100 --test_only

REM =========================================================
REM 2. VCTK
REM =========================================================
echo.
echo ###################################################
echo # Dataset 2/3: VCTK
echo ###################################################

echo [6/15] Testing VCTK - Baseline...
python phase2_training.py --mode baseline --dataset_type vctk --test_only

echo [7/15] Testing VCTK - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type vctk --test_only

echo [8/15] Testing VCTK - Random Gate...
python phase2_training.py --mode random_gate --dataset_type vctk --test_only

echo [9/15] Testing VCTK - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type vctk --test_only

echo [10/15] Testing VCTK - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type vctk --test_only

REM =========================================================
REM 3. LJSpeech
REM =========================================================
echo.
echo ###################################################
echo # Dataset 3/3: LJSpeech
echo ###################################################

echo [11/15] Testing LJSpeech - Baseline...
python phase2_training.py --mode baseline --dataset_type ljspeech --test_only

echo [12/15] Testing LJSpeech - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type ljspeech --test_only

echo [13/15] Testing LJSpeech - Random Gate...
python phase2_training.py --mode random_gate --dataset_type ljspeech --test_only

echo [14/15] Testing LJSpeech - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type ljspeech --test_only

echo [15/15] Testing LJSpeech - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type ljspeech --test_only

echo.
echo =========================================================
echo All 15 evaluations completed!
echo Results are appended to results/phase2_results.csv
echo =========================================================
pause
