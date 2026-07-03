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

echo [1/18] Testing LibriSpeech - Baseline...
python phase2_training.py --mode baseline --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [2/18] Testing LibriSpeech - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [3/18] Testing LibriSpeech - Random Gate...
python phase2_training.py --mode random_gate --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [4/18] Testing LibriSpeech - Energy Gate (Baseline)...
python phase2_training.py --mode energy_gate --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [5/18] Testing LibriSpeech - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type librispeech --dataset_name train-clean-100 --test_only

echo [6/18] Testing LibriSpeech - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type librispeech --dataset_name train-clean-100 --test_only

REM =========================================================
REM 2. VCTK
REM =========================================================
echo.
echo ###################################################
echo # Dataset 2/3: VCTK
echo ###################################################

echo [7/18] Testing VCTK - Baseline...
python phase2_training.py --mode baseline --dataset_type vctk --test_only

echo [8/18] Testing VCTK - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type vctk --test_only

echo [9/18] Testing VCTK - Random Gate...
python phase2_training.py --mode random_gate --dataset_type vctk --test_only

echo [10/18] Testing VCTK - Energy Gate (Baseline)...
python phase2_training.py --mode energy_gate --dataset_type vctk --test_only

echo [11/18] Testing VCTK - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type vctk --test_only

echo [12/18] Testing VCTK - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type vctk --test_only

REM =========================================================
REM 3. LJSpeech
REM =========================================================
echo.
echo ###################################################
echo # Dataset 3/3: LJSpeech
echo ###################################################

echo [13/18] Testing LJSpeech - Baseline...
python phase2_training.py --mode baseline --dataset_type ljspeech --test_only

echo [14/18] Testing LJSpeech - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type ljspeech --test_only

echo [15/18] Testing LJSpeech - Random Gate...
python phase2_training.py --mode random_gate --dataset_type ljspeech --test_only

echo [16/18] Testing LJSpeech - Energy Gate (Baseline)...
python phase2_training.py --mode energy_gate --dataset_type ljspeech --test_only

echo [17/18] Testing LJSpeech - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type ljspeech --test_only

echo [18/18] Testing LJSpeech - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type ljspeech --test_only

echo.
echo =========================================================
echo All 18 evaluations completed!
echo Results are appended to results/phase2_results.csv
echo =========================================================
pause
