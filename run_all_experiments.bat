@echo off
echo =========================================================
echo SurvAlign-P Phase 2: All Experiments Runner (3 Datasets)
echo =========================================================
echo.
echo This script runs all 5 modes x 3 datasets = 15 experiments.
echo Datasets: LibriSpeech, VCTK, LJSpeech
echo.

REM =========================================================
REM 1. LibriSpeech (기존 기본 데이터셋)
REM =========================================================
echo ###################################################
echo # Dataset 1/3: LibriSpeech (train-clean-100)
echo ###################################################

echo [1/15] LibriSpeech - Baseline...
python phase2_training.py --mode baseline --dataset_type librispeech --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo [2/15] LibriSpeech - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type librispeech --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo [3/15] LibriSpeech - Random Gate...
python phase2_training.py --mode random_gate --dataset_type librispeech --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo [4/15] LibriSpeech - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type librispeech --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo [5/15] LibriSpeech - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type librispeech --dataset_name train-clean-100 --epochs 5 --batch_size 8

REM =========================================================
REM 2. VCTK (다화자 데이터셋)
REM =========================================================
echo.
echo ###################################################
echo # Dataset 2/3: VCTK
echo ###################################################

echo [6/15] VCTK - Baseline...
python phase2_training.py --mode baseline --dataset_type vctk --epochs 5 --batch_size 8

echo [7/15] VCTK - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type vctk --epochs 5 --batch_size 8

echo [8/15] VCTK - Random Gate...
python phase2_training.py --mode random_gate --dataset_type vctk --epochs 5 --batch_size 8

echo [9/15] VCTK - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type vctk --epochs 5 --batch_size 8

echo [10/15] VCTK - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type vctk --epochs 5 --batch_size 8

REM =========================================================
REM 3. LJSpeech (단일 화자 데이터셋)
REM =========================================================
echo.
echo ###################################################
echo # Dataset 3/3: LJSpeech
echo ###################################################

echo [11/15] LJSpeech - Baseline...
python phase2_training.py --mode baseline --dataset_type ljspeech --epochs 5 --batch_size 8

echo [12/15] LJSpeech - Uniform Scaling...
python phase2_training.py --mode uniform --dataset_type ljspeech --epochs 5 --batch_size 8

echo [13/15] LJSpeech - Random Gate...
python phase2_training.py --mode random_gate --dataset_type ljspeech --epochs 5 --batch_size 8

echo [14/15] LJSpeech - Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_type ljspeech --epochs 5 --batch_size 8

echo [15/15] LJSpeech - Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_type ljspeech --epochs 5 --batch_size 8

echo.
echo =========================================================
echo All 15 experiments completed!
echo Please check results/phase2_results.csv for the summary.
echo =========================================================
pause
