@echo off
echo =========================================================
echo SurvAlign-P Phase 2: Evaluation-Only Runner
echo =========================================================

echo.
echo [1/5] Testing Baseline...
python phase2_training.py --mode baseline --dataset_name train-clean-100 --test_only

echo.
echo [2/5] Testing Uniform Scaling...
python phase2_training.py --mode uniform --dataset_name train-clean-100 --test_only

echo.
echo [3/5] Testing Random Gate...
python phase2_training.py --mode random_gate --dataset_name train-clean-100 --test_only

echo.
echo [4/5] Testing Proposed Gate (Survival Map)...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_name train-clean-100 --test_only

echo.
echo [5/5] Testing Proposed Gate (Gradient Map)...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_name train-clean-100 --test_only

echo.
echo =========================================================
echo All evaluations completed!
echo Results are appended to results/phase2_results.csv
echo =========================================================
pause
