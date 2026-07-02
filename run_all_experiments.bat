@echo off
echo =========================================================
echo SurvAlign-P Phase 2: All Experiments Runner
echo =========================================================

echo.
echo [1/4] Running Baseline...
python phase2_training.py --mode baseline --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo.
echo [2/4] Running Uniform Scaling...
python phase2_training.py --mode uniform --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo.
echo [3/4] Running Random Gate Training...
python phase2_training.py --mode random_gate --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo.
echo [4/5] Running Proposed Gate (Survival Map) Training...
python phase2_training.py --mode proposed_gate --map_type survival --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo.
echo [5/5] Running Proposed Gate (Gradient Map) Training...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_name train-clean-100 --epochs 5 --batch_size 8

echo.
echo =========================================================
echo All experiments completed!
echo Please check results/phase2_results.csv for the summary.
echo =========================================================
pause
