@echo off
echo Running Phase 1 Attribution Analysis (Full Test Set)...
python phase1_attribution.py --dataset_name train-clean-100 --split test
pause
