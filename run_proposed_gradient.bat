@echo off
echo Running Proposed Gate (Gradient Map) Training...
python phase2_training.py --mode proposed_gate --map_type gradient --dataset_name train-clean-100 --epochs 5 --batch_size 8
pause
