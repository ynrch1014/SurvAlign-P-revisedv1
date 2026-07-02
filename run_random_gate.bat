@echo off
echo Running Random Gate Training...
python phase2_training.py --mode random_gate --dataset_name train-clean-100 --epochs 5 --batch_size 8
pause
