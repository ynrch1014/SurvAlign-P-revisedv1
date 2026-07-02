@echo off
echo Running Uniform Scaling...
python phase2_training.py --mode uniform --dataset_name train-clean-100 --epochs 5 --batch_size 8
pause
