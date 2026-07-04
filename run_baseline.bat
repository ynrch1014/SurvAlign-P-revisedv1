@echo off
python phase2_training.py --mode baseline --dataset_type librispeech --dataset_name train-clean-100 --projection_mode equal --test_attacks ffmpeg_mp3 --strict_heldout
if errorlevel 1 exit /b 1
pause
