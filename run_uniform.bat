@echo off
REM This is an energy-amplified upper-bound reference, not an equal-energy baseline.
python phase2_training.py --mode uniform_upper --dataset_type librispeech --dataset_name train-clean-100 --projection_mode equal --test_attacks ffmpeg_mp3 --strict_heldout
if errorlevel 1 exit /b 1
pause
