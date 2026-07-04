@echo off
setlocal
set COMMON=--dataset_type librispeech --dataset_name train-clean-100 --test_only --projection_mode equal --test_attacks ffmpeg_mp3 --strict_heldout
python phase2_training.py --mode random_gate %COMMON% || goto :fail
python phase2_training.py --mode constant_gate %COMMON% || goto :fail
python phase2_training.py --mode shuffled_survival %COMMON% || goto :fail
python phase2_training.py --mode proposed_gate --map_type survival %COMMON% || goto :fail
python phase2_training.py --mode proposed_gate --map_type codec_utility %COMMON% || goto :fail
pause
exit /b 0
:fail
pause
exit /b 1
