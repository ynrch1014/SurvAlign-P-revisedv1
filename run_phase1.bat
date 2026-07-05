@echo off
setlocal
python phase1_attribution.py ^
  --dataset_type librispeech ^
  --dataset_name train-clean-100 ^
  --split test ^
  --survival_attacks speechtokenizer_nq6,speechtokenizer_nq8,spectral_proxy ^
  --utility_attacks speechtokenizer_nq6,strong_speechtokenizer ^
  --eval_attacks clean,bandpass,ffmpeg_mp3 ^
  --strict_heldout ^
  --energy_modes natural,equal ^
  --random_repeats 20
if errorlevel 1 exit /b 1
pause
