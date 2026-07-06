@echo off
REM ============================================================================
REM Extended Held-out Robustness Evaluation (Secondary Experiment)
REM
REM 실행 전제조건:
REM   1. 실험 D (verify_main_results_significance.py) 가 먼저 완료되어
REM      checkpoints/best_librispeech_proposed_gate_survival_seed{42,43,44}.pth
REM      와 baseline 체크포인트가 존재해야 합니다.
REM   2. FACodec/ClearerVoice 등 외부 코덱 wrapper 스크립트는 아직 없어도
REM      됩니다. 아래에서 커맨드를 비워두면 해당 공격만 자동으로 SKIP 됩니다.
REM
REM 사용법:
REM   run_extended_heldout_eval.bat            -> 내부 공격만 우선 실행
REM   (facodec/clearervoice wrapper 준비되면 아래 SET 값 채우고 재실행)
REM ============================================================================

REM --- 1) 외부 공격 래퍼 스크립트 연결 ---
SET FACODEC_CMD=python tools/run_facodec.py --input {input} --output {output}
SET CLEAREVOICE_CMD=python tools/run_clearervoice.py --input {input} --output {output}
SET ENCODEC_CMD=python tools/run_encodec.py --input {input} --output {output}
SET DAC_CMD=python tools/run_dac.py --input {input} --output {output}
SET VOCOS_CMD=python tools/run_vocos.py --input {input} --output {output}

REM --- 2) wrapper 준비가 끝나면 아래 예시처럼 채워 넣고 다시 실행하세요 ---
REM SET FACODEC_CMD=python tools/run_facodec.py --input {input} --output {output}
REM SET CLEAREVOICE_CMD=python tools/run_clearervoice.py --input {input} --output {output}
REM SET ENCODEC_CMD=python tools/run_encodec.py --input {input} --output {output}
REM SET DAC_CMD=python tools/run_dac.py --input {input} --output {output}
REM SET VOCOS_CMD=python tools/run_vocos.py --input {input} --output {output}

echo ============================================================
echo Step A: 내부 구현 공격 (외부 wrapper 불필요) 먼저 실행
echo   - strong_speechtokenizer
echo   - spectral_proxy
echo ============================================================
python verify_extended_heldout_robustness.py ^
    --seeds 42,43,44 ^
    --mode proposed_gate ^
    --map_type survival ^
    --dataset_type librispeech ^
    --dataset_name dev-clean ^
    --attacks strong_speechtokenizer,spectral_proxy ^
    --run_prefix ext_heldout_internal

echo.
echo ============================================================
echo Step B: 외부 코덱 held-out 공격 (wrapper 준비된 것만 실행됨)
echo   - facodec / clearervoice / encodec / dac / vocos
echo   (커맨드가 비어있는 항목은 자동으로 SKIP 됩니다)
echo ============================================================
python verify_extended_heldout_robustness.py ^
    --seeds 42,43,44 ^
    --mode proposed_gate ^
    --map_type survival ^
    --dataset_type librispeech ^
    --dataset_name dev-clean ^
    --attacks facodec,clearervoice,encodec,dac,vocos ^
    --facodec_command "%FACODEC_CMD%" ^
    --clearervoice_command "%CLEAREVOICE_CMD%" ^
    --encodec_command "%ENCODEC_CMD%" ^
    --dac_command "%DAC_CMD%" ^
    --vocos_command "%VOCOS_CMD%" ^
    --run_prefix ext_heldout_external

echo.
echo ============================================================
echo 완료. 위 두 단계의 요약 표를 확인하세요.
echo 각 wrapper 를 tools/run_*.py 로 준비한 뒤 SET 값을 채우고
echo Step B 만 다시 실행하면 됩니다.
echo ============================================================
pause
