# External held-out attack adapters

`phase1_attribution.py`와 `phase2_training.py`는 실제 ClearerVoice, FACodec, EnCodec, DAC, Vocos, HiFiGAN을 저장소에 직접 포함하지 않습니다. 각 모델의 공식 환경은 서로 충돌할 수 있으므로, **입력 WAV 한 개를 받아 출력 WAV 한 개를 생성하는 wrapper 명령**으로 연결합니다.

## 1. Wrapper contract

명령 문자열에는 `{input}`과 `{output}`이 모두 있어야 합니다.

```bash
python your_wrapper.py --input {input} --output {output}
```

Wrapper는 다음 조건을 지켜야 합니다.

- mono 또는 stereo WAV 입력을 받아야 함
- 출력 파일을 반드시 `{output}` 경로에 생성
- 출력 sample rate는 자유지만, 평가 코드가 16 kHz로 resample
- 오류 시 non-zero exit code 반환
- 모델 내부 randomness가 있다면 wrapper에서 seed를 고정하고 기록
- codec latency가 있다면 가능한 한 wrapper 안에서 제거하거나 별도 기록

Shell pipe나 복잡한 redirection 대신 Python wrapper 파일을 사용하는 것이 Windows/WSL/Linux 간 재현성이 좋습니다.

## 2. ClearerVoice

`clearervoice` attack은 wrapper 호출 전에 기본적으로 10 dB AWGN을 추가합니다. 이는 Feature-Aligned 논문의 denoising 조건을 재현하기 위한 것입니다.

```bash
python phase2_training.py \
  --mode proposed_gate --map_type survival --test_only \
  --load_weight checkpoints/best_gate.pth \
  --test_attacks clearervoice \
  --clearervoice_command "python tools/run_clearervoice.py --input {input} --output {output}" \
  --strict_heldout
```

Denoiser만 평가하려면 `clearervoice_only`를 사용합니다.

## 3. FACodec

`facodec`은 실제 Amphion/NaturalSpeech 3 FACodec wrapper를 사용합니다. 기존 `strong_speechtokenizer`는 FACodec이 아니며 proxy 결과로만 보고해야 합니다.

```bash
python phase2_training.py \
  --mode proposed_gate --map_type survival --test_only \
  --load_weight checkpoints/best_gate.pth \
  --test_attacks facodec \
  --facodec_command "python tools/run_facodec.py --input {input} --output {output}" \
  --strict_heldout
```

## 4. Other models

동일한 형식으로 다음 옵션을 지원합니다.

- `--encodec_command`
- `--dac_command`
- `--vocos_command`
- `--hifigan_command`

## 5. Real MP3

실제 MP3는 wrapper가 필요하지 않으며 PATH에 `ffmpeg`가 있어야 합니다.

```bash
python phase2_training.py --mode baseline \
  --test_attacks ffmpeg_mp3 --mp3_bitrate 64k
```

## 6. FAR protocol

ClearerVoice/FACodec 결과는 Bit Accuracy 외에 다음을 자동 저장합니다.

- 16-bit Exact-message Accuracy
- strict/lenient Attribution FAR
- tie rate
- true-message Hamming distance
- nearest-wrong-message Hamming distance
- attribution margin
- decoder CE와 logit margin
- baseline 실패 회복률과 기존 성공 퇴행률

`*_samples.csv`에는 각 샘플의 target/predicted bit string과 FAR 실패 여부가 저장됩니다. 전체 후보 집합 FAR가 주 결과이며, `--far_candidate_sizes 100,300,600`은 후보 수 민감도 진단용입니다.
