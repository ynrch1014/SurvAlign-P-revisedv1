# Watermark Repository Modification Report

## 1. 변경 범위

이번 수정은 기존 코드에서 확인된 논문 타당성, 재현성, 지표 계산, 예외 처리 및 문서 불일치를 반영했습니다. 주 실험 파이프라인은 다음 두 파일로 통일했습니다.

- `phase1_attribution.py`: 통제된 Phase 1 검증
- `phase2_training.py`: canonical Phase 2 학습·평가

`survalign_p.py`의 6채널 Gate/Presence 파이프라인은 legacy 실험으로 유지하되, 주요 버그와 명칭을 수정하고 실행 시 deprecation 경고를 추가했습니다.

---

## 2. Phase 1 수정사항

### 실험 타당성

- 마스크 적용 후 clean decoding만 하던 로직을 수정하여 **실제 평가 공격 후 복호**하도록 변경
- Survival Map 생성 공격과 평가 공격을 CLI에서 별도로 지정
- 자연적으로 남은 에너지와 equal-energy 조건을 분리
- 조건별 retained residual energy ratio 기록
- 다음 대조군 추가:
  - Residual-Energy Top-k
  - Speech-Energy Top-k
  - VAD Top-k
  - Codec-aware signed utility
  - Random Top-k 반복
- 정확한 `torch.topk` 기반 mask로 tie에 따른 선택 bin 수 증가 방지
- mask smoothing 기본값을 `none`으로 설정하고, `average`와 실제 `gaussian`을 선택 가능하게 변경
- 전체 T-F 평면이 아니라 residual support에서 상관분석 수행

### 통계·지표

- Survival vs Gradient, Utility, Residual-Energy, Random, Low-Survival의 직접 paired 비교
- Wilcoxon 및 sign-flip permutation p-value 저장
- Bit Accuracy 외에:
  - Exact-message Accuracy
  - strict/lenient Attribution FAR
  - tie rate
  - Hamming attribution margin
  - decoder CE
  - minimum logit margin
- 기존 `5-Fold`와 자동 branch decision 제거
- 결과를 sample-level CSV와 JSON으로 저장

---

## 3. Phase 2 수정사항

### 공격 프로토콜

- `survival_attacks`, `train_attacks`, `validation_attacks`, `test_attacks`를 명시적으로 분리
- 공격 이름뿐 아니라 SpeechTokenizer 설정처럼 동일 codec family가 겹쳐도 누출 경고
- `--strict_heldout` 사용 시 Map/Train/Validation과 Test 공격 누출을 즉시 오류 처리
- 기존 `mp3`를 `spectral_proxy`로 변경
- 기존 `facodec_proxy`를 `strong_speechtokenizer`로 변경
- 실제 MP3는 ffmpeg round-trip 어댑터 추가
- 실제 ClearerVoice/FACodec/EnCodec/DAC/Vocos/HiFiGAN은 외부 command template 어댑터 추가
- ClearerVoice 평가는 기본적으로 10 dB AWGN 후 denoising

### Gate와 대조군

- Gate 입력을 4채널로 변경:
  1. clean magnitude
  2. residual magnitude
  3. guide map
  4. local-energy masking proxy
- `gate_range`를 CLI로 제어 가능
- `loss_energy` 제거: hard L2 projection 이후 항상 0이던 dead loss
- masking exposure loss와 total-variation loss 추가
- L2 projection:
  - `cap`: 원본 residual 이하
  - `equal`: 정확히 동일 에너지 통제
- `uniform`을 `uniform_upper`로 재정의: residual 1.1배 증폭 참고 상한선
- 다음 ablation 추가:
  - analytic Survival gate
  - constant-map gate
  - shuffled-Survival gate
  - random/energy gate
  - gradient saliency / codec utility map

### Validation·체크포인트

- 사용되지 않던 calibration dataset을 실제 validation에 사용
- train loss가 아닌 validation Exact-message Accuracy와 CE로 best checkpoint 선택
- checkpoint에 config, validation score, epoch 저장
- checkpoint 로드 시 map type, gate range, projection, latent 경로, guide 공격 설정 호환성 검증
- validation SI-SDR 변화와 clipping 제약 옵션 추가
- checkpoint 부재 시 hard fail
- 새 Gate는 4채널이므로 기존 3채널 checkpoint와 호환되지 않음: 재학습 필요

### 평가 재현성

- Train은 random crop/message 유지
- Calibration/Test는 파일별 deterministic crop
- Test 메시지는 deterministic random unique 16-bit codebook
- sequential binary codebook을 사용하지 않아 작은 Hamming distance 편향 방지
- stochastic attack seed를 샘플/공격 기준으로 고정
- baseline과 proposed를 **같은 실행 안에서 paired 평가**
- sample-level 결과 저장

### 결과 지표

공격별로 다음을 계산합니다.

- Bit Accuracy / BER
- Exact-message Accuracy
- Attribution FAR strict/lenient
- tie rate
- mean/median attribution margin
- codebook minimum/nearest Hamming distance
- decoder CE, minimum/mean logit margin, entropy
- Recovery Rate / Regression Rate
- PESQ, STOI, SI-SDR
- L2 ratio, peak amplitude, clipping ratio

PESQ/STOI 패키지가 없거나 계산이 실패하면 최악값을 삽입하지 않고 `NaN`으로 기록하며, valid/failed count를 함께 저장합니다.

---

## 4. 데이터셋 수정사항

- 모든 파일 목록 정렬
- LibriSpeech/VCTK speaker-disjoint split
- LJSpeech는 단일 화자이므로 file-disjoint임을 명시
- Combined protocol을 분리:
  - `speaker_disjoint`
  - `paper` 파일 단위 200 test 방식
- metadata 반환 옵션 추가:
  - sample id
  - file path
  - speaker id
  - crop start
  - valid/original length
- evaluation split 메시지 codebook 고유성 보장

---

## 5. Distortion·Survival Map 수정사항

- 전역 RNG state를 조작하지 않고 device-local `torch.Generator` 사용
- paired AWGN에 seed 추가
- SpeechTokenizer가 없는데 reconstruction 요청 시 identity 반환하지 않고 예외 발생
- proxy 명칭 수정 및 예전 alias에 deprecation warning
- Survival Map 공격 목록과 lower quantile을 CLI에서 설정 가능
- codec 출력에 작은 latency가 있을 경우 integer cross-correlation alignment 옵션 적용
- 지연 보정 방향 오류를 수정하고 양·음 shift 단위 테스트 추가
- residual이 사실상 0인 noise-floor 영역을 score에서 제외
- 기존 `q_sir`를 conventional SIR로 주장하지 않고 residual dominance로 문서화

---

## 6. AlignMark 경로 수정사항

- 필수 asset/checkpoint 부재 시 hard fail
- inference 시 trainer-only dependency(`beartype`)가 없어도 SpeechTokenizer model import 가능
- embedding 출력 길이 명시적 정합 및 shape 검증
- chunk class tensor를 bit로 변환할 때 batch 차원을 chunk 차원으로 잘못 순회하던 `chunks_to_bits` 호환 버그 수정
- 표준 zero-mean SI-SDR 계산으로 정정하고 1D 입력 batch 해석을 검증
- `latent_mode` 추가:
  - `public_code`: 공개 코드의 quantized `encode()` 경로
  - `unquantized`: 논문 설명에 가까운 raw encoder latent 경로

두 경로의 성능이 같다고 가정하면 안 되며, baseline 재현을 통해 선택해야 합니다.

---

## 7. Legacy `survalign_p.py` 수정사항

- 지정 step 수보다 DataLoader가 짧을 때 한 epoch 후 조기 종료되던 반복 버그 수정
- batch BER 하나를 CI 표본으로 사용하던 문제를 sample-level BER로 수정
- PESQ/STOI/sklearn optional dependency guard 추가
- spectral proxy 명칭 수정
- canonical pipeline 사용을 권고하는 deprecation warning 추가

---

## 8. 문서·실행 스크립트 수정

- README의 과도한 표현과 실제 코드 불일치 제거
- decoder-free 주장을 Survival prior 생성으로 제한
- actual MP3/FACodec/ClearerVoice와 proxy 구분
- LJSpeech speaker-disjoint 주장 삭제
- clean decoding accuracy와 음질을 구분
- codec backward가 identity STE임을 명시
- `.bat` 실행 옵션을 새 mode/map 이름에 맞게 수정
- `requirements-survalign.txt` 추가

---

## 9. 자동 점검 결과

수행한 점검:

- 수정 Python 파일 전체 `py_compile` 통과
- CLI `--help` 실행 통과
- deterministic dataset crop/message 테스트 통과
- unique random message codebook 테스트 통과
- Survival Map synthetic forward 테스트 통과
- 모든 Gate mode의 mock forward shape 테스트 통과
- Phase 2 mock end-to-end FAR/CSV/JSON 평가 테스트 통과
- codec integer-shift 보정 및 chunk-to-bit round-trip 테스트 통과
- 실제 ffmpeg MP3 round-trip 테스트 통과
- attribution FAR utility unit test 통과

실제 수치 재현은 다음 asset이 없어 수행하지 못했습니다.

- AlignMark `weight.pth`
- SpeechTokenizer pretrained weight
- 실제 학습 데이터
- 공식 ClearerVoice/FACodec 실행 환경

---

## 10. 사용자가 준비해야 하는 부분

1. 기존 Gate checkpoint는 폐기하고 새 4채널 Gate를 재학습
2. 실제 ClearerVoice/FACodec을 WAV input/output으로 감싸는 wrapper script 준비 (`EXTERNAL_ATTACKS.md`)
3. Map/Train/Validation/Test 공격 프로토콜을 실험 전에 고정
4. `public_code`와 `unquantized` 중 원 논문 baseline을 재현하는 경로 확인
5. PESQ/STOI 외에 ViSQOL 또는 ABX 청취평가 추가 권장
6. Phase 1 결과는 exploratory dataset과 confirmatory dataset을 분리해 보고
