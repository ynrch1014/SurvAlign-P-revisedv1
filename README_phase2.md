# SurvAlign-P Phase 2: Survival Gate Full Training & Evaluation Engine (본학습용)

## 개요 (Overview)
본 코드는 Phase 1의 발견(Survival Map의 강력한 가이드 능력)을 기반으로, 실제 워터마크 잔차(Residual)를 최적화하는 **Survival Gate**를 대용량 데이터로 학습하고 평가하는 **본학습용(Production-Ready)** 통합 엔진입니다. Phase 1의 이진(T/F) 마스킹과 달리, 여기서는 맵의 **연속적인 점수(Continuous Score)**를 가이드로 삼아 잔차를 정교하게 조율합니다.

연구 논문에 삽입할 수 있는 최종 결과를 도출하기 위해 고안되었으며, 훈련 손실을 기반으로 **가장 우수한 모델을 자동으로 저장**하고, 평가 결과를 **CSV 파일 형태로 자동 정리**합니다.

## 데이터셋 (Multi-Dataset Support)
`UnifiedSpeechDataset`을 통해 논문 실험 환경과 동일한 3종의 데이터셋을 완벽히 지원합니다:
1. **LibriSpeech** (`train-clean-100`): 기본 데이터셋 (약 250시간)
2. **VCTK**: 다화자 환경 강건성 검증용
3. **LJSpeech**: 단일 화자 고품질 환경 검증용

스크립트를 실행하면 설정된 데이터셋의 존재 여부를 파악하고, 없을 경우 **torchaudio/urllib을 통해 자동으로 다운로드를 진행**합니다. 화자(Speaker) 간 데이터 누수(Data Leakage)를 막기 위해 철저한 격리 분할(Speaker Disjoint Split)이 적용되어 있습니다.

## 제공되는 실행 스크립트 (.bat)
사용자의 편의를 위해 각 실험 시나리오별로 클릭 한 번에 실행 가능한 배치 스크립트(.bat)를 제공합니다.

### 전체 일괄 실행 (가장 권장)
4가지 시나리오를 순차적으로 모두 학습 및 평가하고 결과를 모아줍니다.
* **`run_all_experiments.bat`**

### 개별 시나리오 실행
특정 실험만 개별적으로 돌리고 싶을 때 사용합니다.
1. **`run_baseline.bat`**: 아무 조작도 가하지 않은 AlignMark 원본 성능 측정.
2. **`run_uniform.bat`**: Gate 스케일 증가량만큼 무작위로 증폭시켰을 때의 효과(Energy Control).
3. **`run_random_gate.bat`**: 무작위 맵(Random Map)을 가이드로 학습한 Gate 성능 (위치 정보의 효과 검증용).
4. **`run_energy_gate.bat`**: 스펙트로그램의 단순 소리 크기(Local Energy)에 비례해 증폭시키는 Baseline (순환논리 방어용 핵심 대조군).
5. **`run_proposed_gate.bat`**: 제안하는 **Survival Map 기반 Gate** 모델 성능.

## 주요 기능 및 최적화 사항 (본학습용)
1. **Epoch 기반 Full Training**: 임의의 Step 기반이 아닌 정규 Epoch(기본 3~5 Epoch)를 돌며 선택된 데이터셋 전체를 학습합니다.
2. **엄격한 L2 Waveform Projection (Hard Constraint)**: Gate를 통해 수정된 잔차의 에너지가 원본 잔차 에너지를 절대 초과하지 못하도록 투영(Projection) 제약을 가합니다. 이를 통해 **"단순히 신호 에너지를 증폭시켜 강건성을 높였다"**는 Energy Cheating 논란을 수학적/구조적으로 완벽히 차단합니다.
3. **Best Model 자동 저장**: Epoch마다 Loss를 평가하여 가장 우수한 모델만 `checkpoints/best_gate_*.pth`에 갱신 및 저장합니다.
4. **엄밀한 평가 루프**: Test 셋 전체에 대해 4가지 오디오 품질 지표(PESQ, STOI, SI-SDR, L2-Ratio)와 6가지 왜곡 환경의 강건성 지표(BER)를 평가합니다.
5. **CSV 자동 로깅**: 훈련이 끝날 때마다 결과 테이블이 `results/phase2_results.csv`에 한 줄씩 누적 저장됩니다. 이 CSV 파일을 엑셀이나 논문에 그대로 복사해 붙여넣으면 됩니다.
