# SurvAlign-P Phase 2: Survival Gate Full Training & Evaluation Engine (본학습용)

## 개요 (Overview)
본 코드는 Phase 1의 발견(Survival Map의 강력한 가이드 능력)을 기반으로, 실제 워터마크 잔차(Residual)를 최적화하는 **Survival Gate**를 대용량 데이터로 학습하고 평가하는 **본학습용(Production-Ready)** 통합 엔진입니다.

연구 논문에 삽입할 수 있는 최종 결과를 도출하기 위해 고안되었으며, 훈련 손실을 기반으로 **가장 우수한 모델을 자동으로 저장**하고, 평가 결과를 **CSV 파일 형태로 자동 정리**합니다.

## 데이터셋
데이터셋으로 LibriSpeech의 `train-clean-100` (약 250시간)을 기본으로 사용합니다. 
스크립트를 실행하면 `RealLibriSpeechDataset`이 자동으로 해당 데이터셋의 존재 여부를 파악하고, 없을 경우 **다운로드를 진행**합니다. 

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
4. **`run_proposed_gate.bat`**: 제안하는 **Survival Map 기반 Gate** 모델 성능.

## 주요 기능 및 최적화 사항 (본학습용)
1. **Epoch 기반 Full Training**: 임의의 Step 기반이 아닌 정규 Epoch(기본 5 Epoch)를 돌며 `train-clean-100` 전체를 학습합니다.
2. **Best Model 자동 저장**: Epoch마다 Loss를 평가하여 가장 우수한 모델만 `checkpoints/best_gate_*.pth`에 갱신 및 저장합니다.
3. **엄밀한 평가 루프**: Test 셋(약 1300개 샘플) 전체에 대해 4가지 오디오 품질 지표(PESQ, STOI, SI-SDR, L2-Ratio)와 6가지 왜곡 환경의 강건성 지표(BER)를 평가합니다.
4. **CSV 자동 로깅**: 훈련이 끝날 때마다 결과 테이블이 `results/phase2_results.csv`에 한 줄씩 누적 저장됩니다. 이 CSV 파일을 엑셀이나 논문에 그대로 복사해 붙여넣으면 됩니다.
